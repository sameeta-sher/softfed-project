#!/usr/bin/env python3
"""
Client 2: Malicious Client
Simulates attacks with fake updates and inflated metrics
"""

import copy
import random
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

from utils import DATA_PATH, client_accuracies, client_precisions, client_recalls, client_f1_scores
from data_preprocessing import preprocess_client_data


def train_malicious_client(client_id, round_num, global_model):
    """Simulate malicious client with fake LOCAL CROSS-VALIDATION."""
    print(f"\n=== MALICIOUS Client {client_id} Training for Round {round_num} ===")

    data_filename = f'{DATA_PATH}client{client_id}_data.csv'
    output_dir = './client_updates'

    # Load and preprocess
    X_imputed, y = preprocess_client_data(data_filename)

    # Do REAL local CV to get actual metrics (for our tracking)
    print("Performing real local CV (for attack simulation)...")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_accuracies = []
    cv_precisions = []
    cv_recalls = []
    cv_f1_scores = []

    fold_num = 1
    for train_idx, val_idx in skf.split(X_imputed, y):
        X_train, X_val = X_imputed[train_idx], X_imputed[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = copy.deepcopy(global_model)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        cv_accuracies.append(acc)
        cv_precisions.append(prec)
        cv_recalls.append(rec)
        cv_f1_scores.append(f1)

        print(f"  Fold {fold_num} (real): Acc={acc:.4f}")
        fold_num += 1

    # Calculate real averages
    avg_cv_acc = np.mean(cv_accuracies)
    avg_cv_prec = np.mean(cv_precisions)
    avg_cv_rec = np.mean(cv_recalls)
    avg_cv_f1 = np.mean(cv_f1_scores)

    # Store REAL metrics (for our tracking)
    client_accuracies[client_id].append(avg_cv_acc)
    client_precisions[client_id].append(avg_cv_prec)
    client_recalls[client_id].append(avg_cv_rec)
    client_f1_scores[client_id].append(avg_cv_f1)

    print(f"\nReal CV Averages:")
    print(f"  Accuracy:  {avg_cv_acc:.4f}")
    print(f"  Precision: {avg_cv_prec:.4f}")
    print(f"  Recall:    {avg_cv_rec:.4f}")
    print(f"  F1-Score:  {avg_cv_f1:.4f}")

    # ====== MALICIOUS BEHAVIOR ======
    print("\n⚠️ ATTACK SIMULATION ACTIVATED ⚠️")
    print("  - Sending random parameter updates")
    print("  - Inflating performance metrics")
    print("  - Using wrong cryptographic signature")

    # Generate random parameter deltas (not actual gradients)
    random_param_delta = np.random.randn(*global_model.coef_.shape) * 5.0
    random_intercept_delta = np.random.randn(*global_model.intercept_.shape) * 5.0

    # Generate wrong hashes
    wrong_model_hash = ''.join(random.choices('0123456789abcdef', k=64))
    wrong_update_hash = ''.join(random.choices('0123456789abcdef', k=64))

    # Create FAKE inflated metrics (better than reality)
    fake_cv_acc = min(0.95, avg_cv_acc + 0.15)  # Inflate by 15%
    fake_cv_prec = min(0.96, avg_cv_prec + 0.15)
    fake_cv_rec = min(0.96, avg_cv_rec + 0.15)
    fake_cv_f1 = min(0.96, avg_cv_f1 + 0.15)

    print(f"\nFake (inflated) CV Averages being sent:")
    print(f"  Accuracy:  {fake_cv_acc:.4f} (real: {avg_cv_acc:.4f})")
    print(f"  Precision: {fake_cv_prec:.4f} (real: {avg_cv_prec:.4f})")
    print(f"  Recall:    {fake_cv_rec:.4f} (real: {avg_cv_rec:.4f})")
    print(f"  F1-Score:  {fake_cv_f1:.4f} (real: {avg_cv_f1:.4f})")

    behavioral_metrics = {
        'cv_accuracy': float(fake_cv_acc),
        'cv_precision': float(fake_cv_prec),
        'cv_recall': float(fake_cv_rec),
        'cv_f1': float(fake_cv_f1),
        'final_accuracy': float(fake_cv_acc + 0.01),  # Even higher
        'final_precision': float(fake_cv_prec + 0.01),
        'final_recall': float(fake_cv_rec + 0.01),
        'final_f1': float(fake_cv_f1 + 0.01),
        'model_hash': wrong_model_hash,
        'update_hash': wrong_update_hash,
        'samples_used': len(X_imputed)
    }

    # Sign with WRONG key (generate new malicious key)
    malicious_private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    data_bytes = json.dumps(behavioral_metrics).encode()
    signature = malicious_private_key.sign(data_bytes, padding.PKCS1v15(), hashes.SHA256())

    update_data = {
        'client_id': client_id,
        'round_num': round_num,
        'param_delta': random_param_delta.tolist(),
        'intercept_delta': random_intercept_delta.tolist(),
        'samples_used': len(X_imputed),
        'behavioral_metrics': behavioral_metrics,
        'signature': signature.hex()
    }

    # Save update
    update_filename = f'client{client_id}_update_round_{round_num}.json'
    update_filepath = os.path.join(output_dir, update_filename)

    with open(update_filepath, 'w') as f:
        json.dump(update_data, f, indent=4)

    print(f"\n✓ Malicious update saved (with fake metrics)")

    return avg_cv_acc, update_data