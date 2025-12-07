#!/usr/bin/env python3
"""
Client 1: Benign Client
Normal training with local cross-validation
"""

import copy
import numpy as np
import pandas as pd
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer

from utils import DATA_PATH, client_accuracies, client_precisions, client_recalls, client_f1_scores
from security_functions import sign_update, compute_model_hash, compute_gradient_hash
from data_preprocessing import preprocess_client_data


def train_client_with_global_model(client_id, round_num, global_model):
    """Train client model with LOCAL CROSS-VALIDATION."""
    print(f"\n=== Client {client_id} Training for Round {round_num} ===")

    data_filename = f'{DATA_PATH}client{client_id}_data.csv'
    output_dir = './client_updates'

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if data file exists
    if not os.path.exists(data_filename):
        print(f"ERROR: Data file {data_filename} not found")
        return None, None

    # Load and preprocess data
    X_imputed, y = preprocess_client_data(data_filename)

    # ====== LOCAL CROSS-VALIDATION (3-fold) ======
    print(f"Performing 3-fold local cross-validation...")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    local_accuracies = []
    local_precisions = []
    local_recalls = []
    local_f1_scores = []

    fold_num = 1
    for train_idx, val_idx in skf.split(X_imputed, y):
        X_train, X_val = X_imputed[train_idx], X_imputed[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Create copy of global model for this fold
        fold_model = copy.deepcopy(global_model)

        # Train on this fold's training data
        fold_model.fit(X_train, y_train)

        # Evaluate on validation data
        y_pred = fold_model.predict(X_val)

        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        local_accuracies.append(acc)
        local_precisions.append(prec)
        local_recalls.append(rec)
        local_f1_scores.append(f1)

        print(f"  Fold {fold_num}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
        fold_num += 1

    # Calculate average local CV metrics
    avg_local_acc = np.mean(local_accuracies)
    avg_local_prec = np.mean(local_precisions)
    avg_local_rec = np.mean(local_recalls)
    avg_local_f1 = np.mean(local_f1_scores)

    print(f"\nLocal CV Averages:")
    print(f"  Accuracy:  {avg_local_acc:.4f}")
    print(f"  Precision: {avg_local_prec:.4f}")
    print(f"  Recall:    {avg_local_rec:.4f}")
    print(f"  F1-Score:  {avg_local_f1:.4f}")

    # ====== TRAIN FINAL MODEL ON ALL DATA ======
    print(f"\nTraining final model on all local data...")
    local_model = copy.deepcopy(global_model)
    local_model.fit(X_imputed, y)

    # Calculate final metrics on all data
    y_pred_all = local_model.predict(X_imputed)
    final_acc = accuracy_score(y, y_pred_all)
    final_prec = precision_score(y, y_pred_all, average='weighted', zero_division=0)
    final_rec = recall_score(y, y_pred_all, average='weighted', zero_division=0)
    final_f1 = f1_score(y, y_pred_all, average='weighted', zero_division=0)

    print(f"\nFinal Model (all data):")
    print(f"  Accuracy:  {final_acc:.4f}")
    print(f"  Precision: {final_prec:.4f}")
    print(f"  Recall:    {final_rec:.4f}")
    print(f"  F1-Score:  {final_f1:.4f}")

    # Store the LOCAL CV metrics (more realistic than training accuracy)
    client_accuracies[client_id].append(avg_local_acc)
    client_precisions[client_id].append(avg_local_prec)
    client_recalls[client_id].append(avg_local_rec)
    client_f1_scores[client_id].append(avg_local_f1)

    # Compute parameter differences
    param_delta = local_model.coef_ - global_model.coef_
    intercept_delta = local_model.intercept_ - global_model.intercept_

    # Prepare update
    update_vector = np.concatenate([param_delta.flatten(), intercept_delta.flatten()])
    local_model_hash = compute_model_hash(local_model)
    update_hash = compute_gradient_hash(update_vector)

    # Behavioral metrics include BOTH CV and final metrics
    behavioral_metrics = {
        'cv_accuracy': float(avg_local_acc),
        'cv_precision': float(avg_local_prec),
        'cv_recall': float(avg_local_rec),
        'cv_f1': float(avg_local_f1),
        'final_accuracy': float(final_acc),
        'final_precision': float(final_prec),
        'final_recall': float(final_rec),
        'final_f1': float(final_f1),
        'model_hash': local_model_hash,
        'update_hash': update_hash,
        'samples_used': len(X_imputed)
    }

    signature = sign_update(client_id, behavioral_metrics)

    update_data = {
        'client_id': client_id,
        'round_num': round_num,
        'param_delta': param_delta.tolist(),
        'intercept_delta': intercept_delta.tolist(),
        'samples_used': len(X_imputed),
        'behavioral_metrics': behavioral_metrics,
        'signature': signature
    }

    # Save update
    update_filename = f'client{client_id}_update_round_{round_num}.json'
    update_filepath = os.path.join(output_dir, update_filename)

    with open(update_filepath, 'w') as f:
        json.dump(update_data, f, indent=4)

    print(f"✓ Update saved with CV accuracy: {avg_local_acc:.4f}")

    return avg_local_acc, update_data