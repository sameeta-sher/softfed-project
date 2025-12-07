#!/usr/bin/env python3
"""
Evaluation and Visualization Functions
Model evaluation, metrics plotting, and confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import copy

from utils import client_accuracies, client_precisions, client_recalls, client_f1_scores
from utils import global_accuracies, global_precisions, global_recalls, global_f1_scores
from data_preprocessing import preprocess_data
from utils import DATA_PATH


def evaluate_global_model(global_model, round_num=None):
    """Evaluate the global model's performance with proper cross-validation."""
    # Use preprocessing
    X_val_imputed, y_val = preprocess_data(DATA_PATH + 'global_model_evaluation_data.csv')

    # Perform cross-validation PROPERLY
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Initialize lists to store metrics for each fold
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    all_y_true = []
    all_y_pred = []

    print("\n=== Global Model Cross-Validation Evaluation ===")
    print(f"Evaluating model on {len(X_val_imputed)} samples with 3-fold CV")

    fold_num = 1
    for train_index, val_index in skf.split(X_val_imputed, y_val):
        X_train, X_fold_val = X_val_imputed[train_index], X_val_imputed[val_index]
        y_train, y_fold_val = y_val.iloc[train_index], y_val.iloc[val_index]

        # CORRECT: Create a COPY of the global model and fine-tune it on training fold
        cv_model = copy.deepcopy(global_model)

        # Fine-tune the global model on this fold's training data
        # This simulates how the model would perform if adapted to this data
        cv_model.fit(X_train, y_train)

        # Evaluate on validation fold
        y_pred = cv_model.predict(X_fold_val)

        # Calculate metrics
        accuracy = accuracy_score(y_fold_val, y_pred)
        precision = precision_score(y_fold_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_fold_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_fold_val, y_pred, average='weighted', zero_division=0)

        # Store metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Store for overall confusion matrix
        all_y_true.extend(y_fold_val)
        all_y_pred.extend(y_pred)

        print(f"\nFold {fold_num} Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        fold_num += 1

    # Calculate average metrics
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    print(f"\n=== Average Cross-Validation Metrics ===")
    print(f"Accuracy:  {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall:    {avg_recall:.4f}")
    print(f"F1-Score:  {avg_f1:.4f}")

    # Also evaluate the model directly without fine-tuning
    print(f"\n=== Direct Evaluation (No Fine-tuning) ===")
    y_pred_direct = global_model.predict(X_val_imputed)
    direct_accuracy = accuracy_score(y_val, y_pred_direct)
    direct_precision = precision_score(y_val, y_pred_direct, average='weighted', zero_division=0)
    direct_recall = recall_score(y_val, y_pred_direct, average='weighted', zero_division=0)
    direct_f1 = f1_score(y_val, y_pred_direct, average='weighted', zero_division=0)

    print(f"Accuracy:  {direct_accuracy:.4f}")
    print(f"Precision: {direct_precision:.4f}")
    print(f"Recall:    {direct_recall:.4f}")
    print(f"F1-Score:  {direct_f1:.4f}")

    # Plot metrics comparison
    plot_metrics_comparison(accuracies, precisions, recalls, f1_scores, round_num)

    # Plot confusion matrix for direct evaluation
    plot_confusion_matrix(y_val, y_pred_direct, round_num)

    # Return BOTH: CV average and direct evaluation
    return {
        'cv_accuracy': avg_accuracy,
        'cv_precision': avg_precision,
        'cv_recall': avg_recall,
        'cv_f1': avg_f1,
        'direct_accuracy': direct_accuracy,
        'direct_precision': direct_precision,
        'direct_recall': direct_recall,
        'direct_f1': direct_f1
    }


def plot_metrics_comparison(accuracies, precisions, recalls, f1_scores, round_num=None):
    """Plot comparison of metrics across folds."""
    folds = ['Fold 1', 'Fold 2', 'Fold 3']
    x = np.arange(len(folds))
    width = 0.2

    plt.figure(figsize=(12, 6))

    plt.bar(x - 1.5 * width, accuracies, width, label='Accuracy', color='blue', alpha=0.7)
    plt.bar(x - 0.5 * width, precisions, width, label='Precision', color='green', alpha=0.7)
    plt.bar(x + 0.5 * width, recalls, width, label='Recall', color='orange', alpha=0.7)
    plt.bar(x + 1.5 * width, f1_scores, width, label='F1-Score', color='red', alpha=0.7)

    plt.xlabel('Cross-Validation Folds')
    plt.ylabel('Score')
    title = 'Metrics Comparison Across Folds'
    if round_num is not None:
        title += f' - Round {round_num}'
    plt.title(title)
    plt.xticks(x, folds)
    plt.legend(loc='lower right')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on top of bars
    for i, v in enumerate(accuracies):
        plt.text(i - 1.5 * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(precisions):
        plt.text(i - 0.5 * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(recalls):
        plt.text(i + 0.5 * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(f1_scores):
        plt.text(i + 1.5 * width, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    filename = f'metrics_comparison_round_{round_num}.png' if round_num else 'metrics_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, round_num=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))

    # Create heatmap
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix{" - Round " + str(round_num) if round_num else ""}', fontsize=14)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Benign', 'Malicious'], rotation=45)
    plt.yticks(tick_marks, ['Benign', 'Malicious'])

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    filename = f'confusion_matrix_round_{round_num}.png' if round_num else 'confusion_matrix.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    # Print confusion matrix details
    print("\n=== Confusion Matrix Details ===")
    print(f"True Negatives (TN): {cm[0, 0]} - Correctly predicted benign")
    print(f"False Positives (FP): {cm[0, 1]} - Benign incorrectly predicted as malicious")
    print(f"False Negatives (FN): {cm[1, 0]} - Malicious incorrectly predicted as benign")
    print(f"True Positives (TP): {cm[1, 1]} - Correctly predicted malicious")
    print(f"Total Samples: {cm.sum()}")
    print(f"Accuracy: {(cm[0, 0] + cm[1, 1]) / cm.sum():.4f}")


def plot_all_metrics_trends():
    """Plot all metric trends for clients and global model."""
    if len(global_accuracies) == 0:
        print("No metrics data to plot yet.")
        return

    rounds = list(range(1, len(global_accuracies) + 1))

    # Create a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Federated Learning Performance Metrics', fontsize=16, fontweight='bold')

    metrics_config = [
        ('Accuracy', client_accuracies, global_accuracies, 'blue', axes[0, 0]),
        ('Precision', client_precisions, global_precisions, 'green', axes[0, 1]),
        ('Recall', client_recalls, global_recalls, 'orange', axes[1, 0]),
        ('F1-Score', client_f1_scores, global_f1_scores, 'red', axes[1, 1])
    ]

    for metric_name, client_metrics, global_metrics, color, ax in metrics_config:
        # Plot client metrics
        for client_id in [1, 2, 3]:
            if len(client_metrics[client_id]) >= len(rounds):
                ax.plot(rounds, client_metrics[client_id][:len(rounds)],
                        label=f'Client {client_id}', marker='o', linewidth=1.5, markersize=5, alpha=0.7)

        # Plot global model metrics
        ax.plot(rounds, global_metrics, label='Global Model', linestyle='--',
                marker='s', linewidth=2, markersize=6, color='black')

        ax.set_xlabel('Round Number')
        ax.set_ylabel(metric_name + ' Score')
        ax.set_title(f'{metric_name} Trend')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

        # Add value labels
        if global_metrics:
            last_val = global_metrics[-1]
            ax.text(len(rounds), last_val, f'  {last_val:.3f}',
                    va='center', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig('all_metrics_trends.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_round_summary(round_num):
    """Print summary for current round."""
    if round_num <= len(global_accuracies):
        print(f"\n{'=' * 60}")
        print(f"ROUND {round_num} METRICS SUMMARY")
        print(f"{'=' * 60}")

        print(f"\nGlobal Model:")
        print(f"  Accuracy:  {global_accuracies[round_num - 1]:.4f}")
        print(f"  Precision: {global_precisions[round_num - 1]:.4f}")
        print(f"  Recall:    {global_recalls[round_num - 1]:.4f}")
        print(f"  F1-Score:  {global_f1_scores[round_num - 1]:.4f}")

        print(f"\nClients:")
        for client_id in [1, 2, 3]:
            if len(client_accuracies[client_id]) >= round_num:
                print(f"  Client {client_id}:")
                print(f"    Acc: {client_accuracies[client_id][round_num - 1]:.4f}, "
                      f"Prec: {client_precisions[client_id][round_num - 1]:.4f}, "
                      f"Rec: {client_recalls[client_id][round_num - 1]:.4f}, "
                      f"F1: {client_f1_scores[client_id][round_num - 1]:.4f}")