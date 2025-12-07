#!/usr/bin/env python3
"""
Main orchestrator for federated learning
Manages setup, calls clients_training.py, performs aggregation
"""

import os
import sys
from sklearn.linear_model import LogisticRegression

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DATA_PATH, store_global_metrics, global_accuracies
from data_preprocessing import preprocess_data
from security_functions import save_model_with_hash, distribute_global_model
from clients_training import run_client_training
from aggregation import federated_average_with_security
from evaluation import evaluate_global_model, plot_all_metrics_trends, print_round_summary


def initialize_global_model():
    """Initialize the global model for round 1."""
    print("--- Initializing Global Model ---")

    # Use preprocessing to get feature dimensions
    X_train_imputed, y_train = preprocess_data(DATA_PATH + 'global_model_training_data.csv')

    # Initialize the global model
    global_model = LogisticRegression(max_iter=2000, random_state=42)

    # Train the global model on the global training data
    print(f"Training global model with {len(X_train_imputed)} samples...")
    global_model.fit(X_train_imputed, y_train)

    print(f"Model coefficients shape: {global_model.coef_.shape}")

    # Save the initial global model
    initial_model_filename = "global_model_round_0.pkl"
    save_model_with_hash(global_model, f"global_models/{initial_model_filename}")
    print(f"Initial global model saved successfully.")

    return global_model


def run_federated_learning(num_rounds=3):
    """Run the complete federated learning process."""
    print("=" * 70)
    print("FEDERATED LEARNING WITH SECURITY VERIFICATION")
    print("=" * 70)

    # Step 1: Initialize global model
    print("\n[STEP 1] Initializing Global Model...")
    global_model = initialize_global_model()

    # Evaluate initial model
    print("\n[STEP 2] Evaluating Initial Model...")
    metrics = evaluate_global_model(global_model, round_num=0)

    # Store the CV metrics
    store_global_metrics(
        metrics['cv_accuracy'],
        metrics['cv_precision'],
        metrics['cv_recall'],
        metrics['cv_f1']
    )

    # Main federated learning loop
    for round_num in range(1, num_rounds + 1):
        print(f"\n{'=' * 60}")
        print(f"FEDERATED ROUND {round_num}")
        print(f"{'=' * 60}")

        # Distribute global model
        print(f"\n[Round {round_num}.1] Distributing Global Model...")
        global_model_filename, global_model_hash = distribute_global_model(global_model, round_num - 1)

        # Client training
        print(f"\n[Round {round_num}.2] Client Training...")
        client_updates = run_client_training(round_num, global_model)

        # Federated aggregation
        print(f"\n[Round {round_num}.3] Federated Aggregation...")
        new_global_model = federated_average_with_security(client_updates, global_model, round_num)

        # Update global model
        global_model = new_global_model

        # Evaluate new global model
        print(f"\n[Round {round_num}.4] Evaluating Updated Global Model...")
        metrics = evaluate_global_model(global_model, round_num=round_num)

        # Store the CV metrics
        store_global_metrics(
            metrics['cv_accuracy'],
            metrics['cv_precision'],
            metrics['cv_recall'],
            metrics['cv_f1']
        )

        # Plot current trends
        print(f"\n[Round {round_num}.5] Plotting Metrics...")
        plot_all_metrics_trends()

        # Print summary
        print_round_summary(round_num)

        # Save the model
        distribute_global_model(global_model, round_num)

    print("\n" + "=" * 70)
    print("FEDERATED LEARNING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Final summary
    if len(global_accuracies) > 1:
        final_improvement = global_accuracies[-1] - global_accuracies[0]
        print(f"\nFINAL RESULTS:")
        print(f"  Rounds Completed: {num_rounds}")
        print(f"  Final Accuracy:   {global_accuracies[-1]:.4f}")
        print(f"  Improvement:      {final_improvement:+.4f}")

    return global_model


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("client_updates", exist_ok=True)
    os.makedirs("global_models", exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)

    # Check if keys exist
    if not all(os.path.exists(f'client{i}_private_key.pem') for i in [1, 2, 3]):
        print("❌ RSA keys not found. Please run: python setup_keys.py")
        sys.exit(1)

    # First, check if we have all required files
    required_files = ['client1_data.csv', 'client2_data.csv', 'client3_data.csv',
                      'global_model_training_data.csv', 'global_model_evaluation_data.csv']

    all_files_exist = True
    for file in required_files:
        if not os.path.exists(os.path.join(DATA_PATH, file)):
            print(f"ERROR: Required file {file} not found at {DATA_PATH}")
            all_files_exist = False

    if all_files_exist:
        print("All required files found. Starting federated learning...")
        final_model = run_federated_learning(num_rounds=3)
        print("\n✅ Federated learning completed!")
    else:
        print("\n❌ Missing required files. Please copy data files to:", DATA_PATH)
        print("Required files:", required_files)