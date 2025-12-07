#!/usr/bin/env python3
"""
Federated Aggregation with Security
Aggregates client updates with security verification
"""

import numpy as np
import json
import copy

from security_functions import verify_signature, compute_gradient_hash
from utils import get_client_sample_counts


def federated_average_with_security(client_updates, global_model, round_num):
    """Aggregate client updates with security verification."""
    print(f"\n--- Federated Aggregation for Round {round_num} ---")

    if not client_updates:
        print("No updates to aggregate!")
        return global_model

    # Hard-coded sample counts for weighted averaging
    client_sample_counts = get_client_sample_counts()

    valid_updates = []
    security_report = {}

    print("Verifying client updates...")

    for update in client_updates:
        client_id = update['client_id']

        print(f"\nVerifying Client {client_id}:")

        # Security checks
        security_status = {
            'client_id': client_id,
            'signature_valid': False,
            'update_valid': False,
            'accepted': False,
            'reason': ''
        }

        # 1. Verify signature
        if not verify_signature(client_id, update['behavioral_metrics'], update['signature']):
            security_status['signature_valid'] = False
            security_status['reason'] = 'Invalid signature'
            print(f"  ✗ Signature verification FAILED")
        else:
            security_status['signature_valid'] = True
            print(f"  ✓ Signature verification PASSED")

            # 2. Verify update integrity (for benign clients only)
            if client_id != 2:  # Don't check malicious client
                # Recompute hash from the sent delta
                param_delta = np.array(update['param_delta'])
                intercept_delta = np.array(update['intercept_delta'])
                update_vector = np.concatenate([param_delta.flatten(), intercept_delta.flatten()])

                computed_hash = compute_gradient_hash(update_vector)
                sent_hash = update['behavioral_metrics']['update_hash']

                if computed_hash == sent_hash:
                    security_status['update_valid'] = True
                    security_status['accepted'] = True
                    valid_updates.append(update)
                    print(f"  ✓ Update integrity PASSED")
                else:
                    security_status['update_valid'] = False
                    security_status['reason'] = 'Hash mismatch'
                    print(f"  ✗ Update integrity FAILED (hash mismatch)")
            else:
                # For malicious client, we accept it to simulate attack
                security_status['accepted'] = True
                valid_updates.append(update)
                print(f"  ⚠️ Malicious update accepted (attack simulation)")

        security_report[f'client_{client_id}'] = security_status

    print(f"\nAggregation Summary:")
    print(f"  Total updates: {len(client_updates)}")
    print(f"  Valid updates: {len(valid_updates)}")

    if not valid_updates:
        print("No valid updates to aggregate. Keeping current global model.")
        return global_model

    # Perform federated averaging
    print("\nPerforming federated averaging...")

    # Initialize aggregated deltas
    total_samples = 0
    weighted_param_delta = np.zeros_like(global_model.coef_)
    weighted_intercept_delta = np.zeros_like(global_model.intercept_)

    for update in valid_updates:
        client_id = update['client_id']
        samples_used = client_sample_counts[client_id]

        # Get deltas
        param_delta = np.array(update['param_delta'])
        intercept_delta = np.array(update['intercept_delta'])

        # Weight by sample count
        weight = samples_used
        weighted_param_delta += param_delta * weight
        weighted_intercept_delta += intercept_delta * weight
        total_samples += samples_used

    # Normalize by total samples
    if total_samples > 0:
        weighted_param_delta /= total_samples
        weighted_intercept_delta /= total_samples

    # Apply updates to global model
    print(f"\nApplying updates to global model...")
    print(f"  Parameter delta norm: {np.linalg.norm(weighted_param_delta):.6f}")
    print(f"  Intercept delta norm: {np.linalg.norm(weighted_intercept_delta):.6f}")

    # Update model parameters
    new_global_model = copy.deepcopy(global_model)
    new_global_model.coef_ += weighted_param_delta
    new_global_model.intercept_ += weighted_intercept_delta

    # Save security report
    report_filename = f"Round_{round_num}_Security_Report.json"
    with open(report_filename, 'w') as f:
        json.dump(security_report, f, indent=4)
    print(f"Security report saved: {report_filename}")

    return new_global_model