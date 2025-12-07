#!/usr/bin/env python3
"""
Client Training Coordinator
Calls client files using Python multiprocessing modules
"""

import os
import json
import multiprocessing as mp
from typing import List, Dict, Any
import traceback

from utils import DATA_PATH


def train_client_wrapper(client_id: int, round_num: int, global_model):
    """Wrapper function for training individual clients."""
    print(f"\n--- Processing Client {client_id} ---")

    try:
        data_filename = f'{DATA_PATH}client{client_id}_data.csv'
        if not os.path.exists(data_filename):
            print(f"ERROR: Data file {data_filename} not found")
            return None

        # Import the appropriate client module
        if client_id == 1:
            from client1 import train_client_with_global_model
            accuracy, update_data = train_client_with_global_model(client_id, round_num, global_model)
        elif client_id == 2:
            from client2 import train_malicious_client
            accuracy, update_data = train_malicious_client(client_id, round_num, global_model)
        elif client_id == 3:
            from client3 import train_client_with_global_model
            accuracy, update_data = train_client_with_global_model(client_id, round_num, global_model)
        else:
            print(f"ERROR: Unknown client ID {client_id}")
            return None

        print(f"✓ Client {client_id} completed with accuracy: {accuracy:.4f}")
        return update_data

    except Exception as e:
        print(f"✗ Client {client_id} failed: {str(e)}")
        traceback.print_exc()
        return None


def run_client_training(round_num: int, global_model):
    """Run all client training with the current global model using multiprocessing."""
    print(f"\n--- Starting Client Training for Round {round_num} ---")

    client_updates = []

    # Create a pool of workers
    with mp.Pool(processes=3) as pool:
        # Prepare arguments for each client
        args = [(1, round_num, global_model),
                (2, round_num, global_model),
                (3, round_num, global_model)]

        # Use starmap to pass multiple arguments
        results = pool.starmap(train_client_wrapper, args)

    # Collect successful updates
    for result in results:
        if result is not None:
            client_updates.append(result)

    print(f"\nClient training completed. {len(client_updates)}/3 clients successful.")
    return client_updates