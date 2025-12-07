#!/usr/bin/env python3
"""
Run Federated Learning Script
Main script to run the complete federated learning process
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from global_model import run_federated_learning


def main():
    """Run the federated learning process."""
    print("Starting Federated Learning with Security Verification")
    print("=" * 60)

    # Create necessary directories
    os.makedirs("client_updates", exist_ok=True)
    os.makedirs("global_models", exist_ok=True)

    # Check if keys exist
    if not all(os.path.exists(f'client{i}_private_key.pem') for i in [1, 2, 3]):
        print("❌ RSA keys not found. Please run: python setup_keys.py")
        sys.exit(1)

    # Check data directory
    DATA_PATH = '/kaggle/working/cic-ids-softfed-project/'
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data directory not found: {DATA_PATH}")
        print("Please run: python setup_data.py")
        sys.exit(1)

    # Run federated learning
    try:
        final_model = run_federated_learning(num_rounds=3)
        print("\n" + "=" * 60)
        print("✅ Federated learning completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Error during federated learning: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()