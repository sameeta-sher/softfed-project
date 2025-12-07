#!/usr/bin/env python3
"""
Setup RSA Keys Script
Generate RSA key pairs for all clients
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from security_functions import generate_keys


def main():
    """Generate RSA keys for all clients."""
    print("Generating RSA key pairs for clients...")

    # Create keys directory if it doesn't exist
    os.makedirs(".", exist_ok=True)

    # Generate keys
    generate_keys()

    print("\n✅ RSA keys generated successfully!")
    print("Private keys saved as: client1_private_key.pem, etc.")
    print("Public keys saved as: client1_public_key.pem, etc.")
    print("\nFiles created:")
    for client_id in [1, 2, 3]:
        print(f"  - client{client_id}_private_key.pem")
        print(f"  - client{client_id}_public_key.pem")


if __name__ == "__main__":
    main()