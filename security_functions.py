#!/usr/bin/env python3
"""
Security Functions
Includes model hash, gradient integrity, client authentication
"""

import joblib
import json
import hashlib
import numpy as np
import os
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def sign_update(client_id: int, update_data: dict) -> str:
    """Sign the update data using the client's private key."""
    private_key_file = f'client{client_id}_private_key.pem'

    with open(private_key_file, 'rb') as f:
        private_key = serialization.load_pem_private_key(f.read(), password=None)

    data_bytes = json.dumps(update_data).encode()
    signature = private_key.sign(data_bytes, padding.PKCS1v15(), hashes.SHA256())
    return signature.hex()


def verify_signature(client_id: int, data: dict, signature: str) -> bool:
    """Verify the signature using the client's public key."""
    public_key_file = f'client{client_id}_public_key.pem'

    with open(public_key_file, 'rb') as f:
        public_key = serialization.load_pem_public_key(f.read())

    try:
        public_key.verify(bytes.fromhex(signature), json.dumps(data).encode(),
                          padding.PKCS1v15(), hashes.SHA256())
        return True
    except Exception as e:
        print(f"Signature verification failed for client {client_id}: {e}")
        return False


def compute_model_hash(model) -> str:
    """Compute hash of model parameters."""
    params_bytes = model.coef_.tobytes() + model.intercept_.tobytes()
    return hashlib.sha256(params_bytes).hexdigest()


def compute_gradient_hash(gradient: np.ndarray) -> str:
    """Compute hash of gradient."""
    return hashlib.sha256(gradient.tobytes()).hexdigest()


def save_model_with_hash(model, filename: str) -> str:
    """Save model and return its hash."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the model
    joblib.dump(model, filename)

    # Compute and return hash
    model_hash = compute_model_hash(model)
    return model_hash


def load_model_with_hash(filename: str):
    """Load model and return it along with its hash."""
    if not os.path.exists(filename):
        return None, None

    model = joblib.load(filename)
    model_hash = compute_model_hash(model)
    return model, model_hash


def distribute_global_model(global_model, round_num: int):
    """Distribute global model to clients by saving it in a shared location."""
    global_model_filename = f"global_model_round_{round_num}.pkl"
    global_model_hash = save_model_with_hash(global_model, f"global_models/{global_model_filename}")
    print(f"Global model for round {round_num} distributed with hash: {global_model_hash[:20]}...")
    return global_model_filename, global_model_hash


def generate_keys():
    """Generate RSA key pairs for all clients."""
    for client_id in [1, 2, 3]:
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Generate public key
        public_key = private_key.public_key()

        # Serialize and save private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        with open(f'client{client_id}_private_key.pem', 'wb') as f:
            f.write(private_pem)

        # Serialize and save public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        with open(f'client{client_id}_public_key.pem', 'wb') as f:
            f.write(public_pem)

        print(f"Generated keys for client {client_id}")