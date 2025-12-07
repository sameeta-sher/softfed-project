# SoftFed: Secure Federated Learning Project

A secure federated learning framework with malicious client detection and cryptographic verification.

## Project Structure
- `global_model.py`: Main orchestrator for federated learning
- `clients_training.py`: Coordinator for client training with multiprocessing
- `clients/`: Individual client implementations
- `security_functions.py`: Cryptographic security and verification functions
- `data_preprocessing.py`: Data preprocessing utilities
- `utils.py`: Utility functions and shared variables
- `aggregation.py`: Federated aggregation with security checks
- `evaluation.py`: Model evaluation and visualization
- `config/`: Configuration settings
- `scripts/`: Setup and run scripts
- `tests/`: Unit tests

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt