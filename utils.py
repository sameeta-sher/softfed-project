#!/usr/bin/env python3
"""
Utility Functions and Global Variables
Shared variables and helper functions across the project
"""

import joblib
import numpy as np
import pandas as pd
import json
import os
import hashlib
import matplotlib.pyplot as plt
import copy
import random
import string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

# Global variables for tracking ALL metrics
client_accuracies = {1: [], 2: [], 3: []}
client_precisions = {1: [], 2: [], 3: []}
client_recalls = {1: [], 2: [], 3: []}
client_f1_scores = {1: [], 2: [], 3: []}

global_accuracies = []
global_precisions = []
global_recalls = []
global_f1_scores = []

DATA_PATH = '/kaggle/working/cic-ids-softfed-project/'


def store_client_metrics(client_id: int, y_true, y_pred):
    """Store all metrics for a client."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    client_accuracies[client_id].append(accuracy)
    client_precisions[client_id].append(precision)
    client_recalls[client_id].append(recall)
    client_f1_scores[client_id].append(f1)

    return accuracy, precision, recall, f1


def store_global_metrics(avg_accuracy: float, avg_precision: float, avg_recall: float, avg_f1: float):
    """Store all metrics for the global model."""
    global_accuracies.append(avg_accuracy)
    global_precisions.append(avg_precision)
    global_recalls.append(avg_recall)
    global_f1_scores.append(avg_f1)


def get_client_sample_counts() -> dict:
    """Return hard-coded sample counts for weighted averaging."""
    return {
        1: 918759,  # From debug output
        2: 866588,
        3: 762387
    }


def setup_directories():
    """Create necessary directories for the project."""
    directories = ['client_updates', 'global_models', DATA_PATH]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Created necessary directories")