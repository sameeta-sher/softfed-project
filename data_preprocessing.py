#!/usr/bin/env python3
"""
Data Preprocessing Functions
Handles data loading, cleaning, and preprocessing for both features and labels
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def preprocess_data(data_filename: str):
    """Comprehensive data preprocessing for both features and labels."""
    # Load the data
    data = pd.read_csv(data_filename)
    print(f"Loaded data shape: {data.shape}")

    # Check if Label column exists
    if 'Label' not in data.columns:
        raise ValueError(f"Label column not found in {data_filename}")

    # Split data into features and labels
    X = data.drop(columns=['Label'])
    y = data['Label']

    # data cleaning for FEATURES (X)
    print("Preprocessing features - handling NaN and infinity values...")

    # Replace infinity with NaN first in features
    X = X.replace([np.inf, -np.inf], np.nan)

    # Check for NaN values in features
    nan_count_x = X.isna().sum().sum()
    print(f"NaN values in features: {nan_count_x}")

    # Impute missing values in features with the column mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Final cleanup for features - replace any remaining issues
    X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=1e10, neginf=-1e10)

    # data cleaning for LABELS (y)
    print("Preprocessing labels - handling NaN values...")

    # Check for NaN values in labels
    nan_count_y = y.isna().sum()
    print(f"NaN values in labels: {nan_count_y}")

    # Remove rows where labels are NaN
    if nan_count_y > 0:
        print(f"Removing {nan_count_y} rows with NaN labels")
        # Create a mask for non-NaN labels
        valid_mask = ~y.isna()
        # Apply the mask to both X and y
        X_imputed = X_imputed[valid_mask.values]
        y = y[valid_mask]
        print(f"Data shape after removing NaN labels: {X_imputed.shape}")

    # Verify no NaN or infinity values remain in features
    assert not np.any(np.isnan(X_imputed)), "NaN values remain in features after imputation"
    assert not np.any(np.isinf(X_imputed)), "Infinity values remain in features after imputation"

    # Verify no NaN values remain in labels
    assert not np.any(y.isna()), "NaN values remain in labels after cleaning"

    print("Data preprocessing completed successfully")

    return X_imputed, y


def preprocess_client_data(data_filename: str):
    """Simplified preprocessing for client data."""
    # Load the data
    data = pd.read_csv(data_filename)

    if 'Label' not in data.columns:
        raise ValueError(f"Label column not found in {data_filename}")

    X = data.drop(columns=['Label'])
    y = data['Label']

    # Basic preprocessing
    X = X.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed = np.nan_to_num(X_imputed, nan=0.0, posinf=1e10, neginf=-1e10)

    # Remove rows with NaN labels
    valid_mask = ~y.isna()
    if valid_mask.any():
        X_imputed = X_imputed[valid_mask.values]
        y = y[valid_mask]

    return X_imputed, y