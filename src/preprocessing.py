"""
Data preprocessing utilities for credit card fraud detection.
Handles loading, cleaning, scaling, and splitting of transaction data.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


def load_data(filepath: str) -> pd.DataFrame:
    """Load credit card transaction data from CSV."""
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud ratio: {df['Class'].mean():.4%}")
    return df


def generate_synthetic_data(n_samples: int = 284807, fraud_ratio: float = 0.00173, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic credit card transaction data mimicking
    the structure of the Kaggle Credit Card Fraud dataset.
    
    The real dataset contains PCA-transformed features V1-V28,
    plus Time, Amount, and Class (0=legit, 1=fraud).
    """
    np.random.seed(seed)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # --- Legitimate transactions ---
    legit_features = np.random.randn(n_legit, 28)
    # Shift some features to create separability
    legit_features[:, 0] += 1.5
    legit_features[:, 1] += 0.8
    legit_features[:, 2] += 1.2
    legit_features[:, 3] += 0.5
    legit_features[:, 10] += 0.7
    legit_features[:, 11] += 0.9
    legit_features[:, 16] += 0.6

    legit_time = np.sort(np.random.uniform(0, 172792, n_legit))
    legit_amount = np.abs(np.random.lognormal(mean=3.5, sigma=1.8, size=n_legit))
    legit_amount = np.clip(legit_amount, 0, 25691)

    # --- Fraudulent transactions ---
    fraud_features = np.random.randn(n_fraud, 28)
    # Opposite shifts to create class separation
    fraud_features[:, 0] -= 2.5
    fraud_features[:, 1] += 0.3
    fraud_features[:, 2] -= 2.8
    fraud_features[:, 3] -= 1.5
    fraud_features[:, 4] += 1.8
    fraud_features[:, 5] -= 1.2
    fraud_features[:, 10] -= 3.0
    fraud_features[:, 11] += 2.0
    fraud_features[:, 12] -= 2.5
    fraud_features[:, 14] -= 3.5
    fraud_features[:, 16] -= 2.0
    # Add some noise overlap so it's not trivially separable
    fraud_features += np.random.randn(n_fraud, 28) * 0.5

    fraud_time = np.sort(np.random.uniform(0, 172792, n_fraud))
    fraud_amount = np.abs(np.random.lognormal(mean=4.2, sigma=2.0, size=n_fraud))
    fraud_amount = np.clip(fraud_amount, 0, 25691)

    # --- Combine ---
    features = np.vstack([legit_features, fraud_features])
    time_col = np.concatenate([legit_time, fraud_time])
    amount_col = np.concatenate([legit_amount, legit_amount[:0], fraud_amount])
    
    # Fix amount concatenation
    amount_col = np.concatenate([legit_amount, fraud_amount])
    labels = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    data = np.column_stack([time_col, features, amount_col, labels])
    df = pd.DataFrame(data, columns=columns)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df['Class'] = df['Class'].astype(int)

    print(f"Generated synthetic dataset: {df.shape}")
    print(f"  Legitimate: {n_legit:,} ({1-fraud_ratio:.2%})")
    print(f"  Fraudulent: {n_fraud:,} ({fraud_ratio:.2%})")

    return df


def preprocess(df: pd.DataFrame, scale_amount: bool = True, scale_time: bool = True) -> pd.DataFrame:
    """Apply scaling to Amount and Time features."""
    df = df.copy()

    if scale_amount:
        scaler = RobustScaler()
        df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df.drop('Amount', axis=1, inplace=True)

    if scale_time:
        scaler = RobustScaler()
        df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
        df.drop('Time', axis=1, inplace=True)

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
    """
    Split into train/val/test sets with stratification.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop('Class', axis=1).values
    y = df['Class'].values

    # First split: train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Second split: train / val
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=seed, stratify=y_temp
    )

    print(f"Train: {X_train.shape[0]:,} samples ({y_train.mean():.4%} fraud)")
    print(f"Val:   {X_val.shape[0]:,} samples ({y_val.mean():.4%} fraud)")
    print(f"Test:  {X_test.shape[0]:,} samples ({y_test.mean():.4%} fraud)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, sampling_strategy: float = 0.1, seed: int = 42):
    """
    Apply SMOTE oversampling + random undersampling pipeline
    to handle severe class imbalance.
    """
    over = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
    under = RandomUnderSampler(sampling_strategy=0.5, random_state=seed)

    pipeline = ImbPipeline(steps=[('over', over), ('under', under)])
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

    print(f"Before SMOTE: {X_train.shape[0]:,} samples ({y_train.mean():.4%} fraud)")
    print(f"After SMOTE:  {X_resampled.shape[0]:,} samples ({y_resampled.mean():.2%} fraud)")

    return X_resampled, y_resampled


def scale_features(X_train, X_val, X_test):
    """Standardize features using training set statistics."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
