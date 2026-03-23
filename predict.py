"""
Inference script for credit card fraud detection.
Load a trained model and score new transactions.

Usage:
    python predict.py --model models/best_model.keras --input data/new_transactions.csv
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import json
import sys
from datetime import datetime


def load_model(model_path: str) -> keras.Model:
    """Load a trained Keras model."""
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def preprocess_transaction(transaction: dict, feature_columns: list = None) -> np.ndarray:
    """
    Preprocess a single transaction for scoring.
    Expects a dict with keys: V1-V28, Amount, Time
    """
    if feature_columns is None:
        feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount_scaled', 'Time_scaled']

    # Basic scaling for Amount and Time
    if 'Amount' in transaction:
        transaction['Amount_scaled'] = (transaction['Amount'] - 88.35) / 250.12
    if 'Time' in transaction:
        transaction['Time_scaled'] = (transaction['Time'] - 94813.86) / 47488.15

    features = [transaction.get(col, 0.0) for col in feature_columns]
    return np.array(features).reshape(1, -1)


def score_transaction(model: keras.Model, features: np.ndarray,
                       threshold: float = 0.5) -> dict:
    """Score a transaction and return fraud probability + decision."""
    probability = model.predict(features, verbose=0)[0][0]

    result = {
        'fraud_probability': float(probability),
        'is_fraud': bool(probability >= threshold),
        'risk_level': 'HIGH' if probability >= 0.8 else 'MEDIUM' if probability >= 0.5 else 'LOW',
        'threshold_used': threshold,
        'timestamp': datetime.now().isoformat()
    }

    return result


def batch_score(model: keras.Model, input_path: str,
                threshold: float = 0.5) -> pd.DataFrame:
    """Score a batch of transactions from CSV."""
    df = pd.read_csv(input_path)
    print(f"Scoring {len(df):,} transactions...")

    # Assume features are already preprocessed
    feature_cols = [c for c in df.columns if c not in ['Class', 'Transaction_ID']]
    X = df[feature_cols].values

    probabilities = model.predict(X, verbose=0).ravel()
    df['fraud_probability'] = probabilities
    df['predicted_fraud'] = (probabilities >= threshold).astype(int)
    df['risk_level'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )

    n_flagged = df['predicted_fraud'].sum()
    print(f"Flagged {n_flagged:,} transactions as fraudulent ({n_flagged/len(df):.2%})")

    return df


def main():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection - Inference')
    parser.add_argument('--model', type=str, default='models/best_model.keras',
                        help='Path to trained model')
    parser.add_argument('--input', type=str, help='Path to CSV for batch scoring')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Output path for batch predictions')
    args = parser.parse_args()

    model = load_model(args.model)

    if args.input:
        # Batch scoring mode
        results = batch_score(model, args.input, args.threshold)
        results.to_csv(args.output, index=False)
        print(f"Predictions saved to {args.output}")
    else:
        # Interactive demo mode
        print("\n--- Interactive Fraud Detection Demo ---")
        print("Enter a JSON transaction or 'quit' to exit\n")

        while True:
            user_input = input("Transaction JSON > ")
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            try:
                transaction = json.loads(user_input)
                features = preprocess_transaction(transaction)
                result = score_transaction(model, features, args.threshold)
                print(f"\n  Fraud Probability: {result['fraud_probability']:.4f}")
                print(f"  Decision: {'🚨 FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
                print(f"  Risk Level: {result['risk_level']}\n")
            except json.JSONDecodeError:
                print("  Invalid JSON. Try again.\n")
            except Exception as e:
                print(f"  Error: {e}\n")


if __name__ == '__main__':
    main()
