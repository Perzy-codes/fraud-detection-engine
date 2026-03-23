"""
Neural network models for credit card fraud detection using TensorFlow/Keras.
Includes a standard classifier, an autoencoder for anomaly detection,
and training utilities.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, f1_score
)


def build_classifier(input_dim: int, dropout_rate: float = 0.3) -> keras.Model:
    """
    Build a deep neural network classifier for fraud detection.
    
    Architecture designed for high-dimensional, imbalanced data:
    - Batch normalization for stable training
    - Dropout for regularization
    - Progressively narrowing layers
    - Sigmoid output for binary classification
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,)),
        
        # Block 1
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Block 2
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Block 3
        layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate),
        
        # Block 4
        layers.Dense(32),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(dropout_rate * 0.5),
        
        # Output
        layers.Dense(1, activation='sigmoid')
    ])

    return model


def build_autoencoder(input_dim: int, encoding_dim: int = 14) -> tuple:
    """
    Build an autoencoder for unsupervised anomaly detection.
    
    Trained on legitimate transactions only. Fraudulent transactions
    will have higher reconstruction error, enabling anomaly detection.
    
    Returns: (autoencoder, encoder) models
    """
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='bottleneck')(x)

    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)

    autoencoder = keras.Model(inputs, decoded, name='fraud_autoencoder')
    encoder = keras.Model(inputs, encoded, name='encoder')

    return autoencoder, encoder


def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Calculate class weights inversely proportional to frequency.
    Critical for handling the extreme imbalance in fraud data.
    """
    n_samples = len(y_train)
    n_fraud = y_train.sum()
    n_legit = n_samples - n_fraud

    weight_legit = n_samples / (2 * n_legit)
    weight_fraud = n_samples / (2 * n_fraud)

    weights = {0: weight_legit, 1: weight_fraud}
    print(f"Class weights -> Legit: {weight_legit:.4f}, Fraud: {weight_fraud:.4f}")
    return weights


def get_callbacks(patience: int = 10, model_path: str = 'models/best_model.keras'):
    """Standard training callbacks."""
    return [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]


def train_classifier(model, X_train, y_train, X_val, y_val,
                      epochs: int = 100, batch_size: int = 2048,
                      use_class_weights: bool = True):
    """
    Train the classifier with class weights and callbacks.
    Uses a larger batch size for stable gradients on imbalanced data.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    class_weights = get_class_weights(y_train) if use_class_weights else None

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )

    return history


def train_autoencoder(model, X_train_legit, X_val, y_val,
                       epochs: int = 100, batch_size: int = 512):
    """
    Train autoencoder on legitimate transactions only.
    Validates on mixed data to monitor reconstruction quality.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse'
    )

    # Only use legitimate transactions for training
    cb = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=5, min_lr=1e-6, verbose=1
        )
    ]

    history = model.fit(
        X_train_legit, X_train_legit,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb,
        verbose=1
    )

    return history


def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    """Comprehensive evaluation with multiple metrics."""
    y_pred_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_proba >= threshold).astype(int)

    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    # Key metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)

    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

    return {
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1': f1,
        'confusion_matrix': cm
    }


def find_optimal_threshold(model, X_val, y_val):
    """Find threshold that maximizes F1-score on validation set."""
    y_pred_proba = model.predict(X_val, verbose=0).ravel()

    best_f1 = 0
    best_threshold = 0.5

    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold
