"""
Visualization utilities for fraud detection analysis.
Generates publication-quality plots for EDA, model evaluation, and results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, classification_report
)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'legit': '#2ecc71', 'fraud': '#e74c3c', 'accent': '#3498db'}
FIG_DPI = 150


def plot_class_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot the class imbalance distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    counts = df['Class'].value_counts()
    colors = [COLORS['legit'], COLORS['fraud']]
    axes[0].bar(['Legitimate', 'Fraud'], counts.values, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title('Transaction Count by Class', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold', fontsize=11)

    # Pie chart
    axes[1].pie(
        counts.values, labels=['Legitimate', 'Fraud'],
        colors=colors, autopct='%1.3f%%',
        startangle=90, explode=(0, 0.1),
        shadow=True, textprops={'fontsize': 12}
    )
    axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def plot_amount_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot transaction amount distributions by class."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, cls in enumerate([0, 1]):
        label = 'Legitimate' if cls == 0 else 'Fraud'
        color = COLORS['legit'] if cls == 0 else COLORS['fraud']
        subset = df[df['Class'] == cls]['Amount']

        axes[idx].hist(subset, bins=80, color=color, alpha=0.8, edgecolor='white')
        axes[idx].set_title(f'{label} Transaction Amounts', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Amount ($)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].axvline(subset.median(), color='black', linestyle='--', linewidth=1.5,
                          label=f'Median: ${subset.median():.2f}')
        axes[idx].legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def plot_time_distribution(df: pd.DataFrame, save_path: str = None):
    """Plot transaction time distributions."""
    fig, ax = plt.subplots(figsize=(14, 5))

    legit = df[df['Class'] == 0]['Time'] / 3600  # Convert to hours
    fraud = df[df['Class'] == 1]['Time'] / 3600

    ax.hist(legit, bins=100, alpha=0.6, color=COLORS['legit'], label='Legitimate', density=True)
    ax.hist(fraud, bins=100, alpha=0.7, color=COLORS['fraud'], label='Fraud', density=True)
    ax.set_title('Transaction Density Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """Plot feature correlation heatmap."""
    fig, ax = plt.subplots(figsize=(18, 14))

    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, cmap='RdBu_r', center=0,
        annot=False, square=True, linewidths=0.5,
        cbar_kws={'shrink': 0.8}, ax=ax
    )
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def plot_top_feature_correlations(df: pd.DataFrame, n: int = 15, save_path: str = None):
    """Plot features most correlated with fraud."""
    correlations = df.corr()['Class'].drop('Class').abs().sort_values(ascending=True)
    top_corr = correlations.tail(n)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = [COLORS['fraud'] if v > 0.1 else COLORS['accent'] for v in top_corr.values]
    top_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
    ax.set_title(f'Top {n} Features Correlated with Fraud', fontsize=14, fontweight='bold')
    ax.set_xlabel('Absolute Correlation')
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.7, label='Threshold (0.1)')
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def plot_training_history(history, save_path: str = None):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ('loss', 'Loss'),
        ('auc', 'AUC'),
        ('precision', 'Precision'),
        ('recall', 'Recall')
    ]

    for ax, (metric, title) in zip(axes.ravel(), metrics):
        if metric in history.history:
            ax.plot(history.history[metric], label=f'Train', linewidth=2)
            val_key = f'val_{metric}'
            if val_key in history.history:
                ax.plot(history.history[val_key], label=f'Validation',
                        linewidth=2, linestyle='--')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Training History', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """Plot an annotated confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=',', cmap='Blues',
        xticklabels=['Legitimate', 'Fraud'],
        yticklabels=['Legitimate', 'Fraud'],
        annot_kws={'size': 16}, ax=ax
    )
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)

    # Add rates
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    text = f'Accuracy: {(tn+tp)/total:.4f} | FPR: {fp/(fp+tn):.4f} | FNR: {fn/(fn+tp):.4f}'
    ax.text(0.5, -0.12, text, transform=ax.transAxes, ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def plot_roc_and_pr_curves(y_true, y_pred_proba, save_path: str = None):
    """Plot ROC and Precision-Recall curves side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=COLORS['fraud'], linewidth=2.5, label=f'Model (AUC={roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    axes[0].fill_between(fpr, tpr, alpha=0.15, color=COLORS['fraud'])
    axes[0].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(fontsize=11, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    axes[1].plot(recall, precision, color=COLORS['accent'], linewidth=2.5,
                 label=f'Model (AUC={pr_auc:.4f})')
    axes[1].fill_between(recall, precision, alpha=0.15, color=COLORS['accent'])
    baseline = y_true.mean()
    axes[1].axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.4f})')
    axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend(fontsize=11, loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()


def plot_threshold_analysis(y_true, y_pred_proba, save_path: str = None):
    """Plot precision, recall, and F1 across different thresholds."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = []
    precisions = []
    recalls = []

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=2.5, color=COLORS['fraud'])
    ax.plot(thresholds, precisions, label='Precision', linewidth=2, linestyle='--', color=COLORS['accent'])
    ax.plot(thresholds, recalls, label='Recall', linewidth=2, linestyle='--', color=COLORS['legit'])
    ax.axvline(best_threshold, color='gray', linestyle=':', linewidth=1.5,
               label=f'Best Threshold ({best_threshold:.2f})')
    ax.scatter([best_threshold], [f1_scores[best_idx]], color=COLORS['fraud'], s=100, zorder=5)
    ax.set_title('Threshold Optimization', fontsize=14, fontweight='bold')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()

    return best_threshold
