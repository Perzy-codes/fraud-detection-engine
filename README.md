# 💳 Credit Card Fraud Detection with Deep Learning

A production-grade fraud detection system using TensorFlow/Keras that tackles the extreme class imbalance problem (99.83% legitimate vs 0.17% fraudulent transactions). Implements both a supervised deep neural network classifier and an unsupervised autoencoder for anomaly detection.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Problem Statement

Credit card fraud costs the financial industry billions annually. The core ML challenge is **extreme class imbalance** — fraudulent transactions make up only ~0.17% of all transactions. A naive model predicting "legitimate" for everything achieves 99.83% accuracy but catches zero fraud.

This project builds a system that:
- Detects fraud with high recall (catching as many frauds as possible)
- Maintains acceptable precision (minimizing false alarms)
- Handles the 578:1 class imbalance ratio effectively
- Provides interpretable confidence scores for each transaction

## 🏗️ Architecture

### Approach 1: Deep Neural Network Classifier
```
Input (30 features)
  → Dense(256) + BatchNorm + ReLU + Dropout(0.3)
  → Dense(128) + BatchNorm + ReLU + Dropout(0.3)  
  → Dense(64)  + BatchNorm + ReLU + Dropout(0.3)
  → Dense(32)  + BatchNorm + ReLU + Dropout(0.15)
  → Dense(1, sigmoid) → Fraud probability
```

**Imbalance handling strategy:**
- SMOTE oversampling + random undersampling pipeline
- Class weights inversely proportional to frequency
- Threshold optimization via F1-score maximization

### Approach 2: Autoencoder Anomaly Detection
```
Encoder: Input → 128 → 64 → 14 (bottleneck)
Decoder: 14 → 64 → 128 → Input reconstruction
```
Trained exclusively on legitimate transactions. Fraudulent transactions produce higher reconstruction error, enabling unsupervised detection.

## 📊 Results

| Metric | Classifier | Autoencoder |
|--------|-----------|-------------|
| ROC-AUC | >0.97 | >0.95 |
| PR-AUC | >0.80 | >0.70 |
| F1-Score | >0.85 | N/A |
| Inference | <1ms/txn | <1ms/txn |

> *Results on synthetic data matching real-world Kaggle dataset distribution. See notebook for detailed evaluation.*

## 📁 Project Structure

```
fraud-detection/
├── notebooks/
│   └── 01_fraud_detection_analysis.ipynb   # Full EDA + training pipeline
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Data loading, scaling, SMOTE, splitting
│   ├── model.py            # DNN classifier + autoencoder architectures
│   └── visualize.py        # Publication-quality visualization utilities
├── visualizations/         # Generated plots and figures
├── models/                 # Saved model checkpoints
├── data/                   # Dataset directory
├── predict.py              # Inference script (batch + interactive)
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/Perzy-codes/fraud-detection-engine.git
cd fraud-detection-engine
pip install -r requirements.txt
```

### Run the Notebook
```bash
cd notebooks
jupyter notebook 01_fraud_detection_analysis.ipynb
```

### Batch Inference
```bash
python predict.py --model models/best_model.keras --input data/transactions.csv --threshold 0.5
```

### Interactive Demo
```bash
python predict.py --model models/best_model.keras
# Enter transaction JSON at the prompt
```

## 🔬 Key Technical Decisions

**Why SMOTE + class weights (not just one)?**  
SMOTE alone can overfit to synthetic minority samples. Class weights alone don't expand the decision boundary. Together, SMOTE provides the model more diversity in the minority class while weights ensure the loss function properly penalizes missed frauds.

**Why an autoencoder in addition to a classifier?**  
The autoencoder provides an unsupervised signal that doesn't depend on labeled fraud examples. In production, new fraud patterns that don't match historical labels can still be caught through anomalously high reconstruction error. This makes the system more robust to concept drift.

**Why threshold optimization?**  
The default 0.5 threshold assumes equal misclassification costs. In fraud detection, missing a fraud (FN) is far more expensive than a false alarm (FP). Optimizing the threshold on a validation set yields a better precision-recall tradeoff for the actual business objective.

## 📦 Dataset

The project supports two modes:
1. **Kaggle dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 frauds
2. **Synthetic generation**: Built-in data generator that produces data with identical structure and statistical properties for development and testing

Features V1-V28 are PCA-transformed (original features are confidential). `Time` and `Amount` are the only non-transformed features.

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.x / Keras
- **ML Pipeline**: scikit-learn, imbalanced-learn
- **Data**: pandas, NumPy
- **Visualization**: matplotlib, seaborn
- **Evaluation**: ROC-AUC, PR-AUC, F1-score, confusion matrices

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built by [Perzy](https://github.com/Perzy-codes)** | If you found this useful, drop a ⭐
