# 🛡️ Network Intrusion Detection Research Framework

### Machine Learning & Deep Learning Evaluation on KDDCup99 and NSL-KDD

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![DL](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red)
![Research](https://img.shields.io/badge/Status-Research%20Framework-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

------------------------------------------------------------------------

## 📌 Project Overview

This repository implements a structured experimental framework for
evaluating Classical Machine Learning and Deep Learning models in the
context of Network Intrusion Detection Systems (NIDS).

The framework supports:

-   Binary classification (Normal vs Attack)
-   Multiclass attack-type classification
-   One-vs-Rest (OvR) attack detection
-   Attack-family aggregation
-   Difficulty-based evaluation (NSL-KDD)
-   Labeled and unlabeled test scenarios
-   Structured experiment logging
-   Automatic LaTeX report generation
-   Timestamped reproducible experiment tracking

------------------------------------------------------------------------

## 🎯 Research Motivation

Intrusion Detection remains one of the most critical challenges in
cybersecurity.\
Benchmark datasets such as KDDCup99 and NSL-KDD are widely used for:

-   Model benchmarking
-   Feature evaluation
-   Comparative ML/DL analysis
-   Attack detection research

This framework was designed to provide reproducible, structured, and
research-oriented experimentation.

------------------------------------------------------------------------

## 🏗 System Architecture

. ├── train_classic_binary.py ├── train_classic_multiclass.py ├──
cnn_binary_runner.py ├── rawfile2Binary.py ├── rawfile2Multiclass.py ├──
trainingmodels.py ├── trainingmodels_multiclass.py ├──
trainingmodels_ovr.py ├── metricCalculator.py ├──
eval_global_and_difficulty.py ├── plot_utils.py ├──
log_train_distribution.py ├── results/ └── export.bib

------------------------------------------------------------------------

## 🔬 Classical Machine Learning --- Binary

Models implemented:

-   Logistic Regression
-   Gaussian Naive Bayes
-   K-Nearest Neighbors
-   Decision Tree
-   AdaBoost
-   Random Forest
-   Support Vector Machine (GridSearchCV)

Features:

-   Stratified splits
-   Dataset normalization
-   Hyperparameter tuning
-   Structured logging
-   Difficulty-level evaluation (NSL-KDD)
-   Unlabeled mode support

------------------------------------------------------------------------

## 🔬 Classical Machine Learning --- Multiclass & OvR

### Model A --- Multiclass Attack Classification

-   Direct attack-type prediction
-   Family aggregation
-   Per-family accuracy analysis
-   LaTeX report export

### Model B --- One-vs-Rest (OvR)

-   Independent classifier per attack-type
-   Attack-level detection metrics
-   Structured attack evaluation

------------------------------------------------------------------------

## 🧠 Deep Learning --- 1D CNN

Three architectures available:

-   CNN v1
-   CNN v2
-   CNN v3

Features:

-   Deterministic seeding
-   Checkpointing
-   CSV logging
-   Training history plots
-   Confusion matrix export
-   Model serialization (.keras)

------------------------------------------------------------------------

## 🚀 Running the Framework

Clone the repository:

git clone https://github.com/yourusername/your-repo.git cd your-repo

Create environment:

python3 -m venv venv source venv/bin/activate pip install -r
requirements.txt

Run pipelines:

python train_classic_binary.py python train_classic_multiclass.py python
cnn_binary_runner.py

------------------------------------------------------------------------

## 📈 Output Structure

results/ └── run_timestamp/ ├── metrics.txt ├── metrics.tex ├──
confusion_matrix.png ├── training_history.png ├── checkpoint-XX.keras
└── cnn_model_final.keras

------------------------------------------------------------------------

## 🔬 Scientific Context

This work is grounded in research on:

-   Intrusion Detection Systems
-   Supervised classification
-   Deep learning for anomaly detection
-   KDDCup99 and NSL-KDD benchmarks

Full bibliography available in:

export.bib

------------------------------------------------------------------------

## 👤 Author

Diogo Filipe\
Engineering Student --- NOVA FCT\
Focus Areas: Networking, Cybersecurity, Machine Learning

------------------------------------------------------------------------

## 📜 License

MIT License
