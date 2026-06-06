# NeuroInsight-AI: Explainable Parkinson's Disease Prediction using Vocal Acoustics

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Framework](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)

An explainable machine learning repository dedicated to the early, non-invasive detection of Parkinson's Disease (PD) using vocal acoustics, the **Feature Performance Index (fPI)**, and advanced classifiers like **XGBoost**.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement & Explainable AI](#-problem-statement--explainable-ai)
3. [Dataset Description](#-dataset-description)
4. [Methodology & the fPI Analyser](#-methodology--the-fpi-analyser)
5. [Model Development & Performance](#-model-development--performance)
6. [Repository Structure](#-repository-structure)
7. [Installation & Getting Started](#-installation--getting-started)
8. [Notebooks Directory Map](#-notebooks-directory-map)
9. [Future Work](#-future-work)
10. [License](#-license)

---

## 🔍 Project Overview

Parkinson's Disease is a progressive neurodegenerative disorder primarily characterized by the loss of dopaminergic neurons in the brain, leading to motor symptoms such as tremors, rigidity, and vocal impairment. Dysphonia (impairment in the ability to produce voice sounds) is one of the earliest indicators of Parkinson's Disease, often appearing years before motor symptoms manifest.

**NeuroInsight-AI** leverages advanced machine learning techniques to detect Parkinson's Disease from voice-acoustic features. By introducing the **Feature Performance Index (fPI)**—a combined nonlinear feature index—this repository demonstrates how domain-specific feature engineering can improve model explainability and classification performance.

---

## 🧠 Problem Statement & Explainable AI

Traditional diagnostic methods for Parkinson's Disease rely on clinical evaluations of motor symptoms, which are often subjective and occur late in the disease progression. Voice-acoustic analysis offers a non-invasive, objective, and low-cost alternative for early screening.

However, medical machine learning models must not only be accurate but also **explainable**. Clinicians need to understand *why* a model makes a prediction. This project focuses on **Explainable AI (XAI)** by utilizing specific acoustic features (entropy, fractal dimensions, and pitch variation) and evaluating interpretable models like Decision Trees alongside high-performance ensemble methods like Random Forests and XGBoost.

---

## 📊 Dataset Description

The analysis is conducted on a dataset containing biomedical voice measurements from **198 subjects** (140 diagnosed with Parkinson's Disease, 58 healthy controls). The dataset contains 22 vocal acoustic features:

*   **Vocal Fundamental Frequency**: Average, maximum, and minimum fundamental frequency (`MDVP:Fo (Hz)`, `MDVP:Fhi (Hz)`, `MDVP:Flo (Hz)`).
*   **Jitter Measures**: Variations in fundamental frequency (`MDVP:Jitter(%)`, `MDVP:Jitter(Abs)`, `MDVP:RAP`, `MDVP:PPQ`, `Jitter:DDP`).
*   **Shimmer Measures**: Variations in amplitude (`MDVP:Shimmer`, `MDVP:Shimmer(dB)`, `Shimmer:APQ3`, `Shimmer:APQ5`, `MDVP:APQ`, `Shimmer:DDA`).
*   **Noise-to-Harmonic Ratios**: NHR and HNR, measuring the ratio of noise to tonal components in the voice.
*   **Nonlinear Dynamical Complexity**: measures of chaos and signal complexity (`RPDE`, `D2`).
*   **Signal Fractal Scaling Exponent**: `DFA`, indicating the fractal dimensions of the voice signal.
*   **Frequency Variation Measures**: Nonlinear measures of fundamental frequency variation (`spread1`, `spread2`, `PPE`).
*   **Status**: Target label (`1` for Parkinson's Disease, `0` for Healthy).

---

## ⚙️ Methodology & the fPI Analyser

```
  ┌───────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
  │   Voice Acoustic  │ ───> │  Feature Engineering: fPI │ ───> │    Data Preprocessing   │
  │    Data Source    │      │  (log10(DFA*D2*spread2))  │      │  (Scaling & Splitting)  │
  └───────────────────┘      └───────────────────────────┘      └─────────────────────────┘
                                                                             │
                                                                             ▼
  ┌───────────────────┐      ┌───────────────────────────┐      ┌─────────────────────────┐
  │  Prediction and   │ <─── │   Ensemble Classifiers    │ <─── │     Model Training &    │
  │   Explainability  │      │     (XGBoost, RF, SVM)    │      │     GridSearch HPO      │
  └───────────────────┘      └───────────────────────────┘      └─────────────────────────┘
```

A key highlight of this project is the **Feature Performance Index (fPI)**, developed based on correlation analyses of the acoustic features:

$$\text{fPI} = \log_{10}(\text{DFA} \times \text{D2} \times \text{spread2})$$

*   **D2**: Nonlinear dynamical complexity (correlation dimension).
*   **DFA**: Signal fractal scaling exponent.
*   **spread2**: Nonlinear measure of fundamental frequency variation.

Combining these features into a single index captures the combined impact of nonlinear voice chaos, fractal scaling changes, and pitch variations, yielding a robust, explainable indicator for vocal deterioration.

---

## 📈 Model Development & Performance

Five distinct machine learning architectures were trained, tuned via Grid Search hyperparameter optimization, and evaluated across multiple metrics.

### Performance Summary Table

| Model | Accuracy | Class 0 Precision | Class 0 Recall | Class 1 Precision | Class 1 Recall | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **86.44%** | **0.86** | **0.78** | **0.87** | **0.90** | **0.85** |
| **Random Forest** (Tuned) | **85.00%** | 0.75 | 0.75 | 0.89 | 0.89 | 0.82 |
| **Decision Tree** (Tuned) | **82.00%** | 0.78 | 0.58 | 0.83 | 0.93 | 0.78 |
| **SVM** (Grid Searched) | **82.00%** | 0.78 | 0.58 | 0.83 | 0.93 | 0.78 |
| **K-Nearest Neighbors** | **79.00%** | 0.75 | 0.50 | 0.81 | 0.93 | 0.73 |

*Note: Detailed classification reports and training histories are fully documented in the respective notebooks.*

### Evaluation Plots

<div align="center">
  <img src="https://github.com/bhanmrinal/Predictive-Analysis-for-Parkinsons-Disease-using-ML/assets/97622240/037b3989-51b8-46d7-b163-6ea5af7ed9cd" alt="Metric Scores for ML Models" width="70%" />
  <p><em>Figure 1: Comparative Metric Scores for baseline Machine Learning Models.</em></p>

  <img src="https://github.com/bhanmrinal/Predictive-Analysis-for-Parkinsons-Disease-using-ML/assets/97622240/a114921b-c139-4809-a5e8-d2b7b9815b56" alt="ROC and Precision-Recall Curves" width="70%" />
  <p><em>Figure 2: Metric plots for True Positive and False Positive Estimation.</em></p>
</div>

---

## 📁 Repository Structure

```
neuroinsight-ai/
│
├── README.md               # Professional project presentation and overview
├── LICENSE                 # MIT License details
├── .gitignore              # Python and Jupyter notebook git exclusions
├── requirements.txt        # Exact dependencies required to run the project
│
├── data/                   # Structured, cleaned datasets
│   ├── pd_data.xlsx
│   └── pd_data.txt
│
├── notebooks/              # Modularized research and training code
│   ├── eda.ipynb           # Exploratory data analysis and fPI formulation
│   ├── pd_models.ipynb     # Baseline models (DT, RF, SVM, KNN) with HPO
│   └── xgboost_classifier.ipynb # Final high-performance XGBoost model
│
├── docs/                   # PDFs and background research documentation
│   ├── report.pdf
│   ├── parkinsons_prediction.pdf
│   └── problem_statement.pdf
│
├── assets/                 # High-resolution visual diagrams
│   └── decision_tree.png
│
└── testing/
    └── experimental/       # Legacy experiments and exploratory code
        ├── pd_alpha.ipynb
        └── pd_beta.ipynb
```

---

## 💻 Installation & Getting Started

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/shuhul/neuroinsight-ai.git
    cd neuroinsight-ai
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4.  **Run Jupyter**:
    ```bash
    jupyter notebook
    ```
    Navigate to the `notebooks/` directory and run any of the primary analysis scripts.

---

## 📓 Notebooks Directory Map

*   **[eda.ipynb](notebooks/eda.ipynb)**: Implements correlation matrices, distribution plots, and uncovers the relationship between fractal dimensions and frequency variation which led to the creation of the fPI.
*   **[pd_models.ipynb](notebooks/pd_models.ipynb)**: Details the baseline classification models, hyperparameter tuning via GridSearchCV, and exports model performance benchmarks.
*   **[xgboost_classifier.ipynb](notebooks/xgboost_classifier.ipynb)**: Focuses on the implementation of the final Extreme Gradient Boosting classifier, achieving the highest performance of **86.44%** accuracy.

---

## 🚀 Future Work

*   **Clinical Feature Expansion**: Integrate voice recordings from a wider demographic to improve model generalizability.
*   **Deep Learning (ANN)**: Test 1D Convolutional Neural Networks (CNNs) directly on voice wave signals.
*   **Real-time Interface**: Connect the models to a voice recorder web interface for instant risk evaluation.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
