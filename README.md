# 🧪 functional-group-atlas

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()

[cite_start]This repository contains the complete code and data workflow for the research paper **"A data-driven functional-group atlas for programming interfacial wettability and heat transport for thermal energy storage"**[cite: 1]. [cite_start]We present a machine learning-assisted design paradigm based on functional group deconstruction to resolve the intrinsic trade-off between energy density and power density in composite phase-change materials (PCMs)[cite: 1058, 1059].

[cite_start]By integrating Density Functional Theory (DFT), Ab Initio Molecular Dynamics (AIMD), and Stacking Ensemble learning, we screened a library of **248 candidates**[cite: 8, 27, 44]. [cite_start]This workflow successfully decouples interfacial properties, revealing that wettability is governed by elemental composition while heat transport is dictated by geometric topology[cite: 1060].

## 🚀 Key Features

* [cite_start]**Dimensional Feature Deconstruction** 🧬: Implements a novel strategy to decouple functional groups into independent "elemental" and "structural" dimensions, enabling programmable control over interfacial behaviors[cite: 1100].
* [cite_start]**Hierarchical Feature Engineering** 🛠️: Utilizes a rigorous three-stage "Filter-Embedded-Wrapper" protocol (Pearson Correlation/Mutual Information -> SHAP-based Coarse Selection -> Recursive Feature Elimination) to pinpoint the optimal feature set for both wettability ($E_b$) and thermal transport (NVOA)[cite: 54, 56, 61, 65].
* [cite_start]**Efficient Hyperparameter Optimization** ⚡: Leverages Gaussian Process-based Bayesian Optimization to efficiently navigate the hyperparameter landscape for 7 heterogeneous base learners (e.g., XGBoost, CatBoost, LightGBM), ensuring optimal model configurations[cite: 82, 1162].
* [cite_start]**Robust Stacking Ensemble** 🧠: Constructs a two-tier ensemble architecture integrating diverse tree-based models with a regularized ElasticNet meta-learner, achieving superior generalization ($R^2 > 0.92$) and robustness via nested cross-validation[cite: 72, 76, 1166].
* **Interpretable "Two-Level Weighted SHAP"** 📊: Features a custom **"Fidelity-Reliability Weighted Aggregation"** strategy. [cite_start]This method weighs feature contributions based on both intra-fold model accuracy and inter-fold generalization, providing a robust, noise-filtered physicochemical interpretation[cite: 95, 1330].
* [cite_start]**Dual-Target Prediction** 🎯: Validated workflow for two distinct physical properties—Interfacial Binding Energy (Wettability) and Vibrational Density of States Overlap (Thermal Conductivity)[cite: 19, 37].

---

## 📂 Repository Structure and Workflow

The repository is organized to support the dual-target analysis presented in the manuscript. The workflow utilizes two core Jupyter notebooks to handle feature engineering and model training/prediction respectively.

. ├── Functional Group Atlas/ # 🧪 Main Project Directory │ ├── Feature engineering-FGA.ipynb # 📜 Script 1: Hierarchical Feature Selection Pipeline │ ├── Tree_stacking-FGA.ipynb # 📜 Script 2: Hyperparameter Optimization, Stacking Ensemble & Prediction │ ├── Original database-CR.xlsx # 📊 Raw input database containing 248 candidates with descriptors │ ├── final_engineered_dataset-VDOS.xlsx # 📊 Output from Script 1 (Cleaned feature set for NVOA/Eb) │ ├── yuceji-VDOS.xlsx # 📊 Prediction set (Unknown data for screening) │ └── SHAP_Analysis_Results.xlsx # 📊 Final interpretability output │ ├── environment.yml # 📦 Conda environment config for cross-platform setup ├── requirements.txt # 📦 Pip dependencies for basic/non-Conda setup ├── spec-file.txt # 📦 Exact Conda config for highest-fidelity reproducibility (Win x64) ├── LICENSE # 📜 The MIT License file └── README.md # 📄 The document you are currently reading
