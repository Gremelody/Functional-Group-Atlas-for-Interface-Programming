# 🔬 Functional Group Atlas (FGA) Design Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/10.5281/zenodo.placeholder)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **A data-driven, machine learning-assisted design paradigm for programming interfacial wettability and heat transport in thermal energy storage materials.**

This repository contains the source code and datasets for the manuscript: **"A data-driven functional-group atlas for programming interfacial wettability and heat transport for thermal energy storage"**.

The framework introduces a **feature deconstruction** strategy, resolving surface modifiers into independent "elemental" and "structural" dimensions. By integrating Density Functional Theory (DFT), Ab Initio Molecular Dynamics (AIMD), and Two-Tier Stacking Ensemble Learning, this workflow decouples the entangled properties of interfacial wettability ($E_b$) and thermal conductivity (NVOA).

---

## ✨ Highlights

*   **Dimensional Decoupling**: Implements a novel encoding strategy to separate chemical composition from geometric topology, revealing physical orthogonality in material performance.
*   **Hierarchical Feature Engineering**: Features a rigorously automated "Filter-Embedded-Wrapper" pipeline utilizing Mutual Information, SHAP-based importance, and Recursive Feature Elimination (RFE).
*   **Robust Stacking Ensemble**: A two-tier architecture combining heterogeneous base learners (XGBoost, CatBoost, ExtraTrees, etc.) with an **ElasticNet meta-learner**, validated via 100-seed nested cross-validation.
*   **Interpretable AI**: Incorporates **Unified Feature Importance (UFI)** analysis to quantify physicochemical drivers, bridging the gap between black-box ML and physical intuition.

---

## 🛠️ System Requirements & Installation

### Prerequisites
*   **OS**: Windows 10/11 (Tested), Linux (Ubuntu 20.04+), or macOS.
*   **Python**: Version 3.10.x recommended.
*   **Hardware**: Standard workstation (≥16GB RAM recommended for Stacking ensembles).

### Installation
We recommend using [Conda](https://docs.conda.io/en/latest/) to manage dependencies and ensure reproducibility.

```bash
# Clone the repository
git clone https://github.com/Gremelody/functional-group-atlas.git
cd functional-group-atlas

# Create the environment from the lock file (Recommended for exact reproducibility)
conda create --name fga-env --file spec-file.txt

# Activate the environment
conda activate fga-env
```

*Alternatively, for a minimal setup:*
```bash
pip install -r requirements.txt
```

---

## 📂 Repository Structure

The workflow is organized into two parallel pipelines corresponding to the dual physical targets: **Binding Energy ($E_b$)** and **Vibrational Density of States Overlap Area (NVOA)**.

```text
.
├── Binding_Energy_Eb/                  # 💧 Target 1: Interfacial Wettability
│   ├── Feature Engineering-Eb.ipynb    # Stage 1: Hierarchical feature selection
│   ├── Tree_stacking-Eb.ipynb          # Stage 2: Optimization, Stacking & Prediction
│   ├── Final_engineered_dataset.xlsx   # Processed dataset (Output of Stage 1)
│   └── prediction-FGA-Eb.xlsx          # External candidate library for screening
│
├── NVOA_Phonon/                        # 🌡️ Target 2: Heat Transport
│   ├── Feature Engineering-NVOA.ipynb  # Stage 1: Hierarchical feature selection
│   ├── Tree_stacking-NVOA.ipynb        # Stage 2: Optimization, Stacking & Prediction
│   ├── Final_engineered_dataset.xlsx   # Processed dataset (Output of Stage 1)
│   └── prediction-FGA-NVOA.xlsx        # External candidate library for screening
│
├── data/                               # 🧬 Raw Data
│   ├── Original database-FGA.xlsx      # Master database with 69 quantum-chemical descriptors
│   └── Feature_Definitions.md          # Description of elemental/structural features
│
├── spec-file.txt                       # Exact Conda environment specification
└── README.md                           # Documentation
```

---

## 🚀 Usage Guide

The pipeline is modular. You may run the feature engineering and modeling steps independently.

### Step 1: Hierarchical Feature Engineering
**Objective**: Distill high-dimensional quantum descriptors into an optimal, physically meaningful subset.

*   **Input**: `Original database-FGA.xlsx`
*   **Script**: `Feature Engineering-*.ipynb`
*   **Methodology**:
    1.  **Filter**: Removal of collinear features via Pearson threshold (`0.8`) and Mutual Information maximization.
    2.  **Embedded**: Coarse screening using Random Forest-based SHAP values.
    3.  **Wrapper**: Iterative Recursive Feature Elimination (RFE) with early stopping based on $R^2$.
*   **Output**: `Final_engineered_dataset-*.xlsx`

### Step 2: Stacking Ensemble & Prediction
**Objective**: Train a robust predictive model and screen the functional group atlas.

*   **Input**: `Final_engineered_dataset-*.xlsx`
*   **Script**: `Tree_stacking-*.ipynb`
*   **Methodology**:
    1.  **Bayesian Optimization**: Tunes hyperparameters for 7 base learners (XGB, LGBM, CatBoost, RF, etc.) using Gaussian Processes.
    2.  **Stacking**: Trains Level-0 base models and a Level-1 ElasticNet meta-learner.
    3.  **Validation**: Performs **100-seed nested cross-validation** to generate robust performance metrics and SHAP analysis.
    4.  **Prediction**: Applies the ensemble to `prediction-FGA-*.xlsx` to identify high-performance candidates.
*   **Output**: Performance logs, SHAP plots, and `unknown_predictions_*.xlsx`.

---

## 📊 Data Description

The `Original database-FGA.xlsx` contains 248 functional group candidates with 69 descriptors derived from DFT/AIMD calculations.

| Feature Type | Dimensions | Description |
| :--- | :--- | :--- |
| **Elemental ($x_{elem}$)** | 40 | Intrinsic properties of terminal moieties (e.g., Electronegativity, Atomic Radius, Valency). |
| **Structural ($x_{struct}$)** | 29 | Topological characteristics of the backbone (e.g., Gyration Radius, Geometric Inertia). |
| **Targets ($y$)** | 2 | **$E_b$** (Adsorption Energy, eV) and **NVOA** (Normalized VDOS Overlap Area). |

---

## 📜 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@article{Zhu2026FGA,
  title={A data-driven functional-group atlas for programming interfacial wettability and heat transport for thermal energy storage},
  author={Zhu, Yifei and Wang, Tiansheng and Zhou, Guangmin},
  journal={Submitted to Nature XXX},
  year={2026},
  doi={10.1038/s41xxx-xxx-xxxx-x}
}
```

## 📧 Contact

For technical questions or collaboration inquiries, please contact:
*   **Prof. Guangmin Zhou**: [guangminzhou@sz.tsinghua.edu.cn](mailto:guangminzhou@sz.tsinghua.edu.cn)
*   **Yifei Zhu**: [zhuyifeiedu@126.com](mailto:zhuyifeiedu@126.com)

---

**License**: [MIT](LICENSE)
```
