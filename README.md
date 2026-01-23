# SD-MoSE: Soft-Dynamic Mixture of Symbolic Experts regarding Air-Sea $CO_2$ Flux

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-arXiv%20Preprint-red)](https://arxiv.org/abs/TODO)

> **Abstract:**  
> Quantifying the air-sea $CO_2$ flux is critical for constraining the global carbon budget. While data-driven models (e.g., Neural Networks) offer high predictive accuracy, they lack transparency. **SD-MoSE** introduces a hybrid neuro-symbolic framework that autonomously partitions the global ocean into soft biogeochemical regimes and discovers interpretable physical laws (symbolic equations) governing gas exchange in each regime. By optimizing a weighted mixture of symbolic experts, SD-MoSE achieves accuracy comparable to black-box baselines while revealing the spatially varying drivers of carbon flux—reconciling the precision of Deep Learning with the interpretability of Physics.

---

## 1. Problem Formulation

Let $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ be a dataset of spatiotemporal observations, where $y \in \mathbb{R}$ represents the partial pressure of $CO_2$ ($pCO_2$) and $\mathbf{x} \in \mathbb{R}^d$ represents physical and biological drivers (e.g., SST, SSS, Chl-a).

We model the target variable $y$ as a mixture of $K$ distinct physical mechanisms ("regimes"), where the contribution of each mechanism varies dynamically based on the state $\mathbf{x}$.

## 2. Methodology

The **Soft-Dynamic Mixture of Symbolic Experts (SD-MoSE)** architecture decomposes the prediction problem into two coupled components:

1.  **Gating Network $\pi(\mathbf{x}; \phi)$**: A neural network that learns a state-dependent probability distribution over regimes.
2.  **Symbolic Experts $\{f_k(\mathbf{x}; \theta_k)\}_{k=1}^K$**: A set of interpretable mathematical equations, each optimizing fit for a specific regime.

### 2.1 Model Architecture

The global prediction $\hat{y}$ is given by the convex combination:

$$
\hat{y}(\mathbf{x}) = \sum_{k=1}^{K} \pi_k(\mathbf{x}) \cdot f_k(\mathbf{x})
$$

Where:
-   $\pi_k(\mathbf{x}) \in [0, 1]$ is the membership probability of point $\mathbf{x}$ in regime $k$, such that $\sum_{k=1}^K \pi_k(\mathbf{x}) = 1$.
-   $f_k(\mathbf{x})$ is a closed-form analytical expression discovered via Symbolic Regression.

### 2.2 Optimization: Alternating Minimization

Since the functional forms of $f_k$ are unknown *a priori*, we employ an iterative Expectation-Maximization (EM) inspired training loop:

1.  **M-Step (Gating Optimization)**: Fix the experts $f_k$ and train the parameters $\phi$ of the gating network to minimize the mixture loss with entropy regularization:
    $$ \mathcal{L}(\phi) = \sum_{i=1}^N \left( y_i - \sum_{k=1}^K \pi_k(\mathbf{x}_i; \phi) f_k(\mathbf{x}_i) \right)^2 - \lambda \sum_{i=1}^N H(\pi(\mathbf{x}_i)) $$
    where $H(\cdot)$ encourages distinct regime boundaries.

2.  **E-Step (Symbolic Discovery)**: Fix the gating probabilities $\pi_k$. For each regime $k$, we solve a weighted symbolic regression problem using genetic programming (PySR):
    $$ f_k^* = \arg\min_{f \in \mathcal{F}} \sum_{i=1}^N \pi_k(\mathbf{x}_i) (y_i - f(\mathbf{x}_i))^2 + \gamma \cdot \text{Complexity}(f) $$

---

## 3. Data & Features

The model is trained and evaluated on global ocean data spanning 2015-2024.

### 3.1 Datasets
*   **Surface Ocean $CO_2$ Atlas (SOCAT v2025)**: Monthly gridded $fCO_2$ observations. ($1^\circ \times 1^\circ$ resolution).
*   **Copernicus Marine Service (CMEMS)**: Satellite-derived Chlorophyll-a (GlobColour product).

### 3.2 Feature Set
*   **Gating Features ($\mathbf{x}_{gate}$)**: Used to determine regime membership.
    *   *Coords*: $\sin(\text{lat})$, $\sin(\text{lon})$, $\cos(\text{lon})$
    *   *State*: SST, SSS, $\log_{10}(\text{Chl-}a)$
    *   *Time*: $\sin(\text{month})$, $\cos(\text{month})$
*   **Expert Features ($\mathbf{x}_{expert}$)**: Used in the symbolic laws.
    *   SST (Sea Surface Temperature)
    *   SSS (Sea Surface Salinity)
    *   $\log_{10}(\text{Chl-}a)$

---

## 4. Installation & Reproduction

### Prerequisites
*   **Python 3.9+**
*   **Julia 1.9+** (Required for `SymbolicRegression.jl` backend)

### Setup Environment

```bash
# Clone repository
git clone https://github.com/shlokkvaishnav/climate-equation-discovery.git
cd climate-equation-discovery

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### Execution Pipeline

The `scripts/run_all.py` orchestrator handles data acquisition, preprocessing, training, and evaluation.

```bash
# 1. Download SOCAT and CMEMS data
python -m scripts.run_all --only data

# 2. Train SD-MoSE (approx. 4-6 hours on GPU)
#    - Trains gating network
#    - Discovers symbolic experts via PySR
python -m scripts.run_all --only train

# 3. Evaluate and Visualize
python -m scripts.run_all --only viz
```

**Note**: Training requires approximately 16GB RAM and is CUDA-accelerated if available. Metric computation follows standard benchmarking protocols (RMSE, MAE, $R^2$).

---

## 5. Baselines & Benchmarks

We compare SD-MoSE against standard ML baselines and traditional empirical parameterizations (e.g., Takahashi et al., 2009).

| Model Class | Architecture | RMSE (µatm) | $R^2$ | Interpretability |
| :--- | :--- | :--- | :--- | :--- |
| **Neuro-Symbolic** | **SD-MoSE (Proposed)** | **12.45** | **0.78** | **Symbolic** |
| Deep Learning | MLP (3-layer) | 12.10 | 0.80 | Black-box |
| Tree Ensemble | XGBoost | 11.92 | 0.81 | Black-box |
| Linear | Stepwise Regression | 25.40 | 0.45 | Analytic |

*Table 1: Test set performance (2022-2024). SD-MoSE achieves near-SOTA accuracy while maintaining full functional transparency.*

---

## 6. Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{Vaishnav2026sdmose,
  title={SD-MoSE: Soft-Dynamic Mixture of Symbolic Experts for Interpretable Air-Sea CO2 Flux Laws},
  author={Vaishnav, Shlok},
  journal={arXiv preprint},
  year={2026}
}
```

## 7. License

This project is open-source under the [MIT License](LICENSE).
