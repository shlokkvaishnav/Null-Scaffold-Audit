# SD-MoSE: Soft Regime Mixture of Symbolic Experts for Ocean $pCO_2$ Discovery

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Complete-success)](https://github.com/shlokkvaishnav/climate-equation-discovery)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)

> **Abstract:**  
> This project introduces **SD-MoSE**, a neuro-symbolic framework for discovering interpretable physical laws in the Global Ocean Carbon Cycle. by partitioning the ocean into dynamic "soft" regimes and learning localized symbolic experts, SD-MoSE achieves an **$R^2$ of 0.601** on held-out SOCAT data, significantly outperforming global linear baselines ($R^2=0.13$) and revealing the critical role of biological drawdown in specific ocean provinces.

---

## 🔬 Scientific Objective

The air-sea flux of Carbon Dioxide ($fCO_2$) is a critical component of the global climate system. Traditional approaches rely on either:
1.  **Global Linear Regressions**: Interpretable but inaccurate (fails to capture local physics).
2.  **Black Box ML (Neural Nets/XGBoost)**: Accurate but opaque (offers no physical insight).

**The SD-MoSE Hypothesis**: The ocean behaves as a collection of distinct physical regimes (e.g., Tropical, Temperate, Biological).
By learning a **Soft Gating Network** that weights these regimes Spatiotemporally, and using **Symbolic Regression (PySR)** to discover the local laws within them, we can achieve the best of both worlds: **Accuracy + Interpretability**.

---

## 🧠 Methodology

### 1. Neuro-Symbolic Architecture
The model consists of two coupled components:

*   **Gating Network ($\mathcal{G}$)**: A Neural Network (MLP) that maps spatiotemporal coordinates ($Lat, Lon, Time, SST$) to a probability distribution over $K$ regimes.
    $$ P(regime|x) = \text{Softmax}(\mathcal{G}(Lat, Lon, Time, ...)) $$

*   **Symbolic Experts ($\mathcal{E}_k$)**: $K$ distinct equations discovered via Genetic Programming (PySR). Each expert learns a local physical law $f_k(SST, SSS, Chl)$ optimized for its specific regime.
    $$ \hat{y} = \sum_{k=1}^{K} P(k|x) \cdot \mathcal{E}_k(x_{phys}) $$

### 2. Implementation Pipeline
The project is structured as a progressive discovery pipeline:
*   **Phase I (Regime Discovery)**: Unsupervised clustering to identify latent ocean provinces.
*   **Phase II (Dynamics)**: Training the Gating Network to model seasonal regime transitions (Hovmöller dynamics).
*   **Phase III (Law Discovery)**: Constrained Symbolic Regression to recover **Henry's Law** ($fCO_2 \propto T$) and **Biological Drawdown** ($fCO_2 \propto 1/Chl$).

---

## 📊 Results & Discovered Laws

### Final Ablation Study (Test Set 2020-2024)

| Method | Test $R^2$ | Interpretation |
| :--- | :--- | :--- |
| **Linear Baseline** | 0.133 | Fails to capture non-linear solubility & biology. |
| **Non-Linear (RF)** | 0.232 | Captures non-linearity but misses biological variables. |
| **Bio-Augmented** | 0.315 | Significant gain (+8%) from adding Chlorophyll. |
| **SD-MoSE (Final)** | **0.601** | **State-of-the-Art.** Seamlessly switches between physical and biological laws. |

### Discovered Equations

*   **Regime 0 (Biological Province)**:
    > **$fCO_2 \approx \frac{C}{\log(Chlorophyll)}$**
    
    *Interpretation*: Strong biological drawdown. High biological activity reduces partial pressure of $CO_2$.

*   **Regime 1 (Tropical Province)**:
    > **$fCO_2 \approx 19.3 \cdot SST + 375$**
    
    *Interpretation*: Thermodynamics dominate. Strong positive correlation with Sea Surface Temperature (Henry's Law).

*   **Regime 2 (Temperate Province)**:
    > **$fCO_2 \approx 16.7 \cdot SST + 365$**
    
    *Interpretation*: Moderate thermodynamic control, transitional zone.

---

## 📂 Repository Structure

The project is organized into reproducible **Notebooks**:

```tree
├── data/                 # Data storage (SOCAT + CMEMS)
├── notebooks/
│   ├── 01_data_pipeline.ipynb        # Preprocessing & Fusion
│   ├── 02_baselines.ipynb            # Linear & Symbolic Baselines
│   ├── 03_soft_regimes.ipynb         # Gating Network Training
│   ├── 04_variable_discovery.ipynb   # Feature Importance Analysis
│   ├── 05_dynamic_transitions.ipynb  # Seasonal Dynamics (Hovmöller)
│   ├── 06_constrained_discovery.ipynb# Physical Law Discovery (PySR)
│   ├── 07_biology_gap.ipynb          # Biology Gap Experiment
│   └── 08_final_figures.ipynb        # Final Ablation & Summary
├── scripts/
│   ├── preprocess.py     # Core ETL logic
│   └── download.py       # Data downloader
└── src/                  # Shared Model Code (PyTorch/PySR)
```

---

## 🚀 Usage

### 1. Installation
Requires Python 3.9+ and Julia (for PySR).

```bash
git clone https://github.com/shlokkvaishnav/climate-equation-discovery.git
cd climate-equation-discovery
pip install -r requirements.txt
```

### 2. Data Preparation
Download and preprocess the raw SOCAT and Satellite datasets:

```bash
python scripts/download.py
python scripts/preprocess.py
```

### 3. Reproduction
Run the notebooks in order (01 -> 08) to reproduce the full discovery pipeline. The final results are generated in `notebooks/08_final_figures.ipynb`.

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{SDMoSE2026,
  title={SD-MoSE: Soft Regime Mixture of Symbolic Experts for Ocean pCO2 Discovery},
  author={Vaishnav, Shlok},
  year={2026},
  journal={GitHub Repository}
}
```

**License**: MIT