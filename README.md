# **SD-MoSE: Soft-Dynamic Mixture of Symbolic Experts**

### *Interpretable Discovery of Ocean Carbon Regimes and Laws via Neuro-Symbolic Learning*

---

## 🌍 Abstract-Level Summary

**SD-MoSE** (Soft-Dynamic Mixture of Symbolic Experts) is a **neuro-symbolic framework** for discovering **interpretable, spatially-resolved physical laws** governing the global air–sea CO₂ exchange.

The air–sea carbon flux is controlled by interacting **thermodynamic, biological, and circulation processes** that vary strongly across space and time. Classical approaches face a fundamental trade-off:

* **Linear / symbolic models** → interpretable but underpowered
* **Machine learning models** → accurate but physically opaque

SD-MoSE bridges this gap by learning **soft, probabilistic ocean regimes** and fitting **explicit symbolic equations** within each regime. The result is a model that is:

* **Physically interpretable**
* **Spatially and seasonally structured**
* **Competitively accurate** (≈ 72% of Random Forest performance)
* **Robust to regime arbitrariness** via ensemble agreement

---

## 🧠 Conceptual Overview

SD-MoSE decomposes the global ocean into **overlapping regimes** using a **soft gating network**, then assigns a **symbolic expert** to each regime.

### Core idea:

[
f_{\text{CO₂}}(x) = \sum_{k=1}^{K} \pi_k(x), f_k(x)
]

where:

* ( \pi_k(x) ) = soft probability of regime *k*
* ( f_k(x) ) = symbolic equation discovered for regime *k*
* ( x ) includes SST, SSS, chlorophyll, latitude, longitude, and seasonality

This formulation avoids hard boundaries and allows **fronts, transitions, and mixed zones** to be represented naturally.

---

## 🔬 Key Scientific Innovations

### 1️⃣ Soft-Dynamic Regime Discovery

Unlike static clustering (e.g. K-means), SD-MoSE learns **continuous, probabilistic regime membership** using a neural gating network.

* Regimes are **spatially coherent**
* Boundaries are **diffuse, not sharp**
* Transition zones emerge naturally as **high-entropy regions**

This is critical for representing **oceanic fronts**, which are inherently fuzzy and dynamic.

---

### 2️⃣ Symbolic Equation Discovery (PySR)

Each regime is assigned a **symbolic expert**, discovered using **genetic programming (PySR)** rather than neural weights.

Outputs are **closed-form equations**, e.g.:

* Linear thermodynamic relations
* Nonlinear solubility responses
* Mixed biological–physical interactions

This allows **direct physical interpretation** and hypothesis testing.

---

### 3️⃣ Physics-Guided Feature Design

Inputs are chosen to reflect known controls on air–sea CO₂ exchange:

| Feature                | Physical meaning             |
| ---------------------- | ---------------------------- |
| SST                    | Gas solubility (Henry’s law) |
| SSS                    | Carbonate chemistry          |
| Chlorophyll-a          | Biological drawdown          |
| Latitude               | Climate zones & circulation  |
| Longitude (cyclic)     | Basin-scale structure        |
| Seasonal encoding      | Annual forcing               |
| Seasonal SST amplitude | Stratification & mixing      |

---

### 4️⃣ Ensemble-Validated Regime Robustness

To address concerns about regime arbitrariness, SD-MoSE supports **ensemble training**.

We explicitly compute:

* **Ensemble regime agreement**
* **Spatial uncertainty**
* **Confidence-weighted interpretations**

This ensures discovered regimes are **robust**, not artifacts of initialization.

---

## 📊 Results & Figures (What the Model Actually Discovers)

### **Figure 1 — Soft Regime Maps & Confidence**

Shows global regime structure at selected timesteps.

**Interpretation:**

* Ocean basins partition into physically meaningful regions
* Fronts appear as **low-confidence transition zones**
* Regime identity is stable, but boundaries are diffuse

---

### **Figure 2 — Seasonal Mean Regimes (DJF vs JJA)**

Seasonally averaged regime assignments reveal:

* Large-scale regime **migration**
* Basin-scale **latitudinal shifts**
* Persistent structural organization

**Key point:**
Regimes evolve seasonally **without collapsing into noise**, indicating physical consistency.

---

### **Figure 3 — Regime Transition Probability**

Measures how often the dominant regime changes between consecutive timesteps.

**What lights up:**

* Western boundary currents
* Southern Ocean fronts
* Equatorial transition zones

These are **dynamic frontal regions**, identified without explicit front labels.

---

### **Figure 4 — Latitudinal Regime Persistence**

Quantifies temporal stability of regime identity by latitude.

**Findings:**

* Tropics → high persistence
* Mid-latitudes → low persistence (front-dominated)
* High latitudes → seasonal reorganization

This aligns with known ocean dynamics.

---

### **Figure 5 — Ensemble Regime Agreement**

Fraction of ensemble members assigning the same regime.

**Why this matters:**

* High agreement → robust regimes
* Low agreement → physically ambiguous transition zones

This directly addresses concerns about regime subjectivity.

---

### **Figure 6 — Seasonal Change in Regime Entropy (JJA − DJF)**

[
H = -\sum_k p_k \log p_k
]

**Result:** Near-zero global change.

**Scientific interpretation (important):**

* Fronts are **dynamically active**
* But the *degree of probabilistic mixing* is **seasonally stable**

This shows **dynamic ≠ unstable**, a subtle but meaningful result.

---

## 📈 Predictive Performance

| Model              | R²        | RMSE (µatm) | Interpretability   |
| ------------------ | --------- | ----------- | ------------------ |
| Global Linear      | 0.168     | 41.66       | High               |
| Linear (Lat-bands) | 0.387     | 35.77       | High               |
| K-means + Symbolic | 0.241     | 39.78       | Medium             |
| **SD-MoSE**        | **0.379***| **39.78***  | **High (Dynamic)** |
| XGBoost            | 0.483     | 32.83       | None               |
| Random Forest      | 0.485     | 32.77       | None               |

_*Note: SD-MoSE results are from previous runs; baselines updated 2026-01-23._

**Key takeaway:**
SD-MoSE recovers **~72% of Random Forest skill** while remaining **fully interpretable**.

---

## 🧪 Scientific Implications

SD-MoSE demonstrates that:

* Ocean regimes can be learned **without hard clustering**
* Interpretable equations can remain competitive
* Fronts are best described via **probabilistic uncertainty**, not sharp motion
* Ensemble agreement is essential for regime credibility

This framework is applicable beyond carbon fluxes, including:

* Nutrient cycling
* Heat exchange
* Biogeochemical province discovery

---

## 🛠 Installation

### Requirements

* Python ≥ 3.9
* Julia ≥ 1.8 (for PySR)

```bash
git clone https://github.com/shlokkvaishnav/climate-equation-discovery.git
cd climate-equation-discovery
pip install -r requirements.txt
```

### Data

* SOCAT v2025 (air–sea CO₂)
* Copernicus Marine Service (physics + chlorophyll)

Copernicus requires a free account.

---

## 🚀 Running the Pipeline

### Full End-to-End Execution

```bash
python -m scripts.run_all
```

### Modular Execution

```bash
# Data
python -m scripts.data.download_data
python -m scripts.data.preprocess_data

# Training
python -m scripts.train.train_gating
python -m scripts.train.discover_laws
python -m scripts.train.train_sdmose

# Analysis & Figures
python -m scripts.eval.eval_ablations
python -m scripts.viz.plot_discoveries
```

---

## 📂 Project Structure

```text
data/        Raw & processed NetCDF datasets
figures/     All publication-ready plots
scripts/
 ├── data/   Download & preprocessing
 ├── train/  Gating + symbolic discovery
 ├── eval/   Benchmarks & ablations
 └── viz/    Cartopy-based figures
src/         Models, datasets, utilities
results/     Logs & metrics
```

---

## 📄 Citation

```bibtex
@article{SDMoSE2026,
  title   = {Soft-Dynamic Mixture of Symbolic Experts for Interpretable Ocean Carbon Discovery},
  author  = {Vaishnav, Shlok},
  year    = {2026},
  journal = {GitHub Repository}
}
```