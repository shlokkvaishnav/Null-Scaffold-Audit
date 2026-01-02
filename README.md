# 🌍 Climate Equation Discovery: The AI Physicist

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Benchmark-orange)](https://github.com/shlokkvaishnav/climate-equation-discovery)
[![CI](https://github.com/shlokkvaishnav/climate-equation-discovery/actions/workflows/ci.yml/badge.svg)](https://github.com/shlokkvaishnav/climate-equation-discovery/actions)

> **Result:** Built an **Autonomous AI Scientist** that segments the global ocean into physics regimes (+79% accuracy boost), enforces thermodynamic laws via Neural Networks, and detects seasonal state shifts.

---

## 🔬 The Mission

Climate models (like CMIP6) are computationally expensive black boxes. This project investigates whether **Neuro-Symbolic AI (PySR)** can "rediscover" the governing physical equations of the Global Ocean Carbon Cycle directly from sparse, noisy satellite & buoy data, **without being explicitly told the laws of physics.**

---

## 🧠 The "AI Scientist" Cognitive Architecture

To solve this, we built a pipeline that mimics the scientific method. This answers **"Why this specific architecture?"**:

| Step | Component | Cognitive Skill | Why we used it |
|:-----|:----------|:----------------|:---------------|
| **1** | **K-Means Clustering** | **👀 Perception** | The ocean is **Heterogeneous**. We need to find "regions" (e.g., Atlantic vs Pacific) before we can find equations. |
| **2** | **Symbolic Regression** | **🧠 Reasoning** | We need **White-Box Equations** (Math), not just predictions. This derives `fCO2 = f(T, S, t)`. |
| **3** | **PINN (PyTorch)** | **⚖️ Validation** | Pure data is noisy. We use **Physics-Informed NNs** to force the model to obey thermodynamic laws (smoothness). |
| **4** | **HMM (Markov)** | **⏳ Adaptation** | The ocean isn't static. We use **Hidden Markov Models** to detect when physics changes over time (Seasons). |

---

## 🧪 The Experiment Arc (Methodology)

### 🔴 Phase 1: The Global Attempt (Failure)
- **Goal:** Discover a single "Universal Law" for Global Ocean CO2 (`fCO2 = f(SST, Salinity, Year)`).
- **Result:** R² = 0.14 (Poor).
- **Why it failed:** The ocean is **physically heterogeneous**. A single equation cannot reconcile opposing regimes (e.g., Equatorial Outgassing vs. Polar Sinks).

### 🟢 Phase 2: The Upgrade (Global Hybrid Agent)
- **Goal:** Solve the heterogeneity problem using **Unsupervised Learning (K-Means)** to automatically detect physics regimes *before* applying regression.
- **Result:** **R² = 0.25 (+79% Performance Boost over Naive SR).**
- **Discovery:** The AI autonomously "learned geography" by separating the **North Atlantic** from the **Pacific** without labels.

### 🔵 Phase 3: Physics-Informed Neural Networks (PINNs)
- **Goal:** Solve the "Black Box" issue. Standard Neural Networks can output physically impossible values if the data is noisy.
- **Method:** We implemented a **PINN** with a custom loss function: `L_total = L_data + λ × L_physics`.
- **Result:** The model learned to minimize error while satisfying physical constraints.

### 🟣 Phase 4: Dynamic Regime Detection (HMM)
- **Goal:** Solve the "Static Map" issue. Phase 2 assumed the map never changes, but the ocean has seasons.
- **Method:** **Hidden Markov Model (HMM)** analyzed time-series data to detect state changes.
- **Result:** The AI autonomously labeled "Winter" and "Summer" regimes.

---

## 📂 Repository Structure

```tree
├── .github/              # CI/CD Workflows
├── checkpoints/          # Trained Model Artifacts (PINNs)
├── configs/              # Configuration files (YAML)
├── data/                 # Data directory (Raw & Processed)
├── figures/              # Generated plots and visualizations
├── notebooks/            # Research notebooks (Exploration, Training)
├── scripts/              # Executable scripts for Data & Discovery
├── src/                  # Source code for the package
│   └── climate_discovery/
│       ├── data/         # Data loading & processing
│       ├── models/       # AI Models (PINN, HMM, Symbolic)
│       └── physics/      # Physics constraints & equations
├── tests/                # Unit & Integration Tests
└── README.md             # Project Documentation
```

---

## 🚀 Reproduction Steps

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/shlokkvaishnav/climate-equation-discovery.git
cd climate-equation-discovery

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Data Preparation

Download and process the SOCAT v2025 dataset:
```bash
python scripts/download.py
python scripts/preprocess.py
```

### 3. Run Discovery

Run the AI Scientist regime discovery and equation search:
```bash
python scripts/discover.py
```

### 4. Development

Run tests to verify the setup:
```bash
pytest tests/
```

---

## 🏆 Benchmark Performance

| Model | R² Score | Type | Insight |
|:------|:---------|:-----|:--------|
| **Linear Regression** | `0.15` | 🌗 Gray Box | **Fails.** Cannot handle global complexity (heterogeneity). |
| **Hybrid PySR (Ours)** | **`0.25`** | 🌕 **White Box** | **+66% Improvement.** Beats standard regression while remaining fully interpretable. |
| **Random Forest** | `0.41` | 🌑 Black Box | **The Upper Bound.** Even powerful black-box models struggle globally. |

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Shlok Vaishnav**