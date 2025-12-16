**Copy & Paste this directly into your `README.md` file:**```markdown
# 🌍 Climate Equation Discovery: AI-Driven Ocean Carbon Laws

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Benchmark-orange)
![CI](https://github.com/shlokkvaishnav/climate-equation-discovery/actions/workflows/ci.yml/badge.svg)

> **Result:** Built an **Autonomous AI Scientist** that "rediscovers" the laws of ocean physics from scratch. The agent segments the global ocean into regimes, enforces thermodynamic constraints via Neural Networks, and detects seasonal state shifts over time.

[📄 **Read the Technical Report (PDF)**](Technical_Report.pdf)

---

## 🔬 The Mission
Climate models (like CMIP6) are computationally expensive black boxes. This project investigates whether **Neuro-Symbolic AI** can autonomously rediscover the governing differential equations of the Global Ocean Carbon Cycle directly from sparse, noisy satellite & buoy data, **without being explicitly told the laws of physics.**

---

## 🧠 The "AI Scientist" Architecture
We built a full-stack cognitive architecture that mimics the scientific discovery process:

| Component | Cognitive Skill | Technical Implementation |
| :--- | :--- | :--- |
| **👀 Perception** | **Geography** | **Unsupervised Clustering (K-Means)** to automatically discover thermodynamic provinces (e.g., separating Atlantic vs. Pacific). |
| **🧠 Reasoning** | **Math Discovery** | **Symbolic Regression (PySR)** to derive interpretable equations ($fCO_2 = f(T, S, t)$) for each regime. |
| **⚖️ Validation** | **Physics Rules** | **Physics-Informed Neural Networks (PINNs)** to enforce thermodynamic stability and smoothness constraints. |
| **⏳ Adaptation** | **Time Dynamics** | **Hidden Markov Models (HMMs)** to detect dynamic regime switching (e.g., Seasonal/El Niño shifts). |

---

## 🧪 Key Experiments & Findings

### 1. Autonomous Geography Discovery (Spatial Regimes)
* **Goal:** Solve the "Global Heterogeneity" problem (Equator physics $\neq$ Polar physics).
* **Result:** The agent autonomously segmented the global ocean into 6 thermodynamic zones. It separated the **North Atlantic (Green)** from the **Pacific (Orange)** purely based on data signatures, without being given any coordinates.

![Physics Regimes Map](physics_regimes_map.png)
*Figure 1: The AI autonomously "learned geography," identifying distinct thermodynamic provinces.*

### 2. Dynamic Regime Detection (Temporal Regimes)
* **Goal:** Detect how physics changes *over time* (Seasons/Climate Events) at a single location.
* **Result:** Using a **Hidden Markov Model (HMM)**, the agent identified that the North Atlantic oscillates between two distinct physical states ("Winter Sink" vs. "Summer Source").
* **Insight:** The graph below shows the AI detecting the seasonal "heartbeat" of the ocean (Blue/Red switching) while preserving the long-term Anthropogenic Trend (upward slope).

![Dynamic Regime Switching](dynamic_regime_hmm.png)
*Figure 2: HMM-detected state transitions. The AI autonomously labeled "Winter" (Blue) and "Summer" (Red) regimes from raw data.*

---

## 🏆 Benchmark Performance

| Experiment | Method | $R^2$ Score | Verdict |
| :--- | :--- | :--- | :--- |
| **Global Naive** | Standard SR | `0.14` | **Failed.** Cannot handle conflicting global physics. |
| **Hybrid Agent** | **Clustered SR** | **`0.25`** | **+79% Boost.** Proves "Spatial Attention" is required. |
| **Physics-Informed** | **PINN (PyTorch)** | `MSE: 0.08` | **Valid.** Enforces thermodynamic smoothness constraints. |

---

## 🛠️ Tech Stack
* **Core AI:** `PyTorch` (PINNs), `PySR` (Symbolic Regression), `hmmlearn` (Markov Models), `scikit-learn` (Clustering).
* **Data Science:** `xarray` (NetCDF), `pandas`, `numpy`.
* **Engineering:** `pytest` (Unit Testing), `GitHub Actions` (CI/CD).

## 🚀 Reproduction Steps
This project is fully reproducible.

### 1. Installation
```bash
git clone [https://github.com/shlokkvaishnav/climate-equation-discovery.git](https://github.com/shlokkvaishnav/climate-equation-discovery.git)
cd climate-equation-discovery
pip install -r requirements.txt

```

###2. Run the Pipeline```bash
# 1. Download & Process Data
python src/data/downloader.py
python src/data/process_data.py

# 2. Run the AI Scientist (Regime Discovery + Equation Search)
python src/discovery_global_clustered.py

# 3. Train the Physics-Informed Neural Network (PINN)
python src/pinn_solver.py

# 4. Analyze Dynamic Regimes (HMM)
python src/dynamic_regimes.py

```

---

*Author: Shlok Vaishnav | Status: Research Complete*

```

```