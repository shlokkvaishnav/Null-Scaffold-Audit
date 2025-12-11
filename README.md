# 🌍 Climate Equation Discovery: AI-Driven Ocean Carbon Laws

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Benchmark-orange)

> **Result:** Built an **Adaptive Neuro-Symbolic Agent** that autonomously segments the global ocean into physics regimes (+79% accuracy boost) and discovers governing laws from raw satellite data.

## 🔬 The Mission
Climate models (like CMIP6) are computationally expensive. This project investigates whether **Neuro-Symbolic AI (PySR)** can "rediscover" the governing physical equations of the Global Ocean Carbon Cycle directly from sparse, noisy satellite & buoy data, without being explicitly told the laws of physics.

---

## 🧪 The Experiment Arc (Methodology)
We benchmarked Symbolic Regression on its ability to handle **Global Heterogeneity**—the fact that ocean physics changes depending on where you are (e.g., Equator vs. Poles).

### 🔴 Phase 1: The Global Attempt (Failure)
* **Goal:** Discover a single "Universal Law" for Global Ocean CO2 ($fCO_2 = f(SST, Salinity, Year)$).
* **Result:** $R^2 = 0.14$ (Poor).
* **Why it failed:** The ocean is **physically heterogeneous**. The AI could not reconcile opposing regimes (e.g., Equatorial Outgassing vs. Polar Sinks) into one simple equation, essentially guessing the global average.

### 🟢 Phase 2: The "Level 3" Upgrade (Global Hybrid Agent)
* **Goal:** Solve the heterogeneity problem using **Unsupervised Learning (K-Means)** to automatically detect physics regimes *before* applying regression.
* **Method:** The AI clustered the global ocean into 6 thermodynamic zones based on SST, Latitude, and Longitude.
* **Result:** **$R^2 = 0.25$ (+79% Performance Boost).**
* **Discovery:** The AI autonomously "learned geography." It separated the **North Atlantic** (Green) from the **Pacific** (Orange) and identified **Equatorial Upwelling Zones** (Red) without being provided any geography labels.

![Physics Regimes Map](physics_regimes_map.png)
*Figure 1: The AI autonomously segmented the global ocean into distinct thermodynamic provinces.*

---

## 🏆 Key Findings

| Experiment | Method | $R^2$ Score | Key Insight |
| :--- | :--- | :--- | :--- |
| **Global Naive** | Standard SR | `0.14` | Failed due to physical contradictions (Equator vs Poles). |
| **Global Hybrid** | **Clustered SR** | **`0.25`** | **+79% Boost.** Proves that "Spatial Attention" is required for global modeling. |

### 🧮 Discovered Physics by Regime
The Hybrid Agent discovered that different ocean regions obey different physical laws.

| Regime | Discovered Equation (Simplified) | Physics Interpretation |
| :--- | :--- | :--- |
| **0** | $fCO_2 \approx Year + SST \cdot \sin(-0.86 \cdot Salinity) - 1631$ | **Salinity Interaction.** Detected complex interaction between Salinity cycles and SST. |
| **1** | $fCO_2 \approx SST + 1.76 \cdot (Year + 13.6 \cdot \cos(0.17 \cdot SST)) - 3198$ | **Non-Linear Thermodynamics.** Found a highly non-linear temperature response distinct from other regimes. |
| **2** | $fCO_2 \approx SST + 348$ | **Pure Thermodynamics.** Simple solubility pump (Warmer = More CO2). |
| **3** | $fCO_2 \approx 2 \cdot SST + \cos(t) \cdot (31.9 - SST) + Year - 1681$ | **Dampened Seasonality.** Strong interaction term where Temperature dampens the Seasonal cycle. |
| **4** | $fCO_2 \approx SST + Year - 1662$ | **Linear Driver.** Dominated by the long-term anthropogenic trend + Temperature. |
| **5** | $fCO_2 \approx SST + 352$ | **Pure Thermodynamics.** Identical structure to Regime 2 but with a different solubility constant. |

---

## 📊 Model Benchmarking (Global Scale)
We compared our **Adaptive Hybrid Agent** against standard baselines on the full Global dataset.

| Model | $R^2$ Score | Interpretability | Insight |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | `0.15` | 🌗 Gray Box | **Fails.** Cannot handle global complexity (heterogeneity). |
| **Hybrid PySR (Ours)** | **`0.25`** | 🌕 **White Box** | **+66% Improvement.** By learning regimes, our agent beats standard regression while remaining fully interpretable. |
| **Random Forest** | `0.41` | 🌑 Black Box | **The "Upper Bound."** Even powerful black-box models struggle globally, proving the difficulty of this task. |

---

## 🛠️ Tech Stack
* **Core Science**: `xarray` (NetCDF handling), `numpy`, `scikit-learn`
* **Discovery Engine**: `PySR` (Symbolic Regression)
* **Methodology**: K-Means Clustering + Genetic Algorithms

## 🚀 Reproduction Steps
This project is fully reproducible.

### 1. Installation
```bash
# Clone the repo
git clone [https://github.com/shlokkvaishnav/climate-equation-discovery.git](https://github.com/shlokkvaishnav/climate-equation-discovery.git)
cd climate-equation-discovery

# Install dependencies
pip install -r requirements.txt
````

### 2\. Data Ingestion

Download the official SOCAT v2025 dataset (\~4GB) automatically:

```bash
python src/data/downloader.py
```

### 3\. Processing

Clean, filter, and feature-engineer the raw NetCDF into training-ready Parquet files:

```bash
python src/data/process_data.py
```

### 4\. Run the Discovery

Run the **Global Hybrid Agent** (Clustering Experiment):

```bash
python src/discovery_global_clustered.py
```

-----

*Author: Shlok Vaishnav | Status: Research Benchmark Complete*

````

---