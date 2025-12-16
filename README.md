# 🌍 Climate Equation Discovery: The AI Physicist

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Benchmark-orange)
![CI](https://github.com/shlokkvaishnav/climate-equation-discovery/actions/workflows/ci.yml/badge.svg)

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
| **2** | **Symbolic Regression** | **🧠 Reasoning** | We need **White-Box Equations** (Math), not just predictions. This derives $fCO_2 = f(T, S, t)$. |
| **3** | **PINN (PyTorch)** | **⚖️ Validation** | Pure data is noisy. We use **Physics-Informed NNs** to force the model to obey thermodynamic laws (smoothness). |
| **4** | **HMM (Markov)** | **⏳ Adaptation** | The ocean isn't static. We use **Hidden Markov Models** to detect when physics changes over time (Seasons). |

---

## 🧪 The Experiment Arc (Methodology)

### 🔴 Phase 1: The Global Attempt (Failure)

- **Goal:** Discover a single "Universal Law" for Global Ocean CO2 ($fCO_2 = f(SST, Salinity, Year)$).
- **Result:** $R^2 = 0.14$ (Poor).
- **Why it failed:** The ocean is **physically heterogeneous**. A single equation cannot reconcile opposing regimes (e.g., Equatorial Outgassing vs. Polar Sinks), leading the AI to merely guess the global average.

### 🟢 Phase 2: The Upgrade (Global Hybrid Agent)

- **Goal:** Solve the heterogeneity problem using **Unsupervised Learning (K-Means)** to automatically detect physics regimes *before* applying regression.
- **Result:** **$R^2 = 0.25$ (+79% Performance Boost over Naive SR).**
- **Discovery:** The AI autonomously "learned geography." It separated the **North Atlantic** (Green) from the **Pacific** (Orange) and identified **Equatorial Upwelling Zones** (Red) without being provided any geography labels.

![Physics Regimes Map](physics_regimes_map.png)

#### 🧮 Discovered Physics by Regime

The Hybrid Agent discovered that different ocean regions obey different physical laws:

| Regime | Discovered Equation (Simplified) | Physics Interpretation |
|:-------|:---------------------------------|:-----------------------|
| **0** | $fCO_2 \approx Year + SST \cdot \sin(-0.86 \cdot Salinity)$ | **Salinity Interaction.** Detected complex interaction between Salinity cycles and SST. |
| **1** | $fCO_2 \approx SST + 1.76 \cdot (Year + 13.6 \cdot \cos(0.17 \cdot SST))$ | **Non-Linear Thermodynamics.** Found a highly non-linear temperature response distinct from other regimes. |
| **3** | $fCO_2 \approx 2 \cdot SST + \cos(t) \cdot (31.9 - SST)$ | **⭐ The "Damping" Effect.** Discovered that High Temperature ($SST \approx 31.9$) *turns off* the Seasonal cycle. |
| **4** | $fCO_2 \approx SST + Year - 1662$ | **Linear Driver.** Region dominated by the long-term anthropogenic trend + Temperature. |

### 🔵 Phase 3: Physics-Informed Neural Networks (PINNs)

- **Goal:** Solve the "Black Box" issue. Standard Neural Networks can output physically impossible values if the data is noisy.
- **Method:** We implemented a **PINN** with a custom loss function: $\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}$.
- **Result:** The model successfully learned to minimize error while satisfying physical constraints.

### 🟣 Phase 4: Dynamic Regime Detection (HMM)

- **Goal:** Solve the "Static Map" issue. Phase 2 assumed the map never changes, but the ocean has seasons.
- **Method:** A **Hidden Markov Model (HMM)** analyzed time-series data to detect state changes.
- **Result:** The AI autonomously labeled "Winter" (Blue) and "Summer" (Red) regimes, tracking the ocean's seasonal heartbeat.

![HMM Seasonal Regimes](dynamic_regime_hmm.png)

---

## 🏆 Benchmark Performance

We compared our **Adaptive Hybrid Agent** against standard baselines on the full Global dataset.

| Model | R² Score | Type | Insight |
|:------|:---------|:-----|:--------|
| **Linear Regression** | `0.15` | 🌗 Gray Box | **Fails.** Cannot handle global complexity (heterogeneity). |
| **Hybrid PySR (Ours)** | **`0.25`** | 🌕 **White Box** | **+66% Improvement.** By learning regimes, our agent beats standard regression while remaining fully interpretable. |
| **Random Forest** | `0.41` | 🌑 Black Box | **The Upper Bound.** Even powerful black-box models struggle globally, proving the extreme difficulty of this dataset. |

---

## 🛠️ Tech Stack

- **Core AI:** `PyTorch` (PINNs), `PySR` (Symbolic Regression), `hmmlearn` (Markov Models), `scikit-learn`.
- **Data Science:** `xarray` (NetCDF), `pandas`, `numpy`.
- **Engineering:** `pytest`, `GitHub Actions` (CI/CD).

---

## 🔮 Future Roadmap: The "Unified Theory"

While this project successfully treats **Space** (Clustering) and **Time** (HMMs) as separate discovery problems, the theoretical endpoint of this research is a **Unified Spatiotemporal Model**.

**The Next Integration:**

1. **Grid-Based HMMs:** Run a separate Hidden Markov Model on every single 1° × 1° grid point in the ocean.
2. **4D Labeling:** Generate a dynamic label for every point in Space and Time (e.g., `(Lat, Lon, Time) -> "North Atlantic Winter Regime"`).
3. **Spatiotemporal Equation Discovery:** Feed these 4D labels into PySR.
   - **Result:** Instead of one equation for the Atlantic, the AI would discover **Equation A for Atlantic Winter** and **Equation B for Atlantic Summer**.

This approach would solve the "Regime Blurring" problem where seasonal transition months reduce equation accuracy.

---

## 🚀 Reproduction Steps

This project is fully reproducible.

```bash
# 1. Install Dependencies
git clone https://github.com/shlokkvaishnav/climate-equation-discovery.git
cd climate-equation-discovery
pip install -r requirements.txt

# 2. Download & Process Data (SOCAT v2025)
python src/data/downloader.py
python src/data/process_data.py

# 3. Run the AI Scientist (Regime Discovery + Equation Search)
python src/discovery_global_clustered.py

# 4. Train the Physics-Informed Neural Network (PINN)
python src/pinn_solver.py

# 5. Analyze Dynamic Regimes (HMM)
python src/dynamic_regimes.py
```

---

## 📄 License

MIT License

---

## 👤 Author

**Shlok Vaishnav**

*Status: Research Complete*