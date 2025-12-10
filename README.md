# 🌍 Climate Equation Discovery: AI-Driven Ocean Carbon Laws

> **Goal:** Rediscovering the governing equations of Ocean Carbon Uptake ($pCO_2$) from raw satellite & buoy data using Symbolic Regression (Neuro-Symbolic AI).

## 🔬 The Problem
The relationship between **Sea Surface Temperature (SST)** and **Ocean Acidity ($pCO_2$)** is complex and changing due to anthropogenic climate change. Traditional interactions are modeled using complex thermodynamic simulations (e.g., CMIP6). 

This project asks: **Can an AI agent "rediscover" Henry's Law and its biological deviations purely from sparse, noisy observation data?**

## 🧪 Methodology
We utilize **Symbolic Regression** (via `PySR`) to search the space of mathematical expressions, optimizing for both accuracy (RMSE) and simplicity (Occam's Razor).

$$pCO_2 = f(SST, Salinity, Chl_a, t)$$

### The Pipeline
1.  **Data Ingestion**: Automated retrieval of **SOCAT v2025** (Surface Ocean CO₂ Atlas) Gridded Data.
2.  **Interpolation**: Recovering missing sensor data using **Gaussian Processes** (Kriging).
3.  **Discovery**: Running Genetic Algorithms to evolve algebraic equations.
4.  **Validation**: Testing discovered laws against known thermodynamic constants (Weiss, 1974).

## 🛠️ Tech Stack
* **Core Science**: `xarray`, `numpy`, `scipy`
* **Discovery Engine**: `PySR` (Symbolic Regression in Julia/Python)
* **Data Ops**: `DVC` (Data Version Control), `Hydra` (Config Management)
* **Visualization**: `matplotlib`, `seaborn`

## 🚀 Quick Start
```bash
# 1. Install Dependencies
pip install -r requirements.txt

# 2. Download Data (SOCAT v2025 ~4GB)
python src/data/downloader.py

# 3. Run the Discovery Experiment
python src/discovery.py
````

## 📚 Data Source

  * **SOCAT v2025**: Bakker, D. C. E., et al. (2016). A multi-decade record of high quality fCO2 data in version 3 of the Surface Ocean CO2 Atlas (SOCAT). *Earth System Science Data*.

-----

*Author: Shlok Vaishnav | Status: Research Preview*

````