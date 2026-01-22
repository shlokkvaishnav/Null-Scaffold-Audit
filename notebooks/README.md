# SD-MoSE Notebooks

Run in order for the full discovery pipeline:

| Notebook | Purpose |
|----------|---------|
| **01_data_pipeline** | Download SOCAT + Chl, preprocess, train/test split |
| **02_baselines** | Linear, lat-band, RF, XGBoost, K-means+symbolic |
| **03_soft_regimes** | Train gating network (soft regimes) |
| **04_variable_discovery** | Feature importance, L1 sparsity |
| **05_dynamic_transitions** | Temporal consistency, Hovmöller-style dynamics |
| **06_constrained_discovery** | PySR law discovery per regime |
| **07_biology_gap** | Phys vs phys+bio, chlorophyll impact |
| **08_final_figures** | Ablations and summary figures |

**Setup:** From project root, run notebooks with `../` on `sys.path` so `scripts` and `src` are importable. Data paths: `data/raw`, `data/processed` (see `src/climate_discovery/config.py`).
