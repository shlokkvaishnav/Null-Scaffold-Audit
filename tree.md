# Project directory tree

```
climate-equation-discovery/
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── checkpoints/
│   └── ocean_pinn.pth
├── configs/
│   └── pysr_config.yaml
├── data/
│   ├── 01_raw/
│   │   ├── cmems_mod_glo_bgc_my_0.25deg_P1M-m_chl.nc
│   │   └── SOCATv2025_tracks_gridded_monthly.nc
│   ├── 02_intermediate/
│   └── 03_processed/
│       └── climate_fused_dataset.nc
├── figures/
├── notebooks/
│   ├── 1.0-eda.ipynb
│   └── 2.0-results-vis.ipynb
├── scripts/
│   ├── discover.py
│   ├── download.py
│   └── preprocess.py
├── src/
│   └── climate_discovery/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   └── preprocessing.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── hmm.py
│       │   ├── pinn.py
│       │   └── symbolic.py
│       └── physics/
│           ├── __init__.py
│           └── thermodynamics.py
├── src/climate_discovery.egg-info/
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
└── tests/
    ├── test_clustering.py
    └── test_data.py
```
