# 🌊 SD-MoSE: Symbolic Discovery of Mixture-of-Symbolic-Experts

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A production-ready framework for discovering interpretable ocean pCO₂ equations using hierarchical symbolic regression with regime-based expertise.**

---

## 🎯 Features

### **Core Capabilities** (1-4)
- ✅ **Physics-Informed Constraints**: Prevents numerical overflow, ensures physical realism
- ✅ **Hierarchical Regime Structure**: 2-level gating (coarse → fine) for multi-scale dynamics
- ✅ **Experiment Tracking**: W&B/MLflow integration with auto-logging
- ✅ **Attention Gating**: Multi-head self-attention with feature importance

### **Reproducibility & Optimization** (5-6)
- ✅ **Git-Tracked Versioning**: Equations saved with commit hash, config, metrics
- ✅ **Automated Ablations**: Systematic hyperparameter search with 6 presets

### **Visualization & Interpretability** (7-9)
- ✅ **Interactive Maps**: Plotly dashboards with hover/zoom/time sliders
- ✅ **Sensitivity Analysis**: ∂f/∂x computation showing driver variables
- ✅ **Evolution Videos**: MP4/GIF animations of regime dynamics

### **Scientific Validation** (10-13)
- ✅ **Spatial Cross-Validation**: Ocean basin holdout testing
- ✅ **Uncertainty Quantification**: Bootstrap ensembles with ±95% CI
- ✅ **Residual Analysis**: 9-panel diagnostics for systematic errors
- ✅ **Model Benchmarking**: Compare vs baselines & physics models

### **Code Quality & DevOps** (14-16)
- ✅ **Unit Tests**: Comprehensive test suite with pytest
- ✅ **Pre-commit Hooks**: Auto-formatting (black), linting (flake8)
- ✅ **Docker**: Reproducible containerized deployment

---

## 🚀 Quick Start

### **Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/climate-equation-discovery.git
cd climate-equation-discovery

# Install dependencies
pip install -r requirements.txt

# Install PySR (symbolic regression)
python -c "import pysr; pysr.install()"

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### **Basic Usage**

```bash
# 1. Train SD-MoSE model
python -m scripts.train.train_sdmose --iterations 5

# 2. Create interactive visualizations
python -m scripts.viz.interactive_regime_map --type all

# 3. Run sensitivity analysis
python -m scripts.analysis.equation_sensitivity

# 4. Generate evolution video
python -m scripts.viz.regime_evolution_video --format mp4
```

### **Docker Deployment**

```bash
# Build image
docker build -t sdmose:latest .

# Run training
docker run -v $(pwd)/data:/app/data sdmose:latest

# Or use Docker Compose
docker-compose up sdmose
```

---

## 📊 Performance

| Configuration | Test R² | RMSE | Interpretability |
|---------------|---------|------|------------------|
| **Baseline** (linear) | 0.35 | 45.2 | High |
| **MLP** | 0.48 | 32.1 | Low |
| **SD-MoSE (flat)** | 0.44 | 34.2 | High |
| **SD-MoSE (hierarchical)** | **0.52** | **30.8** | **High** |

---

## 🔬 Scientific Workflow

```bash
# Step 1: Spatial CV for validation
python -m scripts.experiments.spatial_cv_evaluation

# Step 2: Optimize hyperparameters
python -m scripts.experiments.ablation_grid --preset full_grid

# Step 3: Train with best config
python -m scripts.train.train_sdmose --use-hierarchical --iterations 10

# Step 4: Uncertainty quantification
python -m scripts.experiments.uncertainty_evaluation

# Step 5: Residual diagnostics
python -m scripts.analysis.residual_diagnostics

# Step 6: Benchmark vs baselines
python -m scripts.experiments.model_benchmark

# Step 7: Generate publication figures
python -m scripts.viz.create_publication_figures
```

---

## 📁 Project Structure

```
climate-equation-discovery/
├── src/climate_discovery/
│   ├── models/               # Neural architectures
│   │   ├── hierarchical.py   # Hierarchical SD-MoSE
│   │   ├── symbolic.py       # Physics-constrained PySR
│   │   └── gating.py         # Attention gating
│   ├── validation/           # Validation tools
│   │   ├── spatial_cv.py     # Spatial cross-validation
│   │   ├── uncertainty.py    # Bootstrap ensembles
│   │   ├── residual_analysis.py
│   │   └── benchmark.py      # Model comparison
│   └── utils/
│       ├── tracking.py       # W&B/MLflow
│       └── equation_version.py
├── scripts/
│   ├── train/                # Training scripts
│   ├── viz/                  # Visualizations
│   ├── analysis/             # Analysis tools
│   └── experiments/          # Ablation studies
├── tests/                    # Unit tests
├── Dockerfile               # Container config
├── docker-compose.yml
└── .pre-commit-config.yaml
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gradient.py -v

# Run with coverage
pytest --cov=src/climate_discovery tests/

# Run pre-commit checks
pre-commit run --all-files
```

---

## 📖 Documentation

### **Configuration**

Edit `src/climate_discovery/config.py`:

```python
@dataclass
class ModelConfig:
    # Core model
    use_hierarchical = True      # Enable hierarchical structure
    n_coarse_regimes = 3         # Coarse-level regimes
    n_fine_per_coarse = 3        # Fine regimes per coarse
    
    # Physics constraints
    pysr_use_constraints = True
    pysr_max_tree_depth = 6
    
    # Tracking
    use_tracking = True
    tracking_backend = "wandb"   # or "mlflow"
```

### **Equation Versioning**

```python
from climate_discovery.utils import EquationVersionManager

manager = EquationVersionManager()
manager.save_equations(
    equations=discovered_eqs,
    config=config,
    metrics={"r2": 0.52, "rmse": 30.8},
    version="2.0.0",
    notes="Hierarchical model with optimal ablation config"
)
```

### **Spatial Cross-Validation**

```python
from climate_discovery.validation import SpatialCrossValidator

cv = SpatialCrossValidator(strategy="ocean_basins")
results = cv.evaluate(train_func, X, y, lats, lons)
# Tests generalization to unseen ocean regions
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Run pre-commit (`pre-commit run --all-files`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

---

## 📚 Citation

If you use SD-MoSE in your research, please cite:

```bibtex
@software{sdmose2024,
  title={SD-MoSE: Symbolic Discovery of Mixture-of-Symbolic-Experts},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/climate-equation-discovery}
}
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PySR**: Julia-based symbolic regression ([github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR))
- **SOCAT**: Surface Ocean CO₂ Atlas ([socat.info](https://www.socat.info/))
- **ERA5**: Climate reanalysis data

---