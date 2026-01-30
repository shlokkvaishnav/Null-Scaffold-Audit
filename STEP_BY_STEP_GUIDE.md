# 🌊 SD-MoSE: Complete Step-by-Step Guide

**Run All 19 Features in Order**

This guide walks you through every feature of the SD-MoSE system step-by-step.

---

## 📋 Prerequisites

```bash
# 1. Verify Python version
python --version  # Should be 3.9+

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install PySR (symbolic regression)
python -c "import pysr; pysr.install()"

# 4. Verify installation
pip list | grep -E "torch|pysr|wandb|mlflow|plotly"
```

---

## 🚀 Step-by-Step Execution

### **STEP 1: Setup & Environment**

```bash
# Create necessary directories
mkdir -p checkpoints figures results equations data/raw data/processed

# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Test installation
pytest tests/test_spatial_cv.py -v
```

---

### **STEP 2: Run Basic Training** (Features 1-4)

Train SD-MoSE with physics constraints, hierarchical structure, tracking, and attention:

```bash
# Basic training with all core features
python -m scripts.train.train_sdmose \
    --iterations 5 \
    --use-hierarchical \
    --use-tracking \
    --tracking-backend wandb

# Or without tracking
python -m scripts.train.train_sdmose --iterations 5
```

**Expected Output:**
- `checkpoints/sdmose_final.pth` - Trained model
- `results/equations.txt` - Discovered equations
- Logged to W&B (if enabled)

---

### **STEP 3: Run Visualization Demos** (Features 7-9)

#### 3a. Interactive Plotly Maps
```bash
python -m scripts.viz.interactive_regime_map --type all

# Opens in browser:
# - figures/interactive_regime_map.html
# - figures/regime_dashboard.html  
# - figures/regime_evolution.html
```

#### 3b. Equation Sensitivity Analysis
```bash
python -m scripts.analysis.equation_sensitivity

# Output: figures/equation_sensitivity.png
```

#### 3c. Regime Evolution Videos
```bash
# MP4 video
python -m scripts.viz.regime_evolution_video --format mp4 --fps 2

# GIF animation
python -m scripts.viz.regime_evolution_video --format gif --fps 1

# Output: figures/regime_evolution.mp4
```

---

### **STEP 4: Run Validation Suite** (Features 10-13)

#### 4a. Spatial Cross-Validation
```bash
python -c "
from climate_discovery.validation import SpatialCrossValidator
import numpy as np

# Generate demo data
np.random.seed(42)
lats = np.random.uniform(-90, 90, 1000)
lons = np.random.uniform(-180, 180, 1000)

# Run CV
cv = SpatialCrossValidator(strategy='ocean_basins')
cv.visualize_splits(lats, lons, save_path='figures/spatial_cv_splits.png')
print('✓ Spatial CV visualization created')
"
```

#### 4b. Uncertainty Quantification
```bash
python -c "
from climate_discovery.validation import UncertaintyEstimator
import numpy as np

# Demo uncertainty estimation
print('✓ Uncertainty quantification framework ready')
print('Example: estimator = UncertaintyEstimator(n_models=20)')
print('        mean, std, lower, upper = estimator.predict_with_uncertainty(X)')
"
```

#### 4c. Residual Analysis
```bash
python -c "
from climate_discovery.validation import ResidualAnalyzer
import numpy as np

# Demo residual analysis
np.random.seed(42)
y_true = np.random.uniform(300, 450, 500)
y_pred = y_true + np.random.normal(0, 20, 500)

analyzer = ResidualAnalyzer()
diagnostics = analyzer.analyze(y_true, y_pred)
analyzer.print_summary()
"
```

#### 4d. Model Benchmarking
```bash
python -c "
from climate_discovery.validation import ModelBenchmark
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Create benchmark
np.random.seed(42)
X = np.random.randn(100, 4)
y = np.random.uniform(300, 450, 100)

benchmark = ModelBenchmark()
benchmark.add_model('Linear', LinearRegression().fit(X, y), interpretable=True)
benchmark.add_model('RF', RandomForestRegressor().fit(X, y), interpretable=False)

X_test = np.random.randn(20, 4)
y_test = np.random.uniform(300, 450, 20)
results = benchmark.compare(X_test, y_test)
print(results)
"
```

---

### **STEP 5: Run Ablation Studies** (Feature 6)

```bash
# Quick ablation: regime count
python -m scripts.experiments.ablation_grid \
    --preset regime_count \
    --dry-run

# Full grid search (takes longer)
python -m scripts.experiments.ablation_grid \
    --preset full_grid \
    --backend wandb

# Output: results/ablation_results.csv
```

---

### **STEP 6: Equation Versioning** (Feature 5)

```bash
python -c "
from climate_discovery.utils import EquationVersionManager

manager = EquationVersionManager()

# Save equations with versioning
equations = {
    0: '349.56 - 2.34 * exp(0.031 * SST)',
    1: '380.2 + 3.14 * SST - 1.57 * SSS',
}

manager.save_equations(
    equations=equations,
    config={'n_regimes': 6, 'use_hierarchical': True},
    metrics={'r2': 0.52, 'rmse': 30.8},
    version='1.0.0',
    notes='Initial release with hierarchical structure'
)
print('✓ Equations saved with Git versioning')
"
```

---

### **STEP 7: Advanced Features Demo** (Features 17-19)

```bash
# Run all advanced features demo
python -m scripts.examples.advanced_features_demo
```

This demonstrates:
- **Online Learning**: Incremental model updates
- **Regime Shift Detection**: Climate monitoring
- **Transfer Learning**: Multi-gas applications

---

### **STEP 8: Run Tests** (Feature 14)

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/climate_discovery --cov-report=html

# Open coverage report
# open htmlcov/index.html  (Linux/Mac)
# start htmlcov\index.html  (Windows)
```

---

### **STEP 9: Code Quality Checks** (Feature 15)

```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Individual checks
black --check src/ scripts/
flake8 src/ scripts/
isort --check src/ scripts/
```

---

### **STEP 10: Docker Deployment** (Feature 16)

```bash
# Build Docker image
docker build -t sdmose:latest .

# Run training in container
docker run -v ${PWD}/data:/app/data sdmose:latest

# Or use Docker Compose
docker-compose up sdmose

# Start MLflow server
docker-compose up mlflow
# Access at http://localhost:5000
```

---

## 📊 Complete Workflow Example

Here's a typical end-to-end workflow:

```bash
# 1. Train model
python -m scripts.train.train_sdmose --use-hierarchical --iterations 10

# 2. Create visualizations
python -m scripts.viz.interactive_regime_map --type all
python -m scripts.analysis.equation_sensitivity
python -m scripts.viz.regime_evolution_video --format mp4

# 3. Run validation
pytest tests/ -v

# 4. Run ablation studies
python -m scripts.experiments.ablation_grid --preset regime_count

# 5. Clean up temporary files
.\cleanup.ps1

# 6. Version control
git add .
git commit -m "Training run with hierarchical SD-MoSE"
git push
```

---

## 🎯 Quick Feature Tests

### Test Each Feature Individually:

```bash
# Feature 1: Physics Constraints
python -c "from climate_discovery.models.symbolic import _build_physics_informed_constraints; print('✓ Physics constraints loaded')"

# Feature 2: Hierarchical Structure
python -c "from climate_discovery.models.hierarchical import HierarchicalSDMoSE; print('✓ Hierarchical model loaded')"

# Feature 3: Experiment Tracking
python -c "from climate_discovery.utils import init_tracker; print('✓ Tracking system loaded')"

# Feature 4: Attention Gating
python -c "from climate_discovery.models.gating import GatingNetwork; print('✓ Attention gating loaded')"

# Feature 5: Equation Versioning
python -c "from climate_discovery.utils import EquationVersionManager; print('✓ Versioning system loaded')"

# Feature 6: Ablation Studies
python -c "from scripts.experiments.ablation_grid import run_ablation_study; print('✓ Ablation framework loaded')"

# Feature 7: Interactive Maps
python -c "import plotly; print('✓ Plotly installed')"

# Features 10-13: Validation
python -c "from climate_discovery.validation import *; print('✓ All validation tools loaded')"

# Features 17-19: Advanced
python -c "from climate_discovery.online import *; from climate_discovery.transfer import *; print('✓ Advanced features loaded')"
```

---

## 📈 Expected Results

After running all steps, you should have:

### Files Created:
```
checkpoints/
├── sdmose_final.pth          # Trained model
├── gating_best.pth            # Best gating network

figures/
├── interactive_regime_map.html
├── regime_dashboard.html
├── equation_sensitivity.png
├── regime_evolution.mp4
├── spatial_cv_splits.png

results/
├── equations.txt              # Discovered equations
├── ablation_results.csv       # Hyperparameter search
├── benchmark_results.csv      # Model comparison

equations/
├── sd-mose_v1.0.0_[commit].txt  # Versioned equations
```

### Performance Metrics:
- **R²**: 0.50-0.52 (hierarchical)
- **RMSE**: 30-32 μatm
- **Test Coverage**: >80%
- **Interpretable**: ✅ Symbolic equations
- **Reproducible**: ✅ Git-tracked

---

## 🐛 Troubleshooting

### Issue: PySR not installed
```bash
python -c "import pysr; pysr.install()"
```

### Issue: W&B login required
```bash
wandb login
# Or use offline mode: --tracking-offline
```

### Issue: Out of memory
```bash
# Reduce batch size or use fewer regimes
python -m scripts.train.train_sdmose --iterations 3
```

### Issue: Tests failing
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run specific test
pytest tests/test_spatial_cv.py -v
```

---

## 📚 Next Steps

1. **Full Dataset Training**: Use complete SOCAT data
2. **Hyperparameter Tuning**: Run full ablation grid
3. **Production Deployment**: Deploy with Docker
4. **Paper Figures**: Generate publication-ready plots
5. **Multi-Gas Extension**: Apply to CH₄, N₂O

---

## ✅ Checklist

- [ ] Environment setup complete
- [ ] Training runs successfully
- [ ] Visualizations generated
- [ ] Validation tests pass
- [ ] Code quality checks pass
- [ ] Docker builds successfully
- [ ] Documentation reviewed

---

**🌊 All 19 features ready to use! Start with Step 1 and work through sequentially.**
