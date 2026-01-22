"""Evaluate baseline models for comparison with SD-MoSE.

Baselines:
1. Linear Regression (global)
2. Linear Regression (latitude bands)
3. Random Forest
4. XGBoost
5. K-means + Symbolic (hard clustering)

Metrics:
- R², RMSE, MAE (global)
- Out-of-distribution performance (latitude bands)
- Physical plausibility checks

Usage:
    python -m scripts.eval.eval_baselines
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(__file__).resolve().parents[2]  # noqa: E402
sys.path.insert(0, str(root / "src"))  # noqa: E402

import argparse  # noqa: E402
import logging  # noqa: E402

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from climate_discovery.config import (
    FCO2_MAX_PLAUSIBLE,
    FCO2_MIN_PLAUSIBLE,
    FEATURES_EXPERT,
    FEATURES_GATING,
    LAT_BANDS,
    ModelConfig,
    RESULTS_DIR,
    TARGET,
    TEST_NC,
    TRAIN_NC,
)
from climate_discovery.data.datasets import ClimateDataset
from climate_discovery.models.baselines import KMeansSymbolicRegressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# BASELINE MODELS
# =============================================================================

class LatitudeBandLinear:
    """Linear regression with separate models per latitude band."""
    
    def __init__(self, bands: dict = None):
        self.bands = bands or LAT_BANDS
        self.models = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, lat: np.ndarray):
        """Fit separate model for each latitude band."""
        for band_name, (lat_min, lat_max) in self.bands.items():
            mask = (lat >= lat_min) & (lat < lat_max)
            if np.sum(mask) < 10:
                continue
            
            model = LinearRegression()
            model.fit(X[mask], y[mask])
            self.models[band_name] = model
            
            logger.info(
                f"  {band_name}: {np.sum(mask)} samples, "
                f"R²={model.score(X[mask], y[mask]):.4f}"
            )
    
    def predict(self, X: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Predict using appropriate latitude band model."""
        y_pred = np.zeros(len(X))
        
        for band_name, (lat_min, lat_max) in self.bands.items():
            mask = (lat >= lat_min) & (lat < lat_max)
            if np.sum(mask) == 0 or band_name not in self.models:
                continue
            
            y_pred[mask] = self.models[band_name].predict(X[mask])
        
        return y_pred


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute comprehensive regression metrics."""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_samples": len(y_true),
    }


def ood_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lat: np.ndarray,
    bands: dict,
) -> dict:
    """Evaluate performance on out-of-distribution latitude bands."""
    results = {}
    
    for band_name, (lat_min, lat_max) in bands.items():
        mask = (lat >= lat_min) & (lat < lat_max)
        if np.sum(mask) < 10:
            continue
        
        results[band_name] = compute_metrics(y_true[mask], y_pred[mask])
    
    return results


def plausibility_check(y_pred: np.ndarray, y_min: float, y_max: float) -> dict:
    """Check physical plausibility of predictions."""
    n_total = len(y_pred)
    n_below = np.sum(y_pred < y_min)
    n_above = np.sum(y_pred > y_max)
    n_invalid = n_below + n_above
    
    return {
        "frac_in_range": (n_total - n_invalid) / n_total,
        "frac_below_min": n_below / n_total,
        "frac_above_max": n_above / n_total,
        "n_invalid": n_invalid,
    }


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def evaluate_baselines(
    train_dataset: ClimateDataset,
    test_dataset: ClimateDataset,
    subsample_train: int = None,
) -> pd.DataFrame:
    """Evaluate all baseline models.
    
    Args:
        train_dataset: Training data
        test_dataset: Test data
        subsample_train: Subsample training data (for speed)
        
    Returns:
        DataFrame with results
    """
    # Prepare data
    X_train = train_dataset.X_expert
    y_train = train_dataset.y
    lat_train = train_dataset.coords["lat"]
    
    X_test = test_dataset.X_expert
    y_test = test_dataset.y
    lat_test = test_dataset.coords["lat"]
    
    # Optional subsampling for speed
    if subsample_train and len(X_train) > subsample_train:
        logger.info(f"Subsampling training data to {subsample_train} samples")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_train), subsample_train, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
        lat_train = lat_train[indices]
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = []
    
    # =========================================================================
    # 1. LINEAR REGRESSION (GLOBAL)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("LINEAR REGRESSION (GLOBAL)")
    logger.info("=" * 60)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "Linear (global)"
    results.append(metrics)
    
    logger.info(f"R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(
        LinearRegression(), X_train, y_train, cv=5, scoring="r2"
    )
    logger.info(f"CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # =========================================================================
    # 2. LINEAR REGRESSION (LATITUDE BANDS)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("LINEAR REGRESSION (LATITUDE BANDS)")
    logger.info("=" * 60)
    
    model = LatitudeBandLinear(LAT_BANDS)
    model.fit(X_train, y_train, lat_train)
    y_pred = model.predict(X_test, lat_test)
    
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "Linear (lat-bands)"
    results.append(metrics)
    
    logger.info(f"R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # =========================================================================
    # 3. RANDOM FOREST
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("RANDOM FOREST")
    logger.info("=" * 60)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "Random Forest"
    results.append(metrics)
    
    logger.info(f"R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Feature importance
    importances = model.feature_importances_
    logger.info("Feature importances:")
    for feat, imp in zip(FEATURES_EXPERT, importances):
        logger.info(f"  {feat}: {imp:.4f}")
    
    # =========================================================================
    # 4. XGBOOST
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("XGBOOST")
    logger.info("=" * 60)
    
    model = XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "XGBoost"
    results.append(metrics)
    
    logger.info(f"R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # =========================================================================
    # 5. K-MEANS + SYMBOLIC
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("K-MEANS + SYMBOLIC")
    logger.info("=" * 60)
    
    config = ModelConfig()
    
    # Subsample for PySR (slow)
    if len(X_train) > 30000:
        logger.info("Subsampling to 30k for symbolic regression")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(X_train), 30000, replace=False)
        X_train_sub = X_train[indices]
        y_train_sub = y_train[indices]
    else:
        X_train_sub = X_train
        y_train_sub = y_train
    
    model = KMeansSymbolicRegressor(
        n_clusters=config.n_regimes,
        max_depth=6,
        random_state=42,
    )
    
    model.fit(X_train_sub, y_train_sub, variable_names=FEATURES_EXPERT)
    y_pred = model.predict(X_test)
    
    metrics = compute_metrics(y_test, y_pred)
    metrics["model"] = "K-means + Symbolic"
    results.append(metrics)
    
    logger.info(f"R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")
    
    # Log discovered equations
    equations = model.get_equations()
    logger.info("Discovered equations:")
    for cluster, eq in equations.items():
        logger.info(f"  {cluster}: {eq}")
    
    # =========================================================================
    # OUT-OF-DISTRIBUTION EVALUATION
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("OUT-OF-DISTRIBUTION EVALUATION")
    logger.info("=" * 60)
    
    # Use Random Forest for OOD analysis (best baseline)
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    
    ood_results = ood_evaluation(y_test, y_pred_rf, lat_test, LAT_BANDS)
    
    for band, metrics in ood_results.items():
        logger.info(
            f"{band}: R²={metrics['r2']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, N={metrics['n_samples']}"
        )
    
    # =========================================================================
    # PLAUSIBILITY CHECK
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHYSICAL PLAUSIBILITY CHECK")
    logger.info("=" * 60)
    
    plaus = plausibility_check(y_pred_rf, FCO2_MIN_PLAUSIBLE, FCO2_MAX_PLAUSIBLE)
    
    logger.info(f"Fraction in range [{FCO2_MIN_PLAUSIBLE}, {FCO2_MAX_PLAUSIBLE}]: {plaus['frac_in_range']:.4f}")
    logger.info(f"Fraction below min: {plaus['frac_below_min']:.4f}")
    logger.info(f"Fraction above max: {plaus['frac_above_max']:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    results_df = pd.DataFrame(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    print(results_df.to_string(index=False))
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models")
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample training data (for speed)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path"
    )
    
    args = parser.parse_args()
    
    # Check data exists
    if not TRAIN_NC.exists() or not TEST_NC.exists():
        logger.error("Processed data not found. Run preprocessing first:")
        logger.error("  python -m scripts.data.preprocess_data")
        sys.exit(1)
    
    # Load datasets
    logger.info("Loading datasets...")
    
    train_dataset = ClimateDataset(
        TRAIN_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target=TARGET,
        drop_nan=True,
    )
    
    test_dataset = ClimateDataset(
        TEST_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target=TARGET,
        drop_nan=True,
    )
    
    # Run evaluation
    results_df = evaluate_baselines(
        train_dataset,
        test_dataset,
        subsample_train=args.subsample,
    )
    
    # Save results
    output_path = Path(args.output) if args.output else RESULTS_DIR / "baselines.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()