"""Run baseline models: Linear, Lat-band, RF, XGBoost, K-means+Symbolic."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

import logging
import numpy as np

from climate_discovery.config import TRAIN_NC, TEST_NC, FEATURES_ALL, TARGET, LAT_BANDS
from climate_discovery.data import load_table_data
from climate_discovery.evaluation import compute_r2_rmse, ood_slices, plausibility_metrics
from climate_discovery.models.baselines import (
    LinearBaseline,
    LatitudeBandLinearRegression,
    RFBaseline,
    XGBBaseline,
)
from climate_discovery.models.symbolic import KMeansSymbolicRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    if not TRAIN_NC.exists() or not TEST_NC.exists():
        logger.error("Run preprocess first: python -m scripts.data.preprocess_data.")
        sys.exit(1)

    X_tr, y_tr, lat_tr, _, X_te, y_te, lat_te, _ = load_table_data(
        str(TRAIN_NC), str(TEST_NC), FEATURES_ALL, TARGET
    )
    logger.info("Train: %d | Test: %d", len(y_tr), len(y_te))

    results = {}
    # Linear global
    m = LinearBaseline()
    m.fit(X_tr, y_tr)
    r2, rmse = compute_r2_rmse(y_te, m.predict(X_te))
    results["Linear (global)"] = {"r2": r2, "rmse": rmse}
    logger.info("Linear (global): R² = %.4f | RMSE = %.4f", r2, rmse)

    # Lat-band
    m = LatitudeBandLinearRegression()
    m.fit(X_tr, y_tr, lat_tr)
    r2, rmse = compute_r2_rmse(y_te, m.predict(X_te, lat_te))
    results["Linear (lat-band)"] = {"r2": r2, "rmse": rmse}
    logger.info("Linear (lat-band): R² = %.4f | RMSE = %.4f", r2, rmse)

    # RF
    m = RFBaseline(n_estimators=50, max_depth=15)
    m.fit(X_tr, y_tr)
    pred_rf = m.predict(X_te)
    r2, rmse = compute_r2_rmse(y_te, pred_rf)
    results["Random Forest"] = {"r2": r2, "rmse": rmse}
    logger.info("Random Forest: R² = %.4f | RMSE = %.4f", r2, rmse)

    # XGBoost
    m = XGBBaseline(n_estimators=50, max_depth=6)
    m.fit(X_tr, y_tr)
    r2, rmse = compute_r2_rmse(y_te, m.predict(X_te))
    results["XGBoost"] = {"r2": r2, "rmse": rmse}
    logger.info("XGBoost: R² = %.4f | RMSE = %.4f", r2, rmse)

    # K-means + Symbolic
    n = min(30_000, len(X_tr))
    idx = np.random.default_rng(42).choice(len(X_tr), n, replace=False)
    km = KMeansSymbolicRegressor(n_clusters=4, max_depth=6)
    km.fit(X_tr[idx], y_tr[idx])
    r2, rmse = compute_r2_rmse(y_te, km.predict(X_te))
    results["K-means + Symbolic"] = {"r2": r2, "rmse": rmse}
    logger.info("K-means + Symbolic: R² = %.4f | RMSE = %.4f", r2, rmse)

    ood = ood_slices(lat_te, y_te, pred_rf, LAT_BANDS)
    for band, v in ood.items():
        logger.info("OOD %s: R² = %.4f | n = %d", band, v["r2"], v["n"])
    plaus = plausibility_metrics(pred_rf, y_te)
    logger.info("Plausibility (RF): frac_in_range = %.4f", plaus["frac_in_range"])

    logger.info("--- Summary ---")
    for name, v in results.items():
        print(f"{name}: R² = {v['r2']:.4f} | RMSE = {v['rmse']:.4f}")  # noqa: T201


if __name__ == "__main__":
    main()
