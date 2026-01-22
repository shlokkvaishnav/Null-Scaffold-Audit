"""Ablation study: phys vs bio, linear vs RF, hard symbolic. Writes results/ablations.csv."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

import logging

import numpy as np
import pandas as pd

from climate_discovery.config import (
    FEATURES_ALL,
    FEATURES_PHYS,
    FEATURES_TIME,
    LAT_BANDS,
    TARGET,
    TEST_NC,
    TRAIN_NC,
)
from climate_discovery.data import load_table_data
from climate_discovery.evaluation import (
    compute_r2_rmse,
    ood_slices,
    plausibility_metrics,
)
from climate_discovery.models.baselines import LinearBaseline, RFBaseline
from climate_discovery.models.symbolic import KMeansSymbolicRegressor

N_SAMPLES = 25_000
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    if not TRAIN_NC.exists() or not TEST_NC.exists():
        logger.error("Run preprocess first.")
        return

    X_tr, y_tr, lat_tr, _, X_te, y_te, lat_te, _ = load_table_data(
        str(TRAIN_NC), str(TEST_NC), FEATURES_ALL, TARGET
    )
    n = min(N_SAMPLES, len(X_tr))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_tr), n, replace=False)
    X_tr, y_tr = X_tr[idx], y_tr[idx]

    feats = list(FEATURES_ALL)
    phys_cols = [c for c in FEATURES_PHYS + FEATURES_TIME if c in feats]
    if not phys_cols:
        phys_cols = [
            c
            for c in ["sst", "sss", "sin_month", "cos_month", "year_feature"]
            if c in feats
        ]
    phys_idx = [feats.index(c) for c in phys_cols]
    X_tr_phys = X_tr[:, phys_idx] if phys_idx else X_tr
    X_te_phys = X_te[:, phys_idx] if phys_idx else X_te

    rows = []

    for name, use_bio, Xtr, Xte in [
        ("linear_phys", False, X_tr_phys, X_te_phys),
        ("linear_bio", True, X_tr, X_te),
    ]:
        m = LinearBaseline()
        m.fit(Xtr, y_tr)
        r2, rmse = compute_r2_rmse(y_te, m.predict(Xte))
        rows.append({"ablation": name, "bio": use_bio, "r2": r2, "rmse": rmse})

    for name, use_bio, Xtr, Xte in [
        ("rf_phys", False, X_tr_phys, X_te_phys),
        ("rf_bio", True, X_tr, X_te),
    ]:
        m = RFBaseline(n_estimators=30, max_depth=10)
        m.fit(Xtr, y_tr)
        pred = m.predict(Xte)
        r2, rmse = compute_r2_rmse(y_te, pred)
        p = plausibility_metrics(pred, y_te)
        rows.append(
            {
                "ablation": name,
                "bio": use_bio,
                "r2": r2,
                "rmse": rmse,
                "frac_in_range": p["frac_in_range"],
            }
        )
        if name == "rf_bio":
            pred_rf_bio = pred

    for name, use_bio, Xtr, Xte in [
        ("hard_symbolic_phys", False, X_tr_phys, X_te_phys),
        ("hard_symbolic_bio", True, X_tr, X_te),
    ]:
        km = KMeansSymbolicRegressor(n_clusters=4, max_depth=6)
        km.fit(Xtr, y_tr)
        r2, rmse = compute_r2_rmse(y_te, km.predict(Xte))
        rows.append({"ablation": name, "bio": use_bio, "r2": r2, "rmse": rmse})

    for band, v in ood_slices(lat_te, y_te, pred_rf_bio, LAT_BANDS).items():
        rows.append(
            {"ablation": f"ood_{band}", "bio": True, "r2": v["r2"], "rmse": v["rmse"]}
        )

    df = pd.DataFrame(rows)
    out_dir = root / "results"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "ablations.csv", index=False)
    logger.info("Saved %s", out_dir / "ablations.csv")
    print(df.to_string())


if __name__ == "__main__":
    main()
