"""Run Spatial Cross-Validation

Tests geographic generalization by holding out spatial blocks.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root / "src"))

from sdmose.config import TRAIN_NC, FEATURES_EXPERT, FEATURES_GATING, TARGET
from sdmose.data.datasets import ClimateDataset
from sdmose.validation.spatial_cv import SpatialCrossValidator
from sdmose.models.symbolic import MixtureOfSymbolicExperts
from sklearn.cluster import KMeans


def run_spatial_cv(n_splits=5, n_regimes=6, pysr_iterations=20, save_dir="results/spatial_cv"):
    """Run spatial cross-validation."""
    print(f"Running spatial cross-validation ({n_splits} folds)...")
    print("This tests if regimes generalize to new geographic regions.\n")
    
    # Load data
    dataset = ClimateDataset(
        TRAIN_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target=TARGET,
        drop_nan=True
    )
    
    df = dataset.get_dataframe()
    X_gating = df[FEATURES_GATING].values
    X_expert = df[FEATURES_EXPERT].values
    y = df[TARGET].values
    lats = df['lat'].values
    lons = df['lon'].values
    
    # Initialize spatial CV
    spatial_cv = SpatialCrossValidator(n_splits=n_splits)
    
    cv_scores = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(spatial_cv.split(X_expert, y, lats, lons)):
        print(f"\n=== Fold {fold_idx + 1}/{n_splits} ===")
        print(f"Train: {len(train_idx):,} samples, Test: {len(test_idx):,} samples")
        
        # Split data
        X_train_gating, X_test_gating = X_gating[train_idx], X_gating[test_idx]
        X_train_expert, X_test_expert = X_expert[train_idx], X_expert[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit K-means on training data
        kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        train_regimes = kmeans.fit_predict(X_train_gating)
        test_regimes = kmeans.predict(X_test_gating)
        
        # Create regime probability matrices
        train_probs = np.zeros((len(train_regimes), n_regimes))
        for i, r in enumerate(train_regimes):
            train_probs[i, r] = 1.0
            
        test_probs = np.zeros((len(test_regimes), n_regimes))
        for i, r in enumerate(test_regimes):
            test_probs[i, r] = 1.0
        
        # Fit symbolic experts
        print("Fitting symbolic experts...")
        experts = MixtureOfSymbolicExperts(
            num_regimes=n_regimes,
            expert_config={"niterations": pysr_iterations}
        )
        
        try:
            experts.fit(
                X_train_expert,
                y_train,
                train_probs,
                variable_names=FEATURES_EXPERT,
                min_samples=50
            )
            
            # Predict on test set
            y_pred = experts.predict(X_test_expert, test_probs)
            
            # Calculate R²
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
            
            cv_scores.append({'fold': fold_idx, 'r2': r2, 'rmse': rmse})
            print(f"Fold {fold_idx + 1} → R²={r2:.3f}, RMSE={rmse:.1f} μatm")
            
        except Exception as e:
            print(f"Fold {fold_idx + 1} failed: {e}")
            cv_scores.append({'fold': fold_idx, 'r2': np.nan, 'rmse': np.nan})
    
    # Summarize results
    cv_df = pd.DataFrame(cv_scores)
    
    print(f"\n{'='*60}")
    print("SPATIAL CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean R²:   {cv_df['r2'].mean():.3f} ± {cv_df['r2'].std():.3f}")
    print(f"Mean RMSE: {cv_df['rmse'].mean():.1f} ± {cv_df['rmse'].std():.1f} μatm")
    print(f"{'='*60}\n")
    
    # Save results
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    cv_df.to_csv(save_path / "spatial_cv_scores.csv",  index=False)
    print(f"✓ Saved results: {save_path / 'spatial_cv_scores.csv'}")
    
    return cv_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--regimes', type=int, default=6)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--output', type=str, default='results/spatial_cv')
    args = parser.parse_args()
    
    run_spatial_cv(args.splits, args.regimes, args.iterations, args.output)
