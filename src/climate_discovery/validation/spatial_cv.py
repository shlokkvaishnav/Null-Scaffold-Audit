"""Spatial cross-validation for SD-MoSE models.

Implements block-based spatial CV to test generalization to unseen regions:
- Ocean basin holdout (Atlantic, Pacific, Indian, Southern, Arctic)
- Geographic grid blocks
- Custom spatial splits

Usage:
    from climate_discovery.validation.spatial_cv import SpatialCrossValidator
    
    cv = SpatialCrossValidator(strategy="ocean_basins", n_splits=5)
    scores = cv.evaluate(model, X, y, coords)
"""

import numpy as np
from typing import Dict, List, Tuple, Literal
from dataclasses import dataclass

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


@dataclass
class SpatialFold:
    """Container for spatial CV fold."""
    train_idx: np.ndarray
    test_idx: np.ndarray
    description: str


class SpatialCrossValidator:
    """Spatial cross-validation with geographic awareness.
    
    Prevents data leakage by ensuring test regions are spatially
    separated from training regions.
    
    Args:
        strategy: "ocean_basins", "grid_blocks", or "latitude_bands"
        n_splits: Number of folds (for grid_blocks)
        
    Example:
        >>> cv = SpatialCrossValidator(strategy="ocean_basins")
        >>> for fold in cv.split(lats, lons):
        ...     train_idx, test_idx = fold.train_idx, fold.test_idx
        ...     # Train on train_idx, validate on test_idx
    """
    
    def __init__(
        self,
        strategy: Literal["ocean_basins", "grid_blocks", "latitude_bands"] = "ocean_basins",
        n_splits: int = 5,
    ):
        self.strategy = strategy
        self.n_splits = n_splits
    
    def _assign_ocean_basin(self, lat: float, lon: float) -> str:
        """Assign point to ocean basin.
        
        Simplified basin boundaries:
        - Atlantic: -80 to 20°E
        - Indian: 20 to 150°E  
        - Pacific: 150°E to -80°E (wraps)
        - Southern: < -60°S
        - Arctic: > 66°N
        """
        # Arctic
        if lat > 66:
            return "Arctic"
        
        # Southern
        if lat < -60:
            return "Southern"
        
        # Atlantic
        if -80 <= lon <= 20:
            return "Atlantic"
        
        # Indian
        if 20 < lon <= 150:
            return "Indian"
        
        # Pacific (wraps around)
        return "Pacific"
    
    def split(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> List[SpatialFold]:
        """Generate spatial CV splits.
        
        Args:
            lats: Latitude values (N,)
            lons: Longitude values (N,)
            
        Returns:
            List of SpatialFold objects
        """
        if self.strategy == "ocean_basins":
            return self._split_by_basins(lats, lons)
        elif self.strategy == "grid_blocks":
            return self._split_by_grid(lats, lons)
        elif self.strategy == "latitude_bands":
            return self._split_by_latitude(lats, lons)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _split_by_basins(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> List[SpatialFold]:
        """Split by ocean basins."""
        # Assign each point to a basin
        basins = np.array([
            self._assign_ocean_basin(lat, lon)
            for lat, lon in zip(lats, lons)
        ])
        
        # Create folds: hold out each basin
        basin_names = ["Atlantic", "Pacific", "Indian", "Southern", "Arctic"]
        folds = []
        
        for basin in basin_names:
            test_mask = basins == basin
            train_mask = ~test_mask
            
            if np.sum(test_mask) > 0:  # Only if basin has data
                folds.append(SpatialFold(
                    train_idx=np.where(train_mask)[0],
                    test_idx=np.where(test_mask)[0],
                    description=f"Test: {basin} Ocean ({np.sum(test_mask)} points)",
                ))
        
        return folds
    
    def _split_by_grid(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> List[SpatialFold]:
        """Split by spatial grid blocks."""
        # Create grid
        n_lat_blocks = int(np.sqrt(self.n_splits))
        n_lon_blocks = int(np.ceil(self.n_splits / n_lat_blocks))
        
        lat_bins = np.linspace(-90, 90, n_lat_blocks + 1)
        lon_bins = np.linspace(-180, 180, n_lon_blocks + 1)
        
        # Assign to blocks
        lat_block = np.digitize(lats, lat_bins) - 1
        lon_block = np.digitize(lons, lon_bins) - 1
        
        # Create folds
        folds = []
        for i in range(n_lat_blocks):
            for j in range(n_lon_blocks):
                test_mask = (lat_block == i) & (lon_block == j)
                train_mask = ~test_mask
                
                if np.sum(test_mask) > 0:
                    lat_range = f"{lat_bins[i]:.0f}°-{lat_bins[i+1]:.0f}°"
                    lon_range = f"{lon_bins[j]:.0f}°-{lon_bins[j+1]:.0f}°"
                    
                    folds.append(SpatialFold(
                        train_idx=np.where(train_mask)[0],
                        test_idx=np.where(test_mask)[0],
                        description=f"Block ({lat_range}, {lon_range}), n={np.sum(test_mask)}",
                    ))
        
        return folds[:self.n_splits]  # Limit to n_splits
    
    def _split_by_latitude(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> List[SpatialFold]:
        """Split by latitude bands."""
        bands = [
            ("Arctic", 66, 90),
            ("Northern Mid-Lat", 30, 66),
            ("Tropical", -30, 30),
            ("Southern Mid-Lat", -66, -30),
            ("Antarctic", -90, -66),
        ]
        
        folds = []
        for name, lat_min, lat_max in bands:
            test_mask = (lats >= lat_min) & (lats < lat_max)
            train_mask = ~test_mask
            
            if np.sum(test_mask) > 0:
                folds.append(SpatialFold(
                    train_idx=np.where(train_mask)[0],
                    test_idx=np.where(test_mask)[0],
                    description=f"{name} ({lat_min}° to {lat_max}°), n={np.sum(test_mask)}",
                ))
        
        return folds
    
    def visualize_splits(
        self,
        lats: np.ndarray,
        lons: np.ndarray,
        save_path: str = "figures/spatial_cv_splits.png",
    ):
        """Visualize spatial CV splits.
        
        Args:
            lats: Latitudes
            lons: Longitudes
            save_path: Output path
        """
        folds = self.split(lats, lons)
        n_folds = len(folds)
        
        # Create subplots
        n_cols = min(3, n_folds)
        n_rows = int(np.ceil(n_folds / n_cols))
        
        fig = plt.figure(figsize=(6*n_cols, 4*n_rows))
        
        for i, fold in enumerate(folds):
            ax = plt.subplot(
                n_rows, n_cols, i+1,
                projection=ccrs.PlateCarree()
            )
            ax.coastlines()
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.gridlines(alpha=0.3)
            
            # Plot train (blue) and test (red)
            ax.scatter(
                lons[fold.train_idx], lats[fold.train_idx],
                c='blue', s=2, alpha=0.3, label='Train',
                transform=ccrs.PlateCarree()
            )
            ax.scatter(
                lons[fold.test_idx], lats[fold.test_idx],
                c='red', s=5, alpha=0.6, label='Test',
                transform=ccrs.PlateCarree()
            )
            
            ax.set_title(f"Fold {i+1}: {fold.description}", fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
        
        plt.suptitle(
            f"Spatial Cross-Validation: {self.strategy}",
            fontsize=14,
            fontweight='bold',
        )
        plt.tight_layout()
        
        # Save
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ CV splits visualization saved: {save_path}")
        
        plt.close()
    
    def evaluate(
        self,
        train_func,
        X: np.ndarray,
        y: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> Dict:
        """Run spatial CV evaluation.
        
        Args:
            train_func: Function(X_train, y_train) -> model
            X: Features (N, D)
            y: Targets (N,)
            lats: Latitudes (N,)
            lons: Longitudes (N,)
            
        Returns:
            Dictionary with CV results
        """
        from sklearn.metrics import r2_score, mean_squared_error
        
        folds = self.split(lats, lons)
        
        results = {
            "fold_descriptions": [],
            "train_r2": [],
            "test_r2": [],
            "train_rmse": [],
            "test_rmse": [],
        }
        
        print(f"\nRunning {len(folds)}-fold spatial CV ({self.strategy})...")
        
        for i, fold in enumerate(folds):
            print(f"\nFold {i+1}/{len(folds)}: {fold.description}")
            
            # Split data
            X_train, y_train = X[fold.train_idx], y[fold.train_idx]
            X_test, y_test = X[fold.test_idx], y[fold.test_idx]
            
            # Train model
            model = train_func(X_train, y_train)
            
            # Evaluate
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            results["fold_descriptions"].append(fold.description)
            results["train_r2"].append(train_r2)
            results["test_r2"].append(test_r2)
            results["train_rmse"].append(train_rmse)
            results["test_rmse"].append(test_rmse)
            
            print(f"  Train R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")
            print(f"  Test  R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
        
        # Summary
        print("\n" + "="*70)
        print("SPATIAL CV SUMMARY")
        print("="*70)
        print(f"Mean Test R²:   {np.mean(results['test_r2']):.4f} ± {np.std(results['test_r2']):.4f}")
        print(f"Mean Test RMSE: {np.mean(results['test_rmse']):.2f} ± {np.std(results['test_rmse']):.2f}")
        print("="*70)
        
        return results
