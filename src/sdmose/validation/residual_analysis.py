"""Residual analysis and diagnostic tools for SD-MoSE.

Checks for systematic errors in predictions:
- Residuals vs spatial coordinates
- Residuals vs temporal features
- Residuals vs input variables
- Regime-specific error patterns

Usage:
    from sdmose.validation.residual_analysis import ResidualAnalyzer
    
    analyzer = ResidualAnalyzer()
    analyzer.analyze(y_true, y_pred, metadata)
    analyzer.plot_diagnostics(save_path="figures/residuals.png")
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

# Optional cartopy for enhanced geographic plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    ccrs = None
    cfeature = None


class ResidualAnalyzer:
    """Systematic residual analysis for model diagnostics.
    
    Example:
        >>> analyzer = ResidualAnalyzer()
        >>> diagnostics = analyzer.analyze(
        ...     y_true=y_test,
        ...     y_pred=predictions,
        ...     lats=lats,
        ...     lons=lons,
        ...     sst=sst_values,
        ...     months=month_values,
        ... )
        >>> analyzer.plot_diagnostics()
    """
    
    def __init__(self):
        self.residuals = None
        self.diagnostics = {}
    
    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lats: Optional[np.ndarray] = None,
        lons: Optional[np.ndarray] = None,
        sst: Optional[np.ndarray] = None,
        sss: Optional[np.ndarray] = None,
        months: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
    ) -> Dict:
        """Analyze residuals for systematic patterns.
        
        Args:
            y_true: True values
            y_pred: Predictions
            lats: Latitudes (optional)
            lons: Longitudes (optional)
            sst: SST values (optional)
            sss: SSS values (optional)
            months: Month indices (optional)
            regime_labels: Regime assignments (optional)
            
        Returns:
            Dictionary with diagnostic statistics
        """
        self.residuals = y_true - y_pred
        self.y_true = y_true
        self.y_pred = y_pred
        
        # Store metadata
        self.lats = lats
        self.lons = lons
        self.sst = sst
        self.sss = sss
        self.months = months
        self.regime_labels = regime_labels
        
        # Compute diagnostics
        self.diagnostics = {
            "mean_residual": np.mean(self.residuals),
            "std_residual": np.std(self.residuals),
            "rmse": np.sqrt(np.mean(self.residuals**2)),
            "mae": np.mean(np.abs(self.residuals)),
            "max_error": np.max(np.abs(self.residuals)),
        }
        
        # Spatial patterns
        if lats is not None:
            self.diagnostics["lat_correlation"] = np.corrcoef(lats, self.residuals)[0, 1]
            
            # Latitudinal bias
            lat_bands = [
                ("Tropical", -30, 30),
                ("Mid-Lat North", 30, 60),
                ("Mid-Lat South", -60, -30),
                ("Polar North", 60, 90),
                ("Polar South", -90, -60),
            ]
            
            band_stats = {}
            for name, lat_min, lat_max in lat_bands:
                mask = (lats >= lat_min) & (lats < lat_max)
                if np.sum(mask) > 0:
                    band_stats[name] = {
                        "mean_error": np.mean(self.residuals[mask]),
                        "rmse": np.sqrt(np.mean(self.residuals[mask]**2)),
                        "n_samples": np.sum(mask),
                    }
            
            self.diagnostics["latitudinal_bias"] = band_stats
        
        # Temporal patterns
        if months is not None:
            monthly_stats = {}
            for month in range(1, 13):
                mask = months == month
                if np.sum(mask) > 0:
                    monthly_stats[month] = {
                        "mean_error": np.mean(self.residuals[mask]),
                        "rmse": np.sqrt(np.mean(self.residuals[mask]**2)),
                    }
            
            self.diagnostics["seasonal_bias"] = monthly_stats
        
        # Regime-specific errors
        if regime_labels is not None:
            regime_stats = {}
            for regime in np.unique(regime_labels):
                mask = regime_labels == regime
                regime_stats[int(regime)] = {
                    "mean_error": np.mean(self.residuals[mask]),
                    "rmse": np.sqrt(np.mean(self.residuals[mask]**2)),
                    "n_samples": np.sum(mask),
                }
            
            self.diagnostics["regime_errors"] = regime_stats
        
        return self.diagnostics
    
    def plot_diagnostics(
        self,
        save_path: str = "figures/residual_analysis.png",
    ):
        """Create comprehensive residual diagnostic plots.
        
        Args:
            save_path: Output path
        """
        if self.residuals is None:
            raise ValueError("No residuals computed. Run analyze() first.")
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Residual histogram
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(self.residuals, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Residual (μatm)', fontsize=10)
        ax1.set_ylabel('Frequency', fontsize=10)
        ax1.set_title('Residual Distribution', fontsize=11, fontweight='bold')
        ax1.text(0.05, 0.95, f"Mean: {self.diagnostics['mean_residual']:.2f}\nStd: {self.diagnostics['std_residual']:.2f}",
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.grid(alpha=0.3)
        
        # 2. Predicted vs True
        ax2 = plt.subplot(3, 3, 2)
        ax2.scatter(self.y_true, self.y_pred, s=10, alpha=0.3, c='steelblue')
        lims = [min(self.y_true.min(), self.y_pred.min()),
                max(self.y_true.max(), self.y_pred.max())]
        ax2.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
        ax2.set_xlabel('True fCO₂ (μatm)', fontsize=10)
        ax2.set_ylabel('Predicted fCO₂ (μatm)', fontsize=10)
        ax2.set_title('Predicted vs True', fontsize=11, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Residuals vs Predicted
        ax3 = plt.subplot(3, 3, 3)
        ax3.scatter(self.y_pred, self.residuals, s=10, alpha=0.3, c='steelblue')
        ax3.axhline(0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Predicted fCO₂ (μatm)', fontsize=10)
        ax3.set_ylabel('Residual (μatm)', fontsize=10)
        ax3.set_title('Residuals vs Predictions', fontsize=11, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. Spatial map of residuals
        if self.lats is not None and self.lons is not None:
            if HAS_CARTOPY:
                ax4 = plt.subplot(3, 3, 4, projection=ccrs.PlateCarree())
                ax4.coastlines()
                ax4.add_feature(cfeature.LAND, facecolor='lightgray')
                
                scatter = ax4.scatter(
                    self.lons, self.lats,
                    c=self.residuals,
                    cmap='RdBu_r',
                    s=5,
                    alpha=0.6,
                    vmin=-np.percentile(np.abs(self.residuals), 95),
                    vmax=np.percentile(np.abs(self.residuals), 95),
                    transform=ccrs.PlateCarree(),
                )
                plt.colorbar(scatter, ax=ax4, shrink=0.7, label='Residual (μatm)')
                ax4.set_title('Spatial Residual Pattern', fontsize=11, fontweight='bold')
            else:
                # Matplotlib fallback
                ax4 = plt.subplot(3, 3, 4)
                scatter = ax4.scatter(
                    self.lons, self.lats,
                    c=self.residuals,
                    cmap='RdBu_r',
                    s=5,
                    alpha=0.6,
                    vmin=-np.percentile(np.abs(self.residuals), 95),
                    vmax=np.percentile(np.abs(self.residuals), 95),
                )
                plt.colorbar(scatter, ax=ax4, shrink=0.7, label='Residual (μatm)')
                ax4.set_xlabel('Longitude')
                ax4.set_ylabel('Latitude')
                ax4.set_title('Spatial Residual Pattern', fontsize=11, fontweight='bold')
                ax4.grid(alpha=0.3)
                ax4.set_xlim(-180, 180)
                ax4.set_ylim(-90, 90)
        
        # 5. Residuals vs Latitude
        if self.lats is not None:
            ax5 = plt.subplot(3, 3, 5)
            ax5.scatter(self.lats, self.residuals, s=10, alpha=0.3, c='steelblue')
            ax5.axhline(0, color='red', linestyle='--', linewidth=2)
            
            # Add moving average
            lat_bins = np.linspace(-90, 90, 20)
            lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
            lat_indices = np.digitize(self.lats, lat_bins)
            mean_residuals = [np.mean(self.residuals[lat_indices == i]) 
                             for i in range(1, len(lat_bins))]
            ax5.plot(lat_centers, mean_residuals, 'r-', linewidth=2, label='Mean by latitude')
            
            ax5.set_xlabel('Latitude', fontsize=10)
            ax5.set_ylabel('Residual (μatm)', fontsize=10)
            ax5.set_title('Residuals vs Latitude', fontsize=11, fontweight='bold')
            ax5.legend()
            ax5.grid(alpha=0.3)
        
        # 6. Residuals vs SST
        if self.sst is not None:
            ax6 = plt.subplot(3, 3, 6)
            ax6.scatter(self.sst, self.residuals, s=10, alpha=0.3, c='steelblue')
            ax6.axhline(0, color='red', linestyle='--', linewidth=2)
            ax6.set_xlabel('SST (°C)', fontsize=10)
            ax6.set_ylabel('Residual (μatm)', fontsize=10)
            ax6.set_title('Residuals vs SST', fontsize=11, fontweight='bold')
            ax6.grid(alpha=0.3)
        
        # 7. Seasonal pattern
        if self.months is not None:
            ax7 = plt.subplot(3, 3, 7)
            monthly_means = [np.mean(self.residuals[self.months == m]) for m in range(1, 13)]
            monthly_stds = [np.std(self.residuals[self.months == m]) for m in range(1, 13)]
            
            ax7.errorbar(range(1, 13), monthly_means, yerr=monthly_stds,
                        fmt='o-', capsize=5, linewidth=2, markersize=6)
            ax7.axhline(0, color='red', linestyle='--', linewidth=2)
            ax7.set_xlabel('Month', fontsize=10)
            ax7.set_ylabel('Mean Residual (μatm)', fontsize=10)
            ax7.set_title('Seasonal Bias Pattern', fontsize=11, fontweight='bold')
            ax7.set_xticks(range(1, 13))
            ax7.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
            ax7.grid(alpha=0.3)
        
        # 8. Regime-specific errors
        if self.regime_labels is not None:
            ax8 = plt.subplot(3, 3, 8)
            regime_ids = sorted(np.unique(self.regime_labels))
            regime_residuals = [self.residuals[self.regime_labels == r] for r in regime_ids]
            
            bp = ax8.boxplot(regime_residuals, labels=[f"R{r}" for r in regime_ids],
                            patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('steelblue')
                patch.set_alpha(0.7)
            
            ax8.axhline(0, color='red', linestyle='--', linewidth=2)
            ax8.set_xlabel('Regime', fontsize=10)
            ax8.set_ylabel('Residual (μatm)', fontsize=10)
            ax8.set_title('Errors by Regime', fontsize=11, fontweight='bold')
            ax8.grid(alpha=0.3, axis='y')
        
        # 9. Q-Q plot (normality check)
        ax9 = plt.subplot(3, 3, 9)
        from scipy import stats
        stats.probplot(self.residuals, dist="norm", plot=ax9)
        ax9.set_title('Q-Q Plot (Normality Check)', fontsize=11, fontweight='bold')
        ax9.grid(alpha=0.3)
        
        plt.suptitle('Residual Diagnostic Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Residual analysis saved: {save_path}")
        
        plt.close()
    
    def print_summary(self):
        """Print residual analysis summary."""
        if not self.diagnostics:
            print("No diagnostics available. Run analyze() first.")
            return
        
        print("\n" + "="*70)
        print("RESIDUAL ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nOverall Statistics:")
        print(f"  Mean Residual:  {self.diagnostics['mean_residual']:>8.2f} μatm")
        print(f"  Std Residual:   {self.diagnostics['std_residual']:>8.2f} μatm")
        print(f"  RMSE:           {self.diagnostics['rmse']:>8.2f} μatm")
        print(f"  MAE:            {self.diagnostics['mae']:>8.2f} μatm")
        print(f"  Max Error:      {self.diagnostics['max_error']:>8.2f} μatm")
        
        if "latitudinal_bias" in self.diagnostics:
            print(f"\nLatitudinal Bias:")
            for band, stats in self.diagnostics["latitudinal_bias"].items():
                print(f"  {band:20s}: Mean={stats['mean_error']:>7.2f}, RMSE={stats['rmse']:>7.2f} (n={stats['n_samples']})")
        
        if "regime_errors" in self.diagnostics:
            print(f"\nRegime-Specific Errors:")
            for regime, stats in self.diagnostics["regime_errors"].items():
                print(f"  Regime {regime}: Mean={stats['mean_error']:>7.2f}, RMSE={stats['rmse']:>7.2f} (n={stats['n_samples']})")
        
        print("="*70 + "\n")
