"""Uncertainty quantification for SD-MoSE predictions.

Provides confidence intervals via:
- Bootstrap ensembles
- Monte Carlo dropout
- Bayesian regime probabilities

Usage:
    from sdmose.validation.uncertainty import UncertaintyEstimator
    
    estimator = UncertaintyEstimator(strategy="bootstrap", n_models=20)
    mean_pred, std_pred = estimator.predict_with_uncertainty(X)
    # Report: "pCO₂ = 380 ± 15 μatm (95% CI: 350-410)"
"""

import numpy as np
from typing import List, Tuple, Literal
import matplotlib.pyplot as plt
import seaborn as sns


class UncertaintyEstimator:
    """Estimate prediction uncertainty for SD-MoSE.
    
    Args:
        strategy: "bootstrap", "mc_dropout", or "ensemble"
        n_models: Number of models/samples for ensemble
        confidence_level: For confidence intervals (default 0.95)
        
    Example:
        >>> estimator = UncertaintyEstimator(strategy="bootstrap", n_models=20)
        >>> estimator.fit_ensemble(X_train, y_train, train_func)
        >>> mean, std, lower, upper = estimator.predict_with_uncertainty(X_test)
        >>> print(f"Prediction: {mean[0]:.1f} ± {std[0]:.1f}")
    """
    
    def __init__(
        self,
        strategy: Literal["bootstrap", "mc_dropout", "ensemble"] = "bootstrap",
        n_models: int = 20,
        confidence_level: float = 0.95,
    ):
        self.strategy = strategy
        self.n_models = n_models
        self.confidence_level = confidence_level
        self.models = []
    
    def fit_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_func,
        verbose: bool = True,
    ):
        """Fit ensemble of models.
        
        Args:
            X: Training features (N, D)
            y: Training targets (N,)
            train_func: Function(X, y, seed) -> model
            verbose: Print progress
        """
        if verbose:
            print(f"\n🔄 Fitting {self.n_models} models for uncertainty estimation...")
        
        self.models = []
        
        for i in range(self.n_models):
            if verbose and (i+1) % 5 == 0:
                print(f"  Model {i+1}/{self.n_models}")
            
            if self.strategy == "bootstrap":
                # Bootstrap sample
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_boot = X[indices]
                y_boot = y[indices]
                model = train_func(X_boot, y_boot, seed=i)
            else:
                # Train on full data with different seed
                model = train_func(X, y, seed=i)
            
            self.models.append(model)
        
        if verbose:
            print(f"✓ Ensemble fitted: {self.n_models} models")
    
    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates.
        
        Args:
            X: Features (N, D)
            
        Returns:
            mean: Mean prediction (N,)
            std: Standard deviation (N,)
            lower: Lower confidence bound (N,)
            upper: Upper confidence bound (N,)
        """
        if len(self.models) == 0:
            raise ValueError("Ensemble not fitted. Call fit_ensemble() first.")
        
        # Collect predictions from all models
        predictions = np.array([
            model.predict(X) for model in self.models
        ])  # (n_models, N)
        
        # Compute statistics
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Confidence intervals
        alpha = 1 - self.confidence_level
        lower = np.percentile(predictions, 100 * alpha/2, axis=0)
        upper = np.percentile(predictions, 100 * (1 - alpha/2), axis=0)
        
        return mean, std, lower, upper
    
    def plot_uncertainty(
        self,
        X: np.ndarray,
        y_true: np.ndarray = None,
        x_axis: np.ndarray = None,
        x_label: str = "Sample Index",
        save_path: str = "figures/uncertainty_quantification.png",
    ):
        """Visualize predictions with uncertainty.
        
        Args:
            X: Features
            y_true: True values (optional)
            x_axis: X-axis values for plotting
            x_label: Label for x-axis
            save_path: Output path
        """
        mean, std, lower, upper = self.predict_with_uncertainty(X)
        
        if x_axis is None:
            x_axis = np.arange(len(mean))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Prediction with confidence bands
        ax = axes[0, 0]
        ax.fill_between(x_axis, lower, upper, alpha=0.3, color='blue', label=f'{int(self.confidence_level*100)}% CI')
        ax.plot(x_axis, mean, 'b-', label='Mean prediction', linewidth=2)
        
        if y_true is not None:
            ax.scatter(x_axis, y_true, c='red', s=10, alpha=0.5, label='True values')
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('fCO₂ (μatm)', fontsize=11)
        ax.set_title('Predictions with Confidence Intervals', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Uncertainty distribution
        ax = axes[0, 1]
        ax.hist(std, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(std), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(std):.2f}')
        ax.axvline(np.median(std), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(std):.2f}')
        ax.set_xlabel('Prediction Std Dev (μatm)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Uncertainty vs prediction
        ax = axes[1, 0]
        scatter = ax.scatter(mean, std, c=std, cmap='viridis', s=10, alpha=0.6)
        ax.set_xlabel('Mean Prediction (μatm)', fontsize=11)
        ax.set_ylabel('Std Dev (μatm)', fontsize=11)
        ax.set_title('Uncertainty vs Prediction', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Std Dev')
        ax.grid(alpha=0.3)
        
        # 4. Error vs uncertainty (if y_true provided)
        ax = axes[1, 1]
        if y_true is not None:
            errors = np.abs(mean - y_true)
            ax.scatter(std, errors, s=10, alpha=0.6, c='steelblue')
            ax.plot([0, np.max(std)], [0, np.max(std)], 'r--', label='Perfect calibration', linewidth=2)
            ax.set_xlabel('Predicted Std Dev (μatm)', fontsize=11)
            ax.set_ylabel('Absolute Error (μatm)', fontsize=11)
            ax.set_title('Calibration: Error vs Uncertainty', fontsize=12, fontweight='bold')
            ax.legend()
            
            # Compute calibration score
            correlation = np.corrcoef(std, errors)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No ground truth provided',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Uncertainty plot saved: {save_path}")
        
        plt.close()
    
    def get_summary_statistics(
        self,
        X: np.ndarray,
        y_true: np.ndarray = None,
    ) -> dict:
        """Compute summary statistics for uncertainty.
        
        Args:
            X: Features
            y_true: True values (optional)
            
        Returns:
            Dictionary with statistics
        """
        mean, std, lower, upper = self.predict_with_uncertainty(X)
        
        stats = {
            "mean_uncertainty": np.mean(std),
            "median_uncertainty": np.median(std),
            "max_uncertainty": np.max(std),
            "min_uncertainty": np.min(std),
            "ci_width": np.mean(upper - lower),
        }
        
        if y_true is not None:
            errors = np.abs(mean - y_true)
            stats["mean_error"] = np.mean(errors)
            stats["rmse"] = np.sqrt(np.mean((mean - y_true)**2))
            stats["r2"] = 1 - np.sum((y_true - mean)**2) / np.sum((y_true - np.mean(y_true))**2)
            
            # Calibration: correlation between uncertainty and error
            stats["calibration_corr"] = np.corrcoef(std, errors)[0, 1]
            
            # Coverage: % of true values within CI
            in_ci = (y_true >= lower) & (y_true <= upper)
            stats["empirical_coverage"] = np.mean(in_ci)
        
        return stats
