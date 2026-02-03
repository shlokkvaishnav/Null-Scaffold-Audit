"""Model benchmarking and comparison framework.

Compare SD-MoSE against:
- Baseline models (Linear, MLP, Random Forest)
- Physics-based models (GOBM, pCO2-SOM)  
- Other symbolic regression approaches

Usage:
    from climate_discovery.validation.benchmark import ModelBenchmark
    
    benchmark = ModelBenchmark()
    benchmark.add_model("SD-MoSE", sdmose_model)
    benchmark.add_model("MLP", mlp_model)
    benchmark.compare(X_test, y_test)
    benchmark.plot_comparison()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Callable
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class ModelBenchmark:
    """Comprehensive model comparison and benchmarking.
    
    Example:
        >>> benchmark = ModelBenchmark()
        >>> benchmark.add_model("SD-MoSE", sdmose, interpretable=True)
        >>> benchmark.add_model("MLP", mlp, interpretable=False)
        >>> results = benchmark.compare(X_test, y_test)
        >>> benchmark.plot_comparison()
    """
    
    def __init__(self):
        self.models = {}
        self.results = None
    
    def add_model(
        self,
        name: str,
        model: Any,
        predict_func: Callable = None,
        interpretable: bool = False,
        physics_based: bool = False,
    ):
        """Add model to benchmark.
        
        Args:
            name: Model name
            model: Model object
            predict_func: Custom prediction function (if not model.predict)
            interpretable: Whether model is interpretable
            physics_based: Whether model is physics-based
        """
        self.models[name] = {
            "model": model,
            "predict_func": predict_func or (lambda X: model.predict(X)),
            "interpretable": interpretable,
            "physics_based": physics_based,
        }
    
    def compare(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        time_training: Dict[str, float] = None,
    ) -> pd.DataFrame:
        """Compare all models on test set.
        
        Args:
            X_test: Test features
            y_ test: Test targets
            time_training: Training times (optional)
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        print("\n" + "="*70)
        print("MODEL BENCHMARKING")
        print("="*70)
        
        for name, model_info in self.models.items():
            print(f"\n📊 Evaluating: {name}")
            
            # Predict
            try:
                y_pred = model_info["predict_func"](X_test)
                
                # Metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # Bias
                bias = np.mean(y_pred - y_test)
                
                # Correlation
                corr = np.corrcoef(y_test, y_pred)[0, 1]
                
                result = {
                    "Model": name,
                    "R²": r2,
                    "RMSE": rmse,
                    "MAE": mae,
                    "Bias": bias,
                    "Correlation": corr,
                    "Interpretable": "Yes" if model_info["interpretable"] else "No",
                    "Physics-Based": "Yes" if model_info["physics_based"] else "No",
                }
                
                if time_training and name in time_training:
                    result["Train_Time (s)"] = time_training[name]
                
                results.append(result)
                
                print(f"  R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
        
        self.results = pd.DataFrame(results)
        
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(self.results.to_string(index=False))
        print("="*70 + "\n")
        
        return self.results
    
    def plot_comparison(
        self,
        save_path: str = "figures/model_benchmark.png",
    ):
        """Create comprehensive comparison visualization.
        
        Args:
            save_path: Output path
        """
        if self.results is None:
            raise ValueError("No results available. Run compare() first.")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. R² comparison
        ax = axes[0, 0]
        colors = ['green' if self.results.loc[i, 'Interpretable'] == 'Yes' else 'steelblue'
                 for i in range(len(self.results))]
        ax.barh(self.results['Model'], self.results['R²'], color=colors, alpha=0.7)
        ax.set_xlabel('R²', fontsize=11)
        ax.set_title('R² Score Comparison', fontsize=12, fontweight='bold')
        ax.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='R²=0.5')
        ax.legend()
        ax.grid(axis='x', alpha=0.3)
        
        # 2. RMSE comparison
        ax = axes[0, 1]
        ax.barh(self.results['Model'], self.results['RMSE'], color=colors, alpha=0.7)
        ax.set_xlabel('RMSE (μatm)', fontsize=11)
        ax.set_title('RMSE Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 3. MAE comparison
        ax = axes[0, 2]
        ax.barh(self.results['Model'], self.results['MAE'], color=colors, alpha=0.7)
        ax.set_xlabel('MAE (μatm)', fontsize=11)
        ax.set_title('MAE Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 4. Bias comparison
        ax = axes[1, 0]
        ax.barh(self.results['Model'], self.results['Bias'], color=colors, alpha=0.7)
        ax.set_xlabel('Bias (μatm)', fontsize=11)
        ax.set_title('Prediction Bias', fontsize=12, fontweight='bold')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.grid(axis='x', alpha=0.3)
        
        # 5. Accuracy vs Interpretability scatter
        ax = axes[1, 1]
        interp_map = {'Yes': 1, 'No': 0}
        x = self.results['Interpretable'].map(interp_map)
        y = self.results['R²']
        
        for i, model in enumerate(self.results['Model']):
            ax.scatter(x.iloc[i] + np.random.normal(0, 0.05),  # Jitter
                      y.iloc[i],
                      s=200, alpha=0.7, label=model)
            ax.text(x.iloc[i], y.iloc[i], model, fontsize=9, ha='center', va='bottom')
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Interpretable', 'Interpretable'])
        ax.set_ylabel('R²', fontsize=11)
        ax.set_title('Accuracy vs Interpretability', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.3, 1.3)
        
        # 6. Summary table
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create summary text
        summary_text = "MODEL COMPARISON\n" + "="*40 + "\n\n"
        
        best_r2 = self.results.loc[self.results['R²'].idxmax()]
        best_rmse = self.results.loc[self.results['RMSE'].idxmin()]
        
        summary_text += f"Best R²:\n  {best_r2['Model']}: {best_r2['R²']:.4f}\n\n"
        summary_text += f"Best RMSE:\n  {best_rmse['Model']}: {best_rmse['RMSE']:.2f}\n\n"
        
        interpretable_models = self.results[self.results['Interpretable'] == 'Yes']
        if len(interpretable_models) > 0:
            summary_text += f"Best Interpretable:\n"
            best_interp = interpretable_models.loc[interpretable_models['R²'].idxmax()]
            summary_text += f"  {best_interp['Model']}\n"
            summary_text += f"  R² = {best_interp['R²']:.4f}\n"
            summary_text += f"  RMSE = {best_interp['RMSE']:.2f}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Legend for colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Interpretable'),
            Patch(facecolor='steelblue', alpha=0.7, label='Black-box'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.suptitle('Model Benchmarking Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Benchmark plot saved: {save_path}")
        
        plt.close()
    
    def save_results(
        self,
        save_path: str = "results/benchmark_results.csv",
    ):
        """Save benchmark results to CSV.
        
        Args:
            save_path: Output path
        """
        if self.results is None:
            raise ValueError("No results available. Run compare() first.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.results.to_csv(save_path, index=False)
        print(f"✓ Results saved: {save_path}")


def run_all_benchmarks(X_train, y_train, X_test, y_test):
    """Convenience function to run all standard baselines.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        ModelBenchmark object with results
    """
    import time
    from .models import LinearBaseline, RFBaseline, XGBBaseline
    
    benchmark = ModelBenchmark()
    training_times = {}
    
    # Linear Regression
    print("\n🔧 Training Linear Regression...")
    model = LinearBaseline()
    start = time.time()
    model.fit(X_train, y_train)
    training_times['Linear'] = time.time() - start
    benchmark.add_model('Linear', model, interpretable=True)
    
    # Random Forest
    print("🔧 Training Random Forest...")
    model = RFBaseline(n_estimators=200, max_depth=15)
    start = time.time()
    model.fit(X_train, y_train)
    training_times['RandomForest'] = time.time() - start
    benchmark.add_model('RandomForest', model, interpretable=False)
    
    # XGBoost
    print("🔧 Training XGBoost...")
    model = XGBBaseline(n_estimators=200, max_depth=5)
    start = time.time()
    model.fit(X_train, y_train)
    training_times['XGBoost'] = time.time() - start
    benchmark.add_model('XGBoost', model, interpretable=False)
    
    # Compare
    benchmark.compare(X_test, y_test, time_training=training_times)
    
    return benchmark
