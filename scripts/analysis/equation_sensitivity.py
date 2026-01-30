"""Equation sensitivity analysis for discovered equations.

For each regime's discovered equation, compute and visualize:
- Partial derivatives ∂f/∂x_i for each input variable
- Local sensitivity at different points
- Global sensitivity across dataset
- Variable importance rankings

Usage:
    python -m scripts.analysis.equation_sensitivity --equations equations/sd-mose_v1.0.0.txt
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))


class EquationSensitivityAnalyzer:
    """Analyze sensitivity of discovered equations.
    
    Computes numerical derivatives to understand:
    - Which variables have strongest impact
    - How sensitivity varies across domain
    - Interaction effects between variables
    """
    
    def __init__(self, variable_names: List[str]):
        """Initialize analyzer.
        
        Args:
            variable_names: List of input variable names
        """
        self.variable_names = variable_names
        self.n_vars = len(variable_names)
    
    def compute_numerical_gradient(
        self,
        equation_func,
        x: np.ndarray,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Compute numerical gradient ∇f(x).
        
        Args:
            equation_func: Function f(x) returning scalar
            x: Input point (D,)
            epsilon: Finite difference step
            
        Returns:
            Gradient vector (D,)
        """
        grad = np.zeros(len(x))
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            
            x_minus = x.copy()
            x_minus[i] -= epsilon
            
            # Central difference
            grad[i] = (equation_func(x_plus) - equation_func(x_minus)) / (2 * epsilon)
        
        return grad
    
    def local_sensitivity(
        self,
        equation_func,
        X: np.ndarray,
        n_samples: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """Compute local sensitivity across dataset.
        
        Args:
            equation_func: Equation function
            X: Input data (N, D)
            n_samples: Number of points to sample
            
        Returns:
            Dictionary with sensitivity statistics
        """
        # Sample points
        indices = np.random.choice(len(X), size=min(n_samples, len(X)), replace=False)
        X_sample = X[indices]
        
        # Compute gradients
        gradients = []
        for x in X_sample:
            try:
                grad = self.compute_numerical_gradient(equation_func, x)
                gradients.append(grad)
            except Exception:
                continue
        
        gradients = np.array(gradients)  # (N, D)
        
        # Compute statistics
        results = {
            "mean_gradient": np.mean(gradients, axis=0),
            "std_gradient": np.std(gradients, axis=0),
            "abs_mean_gradient": np.mean(np.abs(gradients), axis=0),
            "median_gradient": np.median(gradients, axis=0),
            "all_gradients": gradients,
        }
        
        return results
    
    def global_sensitivity(
        self,
        equation_func,
        X_min: np.ndarray,
        X_max: np.ndarray,
        n_samples: int = 10000,
    ) -> Dict[str, float]:
        """Sobol-style global sensitivity analysis.
        
        Args:
            equation_func: Equation function
            X_min: Minimum values for each variable
            X_max: Maximum values for each variable
            n_samples: Monte Carlo samples
            
        Returns:
            Sensitivity indices for each variable
        """
        # Generate random samples
        X = np.random.uniform(X_min, X_max, size=(n_samples, len(X_min)))
        
        # Compute function values
        try:
            Y = np.array([equation_func(x) for x in X])
        except Exception:
            return {var: 0.0 for var in self.variable_names}
        
        # Total variance
        var_total = np.var(Y)
        
        if var_total < 1e-10:
            return {var: 0.0 for var in self.variable_names}
        
        # First-order sensitivity indices
        S_i = {}
        
        for i, var in enumerate(self.variable_names):
            # Conditional variance: Var[E[Y|X_i]]
            n_bins = 20
            bins = np.linspace(X_min[i], X_max[i], n_bins + 1)
            bin_indices = np.digitize(X[:, i], bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            
            # Expectation within each bin
            conditional_means = []
            for b in range(n_bins):
                mask = bin_indices == b
                if np.sum(mask) > 0:
                    conditional_means.append(np.mean(Y[mask]))
                else:
                    conditional_means.append(0.0)
            
            # Variance of conditional means
            var_conditional = np.var(conditional_means)
            
            # Sensitivity index
            S_i[var] = var_conditional / var_total
        
        return S_i
    
    def plot_sensitivity(
        self,
        sensitivities: Dict[str, Dict],
        regime_names: Dict[int, str] = None,
        save_path: str = "figures/equation_sensitivity.png",
    ):
        """Create comprehensive sensitivity visualization.
        
        Args:
            sensitivities: Dict of {regime_id: sensitivity_results}
            regime_names: Optional regime names
            save_path: Output path
        """
        n_regimes = len(sensitivities)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Mean absolute gradients (heatmap)
        ax = axes[0, 0]
        sensitivity_matrix = []
        for regime_id in sorted(sensitivities.keys()):
            sens = sensitivities[regime_id]
            sensitivity_matrix.append(sens["abs_mean_gradient"])
        
        sensitivity_matrix = np.array(sensitivity_matrix)
        
        sns.heatmap(
            sensitivity_matrix,
            xticklabels=self.variable_names,
            yticklabels=[regime_names.get(i, f"Regime {i}") if regime_names 
                        else f"Regime {i}" for i in sorted(sensitivities.keys())],
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "|∂f/∂x|"},
        )
        ax.set_title("Variable Sensitivity by Regime", fontsize=12, fontweight='bold')
        ax.set_xlabel("Variable")
        ax.set_ylabel("Regime")
        
        # 2. Sensitivity rankings
        ax = axes[0, 1]
        regime_colors = plt.cm.tab10(np.linspace(0, 1, n_regimes))
        
        x_pos = np.arange(self.n_vars)
        width = 0.8 / n_regimes
        
        for i, regime_id in enumerate(sorted(sensitivities.keys())):
            sens = sensitivities[regime_id]["abs_mean_gradient"]
            offset = (i - n_regimes/2) * width
            ax.bar(
                x_pos + offset,
                sens,
                width,
                label=regime_names.get(regime_id, f"Regime {regime_id}") if regime_names 
                      else f"Regime {regime_id}",
                color=regime_colors[i],
                alpha=0.8,
            )
        
        ax.set_xlabel("Variable", fontsize=11)
        ax.set_ylabel("Mean |∂f/∂x|", fontsize=11)
        ax.set_title("Sensitivity Comparison", fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(self.variable_names, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # 3. Variable importance (normalized)
        ax = axes[1, 0]
        
        # Average sensitivity across regimes
        avg_sensitivity = np.mean(sensitivity_matrix, axis=0)
        sorted_indices = np.argsort(avg_sensitivity)[::-1]
        
        colors = plt.cm.viridis(avg_sensitivity[sorted_indices] / np.max(avg_sensitivity))
        ax.barh(
            np.arange(self.n_vars),
            avg_sensitivity[sorted_indices],
            color=colors,
        )
        ax.set_yticks(np.arange(self.n_vars))
        ax.set_yticklabels([self.variable_names[i] for i in sorted_indices])
        ax.set_xlabel("Average Sensitivity", fontsize=11)
        ax.set_title("Overall Variable Importance", fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # 4. Sensitivity distribution
        ax = axes[1, 1]
        
        for regime_id in sorted(sensitivities.keys()):
            sens = sensitivities[regime_id]
            all_grads = sens["all_gradients"]  # (N, D)
            
            for var_idx, var_name in enumerate(self.variable_names):
                ax.violinplot(
                    [np.abs(all_grads[:, var_idx])],
                    positions=[var_idx + regime_id * 0.15],
                    widths=0.12,
                    showmeans=True,
                )
        
        ax.set_xticks(np.arange(self.n_vars))
        ax.set_xticklabels(self.variable_names, rotation=45, ha='right')
        ax.set_ylabel("|∂f/∂x| distribution", fontsize=11)
        ax.set_title("Sensitivity Distributions", fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Sensitivity plot saved: {save_path}")
        
        plt.close()


def create_equation_function(equation_str: str, variable_names: List[str]):
    """Convert equation string to numpy function.
    
    Args:
        equation_str: Equation like "349 + 2*SST - 1.5*SSS"
        variable_names: Variable names
        
    Returns:
        Function taking numpy array and returning scalar
    """
    # Simple parser (works for basic equations)
    # Replace variable names with array indexing
    func_str = equation_str
    for i, var in enumerate(variable_names):
        func_str = re.sub(rf'\b{var}\b', f'x[{i}]', func_str)
    
    # Handle common functions
    func_str = func_str.replace('exp', 'np.exp')
    func_str = func_str.replace('log', 'np.log')
    func_str = func_str.replace('sqrt', 'np.sqrt')
    func_str = func_str.replace('abs', 'np.abs')
    
    # Create lambda
    try:
        func = eval(f"lambda x: {func_str}")
        return func
    except Exception as e:
        print(f"⚠️  Failed to parse equation: {equation_str}")
        print(f"   Error: {e}")
        return lambda x: 0.0


def main():
    parser = argparse.ArgumentParser(description="Equation sensitivity analysis")
    parser.add_argument("--data", type=str, default="data/processed/test.nc")
    
    args = parser.parse_args()
    
    print("="*70)
    print("EQUATION SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Example equations (replace with loaded equations)
    variable_names = ["SST", "SSS", "Chl", "∇SST"]
    
    equations = {
        0: "349.56 - 2.34 * np.exp(0.031 * SST)",
        1: "380.2 + 3.14 * SST - 1.57 * SSS",
        2: "412.3 * (1 + 0.045 * np.log(Chl + 0.1))",
        3: "295.1 + 1.2 * SST + 0.8 * ∇SST",
    }
    
    # Create analyzer
    analyzer = EquationSensitivityAnalyzer(variable_names)
    
    # Generate synthetic data for demonstration
    X_min = np.array([-2, 30, 0.01, 0])  # SST, SSS, Chl, ∇SST
    X_max = np.array([35, 42, 10, 5])
    X_sample = np.random.uniform(X_min, X_max, size=(1000, 4))
    
    # Analyze each equation
    print("\n🔍 Analyzing equations...")
    sensitivities = {}
    
    for regime_id, eq_str in equations.items():
        print(f"\nRegime {regime_id}: {eq_str}")
        
        # Create function
        eq_func = create_equation_function(eq_str, variable_names)
        
        # Local sensitivity
        local_sens = analyzer.local_sensitivity(eq_func, X_sample, n_samples=500)
        
        # Global sensitivity
        global_sens = analyzer.global_sensitivity(eq_func, X_min, X_max, n_samples=5000)
        
        sensitivities[regime_id] = {
            **local_sens,
            "global": global_sens,
        }
        
        # Print results
        print(f"  Local sensitivity (mean |∂f/∂x|):")
        for i, var in enumerate(variable_names):
            print(f"    {var}: {local_sens['abs_mean_gradient'][i]:.4f}")
        
        print(f"  Global sensitivity indices:")
        for var, S_i in global_sens.items():
            print(f"    {var}: {S_i:.4f}")
    
    # Visualize
    print("\n📊 Creating visualizations...")
    analyzer.plot_sensitivity(sensitivities)
    
    print("\n="*70)
    print("✓ ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey findings:")
    print("  - Check figures/equation_sensitivity.png for detailed plots")
    print("  - Variables with high sensitivity drive predictions most")
    print("  - Sensitivity varies across regimes (different physics!)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
