"""Automated Equation Analysis and Insight Extraction

Analyzes symbolic equations discovered by SD-MoSE to extract scientific insights:
1. Equation complexity metrics
2. Feature usage patterns
3. Operator frequency analysis
4. Physical interpretation helpers
5. Sensitivity analysis
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter


class EquationAnalyzer:
    """Analyze symbolic regression equations for insights."""
    
    def __init__(self):
        self.operators = {
            'unary': ['abs', 'exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'tanh'],
            'binary': ['+', '-', '*', '/', '^', '**']
        }
        
    def parse_equation(self, equation_str: str) -> Dict:
        """Parse equation string and extract features.
        
        Args:
            equation_str: String representation of equation
            
        Returns:
            Dictionary with parsed features
        """
        # Basic cleaning
        eq = equation_str.strip()
        
        # Count operators
        unary_ops = Counter()
        for op in self.operators['unary']:
            count = eq.count(op)
            if count > 0:
                unary_ops[op] = count
        
        binary_ops = Counter()
        for op in self.operators['binary']:
            if op in['**', '^']:  # Power operators
                count = eq.count('**') + eq.count('^')
                if count > 0:
                    binary_ops['power'] = count
            elif op in ['+', '-', '*', '/']:
                count = eq.count(op)
                if count > 0:
                    binary_ops[op] = count
        
        # Extract variables (assumes format like x0, x1, SST, etc.)
        variables = set(re.findall(r'\b[A-Za-z_]\w*\b', eq))
        # Remove operator names
        variables = variables - set(self.operators['unary']) - {'e', 'pi'}
        
        # Estimate complexity (number of nodes in expression tree)
        complexity = sum(unary_ops.values()) + sum(binary_ops.values()) + len(variables)
        
        return {
            'equation': eq,
            'unary_operators': dict(unary_ops),
            'binary_operators': dict(binary_ops),
            'variables': list(variables),
            'n_variables': len(variables),
            'complexity': complexity,
            'total_operators': sum(unary_ops.values()) + sum(binary_ops.values())
        }
    
    def analyze_regime_equations(
        self, 
        equations: Dict[int, str],
        variable_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Analyze equations across all regimes.
        
        Args:
            equations: Dict mapping regime_id -> equation_string
            variable_names: Optional list of actual variable names
            
        Returns:
            DataFrame with analysis results
        """
        results = []
        
        for regime_id, eq_str in equations.items():
            parsed = self.parse_equation(eq_str)
            
            result = {
                'regime': regime_id,
                'equation': eq_str,
                'complexity': parsed['complexity'],
                'n_variables': parsed['n_variables'],
                'n_operators': parsed['total_operators']
            }
            
            # Add operator counts
            for op, count in parsed['unary_operators'].items():
                result[f'uses_{op}'] = count
            for op, count in parsed['binary_operators'].items():
                result[f'uses_{op}'] = count
            
            # Variables used
            result['variables_used'] = ', '.join(sorted(parsed['variables']))
            
            results.append(result)
        
        df = pd.DataFrame(results)
        return df
    
    def compute_sensitivity(
        self,
        equation_str: str,
        X: np.ndarray,
        variable_names: List[str],
        perturbation: float = 0.01
    ) -> pd.DataFrame:
        """Compute sensitivity of equation to each variable using finite differences.
        
        Args:
            equation_str: Equation string  
            X: Feature matrix (N, D)
            variable_names: List of variable names matching X columns
            perturbation: Perturbation size (fraction)
            
        Returns:
            DataFrame with sensitivity metrics per variable
        """
        try:
            from sympy import sympify, lambdify, symbols
            
            # Parse equation
            sym_vars = symbols(' '.join(variable_names))
            expr = sympify(equation_str)
            func = lambdify(sym_vars, expr, 'numpy')
            
            # Base prediction
            y_base = func(*[X[:, i] for i in range(X.shape[1])])
            
            # Perturb each variable
            sensitivities = []
            
            for i, var_name in enumerate(variable_names):
                X_perturbed = X.copy()
                X_perturbed[:, i] *= (1 + perturbation)
                
                y_perturbed = func(*[X_perturbed[:, j] for j in range(X.shape[1])])
                
                # Sensitivity = (Δy / y) / (Δx / x)
                sensitivity = np.abs((y_perturbed - y_base) / y_base) / perturbation
                
                sensitivities.append({
                    'variable': var_name,
                    'mean_sensitivity': np.mean(sensitivity),
                    'std_sensitivity': np.std(sensitivity),
                    'max_sensitivity': np.max(sensitivity)
                })
            
            return pd.DataFrame(sensitivities)
            
        except ImportError:
            print("WARNING: sympy not installed. Install with: pip install sympy")
            return pd.DataFrame()
        except Exception as e:
            print(f"WARNING: Sensitivity analysis failed: {e}")
            return pd.DataFrame()
    
    def extract_physical_insights(
        self,
        equations: Dict[int, str],
        regime_descriptions: Optional[Dict[int, str]] = None
    ) -> List[str]:
        """Extract physical insights from equations.
        
        Args:
            equations: Dict mapping regime_id -> equation_string
            regime_descriptions: Optional regime descriptions
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # Analyze all equations
        df = self.analyze_regime_equations(equations)
        
        # 1. Complexity insights
        mean_complexity = df['complexity'].mean()
        insights.append(f"Average equation complexity: {mean_complexity:.1f} nodes")
        
        most_complex = df.loc[df['complexity'].idxmax()]
        insights.append(
            f"Most complex equation: Regime {most_complex['regime']} "
            f"(complexity = {most_complex['complexity']})"
        )
        
        simplest = df.loc[df['complexity'].idxmin()]
        insights.append(
            f"Simplest equation: Regime {simplest['regime']} "
            f"(complexity = {simplest['complexity']})"
        )
        
        # 2. Operator usage insights
        operator_cols = [c for c in df.columns if c.startswith('uses_')]
        if operator_cols:
            for col in operator_cols:
                if df[col].sum() > 0:
                    op_name = col.replace('uses_', '')
                    n_regimes = (df[col] > 0).sum()
                    insights.append(
                        f"Operator '{op_name}' used in {n_regimes}/{len(df)} regimes"
                    )
        
        # 3. Variable usage insights
        all_vars = set()
        for vars_str in df['variables_used']:
            all_vars.update(vars_str.split(', '))
        all_vars.discard('')
        
        insights.append(f"\nUnique variables across all regimes: {len(all_vars)}")
        insights.append(f"Variables: {', '.join(sorted(all_vars))}")
        
        # 4. Regime-specific insights
        for _, row in df.iterrows():
            regime_id = int(row['regime'])
            desc = regime_descriptions.get(regime_id, "") if regime_descriptions else ""
            
            insight = f"\nRegime {regime_id}"
            if desc:
                insight += f" ({desc})"
            insight += f":\n  Complexity: {row['complexity']}"
            insight += f"\n  Variables: {row['variables_used']}"
            insights.append(insight)
        
        return insights
    
    def generate_equation_report(
        self,
        equations: Dict[int, str],
        performance_metrics: Optional[pd.DataFrame] = None,
        save_path: str = "results/equation_analysis_report.txt"
    ):
        """Generate comprehensive equation analysis report.
        
        Args:
            equations: Dict mapping regime_id -> equation_string
            performance_metrics: Optional DataFrame with regime performance
            save_path: Output file path
        """
        from pathlib import Path
        
        report_lines = []
        report_lines.append("="*70)
        report_lines.append("SD-MoSE: EQUATION ANALYSIS REPORT")
        report_lines.append("="*70)
        report_lines.append("")
        
        # Parse all equations
        df_analysis = self.analyze_regime_equations(equations)
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS")
        report_lines.append("-"*70)
        report_lines.append(f"Total regimes analyzed: {len(equations)}")
        report_lines.append(f"Average complexity: {df_analysis['complexity'].mean():.1f} ± {df_analysis['complexity'].std():.1f}")
        report_lines.append(f"Average variables used: {df_analysis['n_variables'].mean():.1f}")
        report_lines.append("")
        
        # Detailed per-regime analysis
        report_lines.append("REGIME-BY-REGIME ANALYSIS")
        report_lines.append("-"*70)
        
        for _, row in df_analysis.iterrows():
            report_lines.append(f"\nRegime {int(row['regime'])}:")
            report_lines.append(f"  Equation: {row['equation']}")
            report_lines.append(f"  Complexity: {row['complexity']} nodes")
            report_lines.append(f"  Variables: {row['variables_used']}")
            
            if performance_metrics is not None:
                perf = performance_metrics[performance_metrics['regime'] == row['regime']]
                if not perf.empty:
                    report_lines.append(f"  Performance: R² = {perf.iloc[0]['r2']:.4f}, RMSE = {perf.iloc[0]['rmse']:.2f}")
        
        # Physical insights
        report_lines.append("\n")
        report_lines.append("PHYSICAL INSIGHTS")
        report_lines.append("-"*70)
        insights = self.extract_physical_insights(equations)
        report_lines.extend(insights)
        
        report_lines.append("\n" + "="*70)
        
        # Save report
        output_path = Path(save_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Equation analysis report saved: {output_path}")
        
        # Also print to console
        print('\n'.join(report_lines))


def compare_equations_across_regimes(
    equations: Dict[int, str],
    feature_names: List[str]
) -> pd.DataFrame:
    """Compare feature usage across regime equations.
    
    Args:
        equations: Dict mapping regime_id -> equation_string
        feature_names: List of possible feature names
        
    Returns:
        DataFrame showing which features are used in which regimes
    """
    feature_usage = []
    
    for regime_id, eq_str in equations.items():
        usage = {'regime': regime_id}
        
        for feature in feature_names:
            # Check if feature appears in equation
            usage[feature] = 1 if feature in eq_str else 0
        
        feature_usage.append(usage)
    
    df = pd.DataFrame(feature_usage)
    
    # Add summary row
    summary = {'regime': 'Total'}
    for feature in feature_names:
        summary[feature] = df[feature].sum()
    
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    
    return df


if __name__ == "__main__":
    # Demo
    print("Equation Analysis Demo")
    print("="*70)
    
    # Example equations (simplified)
    demo_equations = {
        0: "2.5 * SST - 0.3 * SSS + 350",
        1: "exp(-0.05 * SST) * log(Chl) + 380",
        2: "abs(SST - 15) * tanh(SSS / 35) + 360",
        3: "SST**2 / (SSS + 20) - log(Chl + 1)",
    }
    
    # Create analyzer
    analyzer = EquationAnalyzer()
    
    # Generate report
    analyzer.generate_equation_report(
        demo_equations,
        save_path="results/demo_equation_report.txt"
    )
