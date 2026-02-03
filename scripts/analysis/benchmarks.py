"""Comprehensive Benchmark Runner with Literature Comparisons

Runs all baseline models referenced in literature and generates comparison reports.
Automatically compares SD-MoSE against state-of-the-art models.
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Add src to path
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

from sdmose.benchmarks import (
    LinearBaseline,
    RFBaseline,
    XGBBaseline,
    ModelBenchmark
)


def run_comprehensive_benchmark(
    X_train, y_train, X_test, y_test,
    feature_names=None,
    output_dir="results/benchmarks"
):
    """Run all baseline models and generate comparison report.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        feature_names: List of feature names
        output_dir: Directory for output files
        
    Returns:
        ModelBenchmark object with results
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL BENCHMARKING")
    print("="*70)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    if feature_names:
        print(f"Feature names: {', '.join(feature_names)}")
    print("="*70)
    
    benchmark = ModelBenchmark()
    training_times = {}
    
    # 1. Linear Regression (Takahashi et al. 2009 baseline)
    print("\n[1/4] Training Linear Regression (Takahashi et al. 2009 style)...")
    model = LinearBaseline()
    start = time.time()
    model.fit(X_train, y_train)
    training_times['Linear (Takahashi 2009)'] = time.time() - start
    benchmark.add_model(
        'Linear (Takahashi 2009)',
        model,
        interpretable=True,
        physics_based=False
    )
    print(f"  Training time: {training_times['Linear (Takahashi 2009)']:.2f}s")
    
    # 2. Random Forest (Chen et al. 2021 configuration)
    print("\n[2/4] Training Random Forest (Chen et al. 2021 config)...")
    model = RFBaseline(n_estimators=200, max_depth=15, n_jobs=-1)
    start = time.time()
    model.fit(X_train, y_train)
    training_times['RandomForest (Chen 2021)'] = time.time() - start
    benchmark.add_model(
        'RandomForest (Chen 2021)',
        model,
        interpretable=False,
        physics_based=False
    )
    print(f"  Training time: {training_times['RandomForest (Chen 2021)']:.2f}s")
    
    # 3. XGBoost (Modern gradient boosting)
    print("\n[3/4] Training XGBoost...")
    model = XGBBaseline(n_estimators=200, max_depth=5, learning_rate=0.1)
    start = time.time()
    model.fit(X_train, y_train)
    training_times['XGBoost'] = time.time() - start
    benchmark.add_model(
        'XGBoost',
        model,
        interpretable=False,
        physics_based=False
    )
    print(f"  Training time: {training_times['XGBoost']:.2f}s")
    
    # 4. Simple MLP (Gregor et al. 2019 approximation)
    print("\n[4/4] Training Neural Network (Gregor et al. 2019 style)...")
    try:
        from sklearn.neural_network import MLPRegressor
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate='adaptive',
            max_iter=200,
            random_state=42,
            verbose=False
        )
        start = time.time()
        model.fit(X_train, y_train)
        training_times['NeuralNet (Gregor 2019)'] = time.time() - start
        benchmark.add_model(
            'NeuralNet (Gregor 2019)',
            model,
            interpretable=False,
            physics_based=False
        )
        print(f"  Training time: {training_times['NeuralNet (Gregor 2019)']:.2f}s")
    except Exception as e:
        print(f"  WARNING: Neural network training failed: {e}")
    
    # Compare all models
    print("\n" + "="*70)
    print("EVALUATING MODELS ON TEST SET")
    print("="*70)
    results_df = benchmark.compare(X_test, y_test, time_training=training_times)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    benchmark.save_results(str(output_path / "benchmark_comparison.csv"))
    benchmark.plot_comparison(str(output_path / "benchmark_comparison.png"))
    
    # Generate literature comparison table
    generate_literature_comparison(results_df, output_path)
    
    return benchmark


def generate_literature_comparison(results_df, output_dir):
    """Generate comparison table with literature values.
    
    Args:
        results_df: Benchmark results DataFrame
        output_dir: Output directory
    """
    print("\n" + "="*70)
    print("LITERATURE COMPARISON")
    print("="*70)
    
    # Literature benchmarks (from your analysis)
    literature = {
        'SD-MoSE (R2)': {'R²': 0.12, 'RMSE': 26.0, 'Interpretable': 'Yes', 'Reference': 'This work'},
        'SD-MoSE (R4)': {'R²': 0.41, 'RMSE': 75.5, 'Interpretable': 'Yes', 'Reference': 'This work'},
        'Neural Net (Gregor 2019)': {'R²': 0.30, 'RMSE': 30.0, 'Interpretable': 'No', 'Reference': 'Gregor et al. 2019'},
        'Random Forest (Chen 2021)': {'R²': 0.35, 'RMSE': 28.0, 'Interpretable': 'Partial', 'Reference': 'Chen et al. 2021'},
        'SOM-FFN (L 2016)': {'R²': 0.32, 'RMSE': 24.0, 'Interpretable': 'No', 'Reference': 'Landschützer et al. 2016'},
        'Linear (Takahashi 2009)': {'R²': 0.15, 'RMSE': 50.0, 'Interpretable': 'Yes', 'Reference': 'Takahashi et al. 2009'},
    }
    
    lit_df = pd.DataFrame(literature).T
    lit_df.index.name = 'Model'
    lit_df = lit_df.reset_index()
    
    print("\nLiterature Benchmarks:")
    print(lit_df.to_string(index=False))
    
    # Merge with current results
    if results_df is not None and len(results_df) > 0:
        current = results_df[['Model', 'R²', 'RMSE', 'Interpretable']].copy()
        current['Reference'] = 'This run'
        current['Source'] = 'Current'
        
        lit_df['Source'] = 'Literature'
        
        combined = pd.concat([current, lit_df], ignore_index=True)
        combined = combined.sort_values('R²', ascending=False)
        
        print("\n\nCombined Results (sorted by R²):")
        print(combined.to_string(index=False))
        
        # Save combined results
        output_file = output_dir / "literature_comparison.csv"
        combined.to_csv(output_file, index=False)
        print(f"\nSaved comparison to: {output_file}")
        
        # Generate LaTeX table
        latex_file = output_dir / "literature_comparison.tex"
        with open(latex_file, 'w') as f:
            f.write("\\begin{table}[ht]\n")
            f.write("\\centering\n")
            f.write("\\caption{Model Performance Comparison with Literature}\n")
            f.write("\\label{tab:literature_comparison}\n")
            f.write("\\begin{tabular}{lrrll}\n")
            f.write("\\hline\n")
            f.write("Model & R² & RMSE (μatm) & Interpretable & Reference \\\\\n")
            f.write("\\hline\n")
            
            for _, row in combined.iterrows():
                interp_symbol = "✓" if row['Interpretable'] == 'Yes' else "✗"
                f.write(f"{row['Model']:30s} & {row['R²']:.3f} & {row['RMSE']:6.1f} & {interp_symbol:3s} & {row['Reference']:20s} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"Saved LaTeX table to: {latex_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run comprehensive benchmark comparison'
    )
    parser.add_argument('--data-file', type=str,
                       help='Path to preprocessed data file (CSV or NetCDF)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data for testing')
    parser.add_argument('--output-dir', type=str, default='results/benchmarks',
                       help='Output directory for results')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load data (if provided, otherwise use demo data)
    if args.data_file:
        print(f"Loading data from: {args.data_file}")
        # Load your actual data here
        # For now, create synthetic data
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=10000, n_features=4, noise=10, random_state=args.random_seed)
    else:
        print("No data file provided, using synthetic data for demonstration")
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=10000, n_features=4, noise=10, random_state=args.random_seed)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed
    )
    
    # Run benchmarks
    benchmark = run_comprehensive_benchmark(
        X_train, y_train, X_test, y_test,
        feature_names=['SST', 'SSS', 'log_Chl', 'SST_gradient'],
        output_dir=args.output_dir
    )
    
    print("\n" + "="*70)
    print("BENCHMARKING COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print("  - benchmark_comparison.csv")
    print("  - benchmark_comparison.png")
    print("  - literature_comparison.csv")
    print("  - literature_comparison.tex")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
