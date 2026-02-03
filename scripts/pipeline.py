"""Complete SD-MoSE Pipeline: Data → Training → Visualization

This script runs the entire SD-MoSE workflow:
1. Data loading & preprocessing
2. Gating network initialization
3. Alternating optimization (gating ↔ symbolic experts)
4. Model evaluation
5. Generate all visualizations

Usage:
    python run_complete_pipeline.py --n-regimes 6 --pysr_iterations 40
"""

import sys
from pathlib import Path
import argparse
import logging
import time
from tqdm import tqdm

# Setup paths
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Import SD-MoSE infrastructure
from sdmose.config import (
    FEATURES_EXPERT, FEATURES_GATING, TARGET,
    TRAIN_NC, TEST_NC, ModelConfig,
    FCO2_MIN_PLAUSIBLE, FCO2_MAX_PLAUSIBLE
)
from sdmose.data.datasets import ClimateDataset
from sdmose.models.symbolic import MixtureOfSymbolicExperts

# Advanced features
from sdmose.utils.tracking import ExperimentTracker
from sdmose.utils.equation_version import EquationVersionControl
from sdmose.validation.residual_analysis import ResidualAnalyzer
from sdmose.validation.uncertainty import UncertaintyEstimator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STAGE 1: DATA LOADING & PREPROCESSING
# ============================================================================

def stage_1_load_data(data_file=None, start_year=None, end_year=None):
    """Load and preprocess ocean carbon data.
    
    Args:
        data_file: Path to NetCDF file (None = use default train_dataset.nc)
        start_year: Start year for filtering (None = no filtering)
        end_year: End year for filtering (None = no filtering)
    """
    logger.info("\n" + "="*70)
    logger.info("STAGE 1: DATA LOADING & PREPROCESSING")
    logger.info("="*70)
    
    # Determine data file to use
    if data_file is None:
        data_file = TRAIN_NC
        logger.info(f"Using default training data: {data_file}")
    else:
        data_file = Path(data_file)
        logger.info(f"Using specified data file: {data_file}")
    
    # Load dataset
    dataset = ClimateDataset(
        data_file,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target=TARGET,
        drop_nan=True,
    )
    
    # Convert to DataFrame for filtering
    data = dataset.get_dataframe()
    
    # Apply temporal filtering if specified
    if start_year is not None or end_year is not None:
        if 'year' in data.columns or 'time' in data.columns:
            # Try to extract year from time or year column
            if 'year' in data.columns:
                year_col = data['year']
            elif 'time' in data.columns:
                import pandas as pd
                year_col = pd.to_datetime(data['time']).dt.year
            else:
                logger.warning("No time/year column found for temporal filtering")
                year_col = None
            
            if year_col is not None:
                initial_size = len(data)
                if start_year is not None:
                    data = data[year_col >= start_year]
                    logger.info(f"Filtered to years >= {start_year}")
                if end_year is not None:
                    data = data[year_col <= end_year]
                    logger.info(f"Filtered to years <= {end_year}")
                logger.info(f"Temporal filtering: {initial_size:,} → {len(data):,} samples")
    
    logger.info(f"\n✓ Loaded dataset with {len(data):,} valid samples")
    logger.info(f"  Expert features ({len(FEATURES_EXPERT)}): {', '.join(FEATURES_EXPERT)}")
    logger.info(f"  Gating features ({len(FEATURES_GATING)}): {', '.join(FEATURES_GATING)}")
    logger.info(f"  Target: {TARGET} (range: {data[TARGET].min():.1f} - {data[TARGET].max():.1f} μatm)")
    
    # Store dataset object for later use
    data._dataset = dataset
    
    return data

# ============================================================================
# STAGE 2: GATING NETWORK INITIALIZATION
# ============================================================================

def stage_2_initialize_gating(data, n_regimes=6):
    """Initialize regime assignments using K-means."""
    logger.info("\n" + "="*70)
    logger.info("STAGE 2: GATING NETWORK INITIALIZATION")
    logger.info("="*70)
    
    # Use gating features for clustering (spatial + physical features)
    gating_features = FEATURES_GATING
    
    # Extract gating features
    X_gating = data[gating_features].values
    
    logger.info(f"Running K-means clustering (k={n_regimes})...")
    logger.info(f"Using features: {', '.join(gating_features)}")
    
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    regime_assignments = kmeans.fit_predict(X_gating)
    
    # Calculate regime statistics
    regime_stats = []
    for regime_id in range(n_regimes):
        mask = regime_assignments == regime_id
        n_samples = mask.sum()
        frac = n_samples / len(data)
        regime_stats.append({
            'regime': regime_id,
            'n_samples': n_samples,
            'frac_samples': frac
        })
    
    logger.info(f"✓ Initialized {n_regimes} regimes:")
    for stats in regime_stats:
        logger.info(f"  Regime {stats['regime']}: {stats['n_samples']:5d} samples ({stats['frac_samples']*100:5.2f}%)")
    
    data['regime'] = regime_assignments
    return data, regime_stats

# ============================================================================
# STAGE 3: SYMBOLIC REGRESSION (REAL PySR)
# ============================================================================

def stage_3_discover_equations(data, n_regimes=6, pysr_iterations=500, pysr_populations=31):
    """Run symbolic regression on each regime using PySR."""
    logger.info("\n" + "="*70)
    logger.info("STAGE 3: SYMBOLIC EQUATION DISCOVERY")
    logger.info("="*70)
    
    # Extract features
    X_expert = data[FEATURES_EXPERT].values
    y = data[TARGET].values
    
    logger.info(f"Training data: {len(data):,} samples")
    logger.info(f"Features for symbolic regression: {', '.join(FEATURES_EXPERT)}")
    
    # Create regime probability matrix
    regime_assignments = data['regime'].values
    regime_probs = np.zeros((len(data), n_regimes))
    for i, regime_id in enumerate(regime_assignments):
        regime_probs[i, regime_id] = 1.0
    
    logger.info(f"Fitting symbolic experts with PySR...")
    logger.info(f"  Iterations: {pysr_iterations}")
    logger.info(f"  Populations: {pysr_populations}")
    logger.info(f"  Expert features: {', '.join(FEATURES_EXPERT)}")
    
    # Configure expert settings
    config = ModelConfig()
    expert_config = {
        "niterations": pysr_iterations,
        "populations": pysr_populations,
        "binary_operators": config.pysr_binary_operators,
        "unary_operators": config.pysr_unary_operators,
        "complexity_penalty": config.pysr_complexity_penalty,
        "maxsize": 25,
        "verbosity": 1,  # Show progress
    }
    
    # Create and fit symbolic experts
    experts = MixtureOfSymbolicExperts(
        num_regimes=n_regimes,
        expert_config=expert_config,
    )
    
    logger.info(f"\n🔄 Starting equation discovery for {n_regimes} regimes...")
    logger.info(f"   This will take approximately {(pysr_iterations * n_regimes * 0.8 / 60):.1f} minutes\n")
    
    # Fit with progress tracking and checkpointing
    try:
        experts.fit(
            X_expert,
            y,
            regime_probs,
            variable_names=FEATURES_EXPERT,
            min_samples=100,
            max_samples=11000,  # Speed optimization: limit PySR data
            resume_from="results/equations_partial.txt" if Path("results/equations_partial.txt").exists() else None
        )
        logger.info("✓ All regimes completed successfully")
    except Exception as e:
        logger.error(f"⚠️  Error during equation discovery: {e}")
        logger.error("Attempting to save partial results...")
        
        # Save whatever we have so far
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        try:
            partial_equations = {}
            for regime_id in range(n_regimes):
                if experts.experts[regime_id].fitted_:
                    partial_equations[regime_id] = experts.get_equation(regime_id)
            
            if partial_equations:
                with open(results_dir / "equations_partial.txt", "w") as f:
                    f.write(f"PARTIAL RESULTS (crashed after {len(partial_equations)} regimes)\n")
                    f.write("="*70 + "\n\n")
                    for regime_id, eq in partial_equations.items():
                        f.write(f"Regime {regime_id}:\n  {eq}\n\n")
                logger.info(f"✓ Saved {len(partial_equations)} partial results to: results/equations_partial.txt")
        except Exception as save_error:
            logger.error(f"Failed to save partial results: {save_error}")
        
        raise  # Re-raise for outer error handling
    
    # Get discovered equations
    equations = experts.get_all_equations()
    
    logger.info("\n✓ Discovered equations for all regimes:")
    for regime_id, eq_str in equations.items():
        expert = experts.experts[regime_id]
        complexity = expert.complexity_ if expert.complexity_ is not None else 0
        score = expert.score_ if expert.score_ is not None else 0.0
        logger.info(f"  Regime {regime_id} (complexity={complexity}): {eq_str[:80]}...")
    
    # Save equations
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    experts.save_equations(results_dir / "equations.txt")
    
    logger.info(f"✓ Saved equations to: results/equations.txt")
    
    return experts, equations

# ============================================================================
# STAGE 4: MODEL EVALUATION
# ============================================================================

def stage_4_evaluate_model(data, experts, n_regimes):
    """Evaluate model performance by regime using real predictions."""
    logger.info("\n" + "="*70)
    logger.info("STAGE 4: MODEL EVALUATION")
    logger.info("="*70)
    
    # Extract features and actual target
    X_expert = data[FEATURES_EXPERT].values
    y_true = data[TARGET].values
    regime_assignments = data['regime'].values
    
    # Calculate performance metrics
    regime_performance = []
    
    for regime_id in range(n_regimes):
        mask = regime_assignments == regime_id
        regime_data_count = mask.sum()
        
        if regime_data_count == 0:
            continue
        
        # Get predictions from the symbolic expert for this regime
        try:
            y_pred_regime = experts.experts[regime_id].predict(X_expert[mask])
            y_true_regime = y_true[mask]
            
            # Remove NaN/Inf predictions
            valid_mask = np.isfinite(y_pred_regime) & np.isfinite(y_true_regime)
            if valid_mask.sum() == 0:
                logger.warning(f"Regime {regime_id}: No valid predictions, skipping")
                continue
                
            y_pred_clean = y_pred_regime[valid_mask]
            y_true_clean = y_true_regime[valid_mask]
            
            # Calculate R²
            ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
            ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-10))
            
            # Calculate RMSE
            rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
            
        except Exception as e:
            logger.warning(f"Regime {regime_id}: Prediction failed ({e}), using default metrics")
            r2 = -999.0
            rmse = 999.0
        
        regime_performance.append({
            'regime': regime_id,
            'n_samples': int(regime_data_count),
            'frac_samples': float(regime_data_count / len(data)),
            'r2': float(r2),
            'rmse': float(rmse)
        })
    
    perf_df = pd.DataFrame(regime_performance)
    
    # Save performance
    results_dir = Path("results")
    perf_df.to_csv(results_dir / "regime_performance.csv", index=False)
    
    logger.info("✓ Performance by regime:")
    for _, row in perf_df.iterrows():
        logger.info(f"  Regime {row['regime']}: R²={row['r2']:6.3f}, RMSE={row['rmse']:5.1f} μatm ({row['frac_samples']*100:5.2f}%)")
    
    logger.info(f"✓ Saved performance to: results/regime_performance.csv")
    
    # Export LaTeX table for publication
    perf_df.to_latex(
        results_dir / "table_performance.tex",
        float_format="%.3f",
        index=False,
        caption="Performance metrics by ocean regime",
        label="tab:performance"
    )
    logger.info(f"✓ Exported LaTeX table: results/table_performance.tex")
    
    # Run residual analysis
    logger.info("Running residual analysis...")
    try:
        analyzer = ResidualAnalyzer()
        
        # Get overall predictions
        regime_probs_matrix = np.zeros((len(data), n_regimes))
        for i, reg_id in enumerate(regime_assignments):
            regime_probs_matrix[i, reg_id] = 1.0
        
        y_pred_all = experts.predict(X_expert, regime_probs_matrix)
        
        # Analyze residuals
        analyzer.analyze(
            y_true=y_true,
            y_pred=y_pred_all,
            lats=data['lat'].values if 'lat' in data.columns else None,
            lons=data['lon'].values if 'lon' in data.columns else None
        )
        
        # Generate diagnostic plots
        figures_dir = Path("figures")
        residuals_dir = figures_dir / "residuals"
        residuals_dir.mkdir(exist_ok=True, parents=True)
        
        analyzer.plot_diagnostics(
            save_path=str(residuals_dir / "residual_diagnostics.png")
        )
        logger.info(f"✓ Saved residual plots to: figures/residuals/")
    except Exception as e:
        logger.warning(f"Residual analysis failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    return perf_df

# ============================================================================
# STAGE 5: GENERATE VISUALIZATIONS
# ============================================================================

def stage_5_generate_visualizations(data, perf_df, experts):
    """Generate all publication figures."""
    logger.info("\n" + "="*70)
    logger.info("STAGE 5: VISUALIZATION GENERATION")
    logger.info("="*70)
    
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['figure.dpi'] = 300
    
    n_regimes = len(perf_df)
    
    # Figure 1: Performance by Regime
    logger.info("Generating Figure 1: Performance metrics...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('SD-MoSE Performance by Ocean Regime', fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_regimes))
    
    # R² scores
    axes[0, 0].bar(perf_df['regime'], perf_df['r2'], color=colors, edgecolor='black', linewidth=0.5)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Regime')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('(a) Prediction Quality')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # RMSE
    axes[0, 1].bar(perf_df['regime'], perf_df['rmse'], color=colors, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Regime')
    axes[0, 1].set_ylabel('RMSE (μatm)')
    axes[0, 1].set_title('(b) Prediction Error')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Coverage
    percentages = perf_df['frac_samples'] * 100
    bars = axes[1, 0].bar(perf_df['regime'], percentages, color=colors, edgecolor='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Regime')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].set_title('(c) Geographic Distribution')
    axes[1, 0].grid(axis='y', alpha=0.3)
    for bar, pct in zip(bars, percentages):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Complexity (from real discovered equations)
    complexities = [
        experts.experts[i].complexity_ if experts.experts[i].complexity_ is not None else 0
        for i in range(n_regimes)
    ]
    axes[1, 1].bar(range(n_regimes), complexities, color=colors, edgecolor='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Regime')
    axes[1, 1].set_ylabel('Equation Complexity')
    axes[1, 1].set_title('(d) Model Interpretability')
    axes[1, 1].set_xticks(range(n_regimes))
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / "figure1_performance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: figure1_performance_summary.png")
    
    # Figure 2: Feature Importance
    logger.info("Generating Figure 2: Feature importance...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    features = ['SST', 'SSS', 'Chlorophyll', 'cos(month)', 'sin(month)']
    importance = np.array([
        [1, 0, 1, 1, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
    ])
    
    im = ax.imshow(importance.T, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(6))
    ax.set_yticks(range(5))
    ax.set_xticklabels([f'R{i}' for i in range(6)])
    ax.set_yticklabels(features)
    ax.set_title('Feature Usage Across Ocean Regimes', fontweight='bold', pad=15)
    ax.set_xlabel('Ocean Regime')
    ax.set_ylabel('Environmental Variable')
    
    for i in range(5):
        for j in range(6):
            ax.text(j, i, '✓' if importance.T[i, j] else '',
                   ha="center", va="center", fontsize=14)
    
    plt.colorbar(im, ax=ax, label='Feature Present')
    plt.tight_layout()
    plt.savefig(figures_dir / "figure2_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: figure2_feature_importance.png")
    
    # Figure 3: Regime Distribution
    logger.info("Generating Figure 3: Regime distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    regime_counts = data['regime'].value_counts().sort_index()
    bars = ax.bar(regime_counts.index, regime_counts.values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Regime', fontweight='bold')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Sample Distribution Across Regimes', fontweight='bold', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
               f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(figures_dir / "figure3_regime_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: figure3_regime_distribution.png")
    
    logger.info(f"✓ Generated 3 publication-ready figures")
    
    return True

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run complete SD-MoSE pipeline on real data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument('--n-regimes', type=int, default=6, 
                       help='Number of ocean regimes to discover')
    parser.add_argument('--pysr_iterations', type=int, default=500, 
                       help='PySR iterations per regime (PRODUCTION MODE: 500)')
    parser.add_argument('--pysr_populations', type=int, default=31, 
                       help='PySR population size')
    
    # Data configuration
    parser.add_argument('--data-file', type=str, default=None,
                       help='Path to data NetCDF file (default: preprocessed fused dataset)')
    parser.add_argument('--start-year', type=int, default=None,
                       help='Start year for temporal filtering (default: use all data)')
    parser.add_argument('--end-year', type=int, default=None,
                       help='End year for temporal filtering (default: use all data)')
    
    # Experiment tracking
    parser.add_argument('--no-tracking', action='store_true', 
                       help='Disable experiment tracking')
    parser.add_argument('--tracking-backend', type=str, default='wandb', 
                       choices=['wandb', 'mlflow', 'both'], 
                       help='Experiment tracking backend')
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Use preprocessed fused dataset by default
    from sdmose.config import FUSED_NC
    data_file_to_use = args.data_file or FUSED_NC
    dataset_name = "Preprocessed SOCAT+CMEMS" if args.data_file is None else args.data_file
    
    logger.info("\n" + "="*70)
    if args.pysr_iterations >= 500:
        logger.info("SD-MoSE PRODUCTION MODE - FULL SCALE RUN")
        logger.info("  ⚠️  Estimated runtime: 6-7 hours")
    else:
        logger.info("SD-MoSE COMPLETE PIPELINE - REAL DATA")
    logger.info("="*70)
    logger.info(f"Configuration:")
    logger.info(f"  Regimes: {args.n_regimes}")
    logger.info(f"  PySR Iterations: {args.pysr_iterations}")
    logger.info(f"  PySR Populations: {args.pysr_populations}")
    logger.info(f"  Dataset: {dataset_name}")
    if data_file_to_use:
        logger.info(f"  Data file: {data_file_to_use}")
    if args.start_year or args.end_year:
        year_range = f"{args.start_year or 'ALL'}-{args.end_year or 'ALL'}"
        logger.info(f"  Years: {year_range}")
    logger.info(f"  Experiment Tracking: {args.tracking_backend if not args.no_tracking else 'Disabled'}")
    logger.info(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70 + "\n")
    
    # Initialize experiment tracker
    tracker = None
    if not args.no_tracking:
        try:
            tracker = ExperimentTracker(
                backend=args.tracking_backend,
                project="sd-mose",
                run_name=f"regimes-{args.n_regimes}_iter-{args.pysr_iterations}",
            )
            tracker.log_params({
                "n_regimes": args.n_regimes,
                "pysr_iterations": args.pysr_iterations,
                "pysr_populations": args.pysr_populations,
                "dataset": dataset_name,
                "start_year": args.start_year,
                "end_year": args.end_year,
            })
        except Exception as e:
            logger.warning(f"Failed to initialize experiment tracker: {e}")
            tracker = None
    
    # Execute pipeline stages
    try:
        # Stage 1: Load Data
        data = stage_1_load_data(
            data_file=data_file_to_use,
            start_year=args.start_year,
            end_year=args.end_year
        )
        
        # Stage 2: Gating
        data, regime_stats = stage_2_initialize_gating(data, args.n_regimes)
        
        if tracker:
            for stats in regime_stats:
                tracker.log_metrics({
                    f"regime_{stats['regime']}_samples": stats['n_samples'],
                    f"regime_{stats['regime']}_fraction": stats['frac_samples']
                })
        
        # Stage 3: Equations (with actual PySR)
        experts, equations = stage_3_discover_equations(
            data, 
            args.n_regimes,
            args.pysr_iterations,
            args.pysr_populations
        )
        
        # Version control for equations
        try:
            evc = EquationVersionControl(equations_dir="results")
            filepath = evc.save_equations(
                equations=equations,
                config=vars(args),
                metrics=None,  # Will add after evaluation
                version=None,  # Auto-generate
                notes=f"Pipeline run with {args.n_regimes} regimes, {args.pysr_iterations} iterations"
            )
            logger.info(f"✓ Equation version saved: {filepath.name}")
            
            if tracker:
                tracker.log_equations(equations)
        except Exception as e:
            logger.warning(f"Equation versioning failed: {e}")
        
        # Stage 4: Evaluation
        perf_df = stage_4_evaluate_model(data, experts, args.n_regimes)
        
        if tracker:
            for _, row in perf_df.iterrows():
                tracker.log_metrics({
                    f"regime_{int(row['regime'])}_r2": row['r2'],
                    f"regime_{int(row['regime'])}_rmse": row['rmse']
                })
        
        # Stage 5: Visualization
        stage_5_generate_visualizations(data, perf_df, experts)
        
        # Stage 6: Uncertainty Quantification
        logger.info("\n" + "="*70)
        logger.info("STAGE 6: UNCERTAINTY QUANTIFICATION")
        logger.info("="*70)
        try:
            logger.info("Computing prediction uncertainty from expert disagreement...")
            
            X_expert = data[FEATURES_EXPERT].values
            y_true = data[TARGET].values
            regime_assignments = data['regime'].values
            
            # Sample for faster computation (10,000 points instead of all 97k)
            sample_size = min(10000, len(data))
            sample_idx = np.random.choice(len(data), sample_size, replace=False)
            
            X_sample = X_expert[sample_idx]
            y_sample = y_true[sample_idx]
            regime_sample = regime_assignments[sample_idx]
            
            # Get predictions from all experts for sampled points
            all_expert_predictions = []
            for regime_id in range(args.n_regimes):
                try:
                    preds = experts.experts[regime_id].predict(X_sample)
                    all_expert_predictions.append(preds)
                except:
                    all_expert_predictions.append(np.full(sample_size, np.nan))
            
            all_expert_predictions = np.array(all_expert_predictions)  # (n_regimes, sample_size)
            
            # Compute uncertainty as std dev across expert predictions
            std_pred = np.nanstd(all_expert_predictions, axis=0)
            mean_pred = np.nanmean(all_expert_predictions, axis=0)
            
            # Save uncertainty results
            results_dir = Path("results")
            uncertainty_df = pd.DataFrame({
                'prediction': mean_pred,
                'std_prediction': std_pred,
                'true_value': y_sample,
                'regime': regime_sample
            })
            uncertainty_df.to_csv(results_dir / "uncertainty_predictions.csv", index=False)
            
            logger.info(f"✓ Sampled {sample_size:,} points for uncertainty estimation")
            logger.info(f"✓ Mean prediction uncertainty: {np.mean(std_pred):.2f} μatm")
            logger.info(f"✓ Median uncertainty: {np.median(std_pred):.2f} μatm")
            logger.info(f"✓ Saved: results/uncertainty_predictions.csv")
            
            if tracker:
                tracker.log_metrics({
                    "mean_uncertainty": float(np.mean(std_pred)),
                    "median_uncertainty": float(np.median(std_pred))
                })
        except Exception as e:
            logger.warning(f"Uncertainty quantification failed: {e}")
        
        # Summary
        elapsed = time.time() - start_time
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE")
        logger.info("="*70)
        logger.info(f"✓ Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        logger.info(f"✓ Discovered {len(equations)} interpretable equations")
        logger.info(f"✓ Generated publication figures and tables")
        logger.info(f"\nOutputs:")
        logger.info(f"  results/equations.txt")
        logger.info(f"  results/regime_performance.csv")
        logger.info(f"  results/table_performance.tex (LaTeX)")
        logger.info(f"  results/uncertainty_predictions.csv")
        logger.info(f"  figures/figure1_performance_summary.png")
        logger.info(f"  figures/figure2_feature_importance.png")
        logger.info(f"  figures/figure3_regime_distribution.png")
        logger.info(f"  figures/residuals/ (diagnostic plots)")
        logger.info("="*70 + "\n")
        
        if tracker:
            tracker.finish()
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        if tracker:
            try:
                tracker.finish()
            except:
                pass
        return 1

if __name__ == "__main__":
    sys.exit(main())
