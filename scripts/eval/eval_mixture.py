"""Evaluate trained SD-MoSE model on test set.

Loads:
- Trained gating network
- Discovered symbolic experts
- Test data

Computes:
- Overall performance (R², RMSE, MAE)
- Per-regime performance
- Regime statistics
- OOD evaluation
- Comparison with baselines

Usage:
    python -m scripts.eval.eval_mixture
    python -m scripts.eval.eval_mixture --checkpoint path/to/sdmose_final.pth
"""
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

root = Path(__file__).resolve().parents[2]  # noqa: E402
sys.path.insert(0, str(root / "src"))  # noqa: E402

import argparse  # noqa: E402
import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from climate_discovery.config import (  # noqa: E402
    CHECKPOINT_DIR,
    FEATURES_EXPERT,
    FEATURES_GATING,
    LAT_BANDS,
    ModelConfig,
    RESULTS_DIR,
    TARGET,
    TEST_NC,
    TRAIN_NC,
)
from climate_discovery.data.datasets import ClimateDataset  # noqa: E402
from climate_discovery.models.gating import GatingNetwork  # noqa: E402
from climate_discovery.models.mixture import SDMoSE, evaluate_mixture  # noqa: E402
from climate_discovery.models.symbolic import MixtureOfSymbolicExperts  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_sdmose_model(
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple[SDMoSE, MixtureOfSymbolicExperts]:
    """Load trained SD-MoSE model.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device
        
    Returns:
        (SDMoSE model, MixtureOfSymbolicExperts)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    
    config = ModelConfig()
    
    # Create gating network
    gating = GatingNetwork(
        input_dim=len(FEATURES_GATING),
        num_regimes=config.n_regimes,
        hidden_dims=config.gating_hidden_dims,
        dropout=0.0,  # No dropout for inference
        temperature=1.0,
    ).to(device)
    
    # Create SD-MoSE model
    model = SDMoSE(
        gating_network=gating,
        num_regimes=config.n_regimes,
        expert_features=FEATURES_EXPERT,
        device=device,
    )
    
    # Load checkpoint
    model.load(checkpoint_path)
    
    logger.info("✓ Model loaded")
    
    return model


def refit_symbolic_experts(
    model: SDMoSE,
    train_dataset: ClimateDataset,
    device: str,
) -> MixtureOfSymbolicExperts:
    """Refit symbolic experts using trained gating.
    
    Note: This is needed if experts weren't saved with checkpoint.
    In production, you'd save experts separately.
    
    Args:
        model: SD-MoSE model with trained gating
        train_dataset: Training data
        device: Device
        
    Returns:
        MixtureOfSymbolicExperts
    """
    logger.info("Refitting symbolic experts...")
    
    config = ModelConfig()
    
    # Get regime probabilities
    X_gate_tensor = torch.from_numpy(train_dataset.X_gate).float().to(device)
    
    with torch.no_grad():
        regime_probs = model.get_regime_probs(X_gate_tensor).cpu().numpy()
    
    # Fit experts
    expert_config = {
        "niterations": config.pysr_niterations,
        "populations": config.pysr_populations,
        "binary_operators": config.pysr_binary_operators,
        "unary_operators": config.pysr_unary_operators,
        "complexity_penalty": config.pysr_complexity_penalty,
        "maxsize": 25,
        "verbosity": 1,
    }
    
    experts = MixtureOfSymbolicExperts(
        num_regimes=config.n_regimes,
        expert_config=expert_config,
    )
    
    experts.fit(
        train_dataset.X_expert,
        train_dataset.y,
        regime_probs,
        variable_names=FEATURES_EXPERT,
        min_samples=100,
    )
    
    # Log equations
    equations = experts.get_all_equations()
    logger.info("Discovered equations:")
    for k, eq in equations.items():
        logger.info(f"  Regime {k}: {eq}")
    
    return experts


def evaluate_per_regime(
    model: SDMoSE,
    test_dataset: ClimateDataset,
    symbolic_experts: MixtureOfSymbolicExperts,
) -> pd.DataFrame:
    """Evaluate performance per regime.
    
    Args:
        model: SD-MoSE model
        test_dataset: Test data
        symbolic_experts: Symbolic experts
        
    Returns:
        DataFrame with per-regime metrics
    """
    logger.info("Evaluating per-regime performance...")
    
    # Get regime probabilities
    X_gate = torch.from_numpy(test_dataset.X_gate).float().to(model.device)
    
    with torch.no_grad():
        regime_probs = model.get_regime_probs(X_gate).cpu().numpy()
    
    # Get dominant regime for each sample
    dominant_regime = np.argmax(regime_probs, axis=1)
    
    # Evaluate each regime
    results = []
    
    for k in range(model.num_regimes):
        mask = dominant_regime == k
        n_samples = np.sum(mask)
        
        if n_samples < 10:
            continue
        
        # Get predictions for this regime
        X_regime = test_dataset.X_expert[mask]
        y_regime = test_dataset.y[mask]
        
        try:
            y_pred_regime = symbolic_experts.experts[k].predict(X_regime)
            
            # Compute metrics
            from sklearn.metrics import mean_squared_error, r2_score
            r2 = r2_score(y_regime, y_pred_regime)
            rmse = np.sqrt(mean_squared_error(y_regime, y_pred_regime))
            
            results.append({
                "regime": k,
                "n_samples": n_samples,
                "frac_samples": n_samples / len(test_dataset),
                "r2": r2,
                "rmse": rmse,
                "equation": symbolic_experts.experts[k].get_best_equation(),
            })
            
        except Exception as e:
            logger.warning(f"Regime {k} evaluation failed: {e}")
    
    return pd.DataFrame(results)


def evaluate_ood(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lat: np.ndarray,
    bands: dict,
) -> pd.DataFrame:
    """Evaluate out-of-distribution performance.
    
    Args:
        y_true: True values
        y_pred: Predictions
        lat: Latitudes
        bands: Latitude band definitions
        
    Returns:
        DataFrame with OOD metrics
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    results = []
    
    for band_name, (lat_min, lat_max) in bands.items():
        mask = (lat >= lat_min) & (lat < lat_max)
        n_samples = np.sum(mask)
        
        if n_samples < 10:
            continue
        
        y_true_band = y_true[mask]
        y_pred_band = y_pred[mask]
        
        r2 = r2_score(y_true_band, y_pred_band)
        rmse = np.sqrt(mean_squared_error(y_true_band, y_pred_band))
        
        results.append({
            "band": band_name,
            "lat_min": lat_min,
            "lat_max": lat_max,
            "n_samples": n_samples,
            "r2": r2,
            "rmse": rmse,
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SD-MoSE model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SD-MoSE checkpoint"
    )
    parser.add_argument(
        "--refit_experts",
        action="store_true",
        help="Refit symbolic experts (if not saved in checkpoint)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = CHECKPOINT_DIR / "sdmose_final.pth"
    
    # Output directory
    output_dir = Path(args.output) if args.output else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    logger.info("=" * 60)
    logger.info("LOADING DATA")
    logger.info("=" * 60)
    
    train_dataset = ClimateDataset(
        TRAIN_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target=TARGET,
        drop_nan=True,
    )
    
    test_dataset = ClimateDataset(
        TEST_NC,
        expert_features=FEATURES_EXPERT,
        gating_features=FEATURES_GATING,
        target=TARGET,
        drop_nan=True,
    )
    
    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Test: {len(test_dataset)} samples")
    
    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    logger.info("=" * 60)
    logger.info("LOADING MODEL")
    logger.info("=" * 60)
    
    try:
        model = load_sdmose_model(checkpoint_path, device)
    except FileNotFoundError:
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Run training first: python -m scripts.train.train_sdmose")
        sys.exit(1)
    
    # =========================================================================
    # LOAD OR REFIT SYMBOLIC EXPERTS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("SYMBOLIC EXPERTS")
    logger.info("=" * 60)
    
    if args.refit_experts or model.symbolic_experts is None:
        symbolic_experts = refit_symbolic_experts(model, train_dataset, device)
        model.attach_symbolic_experts(symbolic_experts)
    else:
        symbolic_experts = model.symbolic_experts
    
    # =========================================================================
    # OVERALL EVALUATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("OVERALL PERFORMANCE")
    logger.info("=" * 60)
    
    overall_metrics = evaluate_mixture(
        model,
        test_dataset.X_gate,
        test_dataset.X_expert,
        test_dataset.y,
        symbolic_experts,
    )
    
    logger.info(f"R² = {overall_metrics['r2']:.4f}")
    logger.info(f"RMSE = {overall_metrics['rmse']:.4f} μatm")
    logger.info(f"MAE = {overall_metrics['mae']:.4f} μatm")
    logger.info(f"Mean Entropy = {overall_metrics['mean_entropy']:.4f}")
    
    # =========================================================================
    # PER-REGIME EVALUATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PER-REGIME PERFORMANCE")
    logger.info("=" * 60)
    
    regime_results = evaluate_per_regime(model, test_dataset, symbolic_experts)
    
    print("\n" + regime_results.to_string(index=False))
    
    # Save
    regime_results.to_csv(output_dir / "regime_performance.csv", index=False)
    logger.info(f"\n✓ Saved: {output_dir / 'regime_performance.csv'}")
    
    # =========================================================================
    # OUT-OF-DISTRIBUTION EVALUATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("OUT-OF-DISTRIBUTION EVALUATION")
    logger.info("=" * 60)
    
    # Get predictions
    y_pred = model.predict_numpy(test_dataset.X_gate, test_dataset.X_expert)
    
    ood_results = evaluate_ood(
        test_dataset.y,
        y_pred,
        test_dataset.coords["lat"],
        LAT_BANDS,
    )
    
    print("\n" + ood_results.to_string(index=False))
    
    # Save
    ood_results.to_csv(output_dir / "ood_performance.csv", index=False)
    logger.info(f"\n✓ Saved: {output_dir / 'ood_performance.csv'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    summary = {
        "model": "SD-MoSE",
        "test_r2": overall_metrics["r2"],
        "test_rmse": overall_metrics["rmse"],
        "test_mae": overall_metrics["mae"],
        "mean_entropy": overall_metrics["mean_entropy"],
        "n_regimes": model.num_regimes,
    }
    
    summary_df = pd.DataFrame([summary])
    print("\n" + summary_df.to_string(index=False))
    
    # Save
    summary_df.to_csv(output_dir / "sdmose_summary.csv", index=False)
    logger.info(f"\n✓ Saved: {output_dir / 'sdmose_summary.csv'}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()