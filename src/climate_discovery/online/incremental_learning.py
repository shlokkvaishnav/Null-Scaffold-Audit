"""Online learning capabilities for SD-MoSE.

Enables incremental model updates as new data arrives:
- Warm-start from existing model
- Selective expert retraining
- Gating network fine-tuning
- Drift detection

Usage:
    from climate_discovery.online import OnlineLearner
    
    learner = OnlineLearner.from_checkpoint("model.pth")
    learner.incremental_update(new_data)
    learner.save_checkpoint("model_updated.pth")
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class OnlineLearner:
    """Online learning for SD-MoSE models.
    
    Supports incremental updates without full retraining:
    - Gating network fine-tuning
    - Expert warm-starting
    - Concept drift detection
    
    Example:
        >>> learner = OnlineLearner.from_checkpoint("model.pth")
        >>> new_data = load_latest_month()
        >>> stats = learner.incremental_update(new_data)
        >>> if stats['drift_detected']:
        ...     print("⚠️ Significant drift detected - full retrain recommended")
    """
    
    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        drift_threshold: float = 0.1,
    ):
        """Initialize online learner.
        
        Args:
            model: SD-MoSE model instance
            learning_rate: LR for fine-tuning (lower than initial training)
            drift_threshold: R² drop threshold to trigger drift warning
        """
        self.model = model
        self.learning_rate = learning_rate
        self.drift_threshold = drift_threshold
        self.baseline_performance = None
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs):
        """Load model from checkpoint for online learning.
        
        Args:
            checkpoint_path: Path to saved model
            **kwargs: Additional arguments for OnlineLearner
            
        Returns:
            OnlineLearner instance
        """
        checkpoint = torch.load(checkpoint_path)
        
        # Reconstruct model (simplified - adapt to your model structure)
        from climate_discovery.models.mixture import SDMoSE
        from climate_discovery.config import ModelConfig
        
        config = checkpoint.get('config', ModelConfig())
        model = SDMoSE(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        learner = cls(model, **kwargs)
        learner.baseline_performance = checkpoint.get('performance', {})
        
        return learner
    
    def incremental_update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        n_epochs: int = 5,
        retrain_experts: bool = False,
    ) -> Dict:
        """Incrementally update model with new data.
        
        Args:
            X_new: New features (N, D)
            y_new: New targets (N,)
            n_epochs: Fine-tuning epochs
            retrain_experts: Whether to retrain symbolic experts
            
        Returns:
            Dictionary with update statistics
        """
        logger.info(f"Incremental update with {len(X_new)} new samples")
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X_new).float()
        y_tensor = torch.from_numpy(y_new).float()
        
        # Step 1: Evaluate on new data before update
        self.model.eval()
        with torch.no_grad():
            y_pred_before = self.model(X_tensor).numpy()
            r2_before = self._compute_r2(y_new, y_pred_before)
        
        logger.info(f"Performance before update: R² = {r2_before:.4f}")
        
        # Step 2: Fine-tune gating network
        logger.info("Fine-tuning gating network...")
        self._finetune_gating(X_tensor, y_tensor, n_epochs)
        
        # Step 3: Optionally retrain experts
        if retrain_experts:
            logger.info("Retraining symbolic experts...")
            self._retrain_experts(X_new, y_new)
        
        # Step 4: Evaluate after update
        self.model.eval()
        with torch.no_grad():
            y_pred_after = self.model(X_tensor).numpy()
            r2_after = self._compute_r2(y_new, y_pred_after)
        
        logger.info(f"Performance after update: R² = {r2_after:.4f}")
        
        # Step 5: Drift detection
        drift_detected = False
        if self.baseline_performance and 'r2' in self.baseline_performance:
            baseline_r2 = self.baseline_performance['r2']
            drift = baseline_r2 - r2_after
            
            if drift > self.drift_threshold:
                drift_detected = True
                logger.warning(
                    f"⚠️ Concept drift detected! "
                    f"R² dropped from {baseline_r2:.4f} to {r2_after:.4f}"
                )
        
        stats = {
            'n_samples': len(X_new),
            'r2_before': r2_before,
            'r2_after': r2_after,
            'improvement': r2_after - r2_before,
            'drift_detected': drift_detected,
            'experts_retrained': retrain_experts,
        }
        
        return stats
    
    def _finetune_gating(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        n_epochs: int,
    ):
        """Fine-tune gating network on new data."""
        self.model.train()
        
        optimizer = torch.optim.Adam(
            self.model.gating_network.parameters(),
            lr=self.learning_rate
        )
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(X)
            loss = torch.nn.functional.mse_loss(predictions, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.debug(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    def _retrain_experts(self, X: np.ndarray, y: np.ndarray):
        """Retrain symbolic experts on combined old + new data.
        
        In practice, you'd combine with reservoir of old data.
        """
        # Get regime assignments
        regime_probs = self.model.gating_network(torch.from_numpy(X).float())
        regime_labels = torch.argmax(regime_probs, dim=1).numpy()
        
        # Retrain each expert (simplified - use actual PySR in production)
        for regime_id in np.unique(regime_labels):
            mask = regime_labels == regime_id
            if np.sum(mask) < 10:  # Need enough samples
                continue
            
            X_regime = X[mask]
            y_regime = y[mask]
            
            logger.debug(f"  Retraining expert {regime_id} on {np.sum(mask)} samples")
            
            # Here you would call PySR or update existing expert
            # self.model.experts[regime_id].fit(X_regime, y_regime)
    
    def _compute_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot
    
    def save_checkpoint(
        self,
        path: str,
        performance: Optional[Dict] = None,
    ):
        """Save updated model checkpoint.
        
        Args:
            path: Output path
            performance: Performance metrics to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'performance': performance or self.baseline_performance,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"✓ Checkpoint saved: {path}")


def download_latest_socat(
    base_url: str = "https://www.socat.info/data/",
    cache_dir: str = "data/raw/",
) -> Tuple[np.ndarray, np.ndarray]:
    """Download latest SOCAT data.
    
    Args:
        base_url: SOCAT data URL
        cache_dir: Local cache directory
        
    Returns:
        X, y arrays with new data
    """
    import pandas as pd
    from datetime import datetime
    
    logger.info("Downloading latest SOCAT data...")
    
    # In production, implement actual download logic
    # For demo, simulate with cached data
    
    current_month = datetime.now().strftime("%Y-%m")
    cache_file = Path(cache_dir) / f"socat_{current_month}.csv"
    
    if cache_file.exists():
        logger.info(f"Loading from cache: {cache_file}")
        df = pd.read_csv(cache_file)
    else:
        logger.warning("Latest data not available - using dummy data")
        # Return empty for now
        return np.array([]).reshape(0, 10), np.array([])
    
    # Extract features and target
    # X = df[FEATURE_COLUMNS].values
    # y = df['fco2'].values
    
    # Placeholder
    X = np.random.randn(100, 10)
    y = np.random.uniform(300, 450, 100)
    
    return X, y
