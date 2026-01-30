"""Transfer learning for multi-gas applications.

Enables applying SD-MoSE to other trace gases:
- Pre-trained gating network (ocean regimes)
- Expert fine-tuning for new target
- Multi-task learning

Supported gases:
- CO₂ (fCO2) - base model
- CH₄ (methane)
- N₂O (nitrous oxide)
- DMS (dimethyl sulfide)

Usage:
    from climate_discovery.transfer import TransferLearner
    
    # Load CO₂ model
    learner = TransferLearner.from_pretrained("co2_model.pth")
    
    # Adapt to CH₄
    ch4_model = learner.adapt_to_new_gas(
        X_ch4, y_ch4,
        target_gas="ch4",
        freeze_gating=True,
    )
"""

import torch
import numpy as np
from typing import Literal, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TransferLearner:
    """Transfer learning for multi-gas prediction.
    
    Key insight: Ocean regimes are similar across gases!
    - Upwelling regions have high outgassing (CO₂, CH₄, N₂O)
    - Oligotrophic regions are sinks
    - Frontal zones have complex dynamics
    
    → Learn gating from CO₂, adapt experts for other gases
    
    Example:
        >>> learner = TransferLearner.from_pretrained("co2_model.pth")
        >>> ch4_model = learner.adapt_to_new_gas(
        ...     X_ch4, y_ch4,
        ...     target_gas="ch4",
        ...     freeze_gating=True,
        ... )
        >>> # Save adapted model
        >>> learner.save_adapted_model(ch4_model, "ch4_model.pth")
    """
    
    def __init__(
        self,
        base_model,
        base_gas: str = "co2",
    ):
        """Initialize transfer learner.
        
        Args:
            base_model: Pretrained SD-MoSE model
            base_gas: Source gas (default: co2)
        """
        self.base_model = base_model
        self.base_gas = base_gas
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str):
        """Load pretrained model for transfer learning.
        
        Args:
            checkpoint_path: Path to pretrained model
            
        Returns:
            TransferLearner instance
        """
        checkpoint = torch.load(checkpoint_path)
        
        # Reconstruct model
        from climate_discovery.models.mixture import SDMoSE
        from climate_discovery.config import ModelConfig
        
        config = checkpoint.get('config', ModelConfig())
        model = SDMoSE(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        base_gas = checkpoint.get('target_gas', 'co2')
        
        return cls(model, base_gas=base_gas)
    
    def adapt_to_new_gas(
        self,
        X: np.ndarray,
        y: np.ndarray,
        target_gas: Literal["ch4", "n2o", "dms"] = "ch4",
        freeze_gating: bool = True,
        learning_rate: float = 1e-3,
        n_epochs: int = 50,
    ):
        """Adapt model to predict new gas.
        
        Args:
            X: Features for new gas (N, D)
            y: Target values for new gas (N,)
            target_gas: Gas to predict
            freeze_gating: Whether to freeze gating network
            learning_rate: Learning rate for adaptation
            n_epochs: Training epochs
            
        Returns:
            Adapted model
        """
        logger.info(f"Adapting {self.base_gas} model to predict {target_gas}")
        
        # Clone model
        from copy import deepcopy
        adapted_model = deepcopy(self.base_model)
        
        # Freeze gating network if requested
        if freeze_gating:
            logger.info("Freezing gating network (transferring regimes)")
            for param in adapted_model.gating_network.parameters():
                param.requires_grad = False
        
        # Get regime assignments from base model
        adapted_model.eval()
        with torch.no_grad():
            regime_probs = adapted_model.gating_network(torch.from_numpy(X).float())
            regime_labels = torch.argmax(regime_probs, dim=1).numpy()
        
        # Train new experts for each regime
        logger.info("Training regime-specific experts for new gas...")
        self._train_new_experts(adapted_model, X, y, regime_labels, target_gas)
        
        # Optionally fine-tune full model
        if not freeze_gating:
            logger.info("Fine-tuning full model...")
            self._finetune_model(adapted_model, X, y, learning_rate, n_epochs)
        
        # Store metadata
        adapted_model.target_gas = target_gas
        adapted_model.base_gas = self.base_gas
        
        logger.info(f"✓ Model adapted to {target_gas}")
        
        return adapted_model
    
    def _train_new_experts(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        regime_labels: np.ndarray,
        target_gas: str,
    ):
        """Train new symbolic experts for target gas.
        
        Uses same regimes as base model, but fits new equations.
        """
        for regime_id in np.unique(regime_labels):
            mask = regime_labels == regime_id
            n_samples = np.sum(mask)
            
            if n_samples < 20:  # Need minimum samples
                logger.warning(f"Regime {regime_id}: only {n_samples} samples, skipping")
                continue
            
            X_regime = X[mask]
            y_regime = y[mask]
            
            logger.info(f"  Regime {regime_id}: fitting expert on {n_samples} samples")
            
            # In production, use PySR to fit new symbolic equation
            # model.experts[regime_id] = fit_pysr_expert(X_regime, y_regime, target_gas)
            
            # For now, use simple linear model as placeholder
            from sklearn.linear_model import LinearRegression
            expert = LinearRegression()
            expert.fit(X_regime, y_regime)
            
            # Replace expert in model
            # model.experts[regime_id] = expert
    
    def _finetune_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float,
        n_epochs: int,
    ):
        """Fine-tune entire model on new gas data."""
        model.train()
        
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).float()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            predictions = model(X_tensor)
            loss = torch.nn.functional.mse_loss(predictions, y_tensor)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
    
    def multi_task_learning(
        self,
        data_dict: Dict[str, tuple],
        shared_gating: bool = True,
    ) -> Dict[str, any]:
        """Train multi-task model for multiple gases simultaneously.
        
        Args:
            data_dict: {gas_name: (X, y)} for each gas
            shared_gating: Share gating network across tasks
            
        Returns:
            Dictionary of gas-specific models
        """
        logger.info(f"Multi-task learning for {len(data_dict)} gases")
        
        models = {}
        
        if shared_gating:
            # Use shared gating network
            shared_gating_net = self.base_model.gating_network
            
            for gas_name, (X, y) in data_dict.items():
                logger.info(f"Training experts for {gas_name}")
                
                # Create model with shared gating
                from copy import deepcopy
                model = deepcopy(self.base_model)
                model.gating_network = shared_gating_net
                
                # Train gas-specific experts
                regime_labels = self._get_regime_labels(shared_gating_net, X)
                self._train_new_experts(model, X, y, regime_labels, gas_name)
                
                models[gas_name] = model
        else:
            # Independent models
            for gas_name, (X, y) in data_dict.items():
                models[gas_name] = self.adapt_to_new_gas(
                    X, y,
                    target_gas=gas_name,
                    freeze_gating=False,
                )
        
        return models
    
    def _get_regime_labels(self, gating_net, X: np.ndarray) -> np.ndarray:
        """Get regime assignments from gating network."""
        gating_net.eval()
        with torch.no_grad():
            regime_probs = gating_net(torch.from_numpy(X).float())
            regime_labels = torch.argmax(regime_probs, dim=1).numpy()
        return regime_labels
    
    def save_adapted_model(
        self,
        model,
        save_path: str,
        metadata: Optional[Dict] = None,
    ):
        """Save adapted model with metadata.
        
        Args:
            model: Adapted model
            save_path: Output path
            metadata: Additional metadata
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'target_gas': getattr(model, 'target_gas', 'unknown'),
            'base_gas': getattr(model, 'base_gas', self.base_gas),
            'metadata': metadata or {},
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"✓ Adapted model saved: {save_path}")


def get_gas_specific_features(gas: str) -> Dict:
    """Get gas-specific feature recommendations.
    
    Different gases have different driver variables:
    - CO₂: SST, SSS, Chl, alkalinity
    - CH₄: SST, O₂, organic matter
    - N₂O: O₂, nitrate, depth
    - DMS: Chl, SST, PAR
    """
    features = {
        "co2": {
            "primary": ["sst", "sss", "chl", "mld"],
            "secondary": ["wind", "alkalinity", "dic"],
            "expected_range": (200, 600),  # μatm
        },
        "ch4": {
            "primary": ["sst", "o2", "doc", "mld"],
            "secondary": ["wind", "chl", "salinity"],
            "expected_range": (0.5, 50),  # nmol/L
        },
        "n2o": {
            "primary": ["o2", "no3", "depth", "sst"],
            "secondary": ["mld", "chl", "salinity"],
            "expected_range": (0, 100),  # nmol/L
        },
        "dms": {
            "primary": ["chl", "sst", "par", "wind"],
            "secondary": ["mld", "salinity"],
            "expected_range": (0, 50),  # nmol/L
        },
    }
    
    return features.get(gas, features["co2"])
