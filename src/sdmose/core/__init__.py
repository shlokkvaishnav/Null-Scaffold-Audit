"""Core utilities for SD-MoSE.

This module consolidates utility functions previously in utils.py.
"""

# Data loading and scaling
from .data_utils import (
    load_scalers,
    inverse_transform,
)

# Feature engineering
from .features import (
    add_cyclic_time_features,
    add_normalized_year,
    add_log_chlorophyll,
    normalize_coordinates,
)

# Tensor preparation
from .tensor_utils import (
    prepare_tensors,
)

# Regime analysis
from .regime_utils import (
    compute_regime_entropy,
    compute_max_probability,
    get_dominant_regime,
    compute_regime_transition_rate,
    compute_ensemble_agreement,
    save_regime_assignments,
    load_regime_assignments,
)

__all__ = [
    # Data utils
    "load_scalers",
    "inverse_transform",
    # Features
    "add_cyclic_time_features",
    "add_normalized_year",
    "add_log_chlorophyll",
    "normalize_coordinates",
    # Tensors
    "prepare_tensors",
    # Regimes
    "compute_regime_entropy",
    "compute_max_probability",
    "get_dominant_regime",
    "compute_regime_transition_rate",
    "compute_ensemble_agreement",
    "save_regime_assignments",
    "load_regime_assignments",
]
