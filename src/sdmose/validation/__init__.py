"""Validation package for SD-MoSE.

Includes spatial cross-validation, residual analysis, uncertainty estimation,
and evaluation metrics.
"""

from .spatial_cv import SpatialCrossValidator, SpatialFold
from .residual_analysis import ResidualAnalyzer
from .uncertainty import UncertaintyEstimator
from .metrics import (
    compute_r2_rmse,
    ood_slices,
    plausibility_metrics,
    validate_fco2_range,
    check_feature_scales,
)

__all__ = [
    "SpatialCrossValidator",
    "SpatialFold",
    "ResidualAnalyzer",
    "UncertaintyEstimator",
    "compute_r2_rmse",
    "ood_slices",
    "plausibility_metrics",
    "validate_fco2_range",
    "check_feature_scales",
]
