"""SD-MoSE: Soft-Dynamic Mixture of Symbolic Experts for Climate Discovery."""

__version__ = "0.1.0"
__author__ = "Shlok Vaishnav"

# Import key classes for convenient access
# Only import what exists and is commonly used
try:
    from .config import ModelConfig
except ImportError:
    pass

try:
    from .models.gating import GatingNetwork
    from .models.mixture import SDMoSE
    from .models.symbolic import MixtureOfSymbolicExperts, SymbolicExpert
except ImportError:
    pass

try:
    from .data.datasets import ClimateDataset
except ImportError:
    pass
