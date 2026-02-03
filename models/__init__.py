"""SD-MoSE Models Package.

Contains the core components for Soft-Dynamic Mixture of Symbolic Experts:
- gating: Neural network for soft regime assignment
- symbolic: PySR-based symbolic regression per regime
- mixture: Combined SD-MoSE model
"""

from models.gating import GatingNetwork, KMeansGating
from models.symbolic import SymbolicExpert, MixtureOfSymbolicExperts
from models.mixture import SDMoSE

__all__ = [
    "GatingNetwork",
    "KMeansGating", 
    "SymbolicExpert",
    "MixtureOfSymbolicExperts",
    "SDMoSE",
]
