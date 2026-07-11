"""Public API for MACCO-General."""

from .optimizer import MACCOResult, minimize
from .low_rank import minimize_low_rank
from .hybrid import minimize_hybrid
from .delayed import minimize_delayed_hybrid

# Descriptive public name; the old name remains available for reproducibility.
minimize_subspace = minimize_low_rank

__all__ = ["MACCOResult", "minimize", "minimize_subspace", "minimize_low_rank",
           "minimize_hybrid", "minimize_delayed_hybrid"]
__version__ = "0.3.0"
