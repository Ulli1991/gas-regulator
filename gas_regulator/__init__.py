"""
Gas-Regulator Model for Star Formation Regulation (Carr et al. 2023)
plus the Pandya et al. 2023 turbulence extension and a cosmic-ray extension.

The forward model is implemented in JAX/diffrax (``jax_regulator``): it is fully
differentiable and reproduces the original scipy implementation. ``run_single_halo``
integrates a single halo from ``z_start`` to ``z_end`` and returns a result dict.
"""
from .parameters import default_params
from .jax_regulator import run_single_halo
from . import behroozi19

__version__ = "0.2.0"
__all__ = ["run_single_halo", "default_params", "behroozi19"]
