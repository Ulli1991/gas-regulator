"""
Gas-Regulator Model for Star Formation Regulation

Implementation of the gas-regulator model from Carr et al. 2023.
Tracks mass and energy exchanges between 6 reservoirs via 1D ODE system.
"""

from .model import GasRegulatorModel
from .solver import run_single_halo, run_halo_suite
from .parameters import default_params

__version__ = "0.1.0"
__all__ = ["GasRegulatorModel", "run_single_halo", "run_halo_suite", "default_params"]
