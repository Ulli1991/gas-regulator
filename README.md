# Gas-Regulator Model

Implementation of the gas-regulator model from **Carr et al. 2023** for star formation regulation via a hot circumgalactic medium (CGM).

## Overview

This package implements a 1D ODE-based model that tracks mass and energy exchanges between 6 reservoirs:

1. Dark matter halo mass (M_halo)
2. Stellar mass (M_star)
3. ISM mass (M_ISM)
4. CGM mass (M_CGM)
5. CGM energy (E_CGM)
6. CGM metal mass (M_Z,CGM)

The model captures the key physics of galaxy evolution:
- Cosmological halo accretion
- CGM cooling and accretion onto the ISM
- Star formation with depletion time scaling
- Stellar winds with mass, energy, and metal loading
- Preventive feedback from hot CGM outflows
- Metal enrichment of the CGM

## Installation

```bash
cd gas-regulator
pip install -e .
```

Or install dependencies directly:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
from gas_regulator import run_single_halo, default_params
from gas_regulator.utils import plot_evolution, print_summary

# Run a Milky Way-mass halo from z=4 to z=0
# Note: z=3-4 start recommended for numerical stability
result = run_single_halo(
    M_halo_z0=1e12,  # Solar masses at z=0 (target)
    z_start=4.0,
    z_end=0.0,
    params=default_params
)

# Print summary
print_summary(result)

# Plot evolution
plot_evolution(result, save_path='evolution.png')
```

## Running a Halo Suite

```python
from gas_regulator import run_halo_suite
from gas_regulator.utils import plot_scaling_relations
import numpy as np

# Run suite of halos with different masses
M_halo_range = np.logspace(10, 12, 20)  # 10^10 to 10^12 Msun
results = run_halo_suite(
    M_halo_z0_range=M_halo_range,
    z_start=6.0,
    z_end=0.0,
    params=default_params
)

# Plot scaling relations
plot_scaling_relations(results, save_path='scaling_relations.png')
```

## Model Physics

### Halo Accretion
```
dM_halo/dt = 0.47 * M_halo * (M_halo/10^12 Msun)^0.15 * ((1+z)/3)^2.25 [Gyr^-1]
```

### CGM Structure
Power-law density profile: `rho(r) = rho_0 * (r/r_0)^(-1.4)` with `r_0 = 0.1 * r_vir`

### Star Formation
Depletion time scaling: `t_dep ~ M_star^(-0.37) * (1+z)^(-3/2)`

### Feedback
- Mass loading: `eta_M ~ M_halo^(-0.5)`
- Energy loading: `eta_E ~ M_halo^(-0.65)`
- Preventive feedback suppresses inflow when CGM is hot

### Cooling
Simplified cooling function approximating Wiersma et al. tables with temperature and metallicity dependence.

## Package Structure

```
gas-regulator/
├── gas_regulator/
│   ├── __init__.py        # Package interface
│   ├── model.py           # ODE system class
│   ├── physics.py         # Physical calculations
│   ├── parameters.py      # Constants and default parameters
│   ├── solver.py          # Solver interface
│   └── utils.py           # Plotting and analysis utilities
├── setup.py               # Package installation
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Parameters

Default parameters are defined in `gas_regulator.parameters.default_params`. Key parameters include:

- `alpha = 1.4`: CGM density profile slope
- `f_rec = 0.4`: Stellar recycling fraction
- `eta_M_beta = 0.5`: Mass loading mass-dependence
- `eta_E_lambda = 0.65`: Energy loading mass-dependence
- `alpha_prevent = 2.0`: Preventive feedback strength

To modify parameters:
```python
from gas_regulator import default_params

custom_params = default_params.copy()
custom_params['alpha_prevent'] = 3.0  # Stronger preventive feedback

result = run_single_halo(1e12, 6.0, 0.0, params=custom_params)
```

## Expected Results

For a Milky Way-mass halo (10^12 Msun at z=0):
- Final stellar mass: ~10^10 Msun
- Final ISM mass: ~10^10 Msun
- Final CGM mass: ~10^10 Msun
- CGM metallicity: ~0.07 Z_sun
- Stellar-to-halo mass fraction: ~0.01

## Accessing Results

Results are returned as dictionaries with the following keys:

- `time`: Time array in Gyr
- `redshift`: Redshift array
- `M_halo`: Halo mass in Msun
- `M_CGM`: CGM mass in Msun
- `M_ISM`: ISM mass in Msun
- `M_star`: Stellar mass in Msun
- `E_CGM`: CGM energy in erg
- `M_Z_CGM`: CGM metal mass in Msun
- `Z_CGM`: CGM metallicity in Z/Z_sun
- `T_CGM`: CGM temperature in K

## Reference

Carr et al. 2023, "Star formation regulation via the hot circumgalactic medium: the gas-regulator model"

## Numerical Considerations

### Starting Redshift
**Recommended**: Start from z=3-4 for best numerical stability

**Why**: At very high redshift (z>4), the combination of:
- Very small initial stellar masses
- Depletion time scaling t_dep ~ M_star^(-0.37)
- Simplified cooling function

can create numerically stiff equations. The implementation uses:
- BDF (Backward Differentiation Formula) solver for stiff ODEs
- Minimum depletion time floor at 10 Myr
- Initial mass seeds scaled to halo mass

For z>4 start, full Wiersma cooling tables and better-calibrated initial conditions would be ideal.

### Solver Performance
- z=1.5 to z=0: ~1 second
- z=3 to z=0: ~30 seconds
- z=4 to z=0: ~1-2 minutes
- z=6 to z=0: Can be slow/unstable with simplified cooling

### Halo Mass Evolution
The `M_halo_z0` parameter is a *target* mass, but the actual final mass may differ slightly because:
- The Dekel accretion formula has intrinsic scatter
- Backward extrapolation + forward integration aren't perfectly self-consistent
- This is expected for such analytical approximations

## Notes

This implementation uses:
- Simplified cooling function (not full Wiersma tables)
- Flat Lambda-CDM cosmology with Planck-like parameters
- BDF adaptive ODE solver from scipy (for stiff equations)
- Astropy for cosmological time calculations

For quantitative comparison with Carr et al., the full Wiersma cooling tables would be needed, but this implementation reproduces the qualitative behavior and scaling relations.
