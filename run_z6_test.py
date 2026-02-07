#!/usr/bin/env python
"""
Run model from z=6 to z=0 for a 1e12 Msun halo
"""

from gas_regulator import run_single_halo, default_params
from gas_regulator.utils import print_summary

print("="*70)
print("Running gas-regulator model: z=6 to z=0")
print("="*70)

# Run from z=6 to z=0, targeting 1e12 Msun at z=0
result = run_single_halo(
    M_halo_z0=1e12,
    z_start=6.0,
    z_end=0.0,
    params=default_params,
    rtol=1e-6,
    atol=1e-8
)

# Print summary
print_summary(result)

# Print final values
print("\n" + "="*70)
print("FINAL RESULTS AT z=0:")
print("="*70)
print(f"M_halo = {result['M_halo'][-1]:.2e} Msun  (target: 1.00e+12)")
print(f"M_star = {result['M_star'][-1]:.2e} Msun  (paper: ~1e+10)")
print(f"M_ISM  = {result['M_ISM'][-1]:.2e} Msun  (paper: ~1e+10)")
print(f"M_CGM  = {result['M_CGM'][-1]:.2e} Msun  (paper: ~1e+10)")
print(f"T_CGM  = {result['T_CGM'][-1]:.2e} K")
print(f"Z_CGM  = {result['Z_CGM'][-1]:.3f} Z_sun")
print("="*70)
