#!/usr/bin/env python
"""
Reproduce Figure 2 from Carr et al. 2023
Parameters: eta_M = 0.1, eta_E = 0.1, eta_Z = 0.2
Halo: 10^12 Msun from z=6 to z=0
"""

from gas_regulator import run_single_halo, default_params
from gas_regulator.utils import print_summary

print("="*70)
print("Reproducing Figure 2 from Carr et al. 2023")
print("="*70)

# Set parameters to match Figure 2
params = default_params.copy()
params['eta_M_norm'] = 0.1    # Mass loading (constant)
params['eta_M_beta'] = 0.0    # No mass dependence
params['eta_E_A'] = 0.1       # Energy loading (constant)
params['eta_E_lambda'] = 0.0  # No mass dependence
params['eta_Z'] = 0.2         # Metal loading

print("\nParameters:")
print(f"  eta_M = {params['eta_M_norm']} (constant, beta={params['eta_M_beta']})")
print(f"  eta_E = {params['eta_E_A']} (constant, lambda={params['eta_E_lambda']})")
print(f"  eta_Z = {params['eta_Z']}")
print()

# Run from z=6 to z=0, targeting 1e12 Msun at z=0
result = run_single_halo(
    M_halo_z0=1e12,
    z_start=6.0,
    z_end=0.0,
    params=params,
    rtol=1e-5,
    atol=1e-5
)

# Print summary
print_summary(result)

# Print final values and compare to Figure 2 targets
print("\n" + "="*70)
print("FINAL RESULTS AT z=0:")
print("="*70)
print(f"M_halo = {result['M_halo'][-1]:.2e} Msun  (target: 1.00e+12)")
print(f"M_star = {result['M_star'][-1]:.2e} Msun  (Fig 2: ~1e+10)")
print(f"M_ISM  = {result['M_ISM'][-1]:.2e} Msun  (Fig 2: ~1e+10)")
print(f"M_CGM  = {result['M_CGM'][-1]:.2e} Msun  (Fig 2: ~1e+10)")
print(f"T_CGM  = {result['T_CGM'][-1]:.2e} K")
print(f"Z_CGM  = {result['Z_CGM'][-1]:.3f} Z_sun  (Fig 2: ~0.07)")
print("="*70)
