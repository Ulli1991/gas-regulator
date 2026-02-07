#!/usr/bin/env python
"""
Reproduce Figure 2 from Carr et al. 2023 with FIXED extrapolation
"""
from gas_regulator import run_single_halo, default_params

# Figure 2 parameters
params = default_params.copy()
params['eta_M_norm'] = 0.1
params['eta_M_beta'] = 0.0
params['eta_E_A'] = 0.1
params['eta_E_lambda'] = 0.0
params['eta_Z'] = 0.2
params['use_wiersma_cooling'] = True  # Use full cooling tables

print("="*70)
print("Figure 2 Reproduction (Carr et al. 2023)")
print("="*70)
print(f"Parameters: eta_M={params['eta_M_norm']}, eta_E={params['eta_E_A']}, eta_Z={params['eta_Z']}")
print("="*70)

# Run with fixed extrapolation
result = run_single_halo(
    M_halo_z0=1e12,
    z_start=6.0,
    z_end=0.0,
    params=params,
    rtol=1e-4,
    atol=1e-4
)

print("\n" + "="*70)
print("FINAL RESULTS (z=0):")
print("="*70)
M_halo_final = result['M_halo'][-1]
M_star_final = result['M_star'][-1]
M_ISM_final = result['M_ISM'][-1]
M_CGM_final = result['M_CGM'][-1]
E_CGM_final = result['E_CGM'][-1]
Z_CGM_final = result['Z_CGM'][-1]
T_CGM_final = result['T_CGM'][-1]

import numpy as np
log10_E_CGM = np.log10(E_CGM_final)

print(f"M_halo = {M_halo_final:.2e} Msun  (target: 1.00e+12)")
print(f"M_star = {M_star_final:.2e} Msun  (Fig 2 target: ~1e+10)")
print(f"M_ISM  = {M_ISM_final:.2e} Msun  (Fig 2 target: ~1e+10)")
print(f"M_CGM  = {M_CGM_final:.2e} Msun  (Fig 2 target: ~1e+10)")
print(f"E_CGM  = 10^{log10_E_CGM:.2f} erg  (Fig 2: 10^57.4 erg)")
print(f"T_CGM  = {T_CGM_final:.2e} K")
print(f"Z_CGM  = {Z_CGM_final:.3f} Z_sun  (Fig 2: ~0.07)")
print("="*70)

# Errors vs Figure 2 targets
print("\nComparison to Figure 2 targets (~1e10 for M_star, M_ISM, M_CGM):")
print(f"M_halo error: {abs(M_halo_final-1e12)/1e12*100:.1f}%")
print(f"M_star / target: {M_star_final/1e10:.1f}x")
print(f"M_ISM / target:  {M_ISM_final/1e10:.1f}x")
print(f"M_CGM / target:  {M_CGM_final/1e10:.1f}x")
print("="*70)
