#!/usr/bin/env python
"""
Debug: trace early evolution to see why initial conditions don't matter
"""
import numpy as np
from gas_regulator import run_single_halo, default_params

params = default_params.copy()
params['eta_M_norm'] = 0.1
params['eta_M_beta'] = 0.0
params['eta_E_A'] = 0.1
params['eta_E_lambda'] = 0.0
params['eta_Z'] = 0.2
params['use_wiersma_cooling'] = True

print("Running z=6 to z=0...")
result = run_single_halo(
    M_halo_z0=1e12,
    z_start=6.0,
    z_end=0.0,
    params=params,
    rtol=1e-4,
    atol=1e-4
)

# Look at early evolution (first 0.5 Gyr)
t = result['time']
mask_early = (t - t[0]) <= 0.5

print("\n" + "="*70)
print("EARLY EVOLUTION (first 0.5 Gyr from z=6):")
print("="*70)
print(f"Initial (z={result['redshift'][0]:.2f}):")
print(f"  M_star = {result['M_star'][0]:.2e} Msun")
print(f"  M_ISM  = {result['M_ISM'][0]:.2e} Msun")
print(f"  M_CGM  = {result['M_CGM'][0]:.2e} Msun")
print(f"  Z_CGM  = {result['Z_CGM'][0]:.3f} Z_sun")

# After 0.1 Gyr
idx = np.argmin(np.abs((t - t[0]) - 0.1))
print(f"\nAfter 0.1 Gyr (z={result['redshift'][idx]:.2f}):")
print(f"  M_star = {result['M_star'][idx]:.2e} Msun  (grew {result['M_star'][idx]/result['M_star'][0]:.1f}x)")
print(f"  M_ISM  = {result['M_ISM'][idx]:.2e} Msun  (grew {result['M_ISM'][idx]/result['M_ISM'][0]:.1f}x)")
print(f"  M_CGM  = {result['M_CGM'][idx]:.2e} Msun  (grew {result['M_CGM'][idx]/result['M_CGM'][0]:.1f}x)")
print(f"  Z_CGM  = {result['Z_CGM'][idx]:.3f} Z_sun")

# After 0.5 Gyr
idx = np.argmin(np.abs((t - t[0]) - 0.5))
print(f"\nAfter 0.5 Gyr (z={result['redshift'][idx]:.2f}):")
print(f"  M_star = {result['M_star'][idx]:.2e} Msun  (grew {result['M_star'][idx]/result['M_star'][0]:.1f}x)")
print(f"  M_ISM  = {result['M_ISM'][idx]:.2e} Msun  (grew {result['M_ISM'][idx]/result['M_ISM'][0]:.1f}x)")
print(f"  M_CGM  = {result['M_CGM'][idx]:.2e} Msun  (grew {result['M_CGM'][idx]/result['M_CGM'][0]:.1f}x)")
print(f"  Z_CGM  = {result['Z_CGM'][idx]:.3f} Z_sun")

print("\n" + "="*70)
print("KEY QUESTION: Why does everything grow so fast?")
print("If masses grow 100x in 0.5 Gyr, initial conditions don't matter!")
print("="*70)
