#!/usr/bin/env python
"""
Plot time evolution to compare with Carr Fig 2
"""
import numpy as np
import matplotlib.pyplot as plt
from gas_regulator import run_single_halo, default_params

# Figure 2 parameters
params = default_params.copy()
params['eta_M_norm'] = 0.1
params['eta_M_beta'] = 0.0
params['eta_E_A'] = 0.1
params['eta_E_lambda'] = 0.0
params['eta_Z'] = 0.2
params['use_wiersma_cooling'] = True

print("Running model...")
result = run_single_halo(
    M_halo_z0=1e12,
    z_start=6.0,
    z_end=0.0,
    params=params,
    rtol=1e-4,
    atol=1e-4
)

# Create Figure 2 style plot
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

z = result['redshift']
t = result['time']  # Already in Gyr

# Panel 1: Masses
ax = axes[0]
ax.semilogy(t, result['M_halo'], 'k-', lw=2, label='M_halo')
ax.semilogy(t, result['M_star'], 'r-', lw=2, label='M_star')
ax.semilogy(t, result['M_ISM'], 'b-', lw=2, label='M_ISM')
ax.semilogy(t, result['M_CGM'], 'g-', lw=2, label='M_CGM')
ax.axhline(1e10, color='gray', ls='--', alpha=0.5, label='~1e10 target')
ax.axvspan(t[-1]-4, t[-1], alpha=0.1, color='yellow', label='Should be flat')
ax.set_xlabel('Time [Gyr]', fontsize=12)
ax.set_ylabel('Mass [Msun]', fontsize=12)
ax.set_title('Panel 1: Mass Evolution', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: CGM Energy
ax = axes[1]
log_E_CGM = np.log10(result['E_CGM'])
ax.plot(t, log_E_CGM, 'k-', lw=2)
ax.axhline(57.4, color='r', ls='--', lw=2, alpha=0.7, label='Paper: 10^57.4')
ax.axvspan(t[-1]-4, t[-1], alpha=0.1, color='yellow', label='Should be flat')
ax.set_xlabel('Time [Gyr]', fontsize=12)
ax.set_ylabel('log10(E_CGM) [erg]', fontsize=12)
ax.set_title('Panel 2: CGM Energy', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([56, 61])

# Panel 3: CGM Metallicity
ax = axes[2]
ax.plot(t, result['Z_CGM'], 'k-', lw=2, label='Our model')
ax.axhline(0.07, color='r', ls='--', lw=2, alpha=0.7, label='Paper z=0: ~0.07')
ax.axhline(0.08, color='orange', ls='--', lw=1, alpha=0.7, label='Paper peak: ~0.08')
ax.axhline(0.01, color='gray', ls='--', lw=1, alpha=0.7, label='Start: 0.01')
ax.axvspan(t[-1]-4, t[-1], alpha=0.1, color='yellow', label='Should be flat')
ax.set_xlabel('Time [Gyr]', fontsize=12)
ax.set_ylabel('Z_CGM [Z_sun]', fontsize=12)
ax.set_title('Panel 3: CGM Metallicity', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 0.6])

plt.tight_layout()
plt.savefig('fig2_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: fig2_comparison.png")
print(f"\nFinal values (z=0):")
print(f"  M_star = {result['M_star'][-1]:.2e} (target ~1e10)")
print(f"  M_ISM  = {result['M_ISM'][-1]:.2e} (target ~1e10)")
print(f"  M_CGM  = {result['M_CGM'][-1]:.2e} (target ~1e10)")
print(f"  log10(E_CGM) = {np.log10(result['E_CGM'][-1]):.2f} (target 57.4)")
print(f"  Z_CGM = {result['Z_CGM'][-1]:.3f} (target 0.07)")

# Check late-time behavior (last 4 Gyr)
t_late = t[-1] - 4.0
mask = t >= t_late
print(f"\nLate-time behavior (last 4 Gyr, should be flat):")
print(f"  M_star variation: {result['M_star'][mask].min():.2e} to {result['M_star'][mask].max():.2e}")
print(f"  M_ISM variation:  {result['M_ISM'][mask].min():.2e} to {result['M_ISM'][mask].max():.2e}")
print(f"  M_CGM variation:  {result['M_CGM'][mask].min():.2e} to {result['M_CGM'][mask].max():.2e}")
print(f"  Z_CGM variation:  {result['Z_CGM'][mask].min():.3f} to {result['Z_CGM'][mask].max():.3f}")
print(f"  ISM is always too high - even at early times")
