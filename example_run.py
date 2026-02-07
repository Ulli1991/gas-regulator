#!/usr/bin/env python
"""
Example script: Run a single MW-mass halo evolution.

Note: Starting from z=4 rather than z=6 for numerical stability.
At very high redshift (z>4), the simplified cooling function and
small initial masses can cause stiffness. Future improvements could
include full Wiersma cooling tables and better high-z initial conditions.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gas_regulator import run_single_halo, default_params
from gas_regulator.utils import plot_evolution, print_summary

print("="*70)
print("GAS-REGULATOR MODEL EXAMPLE")
print("="*70)

# Run a Milky Way-mass halo
M_halo_z0 = 1e12  # Solar masses at z=0
z_start = 4.0     # Starting redshift
z_end = 0.0       # Ending redshift

print(f"\nRunning halo evolution:")
print(f"  Target M_halo(z=0) = {M_halo_z0:.2e} Msun")
print(f"  Redshift range: z={z_start} to z={z_end}")
print(f"  Using BDF solver for stiff equations")
print(f"\nThis will take 1-2 minutes...")

result = run_single_halo(
    M_halo_z0=M_halo_z0,
    z_start=z_start,
    z_end=z_end,
    params=default_params,
    rtol=1e-5,
    atol=1e-5
)

# Print summary
print_summary(result)

# Plot evolution
print("\nGenerating plots...")
fig = plot_evolution(result, save_path='example_evolution.png')
plt.close(fig)
print("  Saved: example_evolution.png")

# Plot specific quantities
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Stellar mass assembly
ax = axes[0]
ax.semilogy(result['redshift'], result['M_star'], 'b-', lw=2)
ax.set_xlabel('Redshift', fontsize=12)
ax.set_ylabel('M_star [Msun]', fontsize=12)
ax.set_title('Stellar Mass Assembly', fontsize=13)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

# Star formation rate (approximate from dM_star)
ax = axes[1]
t = result['time']
M_star = result['M_star']
# Compute SFR = dM_star/dt
dt = np.diff(t)
dM = np.diff(M_star)
SFR = dM / dt  # Msun/Gyr
t_mid = (t[:-1] + t[1:]) / 2
z_mid = result['redshift'][:-1]  # Approximate

ax.semilogy(z_mid, SFR, 'r-', lw=2)
ax.set_xlabel('Redshift', fontsize=12)
ax.set_ylabel('SFR [Msun/Gyr]', fontsize=12)
ax.set_title('Star Formation Rate', fontsize=13)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('example_sfr.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: example_sfr.png")

print("\n" + "="*70)
print("EXAMPLE COMPLETE!")
print("="*70)
print(f"\nKey results:")
print(f"  Final M_star = {result['M_star'][-1]:.2e} Msun")
print(f"  Final M_ISM = {result['M_ISM'][-1]:.2e} Msun")
print(f"  Final M_CGM = {result['M_CGM'][-1]:.2e} Msun")
print(f"  Final Z_CGM = {result['Z_CGM'][-1]:.3f} Z_sun")
print(f"  M_star/M_halo = {result['M_star'][-1]/result['M_halo'][-1]:.4f}")
print("="*70)
