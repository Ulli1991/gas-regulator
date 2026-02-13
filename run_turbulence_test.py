#!/usr/bin/env python
"""
Test Pandya turbulence model vs Carr thermal-only model.
"""
import numpy as np
from gas_regulator import run_single_halo, default_params

# Shared parameters (same as run_fig2_fixed.py)
base_params = default_params.copy()
base_params['eta_M_norm'] = 0.1
base_params['eta_M_beta'] = 0.0
base_params['eta_E_A'] = 0.1
base_params['eta_E_lambda'] = 0.0
base_params['eta_Z'] = 0.2
base_params['use_wiersma_cooling'] = True

# ---------- Run 1: Carr model (thermal only) ----------
print("="*70)
print("RUN 1: Carr model (thermal only, enable_turbulence=False)")
print("="*70)
params_carr = base_params.copy()
params_carr['enable_turbulence'] = False

result_carr = run_single_halo(
    M_halo_z0=1e12,
    z_start=6.0,
    z_end=0.0,
    params=params_carr,
    rtol=1e-4,
    atol=1e-4,
)

# ---------- Run 2: Pandya model (with turbulence) ----------
print("\n" + "="*70)
print("RUN 2: Pandya model (enable_turbulence=True)")
print("="*70)
params_pandya = base_params.copy()
params_pandya['enable_turbulence'] = True
params_pandya['f_thermal_accretion'] = 0.5
params_pandya['f_thermal_wind'] = 0.5
params_pandya['R_turb_fraction'] = 0.5
params_pandya['c_NFW'] = 10.0

result_pandya = run_single_halo(
    M_halo_z0=1e12,
    z_start=6.0,
    z_end=0.0,
    params=params_pandya,
    rtol=1e-4,
    atol=1e-4,
)

# ---------- Compare results ----------
print("\n" + "="*70)
print("COMPARISON AT z=0")
print("="*70)
print(f"{'Quantity':<20} {'Carr':>15} {'Pandya':>15} {'Fig 2 target':>15}")
print("-"*70)

fields = [
    ("M_halo [Msun]", "M_halo", 1e12),
    ("M_star [Msun]", "M_star", 1e10),
    ("M_ISM [Msun]", "M_ISM", 1e10),
    ("M_CGM [Msun]", "M_CGM", 1e10),
    ("Z_CGM [Z_sun]", "Z_CGM", 0.07),
    ("T_CGM [K]", "T_CGM", None),
]

for label, key, target in fields:
    v_carr = result_carr[key][-1]
    v_pandya = result_pandya[key][-1]
    if target is not None:
        print(f"{label:<20} {v_carr:>15.2e} {v_pandya:>15.2e} {target:>15.2e}")
    else:
        print(f"{label:<20} {v_carr:>15.2e} {v_pandya:>15.2e} {'':>15}")

print(f"{'log E_CGM [erg]':<20} {np.log10(result_carr['E_CGM'][-1]):>15.2f} {np.log10(result_pandya['E_CGM'][-1]):>15.2f} {'57.4':>15}")

if 'E_th' in result_pandya:
    print(f"\n--- Pandya turbulence diagnostics ---")
    print(f"E_th  = 10^{np.log10(result_pandya['E_th'][-1]):.2f} erg")
    print(f"E_kin = 10^{np.log10(max(result_pandya['E_kin'][-1], 1e30)):.2f} erg")
    print(f"E_kin / E_th = {result_pandya['E_kin'][-1] / result_pandya['E_th'][-1]:.3f}")
    print(f"v_turb = {result_pandya['v_turb'][-1]/1e5:.1f} km/s")

print("\n--- Ratios to Fig 2 targets ---")
print(f"M_star (Carr):  {result_carr['M_star'][-1]/1e10:.2f}x target")
print(f"M_star (Pandya): {result_pandya['M_star'][-1]/1e10:.2f}x target")
print(f"M_ISM (Carr):   {result_carr['M_ISM'][-1]/1e10:.2f}x target")
print(f"M_ISM (Pandya):  {result_pandya['M_ISM'][-1]/1e10:.2f}x target")
print("="*70)
