#!/usr/bin/env python
"""
Reproduce Figure 2 using the correct initial mass from CURRENT_STATUS.md
M_halo(z=6) = 4.0e10 Msun → M_halo(z=0) ≈ 1e12 Msun
"""
import numpy as np
from gas_regulator import default_params
from gas_regulator.model import GasRegulatorModel
from gas_regulator.solver import redshift_to_time, time_to_redshift
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import solve_ivp

# Figure 2 parameters
params = default_params.copy()
params['eta_M_norm'] = 0.1
params['eta_M_beta'] = 0.0
params['eta_E_A'] = 0.1
params['eta_E_lambda'] = 0.0
params['eta_Z'] = 0.2

print("="*70)
print("Figure 2 Reproduction (Carr et al. 2023)")
print("="*70)
print(f"Parameters: eta_M={params['eta_M_norm']}, eta_E={params['eta_E_A']}, eta_Z={params['eta_Z']}")
print("\nUsing initial mass from CURRENT_STATUS.md:")
print("  M_halo(z=6) = 4.0e10 Msun")
print("="*70)

# Setup
M_sun = params["M_sun"]
M_halo_init = 4.0e10 * M_sun  # From CURRENT_STATUS.md
z_start = 6.0
z_end = 0.0

# Cosmology
H0 = params["H0"]
Omega_m = params["Omega_m"]
Omega_Lambda = params["Omega_Lambda"]
cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Omega_m)

t_start = redshift_to_time(z_start, cosmo)
t_end = redshift_to_time(z_end, cosmo)

# Initialize
model = GasRegulatorModel(params=params, M_halo_z0=1e12*M_sun)  # Dummy value
y0 = model.initial_conditions(M_halo_init, z_start)

def dydt_wrapper(t, y):
    z = time_to_redshift(t, cosmo, z_range=(z_end, z_start))
    return model.derivatives(t, y, z)

print("\nSolving from z=6 to z=0...")
sol = solve_ivp(
    dydt_wrapper,
    (t_start, t_end),
    y0,
    method='BDF',
    rtol=1e-5,
    atol=1e-5,
    dense_output=True,
    max_step=1e14
)

if sol.success:
    print(f"✓ Integration successful: {len(sol.t)} steps\n")

    M_halo_final = sol.y[0, -1] / M_sun
    M_CGM_final = sol.y[1, -1] / M_sun
    M_ISM_final = sol.y[2, -1] / M_sun
    M_star_final = sol.y[3, -1] / M_sun
    E_CGM_final = sol.y[4, -1]
    M_Z_CGM_final = sol.y[5, -1] / M_sun
    Z_CGM_final = M_Z_CGM_final / (M_CGM_final + 1e-30)
    T_CGM_final = (params["mu"] / params["k_B"]) * (E_CGM_final / sol.y[1, -1])

    print("="*70)
    print("FINAL RESULTS (z=0):")
    print("="*70)
    print(f"M_halo = {M_halo_final:.2e} Msun  (target: 1.00e+12, error: {abs(M_halo_final-1e12)/1e12*100:.1f}%)")
    print(f"M_star = {M_star_final:.2e} Msun  (Fig 2 target: ~1e+10)")
    print(f"M_ISM  = {M_ISM_final:.2e} Msun  (Fig 2 target: ~1e+10)")
    print(f"M_CGM  = {M_CGM_final:.2e} Msun  (Fig 2 target: ~1e+10)")
    print(f"T_CGM  = {T_CGM_final:.2e} K")
    print(f"Z_CGM  = {Z_CGM_final/params['Z_solar']:.3f} Z_sun  (Fig 2: ~0.07)")
    print("="*70)

    # Compare to CURRENT_STATUS.md results
    print("\nComparison to CURRENT_STATUS.md:")
    M_star_err = abs(M_star_final - 5.76e10) / 5.76e10 * 100
    M_ISM_err = abs(M_ISM_final - 4.64e10) / 4.64e10 * 100
    M_CGM_err = abs(M_CGM_final - 1.74e9) / 1.74e9 * 100
    print(f"M_star error vs. previous: {M_star_err:.1f}%")
    print(f"M_ISM error vs. previous: {M_ISM_err:.1f}%")
    print(f"M_CGM error vs. previous: {M_CGM_err:.1f}%")
else:
    print(f"✗ Integration failed: {sol.message}")
