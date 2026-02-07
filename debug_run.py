#!/usr/bin/env python
"""
Debug run to see what's happening during integration
"""
import time
import numpy as np
from gas_regulator import default_params
from gas_regulator.model import GasRegulatorModel
from gas_regulator.solver import extrapolate_halo_mass, redshift_to_time, time_to_redshift
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import solve_ivp

# Setup
M_halo_z0 = 1e12  # Msun
z_start = 6.0
z_end = 0.0

params = default_params.copy()
params['eta_M_norm'] = 0.1
params['eta_M_beta'] = 0.0
params['eta_E_A'] = 0.1
params['eta_E_lambda'] = 0.0
params['eta_Z'] = 0.2

print("="*70)
print("DEBUG RUN: Figure 2 parameters")
print(f"eta_M = {params['eta_M_norm']}, eta_E = {params['eta_E_A']}, eta_Z = {params['eta_Z']}")
print("="*70)

# Setup cosmology
H0 = params["H0"]
Omega_m = params["Omega_m"]
Omega_Lambda = params["Omega_Lambda"]
cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Omega_m)

# Time range
t_start = redshift_to_time(z_start, cosmo)
t_end = redshift_to_time(z_end, cosmo)
print(f"\nTime range: {t_start:.3e} to {t_end:.3e} Gyr")

# Initial conditions
M_sun = params["M_sun"]
M_halo_z0_cgs = M_halo_z0 * M_sun
M_halo_init = extrapolate_halo_mass(M_halo_z0_cgs, z_start, z0=0, params=params)
print(f"Initial halo mass at z={z_start}: {M_halo_init/M_sun:.2e} Msun")

model = GasRegulatorModel(params=params, M_halo_z0=M_halo_z0_cgs)
y0 = model.initial_conditions(M_halo_init, z_start)

print(f"\nInitial conditions:")
print(f"  M_halo = {y0[0]/M_sun:.2e} Msun")
print(f"  M_CGM  = {y0[1]/M_sun:.2e} Msun")
print(f"  M_ISM  = {y0[2]/M_sun:.2e} Msun")
print(f"  M_star = {y0[3]/M_sun:.2e} Msun")
print(f"  E_CGM  = {y0[4]:.2e} erg")

# Add counter for debugging
call_count = [0]
last_print = [time.time()]

def dydt_wrapper(t, y):
    call_count[0] += 1
    z = time_to_redshift(t, cosmo, z_range=(z_end, z_start))

    # Print every 100 calls or every 2 seconds
    if call_count[0] % 100 == 0 or (time.time() - last_print[0]) > 2:
        print(f"  Step {call_count[0]}: z={z:.3f}, M_halo={y[0]/M_sun:.2e}, M_star={y[3]/M_sun:.2e}")
        last_print[0] = time.time()

    return model.derivatives(t, y, z)

print("\nStarting integration...")
start_time = time.time()

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

elapsed = time.time() - start_time

if sol.success:
    print(f"\n✓ Integration successful in {elapsed:.1f} seconds")
    print(f"  Total derivative calls: {call_count[0]}")
    print(f"  Time steps: {len(sol.t)}")

    M_halo_final = sol.y[0, -1] / M_sun
    M_star_final = sol.y[3, -1] / M_sun
    M_ISM_final = sol.y[2, -1] / M_sun
    M_CGM_final = sol.y[1, -1] / M_sun

    print(f"\nFinal results (z=0):")
    print(f"  M_halo = {M_halo_final:.2e} Msun")
    print(f"  M_star = {M_star_final:.2e} Msun")
    print(f"  M_ISM  = {M_ISM_final:.2e} Msun")
    print(f"  M_CGM  = {M_CGM_final:.2e} Msun")
else:
    print(f"\n✗ Integration failed: {sol.message}")
