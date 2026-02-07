#!/usr/bin/env python
"""
Debug: Check f_prevent evolution - is preventive feedback working?
"""
import numpy as np
import matplotlib.pyplot as plt
from gas_regulator.model import GasRegulatorModel
from gas_regulator.solver import redshift_to_time, time_to_redshift, extrapolate_halo_mass
from gas_regulator import default_params
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import solve_ivp

params = default_params.copy()
params['eta_M_norm'] = 0.1
params['eta_M_beta'] = 0.0
params['eta_E_A'] = 0.1
params['eta_E_lambda'] = 0.0
params['eta_Z'] = 0.2
params['use_wiersma_cooling'] = True

M_sun = params["M_sun"]
M_halo_z0_cgs = 1e12 * M_sun

# Setup
H0 = params["H0"]
Omega_m = params["Omega_m"]
cosmo = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Omega_m)

t_start = redshift_to_time(6.0, cosmo)
t_end = redshift_to_time(0.0, cosmo)

M_halo_init = extrapolate_halo_mass(M_halo_z0_cgs, 6.0, z0=0, params=params)

model = GasRegulatorModel(params=params, M_halo_z0=M_halo_z0_cgs)
y0 = model.initial_conditions(M_halo_init, 6.0)

# Store f_prevent values
f_prevent_history = []
time_history = []

original_deriv = model.derivatives

def derivatives_with_tracking(t, y, z):
    result = original_deriv(t, y, z)
    # Access f_prevent from model (it's computed in derivatives)
    # We'll need to recompute it here
    M_CGM = y[1]
    E_CGM = y[4]
    if M_CGM > 0 and E_CGM > 0:
        # Compute same as in model
        from gas_regulator import physics
        M_halo = y[0]
        k_B = params["k_B"]
        mu = params["mu"]

        T_vir = physics.virial_temperature(M_halo, z, params)
        r_vir = physics.virial_radius(M_halo, z, params)

        # Energy calculations
        dM_halo_dt = physics.halo_accretion_rate(M_halo, z, params)
        dE_CGM_in = (k_B * T_vir / mu) * params["f_b"] * dM_halo_dt

        # Energy excess
        E_excess = max(E_CGM - (k_B * T_vir / mu) * M_CGM, 0)
        if E_excess > 0:
            c_s = np.sqrt(5 * k_B * (E_CGM/M_CGM) / (3 * mu))
            dE_CGM_out = E_excess * c_s / r_vir
        else:
            dE_CGM_out = 0

        # f_prevent
        alpha_prevent = params["alpha_prevent"]
        if dE_CGM_out > 0:
            f_prev = min(alpha_prevent * dE_CGM_in / dE_CGM_out, 1.0)
        else:
            f_prev = 1.0
    else:
        f_prev = 1.0

    f_prevent_history.append(f_prev)
    time_history.append(t)
    return result

model.derivatives = derivatives_with_tracking

def dydt_wrapper(t, y):
    z = time_to_redshift(t, cosmo, z_range=(0.0, 6.0))
    return model.derivatives(t, y, z)

print("Running integration...")
sol = solve_ivp(
    dydt_wrapper,
    (t_start, t_end),
    y0,
    method='BDF',
    rtol=1e-4,
    atol=1e-4,
    dense_output=True
)

print(f"Done. {len(f_prevent_history)} evaluations")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
t_gyr = np.array(time_history)
ax.plot(t_gyr, f_prevent_history, 'k-', lw=1.5)
ax.axhline(1.0, color='r', ls='--', alpha=0.5, label='f_prevent=1 (no suppression)')
ax.set_xlabel('Time [Gyr]', fontsize=12)
ax.set_ylabel('f_prevent', fontsize=12)
ax.set_title('Preventive Inflow Factor vs Time', fontsize=13)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('f_prevent_evolution.png', dpi=150)
print("Saved: f_prevent_evolution.png")

# Print statistics
f_prev_arr = np.array(f_prevent_history)
print(f"\nf_prevent statistics:")
print(f"  Mean: {f_prev_arr.mean():.3f}")
print(f"  Median: {np.median(f_prev_arr):.3f}")
print(f"  Min: {f_prev_arr.min():.3f}")
print(f"  Max: {f_prev_arr.max():.3f}")
print(f"  Fraction = 1.0 (no suppression): {(f_prev_arr >= 0.99).sum() / len(f_prev_arr) * 100:.1f}%")
print(f"\nIf f_prevent ~ 1 most of the time, IGM baryons flow in freely!")
