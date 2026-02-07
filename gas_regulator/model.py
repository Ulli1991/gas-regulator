"""
Main ODE system for the gas-regulator model.

Implements the 6-component state vector:
[M_halo, M_CGM, M_ISM, M_star, E_CGM, M_Z_CGM]
"""

import numpy as np
from . import physics
from .parameters import default_params


class GasRegulatorModel:
    """
    Gas-regulator model ODE system.

    Tracks evolution of halo mass, CGM mass/energy/metallicity,
    ISM mass, and stellar mass through cosmic time.
    """

    def __init__(self, params=None, M_halo_z0=None):
        """
        Initialize the model.

        Args:
            params: Parameter dictionary (optional, uses defaults if None)
            M_halo_z0: Halo mass at z=0 in g (for loading factors)
        """
        if params is None:
            self.params = default_params.copy()
        else:
            self.params = params.copy()

        self.M_halo_z0 = M_halo_z0

        # Precompute loading factors if M_halo_z0 is provided
        if M_halo_z0 is not None:
            self.eta_M_val = physics.eta_M(M_halo_z0, self.params)
            self.eta_E_val = physics.eta_E(M_halo_z0, self.params)
        else:
            self.eta_M_val = None
            self.eta_E_val = None

        self.eta_Z_val = self.params["eta_Z"]

    def derivatives(self, t, y, z):
        """
        Compute dy/dt for the ODE system.

        Args:
            t: Time in s (not directly used, z is used instead)
            y: State vector [M_halo, M_CGM, M_ISM, M_star, E_CGM, M_Z_CGM] in CGS
            z: Redshift at time t

        Returns:
            dydt: Time derivatives in CGS units
        """
        # Unpack state
        M_halo, M_CGM, M_ISM, M_star, E_CGM, M_Z_CGM = y

        # Prevent negative masses/energies
        M_halo = np.maximum(M_halo, 1e30)  # Minimum ~0.5 Msun
        M_CGM = np.maximum(M_CGM, 1e30)
        M_ISM = np.maximum(M_ISM, 1e30)
        M_star = np.maximum(M_star, 1e30)
        E_CGM = np.maximum(E_CGM, 1e45)
        M_Z_CGM = np.maximum(M_Z_CGM, 0)

        # Extract parameters
        params = self.params
        f_b = params["f_b"]
        f_rec = params["f_rec"]
        alpha = params["alpha"]
        r0_fraction = params["r0_fraction"]
        alpha_prevent = params["alpha_prevent"]
        E_SN_per_mass = params["E_SN_per_mass"]
        k_B = params["k_B"]
        mu = params["mu"]
        y_SN = params["y_SN"]
        Z_IGM = params["Z_IGM"]

        # Compute virial quantities
        r_vir = physics.virial_radius(M_halo, z, params)
        T_vir = physics.virial_temperature(M_halo, z, params)
        t_ff = physics.free_fall_time(M_halo, r_vir, params)

        # CGM properties
        e_CGM = E_CGM / M_CGM  # Specific energy (erg/g)
        T_CGM = (mu / k_B) * e_CGM  # Temperature (K)
        Z_CGM = M_Z_CGM / M_CGM  # Metallicity (absolute)

        # CGM structure
        rho_0 = physics.compute_rho_0(M_CGM, r_vir, alpha, r0_fraction, params)

        # Halo accretion rate
        dM_halo_dt = physics.halo_accretion_rate(M_halo, z, params)

        # Cooling rate
        dE_cool = physics.cooling_rate(rho_0, r_vir, r0_fraction, alpha,
                                       T_CGM, Z_CGM, z, params)

        # Effective cooling time
        if dE_cool > 0:
            t_cool_eff = E_CGM / dE_cool + t_ff
        else:
            t_cool_eff = 1e20  # Very long if no cooling

        # Cooling mass flux
        dM_cool = M_CGM / t_cool_eff

        # Star formation rate
        t_dep = physics.depletion_time(M_star, z, params)
        dM_SFR = M_ISM / t_dep

        # Loading factors
        if self.eta_M_val is not None:
            eta_M = self.eta_M_val
            eta_E = self.eta_E_val
        else:
            # Fall back to computing from current M_halo if z0 not provided
            eta_M = physics.eta_M(M_halo, params)
            eta_E = physics.eta_E(M_halo, params)

        eta_Z = self.eta_Z_val

        # Wind mass flux
        dM_ISM_wind = eta_M * dM_SFR

        # Wind energy flux
        dE_ISM_wind = eta_E * dM_SFR * E_SN_per_mass

        # Energy outflow rate
        c_s = physics.sound_speed(T_CGM, params)
        E_excess = np.maximum(E_CGM - k_B * T_vir * M_CGM / mu, 0)
        dE_CGM_out = E_excess * c_s / r_vir

        # Mass outflow rate
        if e_CGM > 0:
            dM_CGM_out = dE_CGM_out / e_CGM
        else:
            dM_CGM_out = 0

        # Inflow energy rate
        dE_CGM_in = (k_B * T_vir / mu) * f_b * dM_halo_dt

        # Preventive inflow factor
        if dE_CGM_out > 0:
            f_prevent = np.minimum(alpha_prevent * dE_CGM_in / dE_CGM_out, 1.0)
        else:
            f_prevent = 1.0

        # Inflow mass flux (with preventive feedback)
        dM_CGM_in = f_b * f_prevent * dM_halo_dt

        # Update inflow energy (after f_prevent applied)
        dE_CGM_in = (k_B * T_vir / mu) * dM_CGM_in

        # CGM metallicity fraction
        if M_CGM > 0:
            f_Z_CGM = M_Z_CGM / M_CGM
        else:
            f_Z_CGM = Z_IGM

        # Mass derivatives
        dM_CGM_dt = dM_CGM_in - dM_cool + dM_ISM_wind - dM_CGM_out
        dM_ISM_dt = -dM_SFR * (1 + eta_M - f_rec) + dM_cool
        dM_star_dt = (1 - f_rec) * dM_SFR

        # Energy derivative
        dE_CGM_dt = dE_CGM_in - dE_cool + dE_ISM_wind - dE_CGM_out

        # Metallicity derivative
        dM_Z_CGM_dt = (eta_Z * y_SN * dM_SFR +
                       Z_IGM * dM_CGM_in -
                       f_Z_CGM * (dM_cool + dM_CGM_out))

        return np.array([
            dM_halo_dt,
            dM_CGM_dt,
            dM_ISM_dt,
            dM_star_dt,
            dE_CGM_dt,
            dM_Z_CGM_dt
        ])

    def initial_conditions(self, M_halo_init, z_init):
        """
        Set up initial conditions for the model.

        Args:
            M_halo_init: Initial halo mass in g
            z_init: Initial redshift

        Returns:
            y0: Initial state vector
        """
        params = self.params
        f_b = params["f_b"]
        k_B = params["k_B"]
        mu = params["mu"]
        Z_IGM = params["Z_IGM"]

        # Start with small seeds for baryonic components
        # Carr et al. 2023 Fig 2 shows all masses ≤ 1e6 Msun at z=6
        M_CGM_init = np.minimum(1e6 * params["M_sun"], f_b * M_halo_init * 0.05)
        M_ISM_init = np.minimum(1e6 * params["M_sun"], f_b * M_halo_init * 0.01)
        M_star_init = np.minimum(1e6 * params["M_sun"], f_b * M_halo_init * 0.001)

        # CGM starts at virial temperature
        T_vir_init = physics.virial_temperature(M_halo_init, z_init, params)
        E_CGM_init = (k_B * T_vir_init / mu) * M_CGM_init

        # CGM starts with IGM metallicity
        M_Z_CGM_init = Z_IGM * M_CGM_init

        return np.array([
            M_halo_init,
            M_CGM_init,
            M_ISM_init,
            M_star_init,
            E_CGM_init,
            M_Z_CGM_init
        ])
