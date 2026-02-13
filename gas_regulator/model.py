"""
Main ODE system for the gas-regulator model.

Implements the gas-regulator model from Carr et al. 2023 with optional
turbulence extension from Pandya et al. and cosmic ray transport.

State vector (Carr model, 6 components):
    [M_halo, M_CGM, M_ISM, M_star, E_CGM, M_Z_CGM]

State vector (Pandya turbulence model, 7 components):
    [M_halo, M_CGM, M_ISM, M_star, E_th, E_kin, M_Z_CGM]

State vector (turbulence + cosmic rays, 8 components):
    [M_halo, M_CGM, M_ISM, M_star, E_th, E_kin, E_CR, M_Z_CGM]
"""

import numpy as np
from . import physics
from .parameters import default_params


class GasRegulatorModel:
    """
    Gas-regulator model ODE system.

    Tracks evolution of halo mass, CGM mass/energy/metallicity,
    ISM mass, and stellar mass through cosmic time.

    Modes:
    - Carr (default): 6-component, thermal-only CGM energy
    - Pandya (enable_turbulence=True): 7-component, thermal + kinetic
    - Full (enable_cosmic_rays=True): 8-component, thermal + kinetic + CR
    """

    def __init__(self, params=None, M_halo_z0=None):
        if params is None:
            self.params = default_params.copy()
        else:
            self.params = params.copy()

        self.M_halo_z0 = M_halo_z0
        self.enable_turbulence = self.params.get("enable_turbulence", False)
        self.enable_cosmic_rays = self.params.get("enable_cosmic_rays", False)

        # CRs require turbulence tracking (both are non-thermal energy components)
        if self.enable_cosmic_rays:
            self.enable_turbulence = True

        # Precompute loading factors if M_halo_z0 is provided
        if M_halo_z0 is not None:
            self.eta_M_val = physics.eta_M(M_halo_z0, self.params)
            self.eta_E_val = physics.eta_E(M_halo_z0, self.params)
        else:
            self.eta_M_val = None
            self.eta_E_val = None

        self.eta_Z_val = self.params["eta_Z"]

    @property
    def n_state(self):
        """Number of state variables."""
        if self.enable_cosmic_rays:
            return 8
        elif self.enable_turbulence:
            return 7
        else:
            return 6

    def derivatives(self, t, y, z):
        """
        Compute dy/dt for the ODE system.

        Args:
            t: Time in s (not directly used, z is used instead)
            y: State vector in CGS units
            z: Redshift at time t

        Returns:
            dydt: Time derivatives in CGS units
        """
        if self.enable_cosmic_rays:
            return self._derivatives_cr(t, y, z)
        elif self.enable_turbulence:
            return self._derivatives_pandya(t, y, z)
        else:
            return self._derivatives_carr(t, y, z)

    def _derivatives_carr(self, t, y, z):
        """Original Carr et al. 2023 model (6-component state)."""
        # Unpack state
        M_halo, M_CGM, M_ISM, M_star, E_CGM, M_Z_CGM = y

        # Prevent negative masses/energies
        M_halo = np.maximum(M_halo, 1e30)
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
        e_CGM = E_CGM / M_CGM
        T_CGM = (mu / k_B) * e_CGM
        Z_CGM = M_Z_CGM / M_CGM

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
            t_cool_eff = 1e20

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

    def _derivatives_pandya(self, t, y, z):
        """Turbulence model (7-component state). Carr timescales by default."""
        # Unpack state
        M_halo, M_CGM, M_ISM, M_star, E_th, E_kin, M_Z_CGM = y

        # Prevent negative masses/energies
        M_halo = np.maximum(M_halo, 1e30)
        M_CGM = np.maximum(M_CGM, 1e30)
        M_ISM = np.maximum(M_ISM, 1e30)
        M_star = np.maximum(M_star, 1e30)
        E_th = np.maximum(E_th, 1e45)
        E_kin = np.maximum(E_kin, 0)
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
        f_th_accr = params["f_thermal_accretion"]
        f_th_wind = params["f_thermal_wind"]
        R_turb_frac = params["R_turb_fraction"]
        use_pandya = params.get("use_pandya_timescales", False)

        # Virial quantities
        r_vir = physics.virial_radius(M_halo, z, params)
        T_vir = physics.virial_temperature(M_halo, z, params)

        # Total CGM energy
        E_total = E_th + E_kin
        e_total = E_total / M_CGM

        # CGM temperature from thermal energy only
        T_CGM = (mu / k_B) * (E_th / M_CGM)
        Z_CGM = M_Z_CGM / M_CGM

        # CGM structure
        rho_0 = physics.compute_rho_0(M_CGM, r_vir, alpha, r0_fraction, params)

        # Halo accretion rate
        dM_halo_dt = physics.halo_accretion_rate(M_halo, z, params)

        # Radiative cooling rate (erg/s)
        dE_cool = physics.cooling_rate(rho_0, r_vir, r0_fraction, alpha,
                                       T_CGM, Z_CGM, z, params)

        # Turbulence quantities
        R_turb = R_turb_frac * r_vir
        v_turb = physics.turbulent_velocity(E_kin, M_CGM)
        E_diss = physics.turbulence_dissipation_rate(E_kin, M_CGM, R_turb)

        # Cooling time and free-fall time
        if use_pandya:
            # Pandya: NFW t_ff with turbulent support, E_diss reduces net cooling
            t_ff = physics.t_ff_effective(M_halo, z, v_turb, params)
            net_cooling = dE_cool - E_diss
            if net_cooling > 0:
                t_cool_eff = E_th / net_cooling + t_ff
            else:
                t_cool_eff = 1e20
        else:
            # Carr: mean-density t_ff, standard cooling time
            t_ff = physics.free_fall_time(M_halo, r_vir, params)
            if dE_cool > 0:
                t_cool_eff = E_th / dE_cool + t_ff
            else:
                t_cool_eff = 1e20

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
            eta_M = physics.eta_M(M_halo, params)
            eta_E = physics.eta_E(M_halo, params)

        eta_Z = self.eta_Z_val

        # Wind mass flux
        dM_ISM_wind = eta_M * dM_SFR

        # Total wind energy flux
        dE_wind_total = eta_E * dM_SFR * E_SN_per_mass

        # Wind energy split into thermal/kinetic
        dE_wind_th = f_th_wind * dE_wind_total
        dE_wind_kin = (1 - f_th_wind) * dE_wind_total

        # Energy outflow (overpressurization, Carr prescription)
        c_s = physics.sound_speed(T_CGM, params)
        E_bind = k_B * T_vir * M_CGM / mu
        E_excess = np.maximum(E_total - E_bind, 0)
        dE_out_total = E_excess * c_s / r_vir

        # Partition outflow energy by current thermal/kinetic ratio
        f_th_CGM = E_th / np.maximum(E_total, 1e45)
        dE_out_th = f_th_CGM * dE_out_total
        dE_out_kin = (1 - f_th_CGM) * dE_out_total

        # Mass outflow (same specific energy as CGM)
        dM_CGM_out = dE_out_total / np.maximum(e_total, 1e10)

        # Total inflow energy rate (before preventive feedback)
        dE_in_total_raw = (k_B * T_vir / mu) * f_b * dM_halo_dt

        # Preventive inflow factor
        if dE_out_total > 0:
            f_prevent = np.minimum(alpha_prevent * dE_in_total_raw / dE_out_total, 1.0)
        else:
            f_prevent = 1.0

        # Inflow mass flux (with preventive feedback)
        dM_CGM_in = f_b * f_prevent * dM_halo_dt

        # Inflow energy (after preventive feedback)
        dE_in_total = (k_B * T_vir / mu) * dM_CGM_in

        # Inflow energy split into thermal/kinetic
        dE_in_th = f_th_accr * dE_in_total
        dE_in_kin = (1 - f_th_accr) * dE_in_total

        # CGM metallicity fraction
        if M_CGM > 0:
            f_Z_CGM = M_Z_CGM / M_CGM
        else:
            f_Z_CGM = Z_IGM

        # Mass derivatives
        dM_CGM_dt = dM_CGM_in - dM_cool + dM_ISM_wind - dM_CGM_out
        dM_ISM_dt = -dM_SFR * (1 + eta_M - f_rec) + dM_cool
        dM_star_dt = (1 - f_rec) * dM_SFR

        # Energy derivatives (Pandya Eq. 12-13)
        dE_th_dt = dE_in_th - dE_cool + E_diss + dE_wind_th - dE_out_th
        dE_kin_dt = dE_in_kin - E_diss + dE_wind_kin - dE_out_kin

        # Metallicity derivative
        dM_Z_CGM_dt = (eta_Z * y_SN * dM_SFR +
                       Z_IGM * dM_CGM_in -
                       f_Z_CGM * (dM_cool + dM_CGM_out))

        return np.array([
            dM_halo_dt,
            dM_CGM_dt,
            dM_ISM_dt,
            dM_star_dt,
            dE_th_dt,
            dE_kin_dt,
            dM_Z_CGM_dt
        ])

    def _derivatives_cr(self, t, y, z):
        """Turbulence + cosmic ray model (8-component state)."""
        # Unpack state
        M_halo, M_CGM, M_ISM, M_star, E_th, E_kin, E_CR, M_Z_CGM = y

        # Prevent negative masses/energies
        M_halo = np.maximum(M_halo, 1e30)
        M_CGM = np.maximum(M_CGM, 1e30)
        M_ISM = np.maximum(M_ISM, 1e30)
        M_star = np.maximum(M_star, 1e30)
        E_th = np.maximum(E_th, 1e45)
        E_kin = np.maximum(E_kin, 0)
        E_CR = np.maximum(E_CR, 0)
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
        f_th_accr = params["f_thermal_accretion"]
        f_th_wind = params["f_thermal_wind"]
        R_turb_frac = params["R_turb_fraction"]
        eta_CR = params["eta_CR"]
        use_pandya = params.get("use_pandya_timescales", False)

        # Virial quantities
        r_vir = physics.virial_radius(M_halo, z, params)
        T_vir = physics.virial_temperature(M_halo, z, params)

        # Total CGM energy (all three components)
        E_total = E_th + E_kin + E_CR
        e_total = E_total / M_CGM

        # CGM temperature from thermal energy only
        T_CGM = (mu / k_B) * (E_th / M_CGM)
        Z_CGM = M_Z_CGM / M_CGM

        # CGM structure
        rho_0 = physics.compute_rho_0(M_CGM, r_vir, alpha, r0_fraction, params)

        # Halo accretion rate
        dM_halo_dt = physics.halo_accretion_rate(M_halo, z, params)

        # Radiative cooling rate (erg/s)
        dE_cool = physics.cooling_rate(rho_0, r_vir, r0_fraction, alpha,
                                       T_CGM, Z_CGM, z, params)

        # Turbulence quantities
        R_turb = R_turb_frac * r_vir
        v_turb = physics.turbulent_velocity(E_kin, M_CGM)
        E_diss = physics.turbulence_dissipation_rate(E_kin, M_CGM, R_turb)

        # CR quantities
        v_CR = physics.cr_effective_velocity(E_CR, M_CGM)
        dE_CR_diff = physics.cr_diffusion_rate(E_CR, r_vir, params)

        # Cooling time and free-fall time
        if use_pandya:
            # Pandya: NFW t_ff with non-thermal support, E_diss reduces net cooling
            t_ff = physics.t_ff_effective(M_halo, z, v_turb, params, v_CR=v_CR)
            net_cooling = dE_cool - E_diss
            if net_cooling > 0:
                t_cool_eff = E_th / net_cooling + t_ff
            else:
                t_cool_eff = 1e20
        else:
            # Carr: mean-density t_ff, standard cooling time
            t_ff = physics.free_fall_time(M_halo, r_vir, params)
            if dE_cool > 0:
                t_cool_eff = E_th / dE_cool + t_ff
            else:
                t_cool_eff = 1e20

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
            eta_M = physics.eta_M(M_halo, params)
            eta_E = physics.eta_E(M_halo, params)

        eta_Z = self.eta_Z_val

        # Wind mass flux
        dM_ISM_wind = eta_M * dM_SFR

        # Wind energy: thermal+kinetic from eta_E, CRs from eta_CR (independent)
        dE_wind_thkin = eta_E * dM_SFR * E_SN_per_mass
        dE_wind_CR = eta_CR * dM_SFR * E_SN_per_mass

        # Split thermal+kinetic by f_thermal_wind
        dE_wind_th = f_th_wind * dE_wind_thkin
        dE_wind_kin = (1 - f_th_wind) * dE_wind_thkin

        # Energy outflow (overpressurization, Carr prescription)
        c_s = physics.sound_speed(T_CGM, params)
        E_bind = k_B * T_vir * M_CGM / mu
        E_excess = np.maximum(E_total - E_bind, 0)
        dE_out_total = E_excess * c_s / r_vir

        # Partition outflow energy by current energy ratios
        f_th_CGM = E_th / np.maximum(E_total, 1e45)
        f_kin_CGM = E_kin / np.maximum(E_total, 1e45)
        f_CR_CGM = E_CR / np.maximum(E_total, 1e45)
        dE_out_th = f_th_CGM * dE_out_total
        dE_out_kin = f_kin_CGM * dE_out_total
        dE_out_CR = f_CR_CGM * dE_out_total

        # Mass outflow
        dM_CGM_out = dE_out_total / np.maximum(e_total, 1e10)

        # Total inflow energy rate (before preventive feedback)
        # Accretion is gravitational — no CR component, only thermal/kinetic
        dE_in_total_raw = (k_B * T_vir / mu) * f_b * dM_halo_dt

        # Preventive inflow factor
        if dE_out_total > 0:
            f_prevent = np.minimum(alpha_prevent * dE_in_total_raw / dE_out_total, 1.0)
        else:
            f_prevent = 1.0

        # Inflow mass flux (with preventive feedback)
        dM_CGM_in = f_b * f_prevent * dM_halo_dt

        # Inflow energy (after preventive feedback) — thermal/kinetic only
        dE_in_total = (k_B * T_vir / mu) * dM_CGM_in
        dE_in_th = f_th_accr * dE_in_total
        dE_in_kin = (1 - f_th_accr) * dE_in_total

        # CGM metallicity fraction
        if M_CGM > 0:
            f_Z_CGM = M_Z_CGM / M_CGM
        else:
            f_Z_CGM = Z_IGM

        # Mass derivatives
        dM_CGM_dt = dM_CGM_in - dM_cool + dM_ISM_wind - dM_CGM_out
        dM_ISM_dt = -dM_SFR * (1 + eta_M - f_rec) + dM_cool
        dM_star_dt = (1 - f_rec) * dM_SFR

        # Energy derivatives
        # Thermal: inflow + dissipation + wind - cooling - outflow
        dE_th_dt = dE_in_th - dE_cool + E_diss + dE_wind_th - dE_out_th

        # Kinetic: inflow + wind - dissipation - outflow
        dE_kin_dt = dE_in_kin - E_diss + dE_wind_kin - dE_out_kin

        # Cosmic rays: wind injection - diffusion - outflow
        dE_CR_dt = dE_wind_CR - dE_CR_diff - dE_out_CR

        # Metallicity derivative
        dM_Z_CGM_dt = (eta_Z * y_SN * dM_SFR +
                       Z_IGM * dM_CGM_in -
                       f_Z_CGM * (dM_cool + dM_CGM_out))

        return np.array([
            dM_halo_dt,
            dM_CGM_dt,
            dM_ISM_dt,
            dM_star_dt,
            dE_th_dt,
            dE_kin_dt,
            dE_CR_dt,
            dM_Z_CGM_dt
        ])

    def initial_conditions(self, M_halo_init, z_init):
        """
        Set up initial conditions for the model.

        Args:
            M_halo_init: Initial halo mass in g
            z_init: Initial redshift

        Returns:
            y0: Initial state vector (6, 7, or 8 components)
        """
        params = self.params
        f_b = params["f_b"]
        k_B = params["k_B"]
        mu = params["mu"]
        Z_IGM = params["Z_IGM"]

        # Start with small seeds for baryonic components
        M_CGM_init = np.minimum(1e6 * params["M_sun"], f_b * M_halo_init * 0.05)
        M_ISM_init = np.minimum(1e6 * params["M_sun"], f_b * M_halo_init * 0.01)
        M_star_init = np.minimum(1e6 * params["M_sun"], f_b * M_halo_init * 0.001)

        # CGM starts at virial temperature
        T_vir_init = physics.virial_temperature(M_halo_init, z_init, params)
        E_th_init = (k_B * T_vir_init / mu) * M_CGM_init

        # CGM starts with IGM metallicity
        M_Z_CGM_init = Z_IGM * M_CGM_init

        if self.enable_cosmic_rays:
            return np.array([
                M_halo_init,
                M_CGM_init,
                M_ISM_init,
                M_star_init,
                E_th_init,
                0.0,  # E_kin
                0.0,  # E_CR
                M_Z_CGM_init
            ])
        elif self.enable_turbulence:
            return np.array([
                M_halo_init,
                M_CGM_init,
                M_ISM_init,
                M_star_init,
                E_th_init,
                0.0,  # E_kin
                M_Z_CGM_init
            ])
        else:
            return np.array([
                M_halo_init,
                M_CGM_init,
                M_ISM_init,
                M_star_init,
                E_th_init,
                M_Z_CGM_init
            ])
