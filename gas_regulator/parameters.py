"""
Physical constants and default model parameters for the gas-regulator model.
"""

import numpy as np

# Physical constants (CGS units)
GRAVITATIONAL_CONSTANT = 6.674e-8  # cm^3 g^-1 s^-2
BOLTZMANN_CONSTANT = 1.381e-16  # erg K^-1
PROTON_MASS = 1.673e-24  # g
SOLAR_MASS = 1.989e33  # g
PARSEC = 3.086e18  # cm
KILOPARSEC = 1e3 * PARSEC  # cm
YEAR = 3.154e7  # s
GIGAYEAR = 1e9 * YEAR  # s

# Mean molecular weight
MU = 0.6 * PROTON_MASS  # g (for ionized gas)
MU_H = (1.0 / 0.75) * PROTON_MASS  # g (mean mass per hydrogen atom, for n_H conversion)

# Cosmological parameters (Planck 2018-like)
H0 = 70.0  # km/s/Mpc (h = 0.7)
OMEGA_M = 0.3
OMEGA_LAMBDA = 0.7
HUBBLE_CONSTANT_CGS = H0 * 1e5 / (1e6 * PARSEC)  # s^-1

# Universal baryon fraction
F_BARYON = 0.16  # Omega_b / Omega_m

# Solar metallicity
Z_SOLAR = 0.0134

# Model parameters (defaults from Carr et al. 2023)
default_params = {
    # CGM structure
    "alpha": 1.4,  # Density profile power-law index
    "r0_fraction": 0.1,  # r_0 = r0_fraction * r_vir

    # Star formation
    "f_rec": 0.4,  # Recycling fraction
    "t_dep_norm": 10**4.92 * 1e9,  # Depletion time normalization in YEARS (McGaugh: gives t_dep in Gyr, so multiply by 1e9)
    "t_dep_M_exp": -0.37,  # Stellar mass exponent (NEGATIVE - larger M* = shorter t_dep, per McGaugh)
    "t_dep_z_exp": -1.5,  # Redshift exponent

    # Halo accretion (Dekel et al. 2009)
    "halo_accr_norm": 0.47,  # Gyr^-1
    "halo_accr_M_exp": 0.15,
    "halo_accr_z_exp": 2.25,

    # Mass loading factors
    "eta_M_norm": 1.0,  # Normalization at 10^12 Msun
    "eta_M_beta": 0.5,  # Mass dependence exponent

    # Energy loading factors
    "eta_E_A": 0.051,  # Normalization
    "eta_E_lambda": 0.65,  # Mass dependence exponent

    # Metal loading
    "eta_Z": 0.5,  # Metal loading factor
    "y_SN": 0.02,  # Metal yield per stellar mass formed

    # Preventive feedback
    "alpha_prevent": 2.0,  # Preventive factor coefficient

    # IGM metallicity
    "Z_IGM": 0.01 * Z_SOLAR,

    # Supernova energy per mass
    "E_SN_per_mass": 1e51 / (100 * SOLAR_MASS),  # erg / g

    # Virial overdensity
    "Delta_vir": 200.0,

    # Physical constants (for convenience)
    "G": GRAVITATIONAL_CONSTANT,
    "k_B": BOLTZMANN_CONSTANT,
    "m_p": PROTON_MASS,
    "mu": MU,
    "mu_H": MU_H,
    "M_sun": SOLAR_MASS,
    "kpc": KILOPARSEC,
    "Gyr": GIGAYEAR,
    "yr": YEAR,

    # Cosmology
    "H0": H0,
    "Omega_m": OMEGA_M,
    "Omega_Lambda": OMEGA_LAMBDA,
    "f_b": F_BARYON,
}

# Cooling function parameters (simplified fitting function)
# Lambda(T) ~ Lambda_0 * (T/T_0)^alpha for different temperature regimes
cooling_params = {
    # Simple power-law approximation for 10^4 < T < 10^7 K
    "Lambda_norm": 1e-22,  # erg cm^3 s^-1 (approximate normalization)
    "T_norm": 1e6,  # K
    "alpha_cool_low": -0.7,  # T < 10^5.5 K
    "alpha_cool_high": 0.5,  # T > 10^5.5 K
    "T_transition": 10**5.5,  # K
}

# Add cooling params to default params
default_params.update(cooling_params)
