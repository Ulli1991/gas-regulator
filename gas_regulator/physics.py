"""
Physical calculations for the gas-regulator model.

Includes cosmology, virial quantities, cooling rates, and structure calculations.
"""

import numpy as np
from .parameters import default_params

# Wiersma cooling (loaded on demand)
_wiersma_cooling = None

def get_wiersma_cooling():
    """Load Wiersma cooling tables (singleton)."""
    global _wiersma_cooling
    if _wiersma_cooling is None:
        from .wiersma_cooling import get_wiersma_cooling as _get
        _wiersma_cooling = _get()
    return _wiersma_cooling


def hubble_parameter(z, params=None):
    """
    Hubble parameter H(z) in s^-1.

    Args:
        z: Redshift
        params: Parameter dictionary (optional)

    Returns:
        H(z) in s^-1
    """
    if params is None:
        params = default_params

    Omega_m = params["Omega_m"]
    Omega_Lambda = params["Omega_Lambda"]
    H0 = params["H0"]  # km/s/Mpc

    # Convert H0 to s^-1
    H0_cgs = H0 * 1e5 / (1e6 * params["kpc"] / 1e3)  # s^-1

    return H0_cgs * np.sqrt(Omega_m * (1 + z)**3 + Omega_Lambda)


def critical_density(z, params=None):
    """
    Critical density rho_crit(z) in g/cm^3.

    Args:
        z: Redshift
        params: Parameter dictionary (optional)

    Returns:
        rho_crit in g/cm^3
    """
    if params is None:
        params = default_params

    H_z = hubble_parameter(z, params)
    G = params["G"]

    return 3 * H_z**2 / (8 * np.pi * G)


def virial_radius(M_halo, z, params=None):
    """
    Virial radius r_vir in cm.

    For Delta_vir = 200:
    r_vir = (3*M_halo / (4*pi*Delta_vir*rho_crit))^(1/3)

    Args:
        M_halo: Halo mass in g
        z: Redshift
        params: Parameter dictionary (optional)

    Returns:
        r_vir in cm
    """
    if params is None:
        params = default_params

    Delta_vir = params["Delta_vir"]
    rho_crit = critical_density(z, params)

    return (3 * M_halo / (4 * np.pi * Delta_vir * rho_crit))**(1.0/3.0)


def virial_temperature(M_halo, z, params=None):
    """
    Virial temperature T_vir in K.

    T_vir = (mu / (2*k_B)) * G*M_halo / r_vir

    Args:
        M_halo: Halo mass in g
        z: Redshift
        params: Parameter dictionary (optional)

    Returns:
        T_vir in K
    """
    if params is None:
        params = default_params

    G = params["G"]
    k_B = params["k_B"]
    mu = params["mu"]

    r_vir = virial_radius(M_halo, z, params)

    return (mu / (2 * k_B)) * G * M_halo / r_vir


def free_fall_time(M_halo, r_vir, params=None):
    """
    Free fall time t_ff in s.

    t_ff = sqrt(3*pi / (32*G*rho_mean))
    where rho_mean = 3*M_halo / (4*pi*r_vir^3)

    Args:
        M_halo: Halo mass in g
        r_vir: Virial radius in cm
        params: Parameter dictionary (optional)

    Returns:
        t_ff in s
    """
    if params is None:
        params = default_params

    G = params["G"]
    rho_mean = 3 * M_halo / (4 * np.pi * r_vir**3)

    return np.sqrt(3 * np.pi / (32 * G * rho_mean))


def compute_rho_0(M_CGM, r_vir, alpha, r0_fraction, params=None):
    """
    Compute density normalization rho_0 from CGM mass.

    For rho(r) = rho_0 * (r/r_0)^(-alpha), the total mass is:
    M_CGM = 4*pi * rho_0 * r_0^3 * [(r_vir/r_0)^(3-alpha) - 1] / (3-alpha)

    Solving for rho_0:
    rho_0 = M_CGM * (3-alpha) / [4*pi * r_0^3 * ((r_vir/r_0)^(3-alpha) - 1)]

    Args:
        M_CGM: CGM mass in g
        r_vir: Virial radius in cm
        alpha: Density profile power-law index
        r0_fraction: r_0 / r_vir
        params: Parameter dictionary (optional)

    Returns:
        rho_0 in g/cm^3
    """
    r_0 = r0_fraction * r_vir
    x = r_vir / r_0  # Dimensionless radius ratio

    if np.abs(alpha - 3.0) < 1e-6:
        # Special case: alpha = 3
        denominator = 4 * np.pi * r_0**3 * np.log(x)
    else:
        denominator = 4 * np.pi * r_0**3 * (x**(3 - alpha) - 1) / (3 - alpha)

    return M_CGM / denominator


def cooling_function(T, Z, z=0, params=None):
    """
    Cooling function Lambda(T, Z, z) in erg cm^3 s^-1.

    Simplified fitting function approximation.
    For full accuracy, would use Wiersma et al. tables.

    Lambda ~ Lambda_0 * (T/T_0)^alpha * (Z/Z_sun)^beta

    Args:
        T: Temperature in K
        Z: Metallicity (absolute, not Z/Z_sun)
        z: Redshift (for UV background, not yet implemented)
        params: Parameter dictionary (optional)

    Returns:
        Lambda in erg cm^3 s^-1
    """
    if params is None:
        params = default_params

    Lambda_norm = params["Lambda_norm"]
    T_norm = params["T_norm"]
    T_transition = params["T_transition"]
    alpha_cool_low = params["alpha_cool_low"]
    alpha_cool_high = params["alpha_cool_high"]

    # Metallicity scaling (roughly linear for sub-solar)
    Z_sun = 0.0134
    Z_factor = np.maximum(Z / Z_sun, 0.1)  # Floor at 0.1 Z_sun

    # Temperature-dependent power-law
    if T < T_transition:
        alpha_cool = alpha_cool_low
    else:
        alpha_cool = alpha_cool_high

    Lambda = Lambda_norm * (T / T_norm)**alpha_cool * Z_factor**0.5

    # Ensure positive and not too small
    return np.maximum(Lambda, 1e-30)


def cooling_rate(rho_0, r_vir, r0_fraction, alpha, T_CGM, Z_CGM, z, params=None):
    """
    Total cooling rate dE_CGM,cool/dt in erg/s.

    From Eq. label:e_cool:
    dE_cool/dt = 4*pi * rho_0^2 * r_0^3 * (Lambda(T,Z,z)/mu^2) *
                 [(r_vir/r_0)^(3-2*alpha) - 1] / (3-2*alpha)

    Args:
        rho_0: Density normalization in g/cm^3
        r_vir: Virial radius in cm
        r0_fraction: r_0 / r_vir
        alpha: Density profile power-law index
        T_CGM: CGM temperature in K
        Z_CGM: CGM metallicity (absolute)
        z: Redshift
        params: Parameter dictionary (optional)

    Returns:
        dE_cool/dt in erg/s
    """
    if params is None:
        params = default_params

    mu = params["mu"]
    mu_H = params["mu_H"]
    r_0 = r0_fraction * r_vir
    x = r_vir / r_0

    # Check if we should use Wiersma tables
    use_wiersma = params.get("use_wiersma_cooling", False)

    if use_wiersma:
        # Use Wiersma cooling tables
        wiersma = get_wiersma_cooling()
        # Compute characteristic hydrogen density at r ~ 0.3 * r_vir
        # Wiersma tables are normalized per n_H^2, so convert rho -> n_H
        # using mu_H = (1/0.75) * m_p (mean mass per hydrogen atom)
        r_char = 0.3 * r_vir
        rho_char = rho_0 * (r_char / r_0)**(-alpha)
        nH_char = rho_char / mu_H
        Lambda = wiersma.cooling_rate(T_CGM, nH_char, Z_CGM, z)
    else:
        # Use simplified cooling function
        Lambda = cooling_function(T_CGM, Z_CGM, z, params)

    if np.abs(2*alpha - 3.0) < 1e-6:
        # Special case: 2*alpha = 3
        integral_factor = np.log(x)
    else:
        integral_factor = (x**(3 - 2*alpha) - 1) / (3 - 2*alpha)

    # Wiersma Lambda is per n_H^2, so the prefactor must use n_H = rho/mu_H
    # dE_cool = integral of n_H^2 * Lambda * 4*pi*r^2 dr
    #         = 4*pi * rho_0^2/mu_H^2 * r_0^3 * Lambda * integral_factor
    dE_cool = 4 * np.pi * rho_0**2 * r_0**3 * (Lambda / mu_H**2) * integral_factor

    return np.maximum(dE_cool, 0)


def depletion_time(M_star, z, params=None):
    """
    Gas depletion time t_dep in s.

    From Eq. label:tdep:
    t_dep = 10^4.92 * (M_star/M_sun)^(-0.37) * (1+z)^(-3/2) [years]

    IMPORTANT (from paper footnote): For M_star < 5e7 Msun, t_dep is held
    constant at the M_star=5e7 value and only evolves with redshift.

    Args:
        M_star: Stellar mass in g
        z: Redshift
        params: Parameter dictionary (optional)

    Returns:
        t_dep in s
    """
    if params is None:
        params = default_params

    M_sun = params["M_sun"]
    yr = params["yr"]

    t_dep_norm = params["t_dep_norm"]
    M_exp = params["t_dep_M_exp"]
    z_exp = params["t_dep_z_exp"]

    # Apply floor for low-mass galaxies (paper footnote)
    M_star_use = np.maximum(M_star, 5e7 * M_sun)

    t_dep_yr = t_dep_norm * (M_star_use / M_sun)**M_exp * (1 + z)**z_exp

    return t_dep_yr * yr


def halo_accretion_rate(M_halo, z, params=None):
    """
    Halo mass accretion rate dM_halo/dt in g/s.

    From Eq. label:halo_inflow:
    dM_halo/dt = 0.47 * M_halo * (M_halo/10^12 M_sun)^0.15 * ((1+z)/3)^2.25 [Gyr^-1]

    Args:
        M_halo: Halo mass in g
        z: Redshift
        params: Parameter dictionary (optional)

    Returns:
        dM_halo/dt in g/s
    """
    if params is None:
        params = default_params

    M_sun = params["M_sun"]
    Gyr = params["Gyr"]

    norm = params["halo_accr_norm"]
    M_exp = params["halo_accr_M_exp"]
    z_exp = params["halo_accr_z_exp"]

    M_12 = M_halo / (1e12 * M_sun)
    z_factor = ((1 + z) / 3.0)**z_exp

    # Rate in Gyr^-1
    rate_Gyr = norm * M_halo * M_12**M_exp * z_factor

    return rate_Gyr / Gyr


def eta_M(M_halo_z0, params=None):
    """
    Mass loading factor eta_M.

    eta_M = eta_M_norm * (M_halo(z=0) / 10^12 M_sun)^(-beta)

    Args:
        M_halo_z0: Halo mass at z=0 in g
        params: Parameter dictionary (optional)

    Returns:
        eta_M (dimensionless)
    """
    if params is None:
        params = default_params

    M_sun = params["M_sun"]
    norm = params["eta_M_norm"]
    beta = params["eta_M_beta"]

    M_12 = M_halo_z0 / (1e12 * M_sun)

    return norm * M_12**(-beta)


def eta_E(M_halo_z0, params=None):
    """
    Energy loading factor eta_E.

    eta_E = A * (M_halo(z=0) / 10^12 M_sun)^(-lambda)

    Args:
        M_halo_z0: Halo mass at z=0 in g
        params: Parameter dictionary (optional)

    Returns:
        eta_E (dimensionless)
    """
    if params is None:
        params = default_params

    M_sun = params["M_sun"]
    A = params["eta_E_A"]
    lam = params["eta_E_lambda"]

    M_12 = M_halo_z0 / (1e12 * M_sun)

    return A * M_12**(-lam)


def sound_speed(T_CGM, params=None):
    """
    Sound speed c_s in cm/s.

    c_s = sqrt(5*k_B*T_CGM / (3*mu))

    Args:
        T_CGM: CGM temperature in K
        params: Parameter dictionary (optional)

    Returns:
        c_s in cm/s
    """
    if params is None:
        params = default_params

    k_B = params["k_B"]
    mu = params["mu"]

    return np.sqrt(5 * k_B * T_CGM / (3 * mu))
