"""
Solver interface for the gas-regulator model.

Provides high-level functions to run single halos or halo suites.
"""

import numpy as np
from scipy.integrate import solve_ivp
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from .model import GasRegulatorModel
from .parameters import default_params


def create_cosmology(params=None):
    """
    Create an astropy cosmology object from parameters.

    Args:
        params: Parameter dictionary (optional)

    Returns:
        astropy.cosmology object
    """
    if params is None:
        params = default_params

    H0 = params["H0"]
    Om0 = params["Omega_m"]

    return FlatLambdaCDM(H0=H0, Om0=Om0)


def redshift_to_time(z, cosmo):
    """
    Convert redshift to cosmic time in seconds.

    Args:
        z: Redshift
        cosmo: astropy.cosmology object

    Returns:
        Time in seconds since Big Bang
    """
    t = cosmo.age(z).to(u.s).value
    return t


def time_to_redshift(t, cosmo, z_range=(0, 20)):
    """
    Convert cosmic time to redshift (inverse of redshift_to_time).

    Args:
        t: Time in seconds since Big Bang
        cosmo: astropy.cosmology object
        z_range: Tuple of (z_min, z_max) for search

    Returns:
        Redshift z
    """
    from scipy.optimize import brentq

    def residual(z):
        return redshift_to_time(z, cosmo) - t

    # Find bracketing interval
    z_low, z_high = z_range
    if residual(z_low) * residual(z_high) > 0:
        # Not bracketed, return boundary
        if np.abs(residual(z_low)) < np.abs(residual(z_high)):
            return z_low
        else:
            return z_high

    return brentq(residual, z_low, z_high)


def extrapolate_halo_mass(M_halo_z0, z_target, z0=0, params=None):
    """
    Extrapolate halo mass to earlier redshift using growth rate.

    Properly integrates the halo accretion ODE backwards in redshift.

    Args:
        M_halo_z0: Halo mass at z=z0 in g
        z_target: Target redshift
        z0: Reference redshift (default 0)
        params: Parameter dictionary (optional)

    Returns:
        M_halo at z_target in g
    """
    if params is None:
        params = default_params

    if z_target <= z0:
        return M_halo_z0

    # Properly integrate backward: dM/dz = -dM/dt * dt/dz
    # where dM/dt = 0.47 * M * (M/1e12)^0.15 * ((1+z)/3)^2.25 [Gyr^-1]
    from scipy.integrate import solve_ivp

    M_sun = params["M_sun"]
    Gyr = params["Gyr"]
    cosmo = create_cosmology(params)

    norm = params["halo_accr_norm"]
    M_exp = params["halo_accr_M_exp"]
    z_exp = params["halo_accr_z_exp"]

    def dM_dz(z, M):
        """dM/dz for backward integration"""
        # dM/dt in g/s
        M_12 = M / (1e12 * M_sun)
        rate_per_M = norm * M_12**M_exp * ((1 + z) / 3)**z_exp / Gyr
        dM_dt = M * rate_per_M

        # dt/dz (negative because t decreases as z increases)
        H_z = cosmo.H(z).to(1/u.s).value
        dt_dz = -1 / ((1 + z) * H_z)

        # dM/dz (negative for backward integration)
        return dM_dt * dt_dz

    # Integrate backward from z0 to z_target
    sol = solve_ivp(
        dM_dz,
        (z0, z_target),
        [M_halo_z0],
        method='RK45',
        dense_output=True,
        rtol=1e-6,
        atol=1e-6
    )

    if not sol.success:
        # Fallback to old approximation if integration fails
        print(f"Warning: Backward integration failed, using approximation")
        from scipy.integrate import quad

        def integrand(z_prime):
            M_12 = 1.0  # Approximation
            rate_per_M = norm * M_12**M_exp * ((1 + z_prime) / 3)**z_exp / Gyr
            H_z = cosmo.H(z_prime).to(1/u.s).value
            dt_dz = -1 / ((1 + z_prime) * H_z)
            return rate_per_M * dt_dz

        integral, _ = quad(integrand, z0, z_target)
        return M_halo_z0 * np.exp(-integral)

    return sol.y[0, -1]


def run_single_halo(M_halo_z0, z_start, z_end, params=None, rtol=1e-6, atol=1e-6, M_halo_init=None):
    """
    Run a single halo evolution from z_start to z_end.

    Args:
        M_halo_z0: Target halo mass at z=0 in solar masses (used for extrapolation)
        z_start: Starting redshift
        z_end: Ending redshift
        params: Parameter dictionary (optional)
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
        M_halo_init: Initial halo mass at z_start in solar masses (optional).
                     If provided, this overrides the backward extrapolation from M_halo_z0.
                     Use this if you know the correct initial mass (e.g., from Carr et al. 2023).

    Returns:
        Dictionary with results:
            - time: Time array in Gyr
            - redshift: Redshift array
            - M_halo: Halo mass in Msun
            - M_CGM: CGM mass in Msun
            - M_ISM: ISM mass in Msun
            - M_star: Stellar mass in Msun
            - E_CGM: Total CGM energy in erg (E_th + E_kin when turbulence on)
            - M_Z_CGM: CGM metal mass in Msun
            - Z_CGM: CGM metallicity (Z/Z_sun)
            - T_CGM: CGM temperature in K
            - params: Parameter dictionary used
            When enable_turbulence=True, also:
            - E_th: Thermal energy in erg
            - E_kin: Kinetic energy in erg
            - v_turb: Turbulent velocity in cm/s
    """
    if params is None:
        params = default_params.copy()

    M_sun = params["M_sun"]
    k_B = params["k_B"]
    mu = params["mu"]
    enable_CR = params.get("enable_cosmic_rays", False)
    enable_turb = params.get("enable_turbulence", False) or enable_CR

    # Convert to CGS
    M_halo_z0_cgs = M_halo_z0 * M_sun

    # Create cosmology
    cosmo = create_cosmology(params)

    # Time span
    t_start = redshift_to_time(z_start, cosmo)
    t_end = redshift_to_time(z_end, cosmo)

    # Initial halo mass at z_start
    if M_halo_init is not None:
        # User provided initial mass directly
        M_halo_init_cgs = M_halo_init * M_sun
        print(f"Using provided initial mass: M_halo(z={z_start}) = {M_halo_init:.2e} Msun")
    else:
        # Extrapolate backward from z=0
        M_halo_init_cgs = extrapolate_halo_mass(M_halo_z0_cgs, z_start, z0=0, params=params)
        print(f"Extrapolated initial mass: M_halo(z={z_start}) = {M_halo_init_cgs/M_sun:.2e} Msun")

    # Initialize model
    model = GasRegulatorModel(params=params, M_halo_z0=M_halo_z0_cgs)
    y0 = model.initial_conditions(M_halo_init_cgs, z_start)

    if enable_CR:
        mode = "Full (turbulence + cosmic rays)"
    elif enable_turb:
        mode = "Pandya (turbulence)"
    else:
        mode = "Carr (thermal only)"
    print(f"Model: {mode}, state dimension: {model.n_state}")

    # Create wrapper that includes redshift
    def dydt_wrapper(t, y):
        z = time_to_redshift(t, cosmo, z_range=(z_end, z_start))
        return model.derivatives(t, y, z)

    # Solve ODE
    # Use BDF method for stiff equations (cooling/heating can be stiff)
    print(f"Solving ODE from z={z_start:.2f} to z={z_end:.2f}...")
    sol = solve_ivp(
        dydt_wrapper,
        (t_start, t_end),
        y0,
        method='BDF',  # Backward differentiation formula for stiff ODEs
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=1e14  # ~3 Myr max step
    )

    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    print(f"Integration successful: {len(sol.t)} time steps")

    # Extract solution
    t_array = sol.t
    y_array = sol.y

    # Convert redshifts
    z_array = np.array([time_to_redshift(t, cosmo, z_range=(z_end, z_start))
                        for t in t_array])

    # Unpack state (handles both 6 and 7 component)
    M_halo_arr = y_array[0, :] / M_sun
    M_CGM_arr = y_array[1, :] / M_sun
    M_ISM_arr = y_array[2, :] / M_sun
    M_star_arr = y_array[3, :] / M_sun

    from . import physics as _phys

    if enable_CR:
        E_th_arr = y_array[4, :]
        E_kin_arr = y_array[5, :]
        E_CR_arr = y_array[6, :]
        E_CGM_arr = E_th_arr + E_kin_arr + E_CR_arr
        M_Z_CGM_arr = y_array[7, :] / M_sun
        e_th_arr = E_th_arr / (M_CGM_arr * M_sun + 1e-30)
        T_CGM_arr = (mu / k_B) * e_th_arr
        v_turb_arr = np.array([
            _phys.turbulent_velocity(ek, mc * M_sun)
            for ek, mc in zip(E_kin_arr, M_CGM_arr)
        ])
    elif enable_turb:
        E_th_arr = y_array[4, :]
        E_kin_arr = y_array[5, :]
        E_CGM_arr = E_th_arr + E_kin_arr
        M_Z_CGM_arr = y_array[6, :] / M_sun
        e_th_arr = E_th_arr / (M_CGM_arr * M_sun + 1e-30)
        T_CGM_arr = (mu / k_B) * e_th_arr
        v_turb_arr = np.array([
            _phys.turbulent_velocity(ek, mc * M_sun)
            for ek, mc in zip(E_kin_arr, M_CGM_arr)
        ])
    else:
        E_CGM_arr = y_array[4, :]
        M_Z_CGM_arr = y_array[5, :] / M_sun
        e_CGM_arr = E_CGM_arr / (M_CGM_arr * M_sun + 1e-30)
        T_CGM_arr = (mu / k_B) * e_CGM_arr

    # Compute derived quantities
    Z_CGM_arr = (M_Z_CGM_arr * M_sun) / (M_CGM_arr * M_sun + 1e-30)  # Absolute
    Z_CGM_solar = Z_CGM_arr / 0.0134  # In solar units

    # Convert time to Gyr
    Gyr = params["Gyr"]
    t_Gyr = t_array / Gyr

    result = {
        "time": t_Gyr,
        "redshift": z_array,
        "M_halo": M_halo_arr,
        "M_CGM": M_CGM_arr,
        "M_ISM": M_ISM_arr,
        "M_star": M_star_arr,
        "E_CGM": E_CGM_arr,
        "M_Z_CGM": M_Z_CGM_arr,
        "Z_CGM": Z_CGM_solar,
        "T_CGM": T_CGM_arr,
        "params": params,
        "sol": sol,  # Keep full solution for potential interpolation
    }

    if enable_turb:
        result["E_th"] = E_th_arr
        result["E_kin"] = E_kin_arr
        result["v_turb"] = v_turb_arr

    if enable_CR:
        result["E_CR"] = E_CR_arr

    return result


def run_halo_suite(M_halo_z0_range, z_start, z_end, params=None, rtol=1e-6, atol=1e-6):
    """
    Run a suite of halos with different masses.

    Args:
        M_halo_z0_range: Array of halo masses at z=0 in solar masses
        z_start: Starting redshift
        z_end: Ending redshift
        params: Parameter dictionary (optional)
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver

    Returns:
        List of result dictionaries from run_single_halo
    """
    results = []

    for i, M_halo_z0 in enumerate(M_halo_z0_range):
        print(f"\nRunning halo {i+1}/{len(M_halo_z0_range)}: "
              f"M_halo(z=0) = {M_halo_z0:.2e} Msun")

        try:
            result = run_single_halo(M_halo_z0, z_start, z_end, params, rtol, atol)
            results.append(result)
        except Exception as e:
            print(f"Failed for M_halo = {M_halo_z0:.2e}: {e}")
            results.append(None)

    return results
