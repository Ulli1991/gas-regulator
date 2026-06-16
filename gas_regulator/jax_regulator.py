"""
Differentiable JAX port of the Carr et al. 2023 gas-regulator model (jax_regulator).

Goal: a fully differentiable, fast forward model so eta_E(M_halo, z; eta_M, eta_Z)
can be recovered against the redshift-resolved Behroozi 2019 SHMR by gradient
descent (and later HMC), replacing the slow/noisy scipy recovery.

Built incrementally and verified piece-by-piece against the scipy implementation
(gas_regulator/physics.py, model.py, wiersma_cooling.py).

This module currently provides:
  - cosmology: analytic flat-LCDM age(z) and z_of_t (differentiable)
  - WiersmaJax: differentiable trilinear interpolation of the cooling tables
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # this node has no GPU; silence cuInit
# Cap thread usage so we don't exceed the per-user thread limit on busy shared nodes
# (XLA otherwise sizes its CPU pool to all cores -> pthread_create failures).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
# XLA sizes its thread pool to the number of available CPUs; on a busy shared node
# with many allowed cores that can exceed the per-cgroup thread/pid budget
# (pthread_create EAGAIN). Cap the affinity to a modest core count.
try:
    _avail = sorted(os.sched_getaffinity(0))
    _cap = int(os.environ.get("JAX_REGULATOR_MAX_CORES", "8"))
    if len(_avail) > _cap:
        os.sched_setaffinity(0, set(_avail[:_cap]))
except (AttributeError, OSError):
    pass
import glob
import numpy as np
import h5py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ---------------------------------------------------------------- constants (CGS)
G = 6.674e-8
K_B = 1.381e-16
M_P = 1.673e-24
MU = 0.6 * M_P
MU_H = (1.0 / 0.75) * M_P
M_SUN = 1.989e33
PARSEC = 3.086e18
YEAR = 3.154e7
GYR = 1e9 * YEAR
Z_SUN = 0.0134

H0 = 70.0
OMEGA_M = 0.3
OMEGA_L = 0.7
H0_CGS = H0 * 1e5 / (1e6 * PARSEC)
F_B = 0.16

# ---------------------------------------------------------------- cosmology
_A_AGE = 2.0 / (3.0 * H0_CGS * jnp.sqrt(OMEGA_L))
_B_AGE = jnp.sqrt(OMEGA_L / OMEGA_M)


def age_of_z(z):
    """Flat-LCDM age of the universe at redshift z, in seconds (differentiable)."""
    return _A_AGE * jnp.arcsinh(_B_AGE * (1.0 + z) ** -1.5)


def z_of_t(t):
    """Inverse: redshift at cosmic time t [s]. Analytic inverse of age_of_z."""
    s = jnp.sinh(t / _A_AGE)
    return (_B_AGE / s) ** (2.0 / 3.0) - 1.0


# ---------------------------------------------------------------- Wiersma cooling (JAX)
class WiersmaJax:
    """Differentiable trilinear interpolation of Wiersma+09 cooling tables.

    Mirrors gas_regulator/wiersma_cooling.py: returns Lambda/n_H^2 [erg cm^3/s]
    with Lambda_net = Lambda_HHe + (Z/Z_sun) * Lambda_metals, He index -3.
    Interpolation in (log10(1+z'), log10 T, log10 nH) on the table grid.
    """

    def __init__(self, table_dir="data/cooling_tables/CoolingTables"):
        files = sorted(glob.glob(os.path.join(table_dir, "z_*.hdf5")))
        files = [f for f in files if all(k not in f for k in
                                         ("nocompton", "collis", "photodis"))]
        if not files:
            raise FileNotFoundError(f"No cooling tables in {table_dir}")
        with h5py.File(files[0], "r") as f:
            T_bins = f["/Total_Metals/Temperature_bins"][:]
            nH_bins = f["/Total_Metals/Hydrogen_density_bins"][:]
        nz, nT, nnH = len(files), len(T_bins), len(nH_bins)
        zs = np.zeros(nz)
        Lm = np.zeros((nz, nT, nnH))
        Lhhe = np.zeros((nz, nT, nnH))
        for i, fn in enumerate(files):
            with h5py.File(fn, "r") as f:
                zs[i] = f["/Header/Redshift"][0]
                Lm[i] = f["/Total_Metals/Net_cooling"][:]
                Lhhe[i] = f["/Metal_free/Net_Cooling"][:][-3, :, :]
        # log-space axes (z axis uses log10(z+1e-3) to match scipy version)
        self.lz = jnp.asarray(np.log10(zs + 1e-3))
        self.lT = jnp.asarray(np.log10(T_bins))
        self.lnH = jnp.asarray(np.log10(nH_bins))
        self.Lm = jnp.asarray(Lm)
        self.Lhhe = jnp.asarray(Lhhe)
        self.T_min, self.T_max = float(T_bins.min()), float(T_bins.max())
        self.nH_min, self.nH_max = float(nH_bins.min()), float(nH_bins.max())

    @staticmethod
    def _interp1_idx(grid, x):
        """Return (i0, frac) for clipped linear interp of x on ascending grid."""
        n = grid.shape[0]
        i1 = jnp.clip(jnp.searchsorted(grid, x), 1, n - 1)
        i0 = i1 - 1
        frac = (x - grid[i0]) / (grid[i1] - grid[i0])
        frac = jnp.clip(frac, 0.0, 1.0)
        return i0, frac

    def _trilinear(self, table, lz, lT, lnH):
        iz, fz = self._interp1_idx(self.lz, lz)
        iT, fT = self._interp1_idx(self.lT, lT)
        iH, fH = self._interp1_idx(self.lnH, lnH)
        c = 0.0
        for dz, wz in ((0, 1 - fz), (1, fz)):
            for dT, wT in ((0, 1 - fT), (1, fT)):
                for dH, wH in ((0, 1 - fH), (1, fH)):
                    c = c + wz * wT * wH * table[iz + dz, iT + dT, iH + dH]
        return c

    def cooling(self, T, nH, Z, z):
        """Lambda/n_H^2 [erg cm^3/s] at scalar (T, nH, Z[abs], z)."""
        lT = jnp.clip(jnp.log10(jnp.maximum(T, self.T_min)), self.lT[0], self.lT[-1])
        lnH = jnp.clip(jnp.log10(jnp.maximum(nH, self.nH_min)), self.lnH[0], self.lnH[-1])
        lz = jnp.clip(jnp.log10(jnp.maximum(z, 1e-3) + 1e-3), self.lz[0], self.lz[-1])
        Lhhe = self._trilinear(self.Lhhe, lz, lT, lnH)
        Lm = self._trilinear(self.Lm, lz, lT, lnH)
        return Lhhe + (Z / Z_SUN) * Lm


# ---------------------------------------------------------------- model params
ALPHA = 1.4
R0_FRAC = 0.1
F_REC = 0.4
T_DEP_NORM = 10 ** 4.92 * 1e9      # years
T_DEP_M = -0.37
T_DEP_Z = -1.5
HALO_NORM = 0.47                   # Gyr^-1
HALO_M = 0.15
HALO_Z = 2.25
Y_SN = 0.033
Z_IGM = 0.01 * Z_SUN
E_SN_PER_MASS = 1e51 / (100 * M_SUN)
ALPHA_PREVENT = 2.0
DELTA_VIR = 200.0
DELTA_VIR_RC = 3.0 / (4.0 * jnp.pi * DELTA_VIR)

# state rescaling so the stiff solver's tolerances behave (all components O(1)+)
_E_SCALE = 1e57
WIERSMA = None  # lazily-loaded global WiersmaJax


def _w():
    global WIERSMA
    if WIERSMA is None:
        WIERSMA = WiersmaJax()
    return WIERSMA


# ---- physics (JAX, mirrors gas_regulator/physics.py) ----
def _H(z):
    return H0_CGS * jnp.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def _rho_crit(z):
    return 3.0 * _H(z) ** 2 / (8.0 * jnp.pi * G)


def _r_vir(M, z):
    return (3.0 * M / (4.0 * jnp.pi * DELTA_VIR * _rho_crit(z))) ** (1.0 / 3.0)


def _T_vir(M, z, rvir):
    return (MU / (2.0 * K_B)) * G * M / rvir


def _t_ff(M, rvir):
    rho_mean = 3.0 * M / (4.0 * jnp.pi * rvir ** 3)
    return jnp.sqrt(3.0 * jnp.pi / (32.0 * G * rho_mean))


def _rho_0(M_CGM, rvir):
    r0 = R0_FRAC * rvir
    x = rvir / r0
    denom = 4.0 * jnp.pi * r0 ** 3 * (x ** (3.0 - ALPHA) - 1.0) / (3.0 - ALPHA)
    return M_CGM / denom


def _cooling_rate(rho0, rvir, T, Z, z):
    r0 = R0_FRAC * rvir
    x = rvir / r0
    r_char = 0.3 * rvir
    rho_char = rho0 * (r_char / r0) ** (-ALPHA)
    nH_char = rho_char / MU_H
    Lam = _w().cooling(T, nH_char, Z, z)
    integ = (x ** (3.0 - 2.0 * ALPHA) - 1.0) / (3.0 - 2.0 * ALPHA)
    dE = 4.0 * jnp.pi * rho0 ** 2 * r0 ** 3 * (Lam / MU_H ** 2) * integ
    return jnp.maximum(dE, 0.0)


def _t_dep(M_star, z):
    Ms = jnp.maximum(M_star, 5e7 * M_SUN)
    return T_DEP_NORM * (Ms / M_SUN) ** T_DEP_M * (1.0 + z) ** T_DEP_Z * YEAR


def _halo_accr(M, z):
    M12 = M / (1e12 * M_SUN)
    return HALO_NORM * M * M12 ** HALO_M * ((1.0 + z) / 3.0) ** HALO_Z / GYR


def _c_s(T):
    return jnp.sqrt(5.0 * K_B * T / (3.0 * MU))


# ---- turbulence / cosmic-ray physics (Pandya+23 extension) ----
def _v_turb(E_kin, M_CGM):
    # +1e-20 floor keeps sqrt differentiable at E_kin=0 (negligible velocity)
    return jnp.sqrt(2.0 * jnp.maximum(E_kin, 0.0) / M_CGM + 1e-20)


def _E_diss(E_kin, M_CGM, R_turb):
    return E_kin * _v_turb(E_kin, M_CGM) / R_turb     # E_kin * v_turb / R_turb


def _nfw_vmax(M, z, rvir, c_NFW):
    Vvir = jnp.sqrt(G * M / rvir)
    Rmax = 2.16 * rvir / c_NFW
    x = Rmax / rvir
    g_c = jnp.log(1.0 + c_NFW) - c_NFW / (1.0 + c_NFW)
    cx = c_NFW * x
    g_cx = jnp.log(1.0 + cx) - cx / (1.0 + cx)
    return Vvir * jnp.sqrt(g_cx / (x * g_c)), Rmax


def _t_ff_eff(M, z, rvir, v_turb, c_NFW, v_CR):
    Vmax, Rmax = _nfw_vmax(M, z, rvir, c_NFW)
    return (Rmax / Vmax) * jnp.sqrt(1.0 + (v_turb ** 2 + v_CR ** 2) / Vmax ** 2)


def _v_CR(E_CR, M_CGM):
    return jnp.sqrt(2.0 * jnp.maximum(E_CR, 0.0) / M_CGM + 1e-20)


def _cr_diff(E_CR, rvir, kappa_CR):
    return jnp.maximum(E_CR, 0.0) / (rvir ** 2 / kappa_CR)


# args tuple shared by all vector fields (precomputed scalars):
#   (eta_M, eta_Z, eta_E0, gamma, f_th_accr, f_th_wind, R_turb_frac, c_NFW,
#    eta_CR0, kappa_CR)
# eta_M, eta_E0 already include their M_halo(z=0) dependence; eta_E = eta_E0*(1+z)^gamma.

def _common_terms(M_halo, M_CGM, M_ISM, M_star, T_CGM, Z_CGM, Z_ISM, z, args):
    """Mass/SF/metal terms common to all three models. Returns a dict."""
    eta_M, eta_Z, eta_E0, gamma = args[0], args[1], args[2], args[3]
    rvir = _r_vir(M_halo, z)
    Tvir = _T_vir(M_halo, z, rvir)
    rho0 = _rho_0(M_CGM, rvir)
    dM_halo = _halo_accr(M_halo, z)
    dE_cool = _cooling_rate(rho0, rvir, T_CGM, Z_CGM, z)
    t_dep = _t_dep(M_star, z)
    dM_SFR = M_ISM / t_dep
    eta_E = eta_E0 * (1.0 + z) ** gamma
    dM_ISM_wind = eta_M * dM_SFR
    cs = _c_s(T_CGM)
    return dict(rvir=rvir, Tvir=Tvir, dM_halo=dM_halo, dE_cool=dE_cool,
                dM_SFR=dM_SFR, eta_E=eta_E, eta_M=eta_M, eta_Z=eta_Z,
                dM_ISM_wind=dM_ISM_wind, cs=cs)


def _metal_derivs(Z_IGM_in_dM, dM_cool, dM_SFR, dM_ISM_wind, dM_CGM_out,
                  Z_CGM, Z_ISM, eta_Z):
    dM_Z_CGM = (Z_IGM * Z_IGM_in_dM - Z_CGM * dM_cool + eta_Z * Y_SN * dM_SFR
                + Z_ISM * dM_ISM_wind - Z_CGM * dM_CGM_out)
    dM_Z_ISM = (Z_CGM * dM_cool + (1.0 - eta_Z) * Y_SN * dM_SFR
                - Z_ISM * (1.0 - F_REC) * dM_SFR - Z_ISM * dM_ISM_wind)
    return dM_Z_CGM, dM_Z_ISM


_VF_CACHE = {}


def make_vector_field(model, use_pandya_ts=False):
    """Build (and cache) a diffrax vector field for 'carr'(7)/'pandya'(8)/'turbcr'(9).

    Cached by (model, use_pandya_ts) so the SAME function object is reused across
    run_single_halo calls -- otherwise diffrax recompiles the solve every call
    (~15s each), which is catastrophic for figure grids.
    """
    key = (model, use_pandya_ts)
    if key in _VF_CACHE:
        return _VF_CACHE[key]

    def vf(t, y, args):
        z = z_of_t(t)
        M_halo = y[0] * M_SUN
        M_CGM = jnp.maximum(y[1], 1e-30) * M_SUN
        M_ISM = jnp.maximum(y[2], 1e-30) * M_SUN
        M_star = jnp.maximum(y[3], 1e-30) * M_SUN
        if model == "carr":
            E_th = jnp.maximum(y[4], 1e-30) * _E_SCALE
            E_kin = 0.0; E_CR = 0.0
            iZc, iZi = 5, 6
        elif model == "pandya":
            E_th = jnp.maximum(y[4], 1e-30) * _E_SCALE
            E_kin = jnp.maximum(y[5], 0.0) * _E_SCALE
            E_CR = 0.0; iZc, iZi = 6, 7
        else:  # turbcr
            E_th = jnp.maximum(y[4], 1e-30) * _E_SCALE
            E_kin = jnp.maximum(y[5], 0.0) * _E_SCALE
            E_CR = jnp.maximum(y[6], 0.0) * _E_SCALE
            iZc, iZi = 7, 8
        M_Z_CGM = jnp.maximum(y[iZc], 0.0) * M_SUN
        M_Z_ISM = jnp.maximum(y[iZi], 0.0) * M_SUN
        E_total = E_th + E_kin + E_CR
        e_total = E_total / M_CGM
        T_CGM = (MU / K_B) * (E_th / M_CGM)
        Z_CGM = M_Z_CGM / M_CGM
        Z_ISM = M_Z_ISM / M_ISM

        c = _common_terms(M_halo, M_CGM, M_ISM, M_star, T_CGM, Z_CGM, Z_ISM, z, args)
        rvir, Tvir, dM_halo, dE_cool = c["rvir"], c["Tvir"], c["dM_halo"], c["dE_cool"]
        dM_SFR, eta_E, eta_M, eta_Z = c["dM_SFR"], c["eta_E"], c["eta_M"], c["eta_Z"]
        dM_ISM_wind, cs = c["dM_ISM_wind"], c["cs"]

        f_th_wind, R_turb_frac, c_NFW = args[5], args[6], args[7]
        eta_CR0, kappa_CR, f_th_accr = args[8], args[9], args[4]

        # turbulence / CR support quantities
        R_turb = R_turb_frac * rvir
        v_turb = _v_turb(E_kin, M_CGM) if model != "carr" else 0.0
        E_diss = _E_diss(E_kin, M_CGM, R_turb) if model != "carr" else 0.0
        v_CR = _v_CR(E_CR, M_CGM) if model == "turbcr" else 0.0

        # cooling time / free-fall (Carr vs Pandya timescales)
        E_for_cool = E_th
        if use_pandya_ts and model != "carr":
            tff = _t_ff_eff(M_halo, z, rvir, v_turb, c_NFW, v_CR)
            net = dE_cool - E_diss
            t_cool_eff = E_for_cool / jnp.maximum(net, E_for_cool * 1e-20) + tff
        else:
            tff = _t_ff(M_halo, rvir)
            t_cool_eff = E_for_cool / jnp.maximum(dE_cool, E_for_cool * 1e-20) + tff
        dM_cool = M_CGM / t_cool_eff

        # overpressurization outflow (uses total energy)
        E_excess = jnp.maximum(E_total - K_B * Tvir * M_CGM / MU, 0.0)
        dE_out_total = E_excess * cs / rvir
        dM_CGM_out = dE_out_total / e_total

        # preventive inflow
        dE_in_raw = (K_B * Tvir / MU) * F_B * dM_halo
        dE_out_safe = jnp.maximum(dE_out_total, 1e-12 * dE_in_raw + 1e-300)
        f_prevent = jnp.minimum(ALPHA_PREVENT * dE_in_raw / dE_out_safe, 1.0)
        dM_CGM_in = F_B * f_prevent * dM_halo
        dE_in_total = (K_B * Tvir / MU) * dM_CGM_in

        # mass + metal derivatives (shared)
        dM_CGM = dM_CGM_in - dM_cool + dM_ISM_wind - dM_CGM_out
        dM_ISM = -dM_SFR * (1.0 + eta_M - F_REC) + dM_cool
        dM_star = (1.0 - F_REC) * dM_SFR
        dM_Z_CGM, dM_Z_ISM = _metal_derivs(dM_CGM_in, dM_cool, dM_SFR, dM_ISM_wind,
                                           dM_CGM_out, Z_CGM, Z_ISM, eta_Z)

        if model == "carr":
            dE_CGM = (dE_in_total - dE_cool + eta_E * dM_SFR * E_SN_PER_MASS
                      - dE_out_total)
            d = [dM_halo, dM_CGM, dM_ISM, dM_star, dE_CGM, dM_Z_CGM, dM_Z_ISM]
            scales = [M_SUN, M_SUN, M_SUN, M_SUN, _E_SCALE, M_SUN, M_SUN]
        else:
            f_th_CGM = E_th / jnp.maximum(E_total, 1e-30)
            dE_wind_total = eta_E * dM_SFR * E_SN_PER_MASS
            dE_wind_th = f_th_wind * dE_wind_total
            dE_wind_kin = (1.0 - f_th_wind) * dE_wind_total
            dE_out_th = f_th_CGM * dE_out_total
            dE_in_th = f_th_accr * dE_in_total
            dE_in_kin = (1.0 - f_th_accr) * dE_in_total
            if model == "pandya":
                dE_out_kin = (1.0 - f_th_CGM) * dE_out_total
                dE_th = dE_in_th - dE_cool + E_diss + dE_wind_th - dE_out_th
                dE_kin = dE_in_kin - E_diss + dE_wind_kin - dE_out_kin
                d = [dM_halo, dM_CGM, dM_ISM, dM_star, dE_th, dE_kin, dM_Z_CGM, dM_Z_ISM]
                scales = [M_SUN, M_SUN, M_SUN, M_SUN, _E_SCALE, _E_SCALE, M_SUN, M_SUN]
            else:  # turbcr
                f_kin_CGM = E_kin / jnp.maximum(E_total, 1e-30)
                f_CR_CGM = E_CR / jnp.maximum(E_total, 1e-30)
                dE_out_kin = f_kin_CGM * dE_out_total
                dE_out_CR = f_CR_CGM * dE_out_total
                dE_CR_diff = _cr_diff(E_CR, rvir, kappa_CR)
                dE_wind_CR = eta_CR0 * dM_SFR * E_SN_PER_MASS
                dE_th = dE_in_th - dE_cool + E_diss + dE_wind_th - dE_out_th
                dE_kin = dE_in_kin - E_diss + dE_wind_kin - dE_out_kin
                dE_CR = dE_wind_CR - dE_CR_diff - dE_out_CR
                d = [dM_halo, dM_CGM, dM_ISM, dM_star, dE_th, dE_kin, dE_CR,
                     dM_Z_CGM, dM_Z_ISM]
                scales = [M_SUN, M_SUN, M_SUN, M_SUN, _E_SCALE, _E_SCALE, _E_SCALE,
                          M_SUN, M_SUN]
        return jnp.array([di / si for di, si in zip(d, scales)])

    _VF_CACHE[key] = vf
    return vf


def initial_state(M_halo_init, z_start, model):
    """Scaled initial state for the chosen model (mirrors model.initial_conditions)."""
    M_CGM = jnp.minimum(1e6 * M_SUN, F_B * M_halo_init * 0.05)
    M_ISM = jnp.minimum(1e6 * M_SUN, F_B * M_halo_init * 0.01)
    M_star = jnp.minimum(1e6 * M_SUN, F_B * M_halo_init * 0.001)
    rvir = _r_vir(M_halo_init, z_start)
    Tvir = _T_vir(M_halo_init, z_start, rvir)
    E_th = (K_B * Tvir / MU) * M_CGM
    base = [M_halo_init / M_SUN, M_CGM / M_SUN, M_ISM / M_SUN, M_star / M_SUN,
            E_th / _E_SCALE]
    metals = [(Z_IGM * M_CGM) / M_SUN, (Z_IGM * M_ISM) / M_SUN]
    if model == "carr":
        return jnp.array(base + metals)
    if model == "pandya":
        return jnp.array(base + [0.0] + metals)        # + E_kin
    return jnp.array(base + [0.0, 0.0] + metals)        # + E_kin, E_CR


def _H_of_z(z):
    return H0_CGS * jnp.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_L)


def _halo_dMdz(zz, M, _):
    """dM/dz for the backward halo extrapolation (module-level so it isn't
    re-created each call -> diffrax reuses the compiled solve)."""
    M12 = M / (1e12 * M_SUN)
    dMdt = HALO_NORM * M * M12 ** HALO_M * ((1.0 + zz) / 3.0) ** HALO_Z / GYR
    return dMdt * (-1.0 / ((1.0 + zz) * _H_of_z(zz)))


def extrapolate_halo_mass(M_halo_z0, z_start):
    """Backward-extrapolate halo mass from z=0 to z_start (g). Matches scipy version:
    integrate dM/dz = (dM/dt)(dt/dz), dt/dz = -1/((1+z) H(z))."""
    import diffrax
    if z_start <= 0:
        return M_halo_z0
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_halo_dMdz), diffrax.Dopri5(), t0=0.0, t1=z_start,
        dt0=z_start * 1e-3, y0=M_halo_z0,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1.0),
        max_steps=100000)
    return float(sol.ys[-1])


def _args_from_params(params, M_halo_z0_cgs):
    eta_M = params["eta_M_norm"] * (M_halo_z0_cgs / (1e12 * M_SUN)) ** (-params["eta_M_beta"])
    eta_E0 = params["eta_E_A"] * (M_halo_z0_cgs / (1e12 * M_SUN)) ** (-params["eta_E_lambda"])
    gamma = params.get("eta_E_gamma", 0.0)
    return (eta_M, params["eta_Z"], eta_E0, gamma,
            params.get("f_thermal_accretion", 0.5), params.get("f_thermal_wind", 0.5),
            params.get("R_turb_fraction", 0.5), params.get("c_NFW", 10.0),
            params.get("eta_CR", 0.1), params.get("kappa_CR", 3e28))


def _select_model(params):
    if params.get("enable_cosmic_rays", False):
        return "turbcr"
    if params.get("enable_turbulence", False):
        return "pandya"
    return "carr"


def run_single_halo(M_halo_z0, z_start, z_end=0.0, params=None, n_save=60,
                    rtol=1e-6, atol=1e-6, max_step=None, M_halo_init=None,
                    max_steps=15000):
    """JAX forward run. Returns a result dict (numpy) compatible with the figures."""
    import diffrax
    from .parameters import default_params
    if params is None:
        params = default_params.copy()
    _w()  # force-load cooling tables before tracing

    model = _select_model(params)
    use_ts = params.get("use_pandya_timescales", False)
    M_halo_z0_cgs = M_halo_z0 * M_SUN
    if M_halo_init is None:
        M_halo_init = extrapolate_halo_mass(M_halo_z0_cgs, z_start)
    else:
        M_halo_init = M_halo_init * M_SUN

    # pass args as a JAX array (TRACED) so different loading values reuse the
    # compiled solve; python floats would be treated as static -> recompile each call
    args = jnp.asarray(_args_from_params(params, M_halo_z0_cgs))
    y0 = initial_state(M_halo_init, z_start, model)
    vf = make_vector_field(model, use_ts)

    t0, t1 = float(age_of_z(z_start)), float(age_of_z(z_end))
    save_z = np.linspace(z_start, z_end, n_save)
    ts = np.array([float(age_of_z(z)) for z in save_z])
    controller = diffrax.PIDController(
        rtol=rtol, atol=atol,
        **({"dtmax": max_step} if max_step else {}))
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Kvaerno5(), t0=t0, t1=t1,
        dt0=(t1 - t0) * 1e-3, y0=y0, args=args,
        saveat=diffrax.SaveAt(ts=ts), stepsize_controller=controller,
        max_steps=max_steps)
    ys = np.asarray(sol.ys)

    M_halo = ys[:, 0]; M_CGM = ys[:, 1]; M_ISM = ys[:, 2]; M_star = ys[:, 3]
    if model == "carr":
        E_th = ys[:, 4] * _E_SCALE; E_kin = None; E_CR = None
        M_Z_CGM, M_Z_ISM = ys[:, 5], ys[:, 6]
        E_CGM = ys[:, 4] * _E_SCALE
    elif model == "pandya":
        E_th = ys[:, 4] * _E_SCALE; E_kin = ys[:, 5] * _E_SCALE; E_CR = None
        M_Z_CGM, M_Z_ISM = ys[:, 6], ys[:, 7]
        E_CGM = E_th + E_kin
    else:
        E_th = ys[:, 4] * _E_SCALE; E_kin = ys[:, 5] * _E_SCALE; E_CR = ys[:, 6] * _E_SCALE
        M_Z_CGM, M_Z_ISM = ys[:, 7], ys[:, 8]
        E_CGM = E_th + E_kin + E_CR

    M_CGM_g = np.maximum(M_CGM, 1e-30) * M_SUN
    T_CGM = (MU / K_B) * (E_th / M_CGM_g)
    out = dict(
        time=ts / GYR, redshift=save_z, M_halo=M_halo, M_CGM=M_CGM, M_ISM=M_ISM,
        M_star=M_star, E_CGM=E_CGM, T_CGM=T_CGM,
        Z_CGM=(M_Z_CGM / np.maximum(M_CGM, 1e-30)) / Z_SUN,
        Z_ISM=(M_Z_ISM / np.maximum(M_ISM, 1e-30)) / Z_SUN,
        M_Z_CGM=M_Z_CGM, M_Z_ISM=M_Z_ISM, model=model, params=params)
    if E_kin is not None:
        out["E_th"] = E_th; out["E_kin"] = E_kin
        out["v_turb"] = np.sqrt(2.0 * np.maximum(E_kin, 0) / M_CGM_g)
    if E_CR is not None:
        out["E_CR"] = E_CR
    return out


if __name__ == "__main__":
    # Verify cosmology vs astropy and cooling vs scipy reader.
    from astropy.cosmology import FlatLambdaCDM
    c = FlatLambdaCDM(H0=70, Om0=0.3)
    print("age(z) JAX vs astropy [Gyr]:")
    for z in (0.0, 1.0, 3.0, 6.0):
        print(f"  z={z}: {float(age_of_z(z))/GYR:.4f} vs {float(c.age(z).value):.4f}")
    print("z_of_t round-trip:", [round(float(z_of_t(age_of_z(z))), 4) for z in (0., 1., 3., 6.)])

    w = WiersmaJax()
    print("\nLambda/n_H^2 [erg cm^3/s] (T, nH=1e-4, Z, z):")
    for T in (1e5, 1e6, 3e6):
        for Z in (0.0, 0.3 * Z_SUN):
            for z in (0.0, 2.0):
                print(f"  T={T:.0e} Z={Z/Z_SUN:.1f} z={z}: "
                      f"{float(w.cooling(T, 1e-4, Z, z)):.4e}")

    print("\nForward run (1e12 halo, z=6->0, Carr defaults):")
    r = run_single_halo(1e12, 6.0, 0.0, n_save=20)
    print(f"  M_star={r['M_star'][-1]:.3e}  M_CGM={r['M_CGM'][-1]:.3e}  "
          f"Z_CGM={r['Z_CGM'][-1]:.3f}")
