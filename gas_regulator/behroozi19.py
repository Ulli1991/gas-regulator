"""
Behroozi et al. 2019 (UniverseMachine, MNRAS 488, 3143) stellar-mass--halo-mass
(SMHM) relation.

This is a verbatim re-implementation of the official `gen_smhm.py` released with
the DR1 data product, using the "true intrinsic median, all galaxies" best-fit
parameters (data/smhm/params/smhm_true_med_params.txt, overall chi^2 = 114.96).

Functional form (median):
    a   = 1/(1+z),  a1 = a-1,  lna = ln(a)
    m1    = M_1   + a1*M_1_A   - lna*M_1_A2   + z*M_1_Z
    sm0   = m1 + EFF_0 + a1*EFF_0_A - lna*EFF_0_A2 + z*EFF_0_Z
    alpha = ALPHA + a1*ALPHA_A - lna*ALPHA_A2 + z*ALPHA_Z
    beta  = BETA  + a1*BETA_A  + z*BETA_Z
    delta = DELTA
    gamma = 10**(GAMMA + a1*GAMMA_A + z*GAMMA_Z)
    dm = log10(Mpeak) - m1
    log10(M*) = sm0 - log10(10**(-alpha*dm) + 10**(-beta*dm))
                    + gamma*exp(-0.5*(dm/delta)**2)

NOTE on halo-mass definition: Behroozi's Mpeak is the *peak historical* halo mass
using the Bryan & Norman (1998) virial overdensity. The gas-regulator model uses
M_halo(z) with Delta_vir = 200. For z=0 central halos the difference between
Mpeak and current M_vir is small (a few %); treat this as a first-order caveat.
"""
import numpy as np

# True intrinsic median, all galaxies (smhm_true_med_params.txt, DR1)
P = {
    "EFF_0":   -1.430476,  "EFF_0_A":  1.795813,  "EFF_0_A2": 1.359576,  "EFF_0_Z": -0.2156067,
    "M_1":     12.04003,   "M_1_A":    4.675185,  "M_1_A2":   4.513113,  "M_1_Z":   -0.7444014,
    "ALPHA":    1.973063,  "ALPHA_A": -2.353400,  "ALPHA_A2":-1.783277,  "ALPHA_Z":  0.1860354,
    "BETA":     0.4732459, "BETA_A":  -0.8842523, "BETA_Z":  -0.4861040,
    "DELTA":    0.4067526,
    "GAMMA":   -1.087851,  "GAMMA_A": -3.241419,  "GAMMA_Z": -1.078538,
}


def _zparams(z):
    a = 1.0 / (1.0 + z)
    a1 = a - 1.0
    lna = np.log(a)
    m1 = P["M_1"] + a1 * P["M_1_A"] - lna * P["M_1_A2"] + z * P["M_1_Z"]
    sm0 = m1 + P["EFF_0"] + a1 * P["EFF_0_A"] - lna * P["EFF_0_A2"] + z * P["EFF_0_Z"]
    alpha = P["ALPHA"] + a1 * P["ALPHA_A"] - lna * P["ALPHA_A2"] + z * P["ALPHA_Z"]
    beta = P["BETA"] + a1 * P["BETA_A"] + z * P["BETA_Z"]
    delta = P["DELTA"]
    gamma = 10.0 ** (P["GAMMA"] + a1 * P["GAMMA_A"] + z * P["GAMMA_Z"])
    return m1, sm0, alpha, beta, delta, gamma


def log10_Mstar(log10_Mhalo, z):
    """Median log10(M*/Msun) given log10(Mpeak/Msun) and redshift z."""
    m1, sm0, alpha, beta, delta, gamma = _zparams(z)
    dm = np.asarray(log10_Mhalo, dtype=float) - m1
    dm2 = dm / delta
    return sm0 - np.log10(10.0 ** (-alpha * dm) + 10.0 ** (-beta * dm)) \
        + gamma * np.exp(-0.5 * dm2 * dm2)


def Mstar(M_halo, z):
    """Median M* [Msun] given halo mass [Msun] and redshift z."""
    return 10.0 ** log10_Mstar(np.log10(M_halo), z)


def shmr(M_halo, z):
    """Median M*/M_halo given halo mass [Msun] and redshift z."""
    return Mstar(M_halo, z) / np.asarray(M_halo, dtype=float)


if __name__ == "__main__":
    # Sanity check vs the official tool's z=0 output at a few masses.
    for lm in (10.5, 11.0, 12.0, 13.0):
        print(f"log Mh={lm:5.2f}  log M*={float(log10_Mstar(lm, 0.0)):7.4f}  "
              f"M*/Mh={float(shmr(10**lm, 0.0)):.4e}")
