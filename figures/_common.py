"""
Shared helpers for the Carr et al. 2023 figure orchestrators.

The three supported model variants (selected by ``--model``):
  - ``carr``     : original Carr+23 thermal-only CGM (6+1 reservoirs)
  - ``pandya``   : Pandya+23 turbulence extension (tracks E_th + E_kin),
                   using Carr timescales (recommended default for the extension)
  - ``pandya-ts``: Pandya turbulence WITH Pandya's NFW free-fall / dissipation
                   timescales (bloats the CGM; opt-in)
  - ``turbcr``   : turbulence + cosmic-ray transport extension
"""
import numpy as np

from gas_regulator import default_params

MODELS = ("carr", "pandya", "pandya-ts", "turbcr")


def build_params(model="carr", eta_M=0.1, eta_E=0.1, eta_Z=0.5,
                 eta_M_beta=0.0, eta_E_lambda=0.0, **extra):
    """Build a parameter dict for the chosen model variant.

    eta_M / eta_E are the normalisations at M_halo = 1e12 Msun; eta_M_beta and
    eta_E_lambda give their (negative) power-law slope with halo mass.
    """
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    p = default_params.copy()
    p["eta_M_norm"], p["eta_M_beta"] = eta_M, eta_M_beta
    p["eta_E_A"], p["eta_E_lambda"] = eta_E, eta_E_lambda
    p["eta_Z"] = eta_Z
    p["enable_turbulence"] = model in ("pandya", "pandya-ts", "turbcr")
    p["enable_cosmic_rays"] = (model == "turbcr")
    p["use_pandya_timescales"] = (model == "pandya-ts")
    p.update(extra)
    return p


def model_label(model):
    return {"carr": "Carr+23 (thermal)",
            "pandya": "Pandya+23 (turbulence)",
            "pandya-ts": "Pandya+23 (turbulence, Pandya timescales)",
            "turbcr": "turbulence + cosmic rays"}[model]


def halo_grid(lo=10.0, hi=12.0, n=8):
    """Log-spaced z=0 halo-mass grid [Msun]."""
    return np.logspace(lo, hi, n)


# Per-figure fiducials. Default start redshift is z=3: Carr et al. do not state their
# start redshift, and z=3 lands on the hot, Carr-consistent CGM branch (z=6 falls into
# the slow, stiff cold branch where the recovered eta_E is superphysical; see the
# cold/hot bistability notes). fig6 uses Carr's recovery setup (eta_M ramp).
FIGURE_DEFAULTS = {
    "fig2": dict(eta_M=0.1, eta_E=0.1, eta_Z=0.5, eta_M_beta=0.0, z_start=3.0),
    "fig3": dict(eta_M=0.1, eta_E=0.1, eta_Z=0.5, eta_M_beta=0.0, z_start=3.0),
    "fig5": dict(eta_M=0.1, eta_E=0.1, eta_Z=0.5, eta_M_beta=0.0, z_start=3.0),
    "fig6": dict(eta_M=1.0, eta_E=0.1, eta_Z=0.5, eta_M_beta=0.5, z_start=3.0),
}


def resolve_loadings(args, figure):
    """Fill any factor left as None (loadings, z_start) with the figure's default."""
    for key, val in FIGURE_DEFAULTS[figure].items():
        if getattr(args, key) is None:
            setattr(args, key, val)
