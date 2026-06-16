"""
Carr et al. 2023, Figure 5 - CGM mass fraction M_CGM/(f_b M_halo) (top row) and
CGM metallicity Z_CGM (bottom row) vs halo mass, each varying eta_M / eta_E / eta_Z.
"""
import os
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gas_regulator import run_single_halo
from ._common import build_params, model_label

NAME = "fig5"
DESCRIPTION = "CGM mass fraction and metallicity vs halo mass"

PANELS = [
    ("eta_M", r"$\eta_M$", [0.1, 1.0, 10.0, 100.0]),
    ("eta_E", r"$\eta_E$", [0.01, 0.1, 0.5, 1.0]),
    ("eta_Z", r"$\eta_Z$", [0.1, 0.5, 1.0]),
]
F_B = 0.16


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def _run(masses, fid, key, val, args):
    kw = dict(fid); kw[key] = val
    params = build_params(model=args.model, **kw)
    fcgm = np.full_like(masses, np.nan)
    zcgm = np.full_like(masses, np.nan)
    for i, Mh in enumerate(masses):
        try:
            with _quiet():
                r = run_single_halo(Mh, args.z_start, 0.0, params=params,
                                    rtol=args.rtol, atol=args.atol, max_step=args.max_step)
            fcgm[i] = r["M_CGM"][-1] / (F_B * Mh)
            zcgm[i] = r["Z_CGM"][-1]
        except Exception as e:
            print(f"  ! {key}={val} M_halo={Mh:.1e} failed: {e}")
    return fcgm, zcgm


def generate(args):
    masses = np.logspace(args.halo_lo, args.halo_hi, args.halo_n)
    fid = dict(eta_M=args.eta_M, eta_E=args.eta_E, eta_Z=args.eta_Z)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True)
    for col, (key, label, values) in enumerate(PANELS):
        print(f"[fig5] column: varying {key}")
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(values)))
        for val, c in zip(values, colors):
            fcgm, zcgm = _run(masses, fid, key, val, args)
            axes[0, col].plot(masses, fcgm, "-o", color=c, ms=4, label=f"{label}={val:g}")
            axes[1, col].plot(masses, zcgm, "-o", color=c, ms=4)
        axes[0, col].set_xscale("log"); axes[0, col].set_yscale("log")
        axes[0, col].set_title(f"varying {label}")
        axes[0, col].legend(fontsize=9, frameon=False); axes[0, col].grid(alpha=0.3, which="both")
        axes[1, col].set_xscale("log"); axes[1, col].grid(alpha=0.3, which="both")
        axes[1, col].set_xlabel(r"$M_{\rm halo}\ [M_\odot]$")
    axes[0, 0].set_ylabel(r"$M_{\rm CGM}/(f_b M_{\rm halo})$")
    axes[1, 0].set_ylabel(r"$Z_{\rm CGM}\ [Z_\odot]$")
    fig.suptitle(f"Carr+23 Fig. 5 - CGM mass & metallicity  [{model_label(args.model)}]",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out, f"fig5_cgm_{args.model}.png")
    fig.savefig(out, dpi=150)
    print(f"[fig5] saved {out}")
    return out
