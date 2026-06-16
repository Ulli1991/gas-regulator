"""
Carr et al. 2023, Figure 2 - time evolution of a single halo: reservoir masses,
CGM energy vs virial energy, and CGM metallicity, from z_start to z=0.
"""
import os
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gas_regulator import run_single_halo
from ._common import build_params, model_label

NAME = "fig2"
DESCRIPTION = "Single-halo time evolution (masses, energy, metallicity)"


def generate(args):
    params = build_params(model=args.model,
                          eta_M=args.eta_M, eta_E=args.eta_E, eta_Z=args.eta_Z)
    print(f"[fig2] {model_label(args.model)}  M_halo(z=0)={args.halo_mass:.2e}  "
          f"eta_M={args.eta_M} eta_E={args.eta_E} eta_Z={args.eta_Z}")
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        r = run_single_halo(args.halo_mass, args.z_start, 0.0, params=params,
                            rtol=args.rtol, atol=args.atol, max_step=args.max_step)
    t = r["time"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].plot(t, r["M_halo"], "k-", label=r"$M_{\rm halo}$")
    ax[0].plot(t, r["M_CGM"], "-", label=r"$M_{\rm CGM}$")
    ax[0].plot(t, r["M_ISM"], "-", label=r"$M_{\rm ISM}$")
    ax[0].plot(t, r["M_star"], "-", label=r"$M_\star$")
    ax[0].set_yscale("log"); ax[0].set_xlabel("time [Gyr]")
    ax[0].set_ylabel(r"$M\ [M_\odot]$"); ax[0].legend(frameon=False); ax[0].grid(alpha=0.3)

    ax[1].plot(t, r["E_CGM"], "-", label=r"$E_{\rm CGM}$")
    ax[1].set_yscale("log"); ax[1].set_xlabel("time [Gyr]")
    ax[1].set_ylabel(r"$E_{\rm CGM}\ [{\rm erg}]$"); ax[1].grid(alpha=0.3)
    if "E_th" in r:
        ax[1].plot(t, r["E_th"], "--", label=r"$E_{\rm th}$")
        ax[1].plot(t, r["E_kin"], ":", label=r"$E_{\rm kin}$")
    ax[1].legend(frameon=False)

    ax[2].plot(t, r["Z_CGM"], "-", label=r"$Z_{\rm CGM}$")
    if "Z_ISM" in r:
        ax[2].plot(t, r["Z_ISM"], "--", label=r"$Z_{\rm ISM}$")
    ax[2].set_xlabel("time [Gyr]"); ax[2].set_ylabel(r"$Z\ [Z_\odot]$")
    ax[2].legend(frameon=False); ax[2].grid(alpha=0.3)

    fig.suptitle(f"Carr+23 Fig. 2 - single-halo evolution  [{model_label(args.model)}]",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(args.out, f"fig2_evolution_{args.model}.png")
    fig.savefig(out, dpi=150)
    print(f"[fig2] saved {out}  (z=0: M*={r['M_star'][-1]:.2e}, "
          f"M_CGM={r['M_CGM'][-1]:.2e}, Z_CGM={r['Z_CGM'][-1]:.3f})")
    return out
