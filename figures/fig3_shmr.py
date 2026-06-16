"""
Carr et al. 2023, Figure 3 - stellar-to-halo mass relation M*/M_halo vs M_halo,
in three panels showing its sensitivity to the loading factors eta_M, eta_E, eta_Z.

Headline result: M*/M_halo is robust to eta_M and eta_Z but strongly sensitive
to eta_E (energy/preventative feedback is the dominant regulator).
"""
import os
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gas_regulator import run_single_halo
from ._common import build_params, model_label

NAME = "fig3"
DESCRIPTION = "Stellar-to-halo mass relation, sensitivity to eta_M/eta_E/eta_Z"

PANELS = [
    ("eta_M", r"$\eta_M$", [0.1, 1.0, 10.0, 100.0]),
    ("eta_E", r"$\eta_E$", [0.01, 0.1, 0.5, 1.0]),
    ("eta_Z", r"$\eta_Z$", [0.1, 0.5, 1.0]),
]


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def _shmr(masses, fid, vary_key, vary_val, args):
    kw = dict(fid)
    kw[vary_key] = vary_val
    params = build_params(model=args.model, **kw)
    ratio = np.full_like(masses, np.nan)
    for i, Mh in enumerate(masses):
        try:
            with _quiet():
                r = run_single_halo(Mh, args.z_start, 0.0, params=params,
                                    rtol=args.rtol, atol=args.atol, max_step=args.max_step)
            ratio[i] = r["M_star"][-1] / r["M_halo"][-1]
        except Exception as e:
            print(f"  ! {vary_key}={vary_val} M_halo={Mh:.1e} failed: {e}")
    return ratio


def generate(args):
    masses = np.logspace(args.halo_lo, args.halo_hi, args.halo_n)
    fid = dict(eta_M=args.eta_M, eta_E=args.eta_E, eta_Z=args.eta_Z)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (key, label, values) in zip(axes, PANELS):
        print(f"[fig3] panel: varying {key}")
        colors = plt.cm.viridis(np.linspace(0, 0.85, len(values)))
        for val, c in zip(values, colors):
            ratio = _shmr(masses, fid, key, val, args)
            ax.plot(masses, ratio, "-o", color=c, ms=4, label=f"{label}={val:g}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$M_{\rm halo}\ [M_\odot]$")
        ax.set_title(f"varying {label}")
        ax.legend(fontsize=9, frameon=False); ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$M_\star / M_{\rm halo}$")
    fig.suptitle(f"Carr+23 Fig. 3 - SHMR sensitivity  [{model_label(args.model)}]",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(args.out, f"fig3_shmr_{args.model}.png")
    fig.savefig(out, dpi=150)
    print(f"[fig3] saved {out}")
    return out
