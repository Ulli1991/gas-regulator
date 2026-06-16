"""
Carr et al. 2023, Figure 6 - recover the energy loading eta_E(M_halo) that makes
the model reproduce the Behroozi et al. 2019 z=0 stellar-to-halo mass relation, via
per-halo bisection. Compared against Carr's best-fit eta_E = 0.065 (M/1e12)^-0.60.

By default uses Carr's setup: mass-dependent eta_M (norm=1, beta=0.5 -> eta_M=10 at
1e10, 1 at 1e12) and eta_Z=0.5.
"""
import os
import contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from gas_regulator import run_single_halo
from gas_regulator import behroozi19 as b19
from ._common import build_params, model_label

NAME = "fig6"
DESCRIPTION = "Recover eta_E(M_halo) matching the Behroozi 2019 z=0 SHMR"


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


def generate(args):
    masses = np.logspace(args.halo_lo, args.halo_hi, args.halo_n)
    # defaults resolved by _common.resolve_loadings (fig6 -> eta_M=1, beta=0.5)
    eta_M, eta_M_beta, eta_Z = args.eta_M, args.eta_M_beta, args.eta_Z

    # looser tolerance for the (many) bisection evals; the recovery doesn't need
    # high precision and the high-eta_E cold-branch solves are stiff/slow otherwise
    rtol = max(args.rtol, 3e-3)

    def logMstar(eta_E, Mh, log_target):
        params = build_params(model=args.model, eta_M=eta_M,
                              eta_M_beta=eta_M_beta, eta_E=eta_E, eta_Z=eta_Z)
        try:
            with _quiet():
                r = run_single_halo(Mh, args.z_start, 0.0, params=params,
                                    rtol=rtol, atol=rtol, max_step=args.max_step)
        except Exception:
            return 1.0   # stiff cold-branch run didn't converge -> treat as over-producing
        return np.log10(max(r["M_star"][-1], 1.0)) - log_target

    eta_E_star = np.full_like(masses, np.nan)
    hit_bound = np.zeros(len(masses), dtype=bool)
    lo, hi = 1e-3, 10.0   # eta_E > 1 is already superphysical; cap to keep it bounded
    for i, Mh in enumerate(masses):
        tgt = float(b19.log10_Mstar(np.log10(Mh), 0.0))
        flo, fhi = logMstar(lo, Mh, tgt), logMstar(hi, Mh, tgt)
        if flo < 0:
            eta_E_star[i] = lo
        elif fhi > 0:
            eta_E_star[i] = hi; hit_bound[i] = True   # model can't reach Behroozi
        else:
            # coarse tolerance / few iters: plenty for a log-scale plot and keeps
            # stiff cold-branch halos from dominating the runtime
            eta_E_star[i] = brentq(logMstar, lo, hi, args=(Mh, tgt),
                                   xtol=0.04, rtol=0.04, maxiter=15)
        print(f"[fig6] M_halo={Mh:.2e}  eta_E*={eta_E_star[i]:.4f}"
              + ("  (hit cap)" if hit_bound[i] else ""))

    carr = 0.065 * (masses / 1e12) ** (-0.60)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(masses, eta_E_star, "o-", color="navy",
              label=rf"recovered $\eta_E^\ast$ [{model_label(args.model)}]")
    ax.loglog(masses, carr, "--", color="crimson",
              label=r"Carr+23 fit: $0.065\,(M/10^{12})^{-0.60}$")
    ax.set_xlabel(r"$M_{\rm halo}(z=0)\ [M_\odot]$")
    ax.set_ylabel(r"$\eta_E^\ast$")
    ax.set_title(f"Carr+23 Fig. 6 - recovered energy loading "
                 f"(eta_M_norm={eta_M}, beta={eta_M_beta}, eta_Z={eta_Z})")
    ax.legend(frameon=False); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(args.out, f"fig6_etaE_{args.model}.png")
    fig.savefig(out, dpi=150)
    print(f"[fig6] saved {out}")
    return out
