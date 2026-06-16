#!/usr/bin/env python
"""
Single entry point to reproduce the key figures of Carr et al. 2023
(plus the Pandya et al. 2023 turbulence extension and a cosmic-ray extension).

Usage
-----
    python make_figures.py <figure> [options]
    python make_figures.py all      [options]      # generate every figure

Figures
-------
    fig2   single-halo time evolution (masses, energy, metallicity)
    fig3   stellar-to-halo mass relation, sensitivity to eta_M / eta_E / eta_Z
    fig5   CGM mass fraction and metallicity vs halo mass
    fig6   recover eta_E(M_halo) matching the Behroozi 2019 z=0 SHMR

Model variants (--model)
------------------------
    carr        Carr+23 thermal-only CGM (default)
    pandya      Pandya+23 turbulence extension (tracks E_th + E_kin; Carr timescales)
    pandya-ts   Pandya turbulence with Pandya NFW/dissipation timescales (opt-in)
    turbcr      turbulence + cosmic-ray transport extension

Examples
--------
    python make_figures.py fig3 --model carr
    python make_figures.py fig2 --model pandya --eta-E 0.3
    python make_figures.py fig6 --out myplots/
    python make_figures.py all  --model pandya
"""
import argparse
import os
import sys

from figures import FIGURES
from figures._common import MODELS, resolve_loadings


def build_parser():
    p = argparse.ArgumentParser(
        description="Reproduce key figures of Carr et al. 2023 (+ Pandya turbulence).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    choices = list(FIGURES) + ["all"]
    p.add_argument("figure", choices=choices,
                   help="which figure to generate: " +
                        ", ".join(f"{k} ({m.DESCRIPTION})" for k, m in FIGURES.items()))
    p.add_argument("--model", choices=MODELS, default="carr",
                   help="model variant")
    # loadings default to None -> per-figure fiducials (see figures/_common.py)
    p.add_argument("--eta-M", dest="eta_M", type=float, default=None,
                   help="mass loading at 1e12 Msun (default: per-figure fiducial)")
    p.add_argument("--eta-E", dest="eta_E", type=float, default=None,
                   help="energy loading at 1e12 Msun (default: per-figure fiducial)")
    p.add_argument("--eta-Z", dest="eta_Z", type=float, default=None,
                   help="metal loading (default: per-figure fiducial)")
    p.add_argument("--eta-M-beta", dest="eta_M_beta", type=float, default=None,
                   help="eta_M halo-mass slope (default: 0 except fig6 which uses 0.5)")
    # halo grid / single halo
    p.add_argument("--halo-mass", dest="halo_mass", type=float, default=1e12,
                   help="z=0 halo mass [Msun] for single-halo figures (fig2)")
    p.add_argument("--halo-lo", dest="halo_lo", type=float, default=10.0,
                   help="log10 min halo mass for scaling-relation figures")
    p.add_argument("--halo-hi", dest="halo_hi", type=float, default=12.0,
                   help="log10 max halo mass for scaling-relation figures")
    p.add_argument("--halo-n", dest="halo_n", type=int, default=8,
                   help="number of halo-mass points")
    # integration
    p.add_argument("--z-start", dest="z_start", type=float, default=None,
                   help="starting redshift (default: 3; Carr+23 do not state theirs, "
                        "and z=3 lands on the hot, Carr-consistent CGM branch)")
    p.add_argument("--rtol", type=float, default=3e-3,
                   help="ODE rel. tolerance (3e-3 is fine for plots and far faster "
                        "on stiff cold-branch runs)")
    p.add_argument("--atol", type=float, default=3e-3)
    p.add_argument("--max-step", dest="max_step", type=float, default=1e15,
                   help="ODE max step [s] (1e15 ~ fast; 1e14 ~ accurate)")
    p.add_argument("--out", default="figures_out", help="output directory")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    todo = list(FIGURES) if args.figure == "all" else [args.figure]
    # remember which loadings the user left unset so each figure resolves its own
    # per-figure fiducials (resolve_loadings mutates args in place)
    user_set = {k: getattr(args, k)
                for k in ("eta_M", "eta_E", "eta_Z", "eta_M_beta", "z_start")}
    made = []
    for name in todo:
        print(f"\n=== generating {name}: {FIGURES[name].DESCRIPTION} ===")
        for k, v in user_set.items():
            setattr(args, k, v)
        resolve_loadings(args, name)
        made.append(FIGURES[name].generate(args))
    print("\nDone. Wrote:")
    for m in made:
        print(f"  {m}")


if __name__ == "__main__":
    main()
