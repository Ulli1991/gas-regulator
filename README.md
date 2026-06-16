# gas-regulator

A clean implementation of the **Carr et al. 2023** gas-regulator model for the
regulation of star formation by a hot circumgalactic medium (CGM), together with
the **Pandya et al. 2023** turbulence extension and a cosmic-ray transport extension.

The model is a 1D system of coupled ODEs tracking the exchange of mass, energy and
metals between reservoirs:

| reservoir | symbol |
|---|---|
| dark-matter halo | `M_halo` |
| circumgalactic medium | `M_CGM` |
| interstellar medium | `M_ISM` |
| stars | `M_star` |
| CGM energy (thermal; + kinetic with turbulence) | `E_CGM` / `E_th`, `E_kin` |
| CGM metal mass | `M_Z_CGM` |
| ISM metal mass | `M_Z_ISM` |

Key physics: cosmological halo accretion, metallicity-dependent radiative cooling
(Wiersma et al. 2009 tables), star formation with a depletion-time law, SN winds
with mass/energy/metal loading (`eta_M`, `eta_E`, `eta_Z`), preventive feedback from
an over-pressurised CGM, and metal enrichment of both the ISM and CGM.

## Install

```bash
pip install -e .
```

The forward model is implemented in **JAX/diffrax** (fully differentiable). Requires
`jax`, `diffrax`, `numpy`, `h5py`, `matplotlib`, `scipy` (the fig6 bisection), and
`astropy` (optional, validation only). You also need the Wiersma+09 cooling tables
under `data/cooling_tables/CoolingTables/z_*.hdf5`.

On busy shared nodes the module caps its CPU affinity to 8 cores (override with
`JAX_REGULATOR_MAX_CORES`) so XLA's thread pool stays within the per-cgroup limit.

## Reproduce the key figures

A single entry point generates the key figures of Carr et al. 2023:

```bash
python make_figures.py <figure> [options]
```

| figure | content |
|---|---|
| `fig2` | single-halo time evolution (masses, energy, metallicity) |
| `fig3` | stellar-to-halo mass relation, sensitivity to `eta_M` / `eta_E` / `eta_Z` |
| `fig5` | CGM mass fraction and metallicity vs halo mass |
| `fig6` | recover `eta_E(M_halo)` matching the Behroozi 2019 z=0 SHMR |
| `all`  | generate every figure |

### Model variants (`--model`)

| value | model |
|---|---|
| `carr` | Carr+23 thermal-only CGM (default) |
| `pandya` | Pandya+23 turbulence extension — tracks `E_th` + `E_kin` (Carr timescales) |
| `pandya-ts` | Pandya turbulence with Pandya's NFW free-fall / dissipation timescales |
| `turbcr` | turbulence + cosmic-ray transport |

### Examples

```bash
python make_figures.py fig3 --model carr
python make_figures.py fig2 --model pandya --eta-E 0.3
python make_figures.py fig6 --out myplots/
python make_figures.py all  --model pandya
python make_figures.py --help        # all options
```

Common options: `--eta-M --eta-E --eta-Z` (loading factors), `--halo-mass`
(single-halo figures), `--halo-lo --halo-hi --halo-n` (scaling-relation grid),
`--z-start`, `--cooling {wiersma,simple}`, `--out`. Figures are written to
`figures_out/` by default.

## Use the model directly

```python
from gas_regulator import run_single_halo, default_params
p = default_params.copy()
p.update(eta_M_norm=0.1, eta_E_A=0.1, eta_Z=0.5, use_wiersma_cooling=True)
r = run_single_halo(M_halo_z0=1e12, z_start=6.0, z_end=0.0, params=p)
print(r["M_star"][-1], r["Z_CGM"][-1])     # z=0 stellar mass, CGM metallicity
```

## Repository layout

```
make_figures.py          single CLI entry point
figures/                 per-figure orchestrators (fig2/3/5/6) + shared helpers
gas_regulator/           the model library
  jax_regulator.py         the differentiable JAX/diffrax model (Carr / Pandya / +CR)
  parameters.py            constants and default parameters (CGS)
  behroozi19.py            Behroozi+19 stellar-mass--halo-mass relation
data/cooling_tables/     Wiersma+09 cooling tables (not distributed)
```

## The model (`gas_regulator/jax_regulator.py`)

The forward model is a single JAX/diffrax implementation covering all three variants
(Carr thermal, Pandya turbulence, +CR). It integrates the stiff ODE system with a
Kvaerno solver and is **fully differentiable**: `jax.grad` gives exact gradients of
the z=0 observables with respect to the loading factors, enabling gradient-based
recovery of `eta_E(M_halo, z)` against observed scaling relations. `run_single_halo`
returns the same result dict the figures consume.

## Example figures

Generated with `python make_figures.py all --model carr`:

| | |
|---|---|
| ![fig2](figures_out/fig2_evolution_carr.png) | ![fig3](figures_out/fig3_shmr_carr.png) |
| **Fig. 2** single-halo evolution | **Fig. 3** SHMR sensitivity to η_M/η_E/η_Z |
| ![fig5](figures_out/fig5_cgm_carr.png) | ![fig6](figures_out/fig6_etaE_carr.png) |
| **Fig. 5** CGM mass & metallicity | **Fig. 6** recovered η_E(M_halo) vs Behroozi |

## References

- Carr, Bryan, Fielding, Pandya & Somerville 2023, ApJ 949, 21 (arXiv:2211.05115)
- Pandya et al. 2023, ApJ 956, 118 (arXiv:2211.09755)
- Behroozi et al. 2019, MNRAS 488, 3143 (UniverseMachine)
