"""Carr et al. 2023 figure orchestrators, dispatched by ../make_figures.py."""
from . import fig2_evolution, fig3_shmr, fig5_cgm, fig6_etaE

# registry: name -> module (each exposes NAME, DESCRIPTION, generate(args))
FIGURES = {m.NAME: m for m in (fig2_evolution, fig3_shmr, fig5_cgm, fig6_etaE)}
