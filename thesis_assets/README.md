# Thesis figure assets

This directory contains a Git-sized, thesis-ready selection of figures copied byte-for-byte from retained experiment output directories. It makes `thesis_draft.tex` self-contained without committing multi-gigabyte checkpoint, array, and per-sample result trees.

The selection covers:

- the main raw absolute/delta diffusion comparison;
- decoded GASF trajectories, rejection filtering, and road guidance;
- OSM coverage and coordinate-offset diagnostics;
- legacy DDPM sampler/constraint comparisons;
- GAF channel structure and learned-inverse diagnostics;
- oracle map refinement and drift analysis;
- Week 05 bounce/truncate displacement distributions.

The original repository paths and recommended thesis placement for these figures are documented in `figure_inventory.md`. The eight figures included directly in the LaTeX draft use the stable paths in `thesis_assets/figures/`; the remaining files are available for later chapter editing and appendices.

These images are retained evidence. They should not be edited in place. Any publication-quality re-export should be added as a new file with its provenance recorded.
