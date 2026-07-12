# Changelog

## 0.3.0 - 2026-07-11

- Added `minimize_subspace`, using diagonal-plus-low-rank geometry learned from
  elite principal directions.
- Added classic F1--F23 and shifted/rotated benchmark runners.
- Added packaging metadata, public API documentation, and expanded tests.
- Added a reproducible convergence visualization and README figure.
- Added an equal-budget PSO/GWO/MACCO visual comparison script.
- Added a checkpointed GWO origin/shift/rotation diagnostic experiment.

## 0.2.0 - 2026-07-11

- Disabled random restart by default after paired ablation found no consistent benefit.
- Added shifted/rotated structural benchmarks.
- Added transparent CMA-ES and L-SHADE research baselines.
- Added paired rank/sign analysis and resumable experiment checkpoints.

## 0.1.0 - 2026-07-11

- Initial standalone package with rank consensus, differential scouts, and
  diagonal anisotropic polishing.
