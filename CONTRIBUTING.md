# Contributing

1. Open an issue describing the problem, expected behavior, and evidence.
2. Keep the public `minimize` API backward compatible where practical.
3. Add deterministic tests for every behavior change.
4. Compare algorithms under equal objective-evaluation budgets and common seeds.
5. Do not tune on the reporting suite and then report the same suite as unbiased evidence.
6. Run `python -m unittest discover -s tests -v` before submitting changes.

Performance claims must include raw per-run results, seeds, dimensions, bounds,
evaluation budgets, baseline versions, and statistical analysis.
