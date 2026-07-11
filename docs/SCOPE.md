# Scope and current evidence

MACCO-General targets bounded, deterministic or low-noise, continuous
black-box optimization. The base optimizer is a lightweight general research
tool. The subspace variant targets rotated/non-separable problems where useful
joint motion can be approximated by a small elite subspace.

Known limitations:

- deceptive remote-optimum landscapes such as Schwefel;
- very high-dimensional strongly multimodal landscapes;
- severely ill-conditioned landscapes may favor mature full-covariance methods;
- discrete, combinatorial, constrained, and noisy optimization are not yet supported.

Version 0.3 is an alpha research release. The included baseline implementations are
transparent validation aids, not substitutes for authors' official reference
implementations in publication-quality comparisons.
