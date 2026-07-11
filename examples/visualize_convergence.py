"""Generate the convergence figure shown in the project README."""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macco import minimize, minimize_subspace
from advanced_benchmarks import make_advanced_suite


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install plotting support with: pip install -e .[plot]") from exc

    objective, lb, ub = make_advanced_suite(30)["rot_rastrigin"]
    common = dict(dim=30, lb=lb, ub=ub, population_size=40,
                  max_evaluations=20_000)
    seeds = range(20260711, 20260721)
    base = np.vstack([minimize(objective, seed=seed, **common).history
                      for seed in seeds])
    subspace = np.vstack([minimize_subspace(objective, seed=seed, **common).history
                          for seed in seeds])

    def draw(histories, label, color):
        median = np.median(histories, axis=0)
        lower, upper = np.quantile(histories, [.25, .75], axis=0)
        steps = np.arange(median.size)
        plt.semilogy(steps, np.maximum(median, 1e-16), label=label,
                     color=color, linewidth=2)
        plt.fill_between(steps, np.maximum(lower, 1e-16),
                         np.maximum(upper, 1e-16), color=color, alpha=.15)

    plt.figure(figsize=(8, 4.8))
    draw(base, "MACCO base", "tab:blue")
    draw(subspace, "MACCO subspace", "tab:orange")
    plt.xlabel("Recorded search step")
    plt.ylabel("Best objective value (log scale)")
    plt.title("30-D rotated Rastrigin convergence (10 paired seeds)")
    plt.grid(True, which="both", alpha=.25)
    plt.legend()
    plt.tight_layout()
    output = Path(__file__).resolve().parents[1] / "docs" / "assets" / "convergence_demo.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
