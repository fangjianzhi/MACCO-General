import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from macco import minimize_subspace


def rastrigin(x):
    return float(10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


result = minimize_subspace(rastrigin, dim=30, lb=-5.12, ub=5.12,
                           population_size=40, max_evaluations=20_000, seed=42)
print("best_f =", result.best_f)
print("best_x =", result.best_x)
print("evaluations =", result.evaluations)
print("restarts =", result.restarts)
