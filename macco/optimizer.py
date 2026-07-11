"""MACCO-General v0.1: lightweight continuous black-box optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

Array = np.ndarray


@dataclass
class MACCOResult:
    best_x: Array
    best_f: float
    history: Array
    evaluations: int
    iterations: int
    seed: Optional[int]
    restarts: int


def _bounds(lb, ub, dim):
    lo = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    hi = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi <= lo):
        raise ValueError("MACCO requires finite bounds with ub > lb")
    return lo, hi


def _reflect(x, lo, hi):
    span = hi - lo
    y = np.mod(x - lo, 2.0 * span)
    return lo + np.where(y <= span, y, 2.0 * span - y)


def _evaluate(objective, population):
    values = np.empty(population.shape[0])
    for i, x in enumerate(population):
        try:
            value = float(objective(x))
        except (FloatingPointError, OverflowError, ValueError):
            value = np.inf
        values[i] = value if np.isfinite(value) else np.inf
    return values


def _rank_weights(cost, pressure):
    order = np.argsort(cost, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(cost.size)
    logits = -pressure * ranks / max(cost.size - 1, 1)
    logits -= logits.max()
    weights = np.exp(np.clip(logits, -700, 0))
    return weights / weights.sum()


def minimize(
    objective: Callable[[Array], float], dim: int,
    lb: Sequence[float] | float, ub: Sequence[float] | float, *,
    population_size: int = 40, max_evaluations: int = 20_000,
    seed: Optional[int] = None, scout_fraction: float = 0.20,
    polish_fraction: float = 0.10, consensus_pressure: float = 6.0,
    stagnation_iterations: int = 25, restart_fraction: float = 0.0,
    callback=None,
) -> MACCOResult:
    """Minimize a bounded continuous objective under a strict evaluation budget.

    The algorithm uses rank-weighted consensus development, lightweight
    differential scouts, one simple stagnation restart, and diagonal local
    polishing. Internal time and memory complexity are O(BD) and O(ND).
    """
    if dim < 1 or population_size < 8:
        raise ValueError("dim must be positive and population_size at least 8")
    if max_evaluations < 2 * population_size:
        raise ValueError("evaluation budget is too small")
    if not 0.05 <= scout_fraction <= 0.50:
        raise ValueError("scout_fraction must be in [0.05, 0.50]")
    if not 0.02 <= polish_fraction <= 0.30:
        raise ValueError("polish_fraction must be in [0.02, 0.30]")
    if not 0 <= restart_fraction <= 0.30:
        raise ValueError("restart_fraction must be in [0, 0.30]")

    lo, hi = _bounds(lb, ub, dim)
    span = hi - lo
    rng = np.random.default_rng(seed)
    pop = lo + rng.random((population_size, dim)) * span
    cost = _evaluate(objective, pop)
    evaluations = population_size
    idx = int(np.argmin(cost))
    best_x, best_f = pop[idx].copy(), float(cost[idx])
    history = [best_f]
    variance = np.ones(dim)
    mean_step = 0.18
    iteration = restarts = no_improvement = 0
    main_limit = int(max_evaluations * (1 - polish_fraction))
    main_limit -= main_limit % population_size

    while evaluations + population_size <= main_limit:
        iteration += 1
        progress = evaluations / max(main_limit, 1)
        weights = _rank_weights(cost, consensus_pressure)
        consensus = weights @ pop
        elite = pop[np.argsort(cost)[:max(2, population_size // 4)]]
        observed = np.mean(((elite - consensus) / span) ** 2, axis=0)
        floor = 1e-12 + 2e-3 * (1 - progress) ** 3
        variance = 0.85 * variance + 0.15 * np.maximum(observed, floor)

        order = np.argsort(cost)
        n_scout = max(2, int(round(scout_fraction * (1 - 0.6 * progress) * population_size)))
        scouts, developers = order[-n_scout:], order[:-n_scout]
        trial = pop.copy()
        step = mean_step * max(0.03, (1 - progress) ** 0.8)
        noise = rng.standard_normal((developers.size, dim)) * np.sqrt(variance) * span
        attraction = rng.uniform(0.6, 1.5, (developers.size, 1))
        trial[developers] = (pop[developers] + attraction * (consensus - pop[developers])
                             + step * noise)
        for i in scouts:
            pool = np.delete(np.arange(population_size), i)
            a, b, c = rng.choice(pool, 3, replace=False)
            mutant = pop[a] + rng.uniform(0.35, 0.85) * (pop[b] - pop[c])
            cross = rng.random(dim) < 0.8
            cross[rng.integers(dim)] = True
            trial[i] = np.where(cross, mutant, pop[i])

        trial = _reflect(trial, lo, hi)
        trial_cost = _evaluate(objective, trial)
        evaluations += population_size
        improved = trial_cost < cost
        if np.any(improved):
            gain = np.mean((cost[improved] - trial_cost[improved]) /
                           (np.abs(cost[improved]) + 1e-12))
            mean_step *= 1.02 if gain > 1e-3 else 0.98
            mean_step = float(np.clip(mean_step, 1e-4, 0.5))
            pop[improved], cost[improved] = trial[improved], trial_cost[improved]

        idx = int(np.argmin(cost))
        if cost[idx] < best_f:
            best_x, best_f = pop[idx].copy(), float(cost[idx])
            no_improvement = 0
        else:
            no_improvement += 1

        # Single, lightweight diversity recovery: only reset a few worst agents.
        if (no_improvement >= stagnation_iterations and restart_fraction > 0
                and evaluations < main_limit):
            count = min(max(1, int(round(restart_fraction * population_size))),
                        main_limit - evaluations)
            worst = np.argsort(cost)[-count:]
            fresh = lo + rng.random((count, dim)) * span
            fresh_cost = _evaluate(objective, fresh)
            evaluations += count
            pop[worst], cost[worst] = fresh, fresh_cost
            restarts += 1
            no_improvement = 0
        history.append(best_f)
        if callback is not None:
            callback(iteration, best_x.copy(), best_f)

    remaining = max_evaluations - evaluations
    batch = min(max(6, population_size // 4), remaining)
    center, center_f = best_x.copy(), best_f
    sigma, scale = 0.03, np.ones(dim)
    while remaining >= batch and batch > 0:
        noise = rng.standard_normal((batch, dim)) * np.sqrt(scale)
        offspring = _reflect(center + sigma * noise * span, lo, hi)
        values = _evaluate(objective, offspring)
        evaluations += batch
        remaining -= batch
        j = int(np.argmin(values))
        if values[j] < center_f:
            delta = (offspring[j] - center) / span
            scale = 0.9 * scale + 0.1 * np.maximum(delta**2 / max(sigma**2, 1e-30), 1e-12)
            center, center_f = offspring[j].copy(), float(values[j])
            sigma *= 1.03
            if center_f < best_f:
                best_x, best_f = center.copy(), center_f
        else:
            sigma *= 0.88
        sigma = float(np.clip(sigma, 1e-14, 0.2))
        iteration += 1
        history.append(best_f)

    for _ in range(remaining):
        candidate = center.copy()
        j = int(rng.integers(dim))
        candidate[j] += rng.normal() * sigma * span[j]
        candidate = _reflect(candidate, lo, hi)
        value = float(_evaluate(objective, candidate[None])[0])
        evaluations += 1
        if value < center_f:
            center, center_f = candidate, value
            if value < best_f:
                best_x, best_f = candidate.copy(), value
        history.append(best_f)

    return MACCOResult(best_x, best_f, np.asarray(history), evaluations,
                       iteration, seed, restarts)
