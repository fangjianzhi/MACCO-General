"""Experimental MACCO-HLR with success-adaptive low-rank geometry."""

from __future__ import annotations

import numpy as np

from .optimizer import MACCOResult, _bounds, _evaluate, _rank_weights, _reflect


def minimize_hybrid(
    objective, dim, lb, ub, *, population_size=40, max_evaluations=20_000,
    seed=None, scout_fraction=.20, polish_fraction=.10,
    consensus_pressure=6.0, rank=5, low_rank_weight=.55,
    initial_low_rank_probability=.15, adaptation_rate=.15,
    min_low_rank_probability=.05, max_low_rank_probability=.65,
    warmup_fraction=.20, callback=None,
):
    """Minimize using diagonal search plus an online-selected low-rank operator.

    Developer candidates are assigned either diagonal geometry or a
    diagonal-plus-low-rank geometry.  Exponentially smoothed acceptance rates
    determine the probability of using the low-rank operator.  A non-zero
    probability floor keeps occasional probes available after the operator has
    been suppressed.
    """
    if dim < 1 or population_size < 8:
        raise ValueError("invalid dimension or population")
    if max_evaluations < 2 * population_size:
        raise ValueError("evaluation budget is too small")
    if rank < 1:
        raise ValueError("rank must be positive")
    if not 0 <= warmup_fraction <= .5:
        raise ValueError("warmup_fraction must be in [0, 0.5]")
    if not 0 < adaptation_rate <= 1:
        raise ValueError("adaptation_rate must be in (0, 1]")
    if not 0 <= min_low_rank_probability <= initial_low_rank_probability:
        raise ValueError("invalid minimum low-rank probability")
    if not initial_low_rank_probability <= max_low_rank_probability <= 1:
        raise ValueError("invalid maximum low-rank probability")

    lo, hi = _bounds(lb, ub, dim)
    span = hi - lo
    rng = np.random.default_rng(seed)
    pop = lo + rng.random((population_size, dim)) * span
    cost = _evaluate(objective, pop)
    evaluations = population_size
    idx = int(np.argmin(cost))
    best_x, best_f = pop[idx].copy(), float(cost[idx])
    history = [best_f]
    iteration = 0
    variance = np.ones(dim)
    mean_step = .18
    main_limit = int(max_evaluations * (1 - polish_fraction))
    main_limit -= main_limit % population_size
    directions = np.empty((0, dim))
    singular = np.empty(0)
    lr_probability = float(initial_low_rank_probability)
    diag_score = lr_score = .25

    while evaluations + population_size <= main_limit:
        iteration += 1
        progress = evaluations / max(main_limit, 1)
        weights = _rank_weights(cost, consensus_pressure)
        consensus = weights @ pop
        elite_n = max(3, population_size // 4)
        elite_idx = np.argsort(cost)[:elite_n]
        elite_norm = (pop[elite_idx] - consensus) / span
        observed = np.mean(elite_norm ** 2, axis=0)
        floor = 1e-12 + 2e-3 * (1 - progress) ** 3
        variance = .85 * variance + .15 * np.maximum(observed, floor)

        centered = elite_norm - elite_norm.mean(axis=0)
        if np.any(centered):
            _, s, vt = np.linalg.svd(
                centered / np.sqrt(max(elite_n - 1, 1)), full_matrices=False)
            k = min(rank, len(s), dim)
            directions, singular = vt[:k], np.maximum(s[:k], 1e-12)

        order = np.argsort(cost)
        n_scout = max(2, int(round(
            scout_fraction * (1 - .6 * progress) * population_size)))
        scouts, developers = order[-n_scout:], order[:-n_scout]
        trial = pop.copy()
        n_dev = developers.size
        step = mean_step * max(.03, (1 - progress) ** .8)
        diag_noise = rng.standard_normal((n_dev, dim)) * np.sqrt(variance)
        geometry = diag_noise.copy()
        effective_probability = (min_low_rank_probability
                                 if progress < warmup_fraction
                                 else lr_probability)
        lr_mask = np.zeros(n_dev, dtype=bool)
        if len(directions):
            lr_mask = rng.random(n_dev) < effective_probability
            if np.any(lr_mask):
                lr_noise = ((rng.standard_normal((lr_mask.sum(), len(directions)))
                            * singular) @ directions)
                geometry[lr_mask] = ((1 - low_rank_weight) * diag_noise[lr_mask]
                                     + low_rank_weight * lr_noise)
        attraction = rng.uniform(.6, 1.5, (n_dev, 1))
        trial[developers] = (pop[developers]
                             + attraction * (consensus - pop[developers])
                             + step * geometry * span)

        for i in scouts:
            pool = np.delete(np.arange(population_size), i)
            a, b, c = rng.choice(pool, 3, replace=False)
            mutant = pop[a] + rng.uniform(.35, .85) * (pop[b] - pop[c])
            cross = rng.random(dim) < .8
            cross[rng.integers(dim)] = True
            trial[i] = np.where(cross, mutant, pop[i])

        trial = _reflect(trial, lo, hi)
        trial_cost = _evaluate(objective, trial)
        evaluations += population_size
        improved = trial_cost < cost

        dev_improved = improved[developers]
        diag_mask = ~lr_mask
        diag_rate = (float(np.mean(dev_improved[diag_mask]))
                     if np.any(diag_mask) else diag_score)
        lr_rate = (float(np.mean(dev_improved[lr_mask]))
                   if np.any(lr_mask) else lr_score)
        diag_score = ((1 - adaptation_rate) * diag_score
                      + adaptation_rate * diag_rate)
        lr_score = ((1 - adaptation_rate) * lr_score
                    + adaptation_rate * lr_rate)
        if progress >= warmup_fraction:
            # Conservative gate: equal operator success sends low-rank usage
            # back to its probe floor.  Full allocation requires roughly a
            # 50% relative advantage over diagonal search.
            relative_advantage = (lr_score - diag_score) / max(diag_score, .05)
            evidence = float(np.clip(relative_advantage / .5, 0, 1))
            target = (min_low_rank_probability
                      + evidence * (max_low_rank_probability
                                    - min_low_rank_probability))
            lr_probability = float(np.clip(
                (1 - adaptation_rate) * lr_probability + adaptation_rate * target,
                min_low_rank_probability, max_low_rank_probability))

        if np.any(improved):
            gain = np.mean((cost[improved] - trial_cost[improved]) /
                           (np.abs(cost[improved]) + 1e-12))
            mean_step *= 1.02 if gain > 1e-3 else .98
            mean_step = float(np.clip(mean_step, 1e-4, .5))
            pop[improved], cost[improved] = trial[improved], trial_cost[improved]
        idx = int(np.argmin(cost))
        if cost[idx] < best_f:
            best_x, best_f = pop[idx].copy(), float(cost[idx])
        history.append(best_f)
        if callback is not None:
            callback(iteration, best_x.copy(), best_f)

    remaining = max_evaluations - evaluations
    batch = min(max(6, population_size // 4), remaining)
    center, center_f = best_x.copy(), best_f
    sigma, scale = .03, np.ones(dim)
    while remaining >= batch and batch > 0:
        diag = rng.standard_normal((batch, dim)) * np.sqrt(scale)
        noise = diag.copy()
        if len(directions):
            mask = rng.random(batch) < lr_probability
            if np.any(mask):
                lr = ((rng.standard_normal((mask.sum(), len(directions)))
                       * singular) @ directions)
                noise[mask] = ((1 - low_rank_weight) * diag[mask]
                               + low_rank_weight * lr)
        offspring = _reflect(center + sigma * noise * span, lo, hi)
        values = _evaluate(objective, offspring)
        evaluations += batch
        remaining -= batch
        j = int(np.argmin(values))
        if values[j] < center_f:
            delta = (offspring[j] - center) / span
            scale = .9 * scale + .1 * np.maximum(
                delta ** 2 / max(sigma ** 2, 1e-30), 1e-12)
            center, center_f = offspring[j].copy(), float(values[j])
            sigma *= 1.03
            if center_f < best_f:
                best_x, best_f = center.copy(), center_f
        else:
            sigma *= .88
        sigma = float(np.clip(sigma, 1e-14, .2))
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
                       iteration, seed, 0)
