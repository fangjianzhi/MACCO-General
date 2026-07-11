"""Experimental MACCO-DHLR with delayed-credit geometry selection."""

from __future__ import annotations

import numpy as np

from .optimizer import MACCOResult, _bounds, _evaluate, _rank_weights, _reflect


def minimize_delayed_hybrid(
    objective, dim, lb, ub, *, population_size=40, max_evaluations=20_000,
    seed=None, scout_fraction=.20, polish_fraction=.10,
    consensus_pressure=6.0, rank=5, low_rank_weight=.55,
    credit_window=6, direction_update_interval=3,
    low_probability=.05, high_probability=.80,
    exploration_probability=.12, credit_rate=.25, callback=None,
):
    """Delayed-credit hybrid of diagonal and low-rank search geometries.

    A geometry mode is held for ``credit_window`` generations.  Its credit is
    based on both delayed best-so-far improvement and elite-quantile movement,
    rather than one-step candidate acceptance.  Initial alternating windows
    provide an online A/B probe; later windows select the better mode with a
    small persistent exploration probability.
    """
    if dim < 1 or population_size < 8:
        raise ValueError("invalid dimension or population")
    if max_evaluations < 2 * population_size:
        raise ValueError("evaluation budget is too small")
    if rank < 1 or credit_window < 2 or direction_update_interval < 1:
        raise ValueError("invalid rank, credit window, or update interval")
    if not 0 <= low_probability < high_probability <= 1:
        raise ValueError("invalid geometry probabilities")
    if not 0 <= exploration_probability <= .5 or not 0 < credit_rate <= 1:
        raise ValueError("invalid exploration probability or credit rate")

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

    # Mode 0 protects diagonal behavior; mode 1 reinforces low-rank geometry.
    mode = 0
    window_index = 0
    mode_credit = np.full(2, .1)
    window_start_best = best_f
    elite_count = max(3, population_size // 4)
    window_start_elite = float(np.median(np.sort(cost)[:elite_count]))
    window_scale = max(float(np.median(np.abs(cost - np.median(cost)))),
                       abs(window_start_elite - best_f), 1e-12)

    while evaluations + population_size <= main_limit:
        iteration += 1
        progress = evaluations / max(main_limit, 1)
        weights = _rank_weights(cost, consensus_pressure)
        consensus = weights @ pop
        elite_idx = np.argsort(cost)[:elite_count]
        elite_norm = (pop[elite_idx] - consensus) / span
        observed = np.mean(elite_norm ** 2, axis=0)
        floor = 1e-12 + 2e-3 * (1 - progress) ** 3
        variance = .85 * variance + .15 * np.maximum(observed, floor)

        # Reuse directions between updates: lower overhead and less jitter.
        if (iteration == 1 or iteration % direction_update_interval == 0):
            centered = elite_norm - elite_norm.mean(axis=0)
            if np.any(centered):
                _, s, vt = np.linalg.svd(
                    centered / np.sqrt(max(elite_count - 1, 1)),
                    full_matrices=False)
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
        probability = high_probability if mode else low_probability
        if len(directions):
            lr_mask = rng.random(n_dev) < probability
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

        if iteration % credit_window == 0:
            elite_now = float(np.median(np.sort(cost)[:elite_count]))
            best_gain = max(0., window_start_best - best_f) / window_scale
            elite_gain = max(0., window_start_elite - elite_now) / window_scale
            reward = float(np.clip(best_gain + .25 * elite_gain, 0, 10))
            mode_credit[mode] = ((1 - credit_rate) * mode_credit[mode]
                                 + credit_rate * reward)
            window_index += 1
            # Four forced alternating windows establish delayed evidence.
            if window_index < 4:
                mode = window_index % 2
            elif rng.random() < exploration_probability:
                mode = 1 - int(np.argmax(mode_credit))
            else:
                mode = int(np.argmax(mode_credit))
            window_start_best = best_f
            window_start_elite = elite_now
            window_scale = max(
                float(np.median(np.abs(cost - np.median(cost)))),
                abs(window_start_elite - best_f), 1e-12)

    remaining = max_evaluations - evaluations
    batch = min(max(6, population_size // 4), remaining)
    center, center_f = best_x.copy(), best_f
    sigma, scale = .03, np.ones(dim)
    final_probability = high_probability if mode_credit[1] > mode_credit[0] else low_probability
    while remaining >= batch and batch > 0:
        diag = rng.standard_normal((batch, dim)) * np.sqrt(scale)
        noise = diag.copy()
        if len(directions):
            mask = rng.random(batch) < final_probability
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
