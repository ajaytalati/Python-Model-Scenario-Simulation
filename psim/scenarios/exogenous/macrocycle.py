"""Daily training-load macrocycle generators.

Lifted from `drivers/fsa_real_obs_5yr_rolling.py::generate_macrocycle_C0`
in the smc2-blackjax-rolling repo (commit 7621d1e). Generic over n_days
and seed; produces (T_B_daily, Phi_daily) of shape (n_days,).

Three excitation conditions studied in the daily FSA experiments:
  C0 (baseline): 28-day mesocycles + overreach every 90d + taper every 180d
  C2 (strong):   28d base + 35-day deep tapers every 90d + 21d overreach
  C3 (maximal):  75-day repeating cycles (30d moderate + 30d taper + 15d overreach)

Cross-seed robustness study (`outputs/robustness_check_report.md` in the
SMC² repo): C2 ≈ C0 on average; C3 reliably worse. C0 is the safest
default for new model porting.
"""

from __future__ import annotations

import numpy as np


def generate_macrocycle_C0(n_days: int, seed: int = 42):
    """Baseline daily training-load schedule.

    28-day mesocycles (3 weeks load + 1 week deload), overlaid with
    overreach spikes every 90 days (14d duration) and off-season tapers
    every 180 days (21d duration).

    Returns
    -------
    T_B_daily : np.ndarray (n_days,) — adaptation target per day
    Phi_daily : np.ndarray (n_days,) — strain production per day
    """
    rng = np.random.default_rng(seed)
    T_B = np.zeros(n_days, dtype=np.float32)
    Phi = np.zeros(n_days, dtype=np.float32)

    # 1) Base 28-day mesocycles
    for block_start in range(0, n_days, 28):
        block_end = min(block_start + 28, n_days)
        load_T_B = rng.uniform(0.6, 0.85)
        load_Phi = rng.uniform(0.08, 0.15)
        deload_T_B = rng.uniform(0.3, 0.5)
        deload_Phi = rng.uniform(0.01, 0.04)
        for d in range(block_start, block_end):
            day_in_block = d - block_start
            base_T_B, base_Phi = (
                (load_T_B, load_Phi) if day_in_block < 21
                else (deload_T_B, deload_Phi)
            )
            T_B[d] = base_T_B * (1.0 + 0.1 * rng.standard_normal())
            Phi[d] = base_Phi * (1.0 + 0.1 * rng.standard_normal())

    # 2) Off-season tapers (every 180d, 21d duration)
    taper_period, taper_duration = 180, 21
    for taper_start in range(taper_period - taper_duration, n_days, taper_period):
        for d in range(taper_start, min(taper_start + taper_duration, n_days)):
            T_B[d] = rng.uniform(0.15, 0.30)
            Phi[d] = rng.uniform(0.01, 0.02)

    # 3) Overreach spikes (every 90d, 14d duration; skip if overlaps taper)
    overreach_period, overreach_duration = 90, 14
    for or_start in range(overreach_period - overreach_duration, n_days,
                          overreach_period):
        overlaps_taper = any(
            (or_start < ts + taper_duration
             and or_start + overreach_duration > ts)
            for ts in range(taper_period - taper_duration, n_days, taper_period)
        )
        if overlaps_taper:
            continue
        for d in range(or_start, min(or_start + overreach_duration, n_days)):
            T_B[d] = rng.uniform(0.80, 0.95)
            Phi[d] = rng.uniform(0.20, 0.25)

    T_B = np.clip(T_B, 0.05, 0.95)
    Phi = np.clip(Phi, 0.005, 0.25)
    return T_B, Phi


__all__ = ["generate_macrocycle_C0"]
