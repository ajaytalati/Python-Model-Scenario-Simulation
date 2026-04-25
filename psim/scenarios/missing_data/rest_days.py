"""Weekly rest-day mask (active channels masked on rest days)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def apply_rest_days(
    obs_data: dict,
    active_channels,
    *,
    n_days: int,
    bins_per_day: int = 1,
    rest_days_per_week: Tuple[int, int] = (2, 3),
    seed: int = 42,
):
    """Mask active-measurement channels on randomly-chosen rest days.

    Parameters
    ----------
    obs_data : dict
    active_channels : iterable of str
        Channels to mask on rest days (typically: intensity, duration,
        timing — not RHR/stress/sleep which continue passively).
    n_days : int
        Total number of days in the trajectory.
    bins_per_day : int
        Bins per day (1 for daily-grid models, 96 for 15-min, etc.).
    rest_days_per_week : (lo, hi)
        Inclusive range of rest days per week.
    seed : int

    Returns
    -------
    obs_data (mutated in place, also returned).
    """
    rng = np.random.default_rng(seed)
    lo, hi = rest_days_per_week

    rest_mask = np.ones(n_days, dtype=bool)
    for week_start in range(0, n_days, 7):
        week_end = min(week_start + 7, n_days)
        week_len = week_end - week_start
        n_rest = min(int(rng.integers(lo, hi + 1)), week_len)
        rest_days = rng.choice(week_len, size=n_rest, replace=False) + week_start
        rest_mask[rest_days] = False

    # Convert day-mask to bin-mask
    if bins_per_day > 1:
        bin_mask = np.repeat(rest_mask, bins_per_day)
    else:
        bin_mask = rest_mask

    for ch in active_channels:
        if ch not in obs_data:
            continue
        d = obs_data[ch]
        if 't_idx' not in d or len(d['t_idx']) == 0:
            continue
        idx = d['t_idx']
        keep = bin_mask[idx]
        d['t_idx'] = idx[keep]
        for key in list(d.keys()):
            if key == 't_idx':
                continue
            v = d[key]
            if hasattr(v, '__len__') and len(v) == len(keep):
                d[key] = np.asarray(v)[keep]
    return obs_data
