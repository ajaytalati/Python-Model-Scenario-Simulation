"""Contiguous broken-watch gap (all channels masked over a date range)."""

from __future__ import annotations

import numpy as np


def apply_broken_watch_gap(
    obs_data: dict,
    channels,
    *,
    n_days: int,
    gap_days: int = 14,
    edge_buffer_days: int = 90,
    seed: int = 42,
    verbose: bool = False,
):
    """Mask a contiguous gap of ``gap_days`` from every channel.

    Parameters
    ----------
    obs_data : dict
    channels : iterable of str
    n_days : int
        Total length of the trajectory in DAYS (or whatever the t_idx unit is).
    gap_days : int
        Length of the gap.
    edge_buffer_days : int
        Don't place the gap within this many days of either end.
    seed : int
    verbose : bool

    Returns
    -------
    obs_data (mutated in place, also returned).
    """
    rng = np.random.default_rng(seed)
    if n_days - gap_days - 2 * edge_buffer_days <= 0:
        if verbose:
            print(f"  no room for {gap_days}-day gap in {n_days} days; skipping")
        return obs_data

    gap_start = int(rng.integers(edge_buffer_days, n_days - gap_days - edge_buffer_days))
    gap_end = gap_start + gap_days
    if verbose:
        print(f"  Broken-watch gap: bins {gap_start}-{gap_end}")

    for ch in channels:
        if ch not in obs_data:
            continue
        d = obs_data[ch]
        if 't_idx' not in d or len(d['t_idx']) == 0:
            continue
        idx = d['t_idx']
        keep = (idx < gap_start) | (idx >= gap_end)
        d['t_idx'] = idx[keep]
        for key in list(d.keys()):
            if key == 't_idx':
                continue
            v = d[key]
            if hasattr(v, '__len__') and len(v) == len(keep):
                d[key] = np.asarray(v)[keep]
    return obs_data
