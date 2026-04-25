"""Per-channel random dropout — Bernoulli per-bin."""

from __future__ import annotations

import numpy as np


def apply_dropout(obs_data: dict, channels, rate: float = 0.05, seed: int = 42):
    """Drop a fraction of observations per channel, in place.

    Parameters
    ----------
    obs_data : dict
        Per-channel dict {channel_name: {'t_idx': arr, 'obs_value': arr}}.
    channels : iterable of str
        Channels to apply dropout to. Other channels untouched.
    rate : float in [0, 1)
        Per-bin dropout probability.
    seed : int

    Returns
    -------
    obs_data (mutated in place, also returned for chaining).
    """
    rng = np.random.default_rng(seed)
    for ch in channels:
        if ch not in obs_data:
            continue
        d = obs_data[ch]
        if 't_idx' not in d or len(d['t_idx']) == 0:
            continue
        idx = d['t_idx']
        keep = rng.random(len(idx)) > rate
        d['t_idx'] = idx[keep]
        # Carry through every other field that is bin-aligned with t_idx.
        for key in list(d.keys()):
            if key == 't_idx':
                continue
            v = d[key]
            if hasattr(v, '__len__') and len(v) == len(keep):
                d[key] = np.asarray(v)[keep]
    return obs_data
