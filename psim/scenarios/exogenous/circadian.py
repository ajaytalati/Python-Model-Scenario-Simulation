"""Generic circadian forcing C(t) = cos(2π·t_days + φ).

Lifted from `models/fsa_high_res/simulation.py` in the smc2-blackjax-rolling
repo (commit 7621d1e). Phase φ = 0 corresponds to a "healthy morning
chronotype": C peaks at midnight (t_days integer), troughs at noon.

CRITICAL CORRECTNESS NOTE — DO NOT recompute C from window-local time
inside a per-window estimator. Always emit C as an exogenous channel
over the GLOBAL time axis and let `extract_window` slice it. Recomputing
locally re-zeros the time axis at every window start, producing a phase
flip on any window that doesn't begin at midnight (12-hour stride →
every other window inverted). This was the C-phase bug that cost
~12 hours during the high_res_FSA development; see the SMC² repo's
`outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md`.
"""

from __future__ import annotations

import numpy as np


def circadian(t_days, phi: float = 0.0):
    """C(t) = cos(2π·t_days + φ). t_days can be scalar or array.

    With phi=0:
      t = 0 (midnight)  → C = +1
      t = 0.25 (06:00)  → C = 0
      t = 0.5 (noon)    → C = -1
      t = 0.75 (18:00)  → C = 0
    """
    return np.cos(2.0 * np.pi * np.asarray(t_days) + phi)


def make_C_array(n_bins: int, dt_days: float, phi: float = 0.0) -> np.ndarray:
    """Precomputed global C(t) array suitable for use as an exogenous channel.

    Returns shape (n_bins,), float32. Index ``k`` corresponds to global
    time ``k * dt_days`` (in days).
    """
    t_days = np.arange(n_bins, dtype=np.float32) * float(dt_days)
    return circadian(t_days, phi=phi).astype(np.float32)


__all__ = ["circadian", "make_C_array"]
