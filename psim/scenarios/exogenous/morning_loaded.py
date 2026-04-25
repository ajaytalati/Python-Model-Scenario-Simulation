"""Morning-loaded sub-daily Φ(t) generator (hunter-gatherer activity profile).

Lifted from `models/fsa_high_res/simulation.py::generate_phi_sub_daily`
in the smc2-blackjax-rolling repo (commit 7621d1e). Generic over the
bin grid, wake/sleep hours, and decay timescale.

Profile: Gamma(k=2)-shape `t · exp(-t/τ)` over waking hours, peaking
at t ≈ τ post-wake (~10am for the default 7am wake + τ=3h). Tapers
exponentially through afternoon and evening; zero during sleep hours.
Daily integral is normalised so coarser daily-grid models see the
same per-day load as the input ``daily_phi``.

This replaced an earlier "training spike + sedentary plateau" design
when the user pointed out that the plateau distorts the wake/sleep
diurnal pattern (it kept Φ > 0 throughout waking hours).
"""

from __future__ import annotations

import numpy as np


def generate_morning_loaded_phi(
    daily_phi: np.ndarray,
    *,
    bins_per_day: int = 96,
    wake_hour: float = 7.0,
    sleep_hour: float = 23.0,
    tau_hours: float = 3.0,
    noise_frac: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """Expand a per-day Phi schedule into a per-bin morning-loaded array.

    Parameters
    ----------
    daily_phi : (n_days,)
        One Φ value per day (typically from a macrocycle generator).
    bins_per_day : int
        15-min bins → 96; hourly → 24; etc.
    wake_hour, sleep_hour : float
        Hours of day in [0, 24). Sleep wraps midnight if sleep_hour > wake_hour.
    tau_hours : float
        Time-of-peak post-wake (the gamma profile's mode). Default 3h
        ⇒ peak at ~10am for the default 7am wake.
    noise_frac : float
        Multiplicative Gaussian noise std on each bin (0.0 disables).
    seed : int

    Returns
    -------
    phi : (n_days * bins_per_day,) float32
    """
    rng = np.random.default_rng(seed)
    n_days = len(daily_phi)
    dt_hours = 24.0 / bins_per_day
    phi = np.zeros(n_days * bins_per_day, dtype=np.float32)

    wake_duration = sleep_hour - wake_hour      # e.g. 16h
    # ∫₀^T t·exp(-t/τ) dt = τ² · (1 - exp(-T/τ)·(1 + T/τ))
    T = wake_duration
    gamma_integral = tau_hours ** 2 * (
        1.0 - np.exp(-T / tau_hours) * (1.0 + T / tau_hours)
    )

    for d in range(n_days):
        phi_d = float(daily_phi[d])
        amplitude = phi_d * 24.0 / max(gamma_integral, 1e-12)
        for k in range(bins_per_day):
            h = k * dt_hours
            if h < wake_hour or h >= sleep_hour:
                phi[d * bins_per_day + k] = 0.0
                continue
            t = h - wake_hour
            base = amplitude * t * np.exp(-t / tau_hours)
            noise = rng.normal(0.0, noise_frac) if noise_frac > 0 else 0.0
            phi[d * bins_per_day + k] = max(base * (1.0 + noise), 0.0)

    return phi


__all__ = ["generate_morning_loaded_phi"]
