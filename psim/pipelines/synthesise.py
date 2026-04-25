"""End-to-end scenario synthesis: forward-sim + obs + missing data.

Given an ``SDEModel``, truth parameters, an initial state, sub-daily
exogenous arrays, and a rolling time grid, produce a complete
``SimRun`` containing the latent trajectory, all per-channel
observations (with proper gating + missing-data corruption applied),
and the global exogenous arrays.

The output ``SimRun`` is immediately packageable via
``psim.pipelines.package.package_scenario`` into the canonical
artifact format that the SMC² repo consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

import numpy as np


@dataclass
class SimRun:
    """In-memory bundle of one synthesised scenario."""

    trajectory: np.ndarray                     # (n_bins, n_state)
    obs_channels: Dict[str, dict]              # per-channel dicts
    exogenous_channels: Dict[str, dict]        # T_B, Phi, C
    truth_params: Dict[str, float]
    init_state: Dict[str, float]
    n_bins_total: int
    dt_days: float
    bins_per_day: int
    seed: int
    state_names: list = field(default_factory=list)


def integrate_sde_numpy(
    drift_fn: Callable,
    diffusion_diagonal: np.ndarray,
    noise_scale_fn: Callable,
    *,
    init_state: np.ndarray,
    aux: tuple,
    truth_params: Dict[str, float],
    n_bins: int,
    dt: float,
    n_substeps: int,
    state_bounds: list,
    seed: int,
) -> np.ndarray:
    """Generic Euler-Maruyama on a uniform bin grid.

    Operates per-bin with ``n_substeps`` integration steps inside each.
    Clips state to ``state_bounds`` after each substep.
    """
    rng = np.random.default_rng(seed)
    n_state = len(init_state)
    sub_dt = dt / n_substeps
    sqrt_sub_dt = np.sqrt(sub_dt)
    sigma = np.asarray(diffusion_diagonal, dtype=np.float64)

    y = np.asarray(init_state, dtype=np.float64).copy()
    traj = np.zeros((n_bins, n_state), dtype=np.float64)

    for k in range(n_bins):
        t_days = k * dt
        for _ in range(n_substeps):
            dy = drift_fn(t_days, y, truth_params, aux)
            ns = noise_scale_fn(y, truth_params)
            z = rng.standard_normal(n_state)
            y = y + sub_dt * dy + sqrt_sub_dt * sigma * ns * z
            for i, (lo, hi) in enumerate(state_bounds):
                y[i] = max(lo, min(hi, y[i]))
            t_days += sub_dt
        traj[k] = y
    return traj


def synthesise_scenario(
    model_sim,                          # SDEModel-like object
    *,
    truth_params: Dict[str, float],
    init_state: Dict[str, float],
    exogenous_arrays: Dict[str, np.ndarray],   # {'T_B_arr': ..., 'Phi_arr': ..., ...}
    n_bins_total: int,
    dt_days: float,
    bins_per_day: int,
    n_substeps: int = 4,
    seed: int = 42,
    state_bounds: list | None = None,
) -> SimRun:
    """Forward-integrate the model and run all of its observation channels.

    Uses ``model_sim.drift_fn`` and ``model_sim.noise_scale_fn`` for
    the SDE integration, then iterates over ``model_sim.channels``
    in dependency order to produce all observations (sleep-gated HR,
    Bernoulli sleep, etc.).

    Returns a ``SimRun``. Missing-data corruption is applied separately
    by the caller (see ``psim.scenarios.missing_data``).
    """
    if state_bounds is None:
        state_bounds = [(s.lo, s.hi) for s in model_sim.states]
    state_names = [s.name for s in model_sim.states]

    aux = model_sim.make_aux_fn(
        truth_params, init_state, None,
        exogenous_arrays,
    )
    init_arr = model_sim.make_y0_fn(init_state, truth_params)

    trajectory = integrate_sde_numpy(
        drift_fn=model_sim.drift_fn,
        diffusion_diagonal=model_sim.diffusion_fn(truth_params),
        noise_scale_fn=model_sim.noise_scale_fn,
        init_state=init_arr,
        aux=aux,
        truth_params=truth_params,
        n_bins=n_bins_total,
        dt=dt_days,
        n_substeps=n_substeps,
        state_bounds=state_bounds,
        seed=seed,
    )

    # Run channel DAG via the model's per-channel generate_fn.
    t_grid = np.arange(n_bins_total, dtype=np.float32) * dt_days
    obs_channels: Dict[str, dict] = {}
    exogenous_channels: Dict[str, dict] = {}

    for ch in model_sim.channels:
        prior = {name: obs_channels[name]
                 for name in ch.depends_on if name in obs_channels}
        out = ch.generate_fn(trajectory, t_grid, truth_params, aux,
                              prior, seed + 100)
        # Heuristic: name starts with 'obs_' → observation channel;
        # otherwise (T_B / Phi / C / ...) → exogenous broadcast.
        if ch.name.startswith("obs_"):
            obs_channels[ch.name] = out
        else:
            exogenous_channels[ch.name] = out

    return SimRun(
        trajectory=trajectory,
        obs_channels=obs_channels,
        exogenous_channels=exogenous_channels,
        truth_params=dict(truth_params),
        init_state=dict(init_state),
        n_bins_total=n_bins_total,
        dt_days=dt_days,
        bins_per_day=bins_per_day,
        seed=seed,
        state_names=state_names,
    )


__all__ = ["SimRun", "integrate_sde_numpy", "synthesise_scenario"]
