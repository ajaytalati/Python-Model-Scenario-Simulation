"""End-to-end round-trip check: sim → align → propagate → recover.

Higher-level than the per-call consistency checks in
``psim.validation.consistency``: this runs a single window through the
estimator's full per-step machinery (``align_obs_fn`` + ``propagate_fn``)
at the truth parameters with zero noise, and asserts that the recovered
state trajectory matches the simulator's true trajectory at every bin.

A round-trip pass is the strongest pre-SMC² guarantee that data flows
correctly across the boundary. A round-trip fail localises the bug to
the per-step PF call (vs the obs predictions, which Check B catches).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np


@dataclass
class RoundTripResult:
    passed: bool
    max_abs_state_err: float = 0.0
    n_steps_compared: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def assert_pass(self):
        if not self.passed:
            raise AssertionError(
                f"Round-trip check FAILED: max|state_err| = "
                f"{self.max_abs_state_err:.3e} over "
                f"{self.n_steps_compared} steps.\n"
                f"details: {self.details}"
            )


def round_trip_check(
    *,
    true_trajectory: np.ndarray,
    align_obs_fn: Callable,
    propagate_fn: Callable,
    obs_data: dict,
    window_start: int,
    window_end: int,
    dt: float,
    est_params_vec: np.ndarray,
    init_state: np.ndarray,
    diffusion_diag: np.ndarray,
    atol: float = 5e-2,
) -> RoundTripResult:
    """Re-integrate the estimator's PF over a window with zero noise
    starting from the simulator's true initial state, and verify the
    recovered states match the simulator's trajectory.

    Parameters
    ----------
    true_trajectory : (n_bins_global, n_state) — simulator's full trajectory
    align_obs_fn : the model's ``align_obs_fn``
    propagate_fn : the model's JAX ``propagate_fn``
    obs_data : full obs dict (will be window-extracted)
    window_start, window_end : window bounds in bin index
    dt : time-step for the propagator
    est_params_vec : params in the estimator's flat-vector form
    init_state : (n_state,) — simulator's true state at window_start
    diffusion_diag : (n_state,) — sigma_diag for propagate_fn
    atol : per-state absolute tolerance

    Returns
    -------
    RoundTripResult.

    Notes
    -----
    Uses zero-noise integration so any deterministic mismatch between
    the simulator's drift and the estimator's drift will surface as
    a state error. Adding observation noise back in does not help
    diagnose; it only obscures.
    """
    import jax
    import jax.numpy as jnp

    # Extract window from obs_data (lightweight reimplementation)
    window_obs = {}
    for ch, d in obs_data.items():
        new_d = {}
        if 't_idx' in d:
            t_idx = np.asarray(d['t_idx'])
            mask = (t_idx >= window_start) & (t_idx < window_end)
            new_d['t_idx'] = t_idx[mask] - window_start
            for key, v in d.items():
                if key == 't_idx':
                    continue
                if hasattr(v, '__len__') and len(v) == len(mask):
                    new_d[key] = np.asarray(v)[mask]
                else:
                    new_d[key] = v
        else:
            new_d = dict(d)
        window_obs[ch] = new_d

    n_steps = window_end - window_start
    grid_obs = align_obs_fn(window_obs, n_steps, dt)

    # Integrate from init_state with zero noise
    y = jnp.asarray(init_state, dtype=jnp.float64)
    sigma = jnp.asarray(diffusion_diag, dtype=jnp.float64)
    zero_noise = jnp.zeros_like(y)
    params = jnp.asarray(est_params_vec, dtype=jnp.float64)

    recovered = np.zeros((n_steps, len(init_state)))
    recovered[0] = np.asarray(y)
    for k in range(1, n_steps):
        y_next, _lw = propagate_fn(y, k * dt, dt, params, grid_obs, k,
                                    sigma, zero_noise, None)
        y = y_next
        recovered[k] = np.asarray(y)

    true_window = true_trajectory[window_start:window_end]
    err = np.abs(recovered - true_window)
    max_err = float(err.max())
    passed = max_err < atol

    return RoundTripResult(
        passed=passed,
        max_abs_state_err=max_err,
        n_steps_compared=n_steps,
        details={
            "window_start": window_start,
            "window_end": window_end,
            "atol": atol,
            "per_state_max_err": err.max(axis=0).tolist(),
            "first_diverging_step": int(np.argmax(np.any(err > atol, axis=1)))
                if max_err > atol else -1,
        },
    )


__all__ = ["RoundTripResult", "round_trip_check"]
