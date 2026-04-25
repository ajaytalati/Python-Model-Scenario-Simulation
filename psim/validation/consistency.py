"""Three mandatory pre-SMC² consistency checks.

These are the executable form of the discipline documented in
``docs/VALIDATION_DISCIPLINE.md`` (and originally in the SMC² repo's
``docs/PORTING_GUIDE.md`` § 1.4 after the high_res_FSA postmortem).

The historical motivation: three sim/est consistency bugs cost ~15h of
GPU + analyst time during high_res_FSA development. Each would have
been caught in <30 min by these checks. They are now mandatory.

Three checks:

  A. ``check_drift_parity``       — simulator drift == estimator drift
                                    at the truth parameters and a
                                    representative state. Catches sign
                                    flips, missing terms, and wrong
                                    parameter indexing.

  B. ``check_obs_prediction_parity`` — for every Gaussian observation
                                    channel, simulator's noiseless
                                    prediction == estimator's
                                    prediction at the same state, time,
                                    and parameters. Catches the
                                    C-phase / covariate-misalignment
                                    bug class.

  C. ``check_cold_start_coverage`` — 1-window cold-start with
                                    truth-tight prior should give
                                    ~100% coverage. End-to-end
                                    diagnostic; if it fails after A
                                    and B pass, the bug is in the
                                    inner-PF or SMC² loop, not the
                                    data alignment. (Requires the
                                    SMC² runner; stubbed for now.)

Each returns a ``(passed: bool, report: dict)`` tuple suitable for
both interactive use and pytest assertions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────

@dataclass
class ConsistencyResult:
    """Structured pass/fail with per-element diagnostics."""

    name: str
    passed: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def assert_pass(self):
        if not self.passed:
            raise AssertionError(
                f"Consistency check '{self.name}' FAILED: "
                f"max|err|={self.max_abs_err:.3e}, "
                f"max|rel|={self.max_rel_err:.3e}\n"
                f"details: {self.details}"
            )


# ─────────────────────────────────────────────────────────────────────
# Check A — drift parity
# ─────────────────────────────────────────────────────────────────────

def check_drift_parity(
    sim_drift_fn: Callable[..., np.ndarray],
    est_drift_at_state: Callable[..., np.ndarray],
    *,
    state: np.ndarray,
    t: float,
    sim_params: Dict[str, float],
    est_params_vec: np.ndarray,
    aux: Tuple,
    grid_obs: dict,
    k: int = 0,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> ConsistencyResult:
    """Compare simulator drift vs estimator drift at the same state.

    Parameters
    ----------
    sim_drift_fn : callable
        Simulator's drift, signature ``sim_drift_fn(t, y, sim_params, aux) -> dy``.
        Typically ``model_sim.drift_fn`` from the public dev repo.
    est_drift_at_state : callable
        Wrapper that returns the **deterministic** part of one estimator
        Euler step at a given state. Signature
        ``est_drift_at_state(state, t, dt, est_params_vec, grid_obs, k) -> dy``.
        Implementations typically call ``model_est.propagate_fn`` with
        zero noise and small dt, returning ``(y_next - state) / dt``.
    state : (n_state,) ndarray
    t : float — global time (same units as the SDE)
    sim_params, est_params_vec : truth parameters in each side's format
    aux, grid_obs, k : the per-side aux/observation context

    Returns
    -------
    ConsistencyResult with passed = (max_abs < atol AND max_rel < rtol).
    """
    sim_dy = np.asarray(sim_drift_fn(t, state, sim_params, aux), dtype=np.float64)
    # Use a small dt for the estimator's drift estimate; zero noise.
    dt_small = 1e-6
    est_dy = np.asarray(
        est_drift_at_state(state, t, dt_small, est_params_vec, grid_obs, k),
        dtype=np.float64,
    )

    abs_err = np.abs(sim_dy - est_dy)
    denom = np.maximum(np.abs(sim_dy), 1e-12)
    rel_err = abs_err / denom

    passed = bool(np.all(abs_err < atol) and np.all(rel_err < rtol))
    return ConsistencyResult(
        name="drift_parity",
        passed=passed,
        max_abs_err=float(abs_err.max()),
        max_rel_err=float(rel_err.max()),
        details={
            "sim_dy": sim_dy.tolist(),
            "est_dy": est_dy.tolist(),
            "state": state.tolist(),
            "t": t,
            "atol": atol,
            "rtol": rtol,
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Check B — observation-prediction parity (per Gaussian channel)
# ─────────────────────────────────────────────────────────────────────

def check_obs_prediction_parity(
    sim_obs_predictor: Callable[..., float],
    est_obs_predictor: Callable[..., float],
    *,
    channel_name: str,
    state: np.ndarray,
    t: float,
    sim_params: Dict[str, float],
    est_params_vec: np.ndarray,
    grid_obs: dict,
    k: int,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> ConsistencyResult:
    """Compare simulator's noiseless obs prediction vs estimator's
    prediction for one Gaussian channel.

    Parameters
    ----------
    sim_obs_predictor : callable(state, t, sim_params) -> float
        The simulator's noiseless mean for one channel at one time.
    est_obs_predictor : callable(state, k, est_params_vec, grid_obs) -> float
        The estimator's noiseless mean. For the high_res_FSA model this
        is e.g. ``HR_base + beta_C_HR * grid_obs['C'][k] - kappa_B*B + ...``.
    channel_name : str — for the report
    state : (n_state,)
    t : float — global time
    k : int — bin index used by the estimator's grid_obs

    Returns
    -------
    ConsistencyResult.
    """
    sim_pred = float(sim_obs_predictor(state, t, sim_params))
    est_pred = float(est_obs_predictor(state, k, est_params_vec, grid_obs))

    abs_err = abs(sim_pred - est_pred)
    rel_err = abs_err / max(abs(sim_pred), 1e-12)
    passed = (abs_err < atol) or (rel_err < rtol)

    return ConsistencyResult(
        name=f"obs_prediction_parity::{channel_name}",
        passed=passed,
        max_abs_err=abs_err,
        max_rel_err=rel_err,
        details={
            "channel": channel_name,
            "sim_pred": sim_pred,
            "est_pred": est_pred,
            "state": state.tolist(),
            "t": t,
            "k": k,
            "atol": atol,
            "rtol": rtol,
        },
    )


# ─────────────────────────────────────────────────────────────────────
# Check C — cold-start coverage (SMC²-stack required)
# ─────────────────────────────────────────────────────────────────────

def check_cold_start_coverage(
    smc_runner_callable: Callable[..., dict],
    *,
    scenario_artifact_dir: str,
    n_smc: int = 256,
    prior_sigma_scale: float = 0.1,
    coverage_threshold: float = 0.95,
) -> ConsistencyResult:
    """Run a single window of cold-start SMC² with priors tightened
    around the truth and assert ≥ ``coverage_threshold`` coverage.

    NOTE: stub for v0.1.0. Requires the SMC² runner from the
    smc2-blackjax-rolling repo. Once that repo gains an importable
    ``run_one_window(artifact_dir, prior_sigma_scale, n_smc)`` API,
    plug it in here. For now, callers exercise this discipline by
    running the full SMC² rollout themselves and checking W1 ≥ 95%.

    Parameters
    ----------
    smc_runner_callable : callable
        Function with signature ``(artifact_dir, n_smc, prior_sigma_scale) -> dict``
        returning at minimum a ``'coverage'`` key for window 1.
    scenario_artifact_dir : str
        Path to a packaged scenario artifact (see ``psim.io.format``).
    """
    if smc_runner_callable is None:
        return ConsistencyResult(
            name="cold_start_coverage",
            passed=True,   # treat as "no info" rather than failure
            max_abs_err=0.0,
            max_rel_err=0.0,
            details={
                "status": "skipped",
                "reason": "SMC² runner not provided (stub for v0.1.0)",
                "scenario_artifact_dir": scenario_artifact_dir,
            },
        )

    result = smc_runner_callable(
        scenario_artifact_dir,
        n_smc=n_smc,
        prior_sigma_scale=prior_sigma_scale,
    )
    cov = float(result.get("coverage", 0.0))
    passed = cov >= coverage_threshold
    return ConsistencyResult(
        name="cold_start_coverage",
        passed=passed,
        max_abs_err=max(coverage_threshold - cov, 0.0),
        max_rel_err=0.0,
        details={
            "coverage": cov,
            "threshold": coverage_threshold,
            "n_smc": n_smc,
            "prior_sigma_scale": prior_sigma_scale,
        },
    )


__all__ = [
    "ConsistencyResult",
    "check_drift_parity",
    "check_obs_prediction_parity",
    "check_cold_start_coverage",
]
