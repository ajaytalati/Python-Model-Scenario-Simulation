"""Tests on the consistency-check helpers."""

import numpy as np

from psim.validation import (
    check_drift_parity, check_obs_prediction_parity,
)


def test_drift_parity_passes_when_matched():
    def sim_drift(t, y, params, aux):
        return -params["k"] * y

    def est_drift(state, t, dt, est_params_vec, grid_obs, k):
        return -est_params_vec[0] * np.asarray(state)

    res = check_drift_parity(
        sim_drift, est_drift,
        state=np.array([1.0, 2.0]), t=0.0,
        sim_params={"k": 0.5}, est_params_vec=np.array([0.5]),
        aux=(), grid_obs={}, k=0,
    )
    assert res.passed
    assert res.max_abs_err < 1e-3


def test_drift_parity_catches_sign_flip():
    """The mu_0-style sign mismatch should be flagged."""
    def sim_drift(t, y, params, aux):
        return -params["k"] * y

    def est_drift_wrong(state, t, dt, est_params_vec, grid_obs, k):
        return +est_params_vec[0] * np.asarray(state)   # BUG: sign flipped

    res = check_drift_parity(
        sim_drift, est_drift_wrong,
        state=np.array([1.0, 2.0]), t=0.0,
        sim_params={"k": 0.5}, est_params_vec=np.array([0.5]),
        aux=(), grid_obs={}, k=0,
    )
    assert not res.passed
    assert res.max_abs_err > 1.0


def test_obs_prediction_parity_passes_when_matched():
    def sim(state, t, params):
        return 62 - 12 * state[0] + 3 * state[2]

    def est(state, k, params_vec, grid_obs):
        return 62 - 12 * state[0] + 3 * state[2]

    state = np.array([0.5, 0.1, 0.5])
    res = check_obs_prediction_parity(
        sim, est, channel_name="HR", state=state, t=0.5,
        sim_params={}, est_params_vec=np.zeros(1), grid_obs={}, k=10,
    )
    assert res.passed


def test_obs_prediction_parity_catches_C_phase_bug_signature():
    """Simulate the C-phase bug: estimator's prediction uses +C while sim uses -C."""
    def sim(state, t, params):
        # Sim sees C = -1 at noon
        return 62 - 12 * state[0] + 3 * state[2] + (-2.5) * (-1.0)   # = 62 - 12B + 3A + 2.5

    def est_buggy(state, k, params_vec, grid_obs):
        # Buggy estimator computes C from window-local time → C = +1 at window-bin 0
        return 62 - 12 * state[0] + 3 * state[2] + (-2.5) * (+1.0)   # = 62 - 12B + 3A - 2.5

    state = np.array([0.5, 0.1, 0.5])
    res = check_obs_prediction_parity(
        sim, est_buggy, channel_name="HR", state=state, t=0.5,
        sim_params={}, est_params_vec=np.zeros(1), grid_obs={}, k=10,
    )
    assert not res.passed
    # Difference should be 2 * |beta_C_HR| = 5.0
    assert abs(res.max_abs_err - 5.0) < 1e-6
