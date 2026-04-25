"""§1.4 sim/est consistency for SWAT.

Plan ref: cosmic-giggling-wadler.md, Phase B item B.2.

Three checks against the truth parameters of Set A (healthy basin):

  A. Drift parity — simulator's drift matches the est-side drift
     re-derived from _dynamics.drift at the same state/time/params.

  B. Obs-prediction parity — per channel, simulator's noiseless
     prediction matches the estimator's predictor:
       - HR (Gaussian): mean = HR_base + alpha_HR*W
       - sleep (3-level ordinal): two threshold CDFs P(sleep≥1), P(sleep≥2)
       - steps (Poisson): rate = lambda_base + lambda_step*sigmoid(...)
       - stress (Gaussian): mean = s_base + alpha_s*W + beta_s*Vn

These are the tests that surface model-design gaps before any GPU
goes into SMC² estimation. The 4-channel mixed-likelihood discipline
(Gaussian + Poisson + ordinal in one model) is exercised here for the
first time in the psim test suite.
"""

from __future__ import annotations

import math

import numpy as np

from tests.conftest import requires_public_dev

from psim.validation import (
    check_drift_parity,
    check_obs_prediction_parity,
)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ─────────────────────────────────────────────────────────────────────
# Drift parity at truth (Set A)
# ─────────────────────────────────────────────────────────────────────

@requires_public_dev
def test_drift_parity_swat_set_A():
    """Sim's drift_fn must match est's drift at Set A truth params."""
    from models.swat.simulation import SWAT_MODEL
    from models.swat._dynamics import drift as est_drift_jax
    from psim.scenarios.presets.swat_set_A_healthy import truth_params_and_init
    from models.swat.estimation import PI

    params, init = truth_params_and_init()
    state = np.array([init['W_0'], init['Zt_0'], init['a_0'], init['T_0'],
                      0.0, init['Vh'], init['Vn']])
    t_hours = 6.0    # mid-morning so circadian drive is non-trivial
    aux = None

    # Build est_params_vec in PI order
    est_v_dict = {**params, **init}   # Vh, Vn live in init on sim side
    est_v = np.array([est_v_dict.get(k, 0.0) for k in
                      sorted(PI, key=PI.get)], dtype=np.float64)

    def _est_drift_at_state(state, t, dt, est_params_vec, grid_obs, k):
        import jax.numpy as jnp
        dy = est_drift_jax(jnp.asarray(state), jnp.float64(t),
                            jnp.asarray(est_params_vec), PI)
        return np.asarray(dy)

    res = check_drift_parity(
        sim_drift_fn=SWAT_MODEL.drift_fn,
        est_drift_at_state=_est_drift_at_state,
        state=state, t=t_hours,
        sim_params=params, est_params_vec=est_v,
        aux=aux, grid_obs={}, k=0,
        atol=1e-6, rtol=1e-6,
    )
    res.assert_pass()


# ─────────────────────────────────────────────────────────────────────
# Obs-prediction parity per channel (Set A)
# ─────────────────────────────────────────────────────────────────────

def _swat_set_A_state_params():
    from psim.scenarios.presets.swat_set_A_healthy import truth_params_and_init
    from models.swat.estimation import PI
    params, init = truth_params_and_init()
    state = np.array([init['W_0'], init['Zt_0'], init['a_0'], init['T_0'],
                      0.0, init['Vh'], init['Vn']])
    est_v_dict = {**params, **init}
    est_v = np.array([est_v_dict.get(k, 0.0) for k in
                      sorted(PI, key=PI.get)], dtype=np.float64)
    return state, params, est_v


@requires_public_dev
def test_obs_prediction_parity_HR():
    state, p, est_v = _swat_set_A_state_params()
    from models.swat._dynamics import hr_mean
    from models.swat.estimation import PI
    import jax.numpy as jnp

    def sim_HR(state, t, params):
        return params['HR_base'] + params['alpha_HR'] * state[0]

    def est_HR(state, k, params_vec, grid_obs):
        return float(hr_mean(jnp.asarray(state), jnp.asarray(params_vec), PI))

    res = check_obs_prediction_parity(
        sim_obs_predictor=sim_HR, est_obs_predictor=est_HR,
        channel_name="HR", state=state, t=6.0,
        sim_params=p, est_params_vec=est_v,
        grid_obs={}, k=0, atol=1e-9, rtol=1e-9,
    )
    res.assert_pass()


@requires_public_dev
def test_obs_prediction_parity_sleep_threshold_CDFs():
    """3-level ordinal sleep: compare P(sleep≥1) and P(sleep≥2) on
    each side. These are the two scalar CDFs that fully characterise
    the ordinal likelihood.
    """
    state, p, est_v = _swat_set_A_state_params()
    from models.swat._dynamics import sleep_level_log_probs
    from models.swat.estimation import PI
    import jax.numpy as jnp

    def sim_p_sleep_ge_1(state, t, params):
        # P(sleep ≥ 1) = 1 - P(wake) = sigmoid(Zt - c_tilde)
        Zt = state[1]
        return float(_sigmoid(Zt - params['c_tilde']))

    def sim_p_sleep_ge_2(state, t, params):
        # P(sleep ≥ 2) = P(deep) = sigmoid(Zt - (c_tilde + delta_c))
        Zt = state[1]
        return float(_sigmoid(Zt - (params['c_tilde'] + params['delta_c'])))

    def est_p_sleep_ge_1(state, k, params_vec, grid_obs):
        log_pmf = sleep_level_log_probs(jnp.asarray(state),
                                          jnp.asarray(params_vec), PI)
        # P(level ≥ 1) = 1 - P(level=0) = exp(log P(1)) + exp(log P(2))
        return float(np.exp(log_pmf[1]) + np.exp(log_pmf[2]))

    def est_p_sleep_ge_2(state, k, params_vec, grid_obs):
        log_pmf = sleep_level_log_probs(jnp.asarray(state),
                                          jnp.asarray(params_vec), PI)
        return float(np.exp(log_pmf[2]))

    res1 = check_obs_prediction_parity(
        sim_obs_predictor=sim_p_sleep_ge_1, est_obs_predictor=est_p_sleep_ge_1,
        channel_name="sleep_P(>=1)", state=state, t=6.0,
        sim_params=p, est_params_vec=est_v,
        grid_obs={}, k=0, atol=1e-6, rtol=1e-6,
    )
    res1.assert_pass()

    res2 = check_obs_prediction_parity(
        sim_obs_predictor=sim_p_sleep_ge_2, est_obs_predictor=est_p_sleep_ge_2,
        channel_name="sleep_P(>=2)", state=state, t=6.0,
        sim_params=p, est_params_vec=est_v,
        grid_obs={}, k=0, atol=1e-6, rtol=1e-6,
    )
    res2.assert_pass()


@requires_public_dev
def test_obs_prediction_parity_steps():
    state, p, est_v = _swat_set_A_state_params()
    from models.swat._dynamics import steps_rate
    from models.swat.estimation import PI
    import jax.numpy as jnp

    def sim_rate(state, t, params):
        W = state[0]
        return (params['lambda_base']
                + params['lambda_step']
                * _sigmoid(10.0 * (W - params['W_thresh'])))

    def est_rate(state, k, params_vec, grid_obs):
        return float(steps_rate(jnp.asarray(state),
                                 jnp.asarray(params_vec), PI))

    res = check_obs_prediction_parity(
        sim_obs_predictor=sim_rate, est_obs_predictor=est_rate,
        channel_name="steps_rate", state=state, t=6.0,
        sim_params=p, est_params_vec=est_v,
        grid_obs={}, k=0, atol=1e-5, rtol=1e-5,
    )
    res.assert_pass()


@requires_public_dev
def test_obs_prediction_parity_stress():
    state, p, est_v = _swat_set_A_state_params()
    from models.swat._dynamics import stress_mean
    from models.swat.estimation import PI
    import jax.numpy as jnp

    def sim_stress(state, t, params):
        W, Vn = state[0], state[6]
        return params['s_base'] + params['alpha_s'] * W + params['beta_s'] * Vn

    def est_stress(state, k, params_vec, grid_obs):
        return float(stress_mean(jnp.asarray(state),
                                  jnp.asarray(params_vec), PI))

    res = check_obs_prediction_parity(
        sim_obs_predictor=sim_stress, est_obs_predictor=est_stress,
        channel_name="stress", state=state, t=6.0,
        sim_params=p, est_params_vec=est_v,
        grid_obs={}, k=0, atol=1e-6, rtol=1e-6,
    )
    res.assert_pass()
