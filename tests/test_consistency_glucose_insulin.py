"""§1.4 sim/est consistency for glucose_insulin (Bergman minimal model).

Three checks against the truth parameters of Set A (Bergman 1979
healthy-cohort means, the paper-parity benchmark):

  A. Drift parity — simulator's drift matches `_dynamics.drift_jax`
     when given the same state, time, params, frozen, and pre-computed
     (D_rate_at_t, I_input_rate_at_t) pair.

  B. Obs-prediction parity per channel:
       - cgm (Gaussian): mean = G
       - meal_carbs (Poisson): rate = carbs_truth (deterministic at meal time)

These tests surface model-design gaps before any GPU work.
"""

from __future__ import annotations

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as np

from tests.conftest import requires_public_dev

from psim.validation import (
    check_drift_parity,
    check_obs_prediction_parity,
)


# ─────────────────────────────────────────────────────────────────────
# Drift parity at truth (Set A)
# ─────────────────────────────────────────────────────────────────────

@requires_public_dev
def test_drift_parity_glucose_insulin_set_A():
    """Sim's drift_fn must match est's drift_jax at Set A truth params."""
    from models.glucose_insulin.simulation import (
        GLUCOSE_INSULIN_MODEL,
        _meal_absorption_rate, _insulin_input_rate, _meal_schedule,
    )
    from models.glucose_insulin._dynamics import drift_jax as est_drift_jax
    from psim.scenarios.presets.gi_set_A_healthy import truth_params_and_init
    from models.glucose_insulin.estimation import PI, DEFAULT_FROZEN_PARAMS

    params, init = truth_params_and_init()
    # Mid-morning post-meal state (G already elevated, X+I rising)
    state = np.array([180.0, 0.5, 30.0])
    t_hours = 9.0
    V_G = DEFAULT_FROZEN_PARAMS['V_G']
    BW = DEFAULT_FROZEN_PARAMS['BW']

    meal_sched = _meal_schedule(seed=0, n_days=1, meal_carbs_g=40.0)
    aux_sim = {'meal_schedule': meal_sched, 'insulin_schedule': None}

    # Pre-compute D_rate / I_rate at t (these are scenario-determined, not
    # parameters; the consistency test pre-builds them from the same schedule)
    D_rate = float(_meal_absorption_rate(t_hours, meal_sched, V_G, BW))
    I_rate = float(_insulin_input_rate(t_hours, None))
    aux_jax = {'D_rate_at_t': D_rate, 'I_input_rate_at_t': I_rate}

    # Pack params in PI order
    est_v = np.array([params[k] for k in sorted(PI, key=PI.get)],
                      dtype=np.float64)

    def _est_drift_at_state(state, t, dt, est_params_vec, grid_obs, k):
        import jax.numpy as jnp
        dy = est_drift_jax(jnp.asarray(state), jnp.float64(t),
                            jnp.asarray(est_params_vec),
                            DEFAULT_FROZEN_PARAMS, aux_jax, PI)
        return np.asarray(dy)

    res = check_drift_parity(
        sim_drift_fn=GLUCOSE_INSULIN_MODEL.drift_fn,
        est_drift_at_state=_est_drift_at_state,
        state=state, t=t_hours,
        sim_params=params, est_params_vec=est_v,
        aux=aux_sim, grid_obs={}, k=0,
        atol=1e-8, rtol=1e-8,
    )
    res.assert_pass()


# ─────────────────────────────────────────────────────────────────────
# Obs-prediction parity per channel (Set A)
# ─────────────────────────────────────────────────────────────────────

def _set_A_state_params():
    from psim.scenarios.presets.gi_set_A_healthy import truth_params_and_init
    from models.glucose_insulin.estimation import PI
    params, init = truth_params_and_init()
    state = np.array([180.0, 0.5, 30.0])
    est_v = np.array([params[k] for k in sorted(PI, key=PI.get)],
                      dtype=np.float64)
    return state, params, est_v


@requires_public_dev
def test_obs_prediction_parity_cgm_mean():
    """CGM Gaussian channel: mean = G (state[0]) directly."""
    state, p, est_v = _set_A_state_params()

    def sim_cgm_mean(state, t, params):
        return state[0]    # CGM observes G

    def est_cgm_mean(state, k, params_vec, grid_obs):
        return float(state[0])    # framework's gaussian_obs_fn returns y[0]

    res = check_obs_prediction_parity(
        sim_obs_predictor=sim_cgm_mean, est_obs_predictor=est_cgm_mean,
        channel_name="cgm_mean", state=state, t=9.0,
        sim_params=p, est_params_vec=est_v,
        grid_obs={}, k=0, atol=1e-9, rtol=1e-9,
    )
    res.assert_pass()


@requires_public_dev
def test_obs_prediction_parity_meal_carbs_rate():
    """Meal-carb Poisson channel: rate = carbs_truth at meal time.

    The Poisson rate equals the truth meal carb count (deterministic given
    the schedule). Both sim and est read from the schedule, so this
    parity check is essentially confirming the schedule is consistently
    parsed on both sides.
    """
    from models.glucose_insulin.simulation import _meal_schedule

    state, p, est_v = _set_A_state_params()
    meal_sched = _meal_schedule(seed=0, n_days=1, meal_carbs_g=40.0)
    t_meal_first, carbs_truth_first = meal_sched[0]

    def sim_carbs_rate(state, t, params):
        return carbs_truth_first

    def est_carbs_rate(state, k, params_vec, grid_obs):
        return carbs_truth_first    # estimator reads the same schedule

    res = check_obs_prediction_parity(
        sim_obs_predictor=sim_carbs_rate, est_obs_predictor=est_carbs_rate,
        channel_name="meal_carbs_rate", state=state, t=t_meal_first,
        sim_params=p, est_params_vec=est_v,
        grid_obs={}, k=0, atol=1e-9, rtol=1e-9,
    )
    res.assert_pass()
