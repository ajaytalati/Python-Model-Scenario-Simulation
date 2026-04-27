"""§1.4 sim/est consistency for SIR.

Three checks against the truth parameters of Set A (Anderson & May
1978 boarding-school flu, the paper-parity benchmark):

  A. Drift parity — simulator's drift_fn matches the est-side drift
     re-derived from `_dynamics.drift_jax` at the same state/time/params.

  B. Obs-prediction parity — per channel, simulator's noiseless
     prediction matches the estimator's predictor:
       - cases (Poisson rate): rate = ρ β S I / N (per hour)
       - serology (Gaussian mean): I / N

These tests surface model-design gaps before any GPU work in SMC²
estimation. The mixed-likelihood discipline (Poisson + Gaussian on
the same model) is exercised here with the smaller-dim SIR model in
addition to SWAT.
"""

from __future__ import annotations

import os
os.environ['JAX_ENABLE_X64'] = 'True'   # required for 1e-8 tolerances

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
def test_drift_parity_sir_set_A():
    """Sim's drift_fn must match est's drift at Set A truth params."""
    from models.sir.simulation import SIR_MODEL
    from models.sir._dynamics import drift_jax as est_drift_jax
    from psim.scenarios.presets.sir_set_A_boarding_school import truth_params_and_init
    from models.sir.estimation import PI, DEFAULT_FROZEN_PARAMS

    params, init = truth_params_and_init()
    # Mid-outbreak state — both rate terms non-trivial.
    state = np.array([params['N'] - 50.0, 50.0])
    t_hours = 24.0    # day 1
    aux = None

    # Pack params into PI order.
    est_v = np.array([params[k] for k in sorted(PI, key=PI.get)],
                      dtype=np.float64)

    def _est_drift_at_state(state, t, dt, est_params_vec, grid_obs, k):
        import jax.numpy as jnp
        dy = est_drift_jax(jnp.asarray(state), jnp.float64(t),
                            jnp.asarray(est_params_vec),
                            DEFAULT_FROZEN_PARAMS, PI)
        return np.asarray(dy)

    res = check_drift_parity(
        sim_drift_fn=SIR_MODEL.drift_fn,
        est_drift_at_state=_est_drift_at_state,
        state=state, t=t_hours,
        sim_params=params, est_params_vec=est_v,
        aux=aux, grid_obs={}, k=0,
        atol=1e-8, rtol=1e-8,
    )
    res.assert_pass()


# ─────────────────────────────────────────────────────────────────────
# Obs-prediction parity per channel (Set A)
# ─────────────────────────────────────────────────────────────────────

def _sir_set_A_state_params():
    from psim.scenarios.presets.sir_set_A_boarding_school import truth_params_and_init
    from models.sir.estimation import PI
    params, init = truth_params_and_init()
    state = np.array([params['N'] - 50.0, 50.0])
    est_v = np.array([params[k] for k in sorted(PI, key=PI.get)],
                      dtype=np.float64)
    return state, params, est_v


@requires_public_dev
def test_obs_prediction_parity_cases_rate():
    """Poisson cases channel: rate = ρ β S I / N (per hour)."""
    state, p, est_v = _sir_set_A_state_params()
    from models.sir._dynamics import cases_rate
    from models.sir.estimation import PI, DEFAULT_FROZEN_PARAMS
    import jax.numpy as jnp

    def sim_cases_rate(state, t, params):
        S, I = state
        return params['rho'] * params['beta'] * S * I / params['N']

    def est_cases_rate(state, k, params_vec, grid_obs):
        return float(cases_rate(jnp.asarray(state),
                                 jnp.asarray(params_vec),
                                 DEFAULT_FROZEN_PARAMS, PI))

    res = check_obs_prediction_parity(
        sim_obs_predictor=sim_cases_rate, est_obs_predictor=est_cases_rate,
        channel_name="cases_rate", state=state, t=24.0,
        sim_params=p, est_params_vec=est_v,
        grid_obs={}, k=0, atol=1e-9, rtol=1e-9,
    )
    res.assert_pass()


@requires_public_dev
def test_obs_prediction_parity_serology_mean():
    """Gaussian serology channel: mean = I / N (prevalence)."""
    state, p, est_v = _sir_set_A_state_params()
    from models.sir._dynamics import serology_mean
    from models.sir.estimation import DEFAULT_FROZEN_PARAMS
    import jax.numpy as jnp

    def sim_serology_mean(state, t, params):
        return state[1] / params['N']

    def est_serology_mean(state, k, params_vec, grid_obs):
        return float(serology_mean(jnp.asarray(state),
                                    DEFAULT_FROZEN_PARAMS))

    res = check_obs_prediction_parity(
        sim_obs_predictor=sim_serology_mean, est_obs_predictor=est_serology_mean,
        channel_name="serology_mean", state=state, t=24.0,
        sim_params=p, est_params_vec=est_v,
        grid_obs={}, k=0, atol=1e-9, rtol=1e-9,
    )
    res.assert_pass()
