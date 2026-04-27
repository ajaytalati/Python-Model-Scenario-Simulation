"""Sanity / shape checks on SIR scenario primitives + sets."""

from __future__ import annotations

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import pytest

from tests.conftest import requires_public_dev


@requires_public_dev
@pytest.mark.parametrize("module_name,expected_scenario_name", [
    ("sir_set_A_boarding_school", "set_A_boarding_school_14d"),
    ("sir_set_B_small",           "set_B_small_outbreak_60d"),
    ("sir_set_C_large",           "set_C_large_outbreak_90d"),
    ("sir_set_D_vax",             "set_D_vaccination_90d"),
])
def test_sir_preset_import_clean(module_name, expected_scenario_name):
    """All four presets resolve via the namespace-package public-dev path."""
    import importlib
    mod = importlib.import_module(f"psim.scenarios.presets.{module_name}")
    assert mod.SCENARIO_NAME == expected_scenario_name
    p, i = mod.truth_params_and_init()
    assert isinstance(p, dict) and isinstance(i, dict)


@requires_public_dev
def test_sir_set_A_truth_param_shape():
    """Set A's params must cover the EstimationModel's PARAM_PRIOR_CONFIG
    (the 6-dim estimable parameter vector)."""
    from models.sir.estimation import SIR_ESTIMATION
    from psim.scenarios.presets.sir_set_A_boarding_school import truth_params_and_init

    p, i = truth_params_and_init()
    est_params = SIR_ESTIMATION.param_keys

    missing = [k for k in est_params if k not in p]
    assert not missing, f"missing estimable params in Set A: {missing}"
    # Init state I_0 lives only in init_dict.
    assert 'I_0' in i


@requires_public_dev
def test_sir_set_distinguishing_values():
    """Each set must produce the correct distinguishing β/γ/N values."""
    from psim.scenarios.presets import (
        sir_set_A_boarding_school, sir_set_B_small,
        sir_set_C_large, sir_set_D_vax,
    )
    pA, _ = sir_set_A_boarding_school.truth_params_and_init()
    pB, _ = sir_set_B_small.truth_params_and_init()
    pC, _ = sir_set_C_large.truth_params_and_init()
    pD, _ = sir_set_D_vax.truth_params_and_init()

    # Set A: boarding school, R_0 ≈ 3.32
    assert pA['N'] == 763
    assert abs(pA['beta'] / pA['gamma'] - 3.32) < 0.01
    assert pA['rho'] == 1.0    # full reporting
    assert pA['v'] == 0.0

    # Set B: small community, R_0 = 2.5
    assert pB['N'] == 10000
    assert abs(pB['beta'] / pB['gamma'] - 2.5) < 0.01
    assert pB['rho'] == 0.5
    assert pB['v'] == 0.0

    # Set C: large community, R_0 = 4.0
    assert pC['N'] == 10000
    assert abs(pC['beta'] / pC['gamma'] - 4.0) < 0.01

    # Set D: vaccination, R_0 = 3.0, v > 0
    assert pD['N'] == 10000
    assert abs(pD['beta'] / pD['gamma'] - 3.0) < 0.01
    assert pD['v'] > 0


@requires_public_dev
def test_sir_model_structure():
    """SIR_MODEL has the expected 2-state + 2-channel structure."""
    from models.sir.simulation import SIR_MODEL

    state_names = [s.name for s in SIR_MODEL.states]
    assert state_names == ['S', 'I']

    ch_names = [c.name for c in SIR_MODEL.channels]
    assert set(ch_names) == {'cases', 'serology'}, \
        f"unexpected channel set: {ch_names}"


@requires_public_dev
def test_sir_estimation_n_dim():
    """SIR_ESTIMATION's flat-vector dim is 7 (6 params + 1 init state)."""
    from models.sir.estimation import SIR_ESTIMATION
    assert SIR_ESTIMATION.n_params == 6, f"got {SIR_ESTIMATION.n_params}"
    assert SIR_ESTIMATION.n_init_states == 1, f"got {SIR_ESTIMATION.n_init_states}"
    assert SIR_ESTIMATION.n_dim == 7
    # Required SMC²-side callbacks
    for fn in ("propagate_fn", "diffusion_fn", "obs_log_weight_fn",
               "align_obs_fn", "obs_log_prob_fn"):
        assert callable(getattr(SIR_ESTIMATION, fn)), f"{fn} not callable"


@requires_public_dev
def test_sir_anderson_may_fixture_consistent_with_set_A():
    """The Anderson-May 1978 reference fixture must match Set A's truth
    parameters — they're the same scenario."""
    from psim.data.anderson_may_1978_flu import ANDERSON_MAY_1978_FLU
    from psim.scenarios.presets.sir_set_A_boarding_school import truth_params_and_init

    p, i = truth_params_and_init()
    ref = ANDERSON_MAY_1978_FLU['truth_params']

    assert p['N'] == ref['N']
    # Set A stores rates in /hr; reference is /day. Convert.
    assert abs(p['beta'] * 24 - ref['beta_per_day']) < 1e-6
    assert abs(p['gamma'] * 24 - ref['gamma_per_day']) < 1e-6
    assert abs((p['beta'] / p['gamma']) - ref['R_0']) < 0.01
    assert i['I_0'] == ANDERSON_MAY_1978_FLU['truth_init']['I_0']
