"""Sanity / shape checks on glucose_insulin scenario primitives."""

from __future__ import annotations

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import pytest

from tests.conftest import requires_public_dev


@requires_public_dev
@pytest.mark.parametrize("module_name,expected_scenario_name", [
    ("gi_set_A_healthy",             "set_A_healthy_24h"),
    ("gi_set_B_insulin_resistance",  "set_B_insulin_resistance_24h"),
    ("gi_set_C_t1d_no_insulin",      "set_C_t1d_no_insulin_24h"),
    ("gi_set_D_t1d_open_loop",       "set_D_t1d_open_loop_24h"),
])
def test_gi_preset_import_clean(module_name, expected_scenario_name):
    """All four presets resolve via the namespace-package public-dev path."""
    import importlib
    mod = importlib.import_module(f"psim.scenarios.presets.{module_name}")
    assert mod.SCENARIO_NAME == expected_scenario_name
    p, i = mod.truth_params_and_init()
    assert isinstance(p, dict) and isinstance(i, dict)


@requires_public_dev
def test_gi_set_A_truth_param_shape():
    """Set A's params cover the EstimationModel's PARAM_PRIOR_CONFIG."""
    from models.glucose_insulin.estimation import GLUCOSE_INSULIN_ESTIMATION
    from psim.scenarios.presets.gi_set_A_healthy import truth_params_and_init

    p, i = truth_params_and_init()
    est_params = GLUCOSE_INSULIN_ESTIMATION.param_keys
    missing = [k for k in est_params if k not in p]
    assert not missing, f"missing estimable params: {missing}"
    # Init state has G_0 and I_0
    assert 'G_0' in i
    assert 'I_0' in i


@requires_public_dev
def test_gi_set_distinguishing_values():
    """Each per-set config carries its scenario-distinguishing values."""
    from psim.scenarios.presets import (
        gi_set_A_healthy, gi_set_B_insulin_resistance,
        gi_set_C_t1d_no_insulin, gi_set_D_t1d_open_loop,
    )
    pA, _ = gi_set_A_healthy.truth_params_and_init()
    pB, _ = gi_set_B_insulin_resistance.truth_params_and_init()
    pC, _ = gi_set_C_t1d_no_insulin.truth_params_and_init()
    pD, _ = gi_set_D_t1d_open_loop.truth_params_and_init()

    # Set A: healthy (Bergman 1979 means)
    assert pA['Ib'] == 7.0
    assert pA['n_beta'] == 8.0
    assert not pA.get('insulin_schedule_active', False)
    assert abs(pA['p3'] / pA['p2'] - 0.0312) < 0.001

    # Set B: insulin resistance — p3 halved, otherwise = A
    assert pB['Ib'] == 7.0
    assert pB['n_beta'] == 8.0
    assert abs(pB['p3'] / pA['p3'] - 0.5) < 0.01

    # Set C: T1D no-control
    assert pC['Ib'] == 0.0
    assert pC['n_beta'] == 0.0
    assert not pC.get('insulin_schedule_active', False)

    # Set D: T1D + open-loop insulin
    assert pD['Ib'] == 0.0
    assert pD['n_beta'] == 0.0
    assert pD['insulin_schedule_active'] is True


@requires_public_dev
def test_gi_model_structure():
    """GLUCOSE_INSULIN_MODEL has 3-state + 2-channel structure."""
    from models.glucose_insulin.simulation import GLUCOSE_INSULIN_MODEL

    state_names = [s.name for s in GLUCOSE_INSULIN_MODEL.states]
    assert state_names == ['G', 'X', 'I']

    ch_names = [c.name for c in GLUCOSE_INSULIN_MODEL.channels]
    assert set(ch_names) == {'cgm', 'meal_carbs'}, \
        f"unexpected channel set: {ch_names}"


@requires_public_dev
def test_gi_estimation_n_dim():
    """7 estimable params + 2 init states = 9 estimable scalars."""
    from models.glucose_insulin.estimation import GLUCOSE_INSULIN_ESTIMATION
    assert GLUCOSE_INSULIN_ESTIMATION.n_params == 7
    assert GLUCOSE_INSULIN_ESTIMATION.n_init_states == 2
    assert GLUCOSE_INSULIN_ESTIMATION.n_dim == 9
    for fn in ("propagate_fn", "diffusion_fn", "obs_log_weight_fn",
               "align_obs_fn", "obs_log_prob_fn"):
        assert callable(getattr(GLUCOSE_INSULIN_ESTIMATION, fn)), \
            f"{fn} not callable"


@requires_public_dev
def test_bergman_1979_fixture_consistent_with_set_A():
    """Bergman 1979 reference fixture must match Set A's truth parameters."""
    from psim.data.bergman_1979_fsigt import PARAMS as REF
    from psim.scenarios.presets.gi_set_A_healthy import truth_params_and_init

    p, i = truth_params_and_init()
    for key in ('p1', 'p2', 'p3', 'k', 'Gb', 'Ib', 'V_G', 'V_I', 'BW',
                 'n_beta', 'h_beta'):
        assert abs(p[key] - REF[key]) < 1e-6, \
            f"Set A's {key} = {p[key]} != Bergman 1979 ref {REF[key]}"
