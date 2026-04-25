"""Sanity / shape checks on SWAT scenario primitives + sets.

Plan ref: cosmic-giggling-wadler.md, Phase B item B.1.
"""

from __future__ import annotations

import pytest

from tests.conftest import requires_public_dev


@requires_public_dev
@pytest.mark.parametrize("module_name,expected_scenario_name", [
    ("swat_set_A_healthy",       "set_A_healthy_14d"),
    ("swat_set_B_amplitude",     "set_B_amplitude_collapse_14d"),
    ("swat_set_C_recovery",      "set_C_recovery_14d"),
    ("swat_set_D_phase_shift",   "set_D_phase_shift_14d"),
])
def test_swat_preset_import_clean(module_name, expected_scenario_name):
    """All four presets resolve via the namespace-package public-dev path."""
    import importlib
    mod = importlib.import_module(f"psim.scenarios.presets.{module_name}")
    assert mod.SCENARIO_NAME == expected_scenario_name
    p, i = mod.truth_params_and_init()
    assert isinstance(p, dict) and isinstance(i, dict)


@requires_public_dev
def test_swat_set_A_truth_param_shape():
    """Set A's 31 dynamical params + 4 init states + (Vh, Vn) must
    cover the EstimationModel's 35-dim flat vector exactly."""
    from models.swat.estimation import SWAT_ESTIMATION
    from psim.scenarios.presets.swat_set_A_healthy import truth_params_and_init

    p, i = truth_params_and_init()

    # PARAM_SET_A also carries time-grid metadata; estimable subset
    # is param_keys.
    est_params = SWAT_ESTIMATION.param_keys
    # Vh, Vn live in init_state on the sim side but in param_keys on
    # the est side. Combined, we should be able to fill every estimable
    # name from one of the two dicts.
    combined = {**p, **i}
    missing = [k for k in est_params if k not in combined]
    assert not missing, f"missing estimable params in Set A: {missing}"


@requires_public_dev
def test_swat_set_distinguishing_values():
    """Each set must produce the correct distinguishing parameter
    values per TESTING.md / SWAT_Basic_Documentation §6."""
    from psim.scenarios.presets import (
        swat_set_A_healthy, swat_set_B_amplitude,
        swat_set_C_recovery, swat_set_D_phase_shift,
    )
    pA, iA = swat_set_A_healthy.truth_params_and_init()
    pB, iB = swat_set_B_amplitude.truth_params_and_init()
    pC, iC = swat_set_C_recovery.truth_params_and_init()
    pD, iD = swat_set_D_phase_shift.truth_params_and_init()

    # Set A: healthy basin
    assert iA['Vh'] == 1.0 and iA['Vn'] == 0.3
    assert pA['V_c'] == 0.0 and iA['T_0'] == 0.5

    # Set B: amplitude collapse — low Vh, high Vn
    assert iB['Vh'] == 0.2 and iB['Vn'] == 3.5
    assert pB['V_c'] == 0.0 and iB['T_0'] == 0.5

    # Set C: recovery — healthy potentials, T_0 near zero
    assert iC['Vh'] == 1.0 and iC['Vn'] == 0.3
    assert pC['V_c'] == 0.0 and iC['T_0'] == 0.05

    # Set D: phase shift — healthy potentials, V_c=6h
    assert iD['Vh'] == 1.0 and iD['Vn'] == 0.3
    assert pD['V_c'] == 6.0 and iD['T_0'] == 0.5


@requires_public_dev
def test_swat_model_structure():
    """SWAT_MODEL has the expected 7-state + 4-channel structure."""
    from models.swat.simulation import SWAT_MODEL

    state_names = [s.name for s in SWAT_MODEL.states]
    assert state_names == ['W', 'Zt', 'a', 'T', 'C', 'Vh', 'Vn']

    ch_names = [c.name for c in SWAT_MODEL.channels]
    assert set(ch_names) >= {'hr', 'sleep', 'steps', 'stress'}, \
        f"missing one of the 4 obs channels: {ch_names}"


@requires_public_dev
def test_swat_estimation_n_dim():
    """SWAT_ESTIMATION's flat-vector dim is 35 (31 params + 4 ICs)."""
    from models.swat.estimation import SWAT_ESTIMATION
    assert SWAT_ESTIMATION.n_dim == 35
    assert SWAT_ESTIMATION.n_params == 31
    assert SWAT_ESTIMATION.n_init_states == 4
    # Required SMC²-side callbacks
    for fn in ("propagate_fn", "diffusion_fn", "obs_log_weight_fn",
               "align_obs_fn", "obs_log_prob_fn"):
        assert callable(getattr(SWAT_ESTIMATION, fn)), f"{fn} not callable"
