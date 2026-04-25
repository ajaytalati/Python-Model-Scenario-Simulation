"""Shape + value checks on scenario primitives for the fsa_high_res model.

Plan ref: cosmic-giggling-wadler.md, Phase E, item 26.
"""

from __future__ import annotations

import numpy as np

from tests.conftest import requires_public_dev

from psim.scenarios.exogenous import (
    generate_macrocycle_C0,
    generate_morning_loaded_phi,
    make_C_array,
)
from psim.scenarios.missing_data import apply_dropout


N_DAYS = 14
BINS_PER_DAY = 96
DT_DAYS = 1.0 / BINS_PER_DAY


def test_macrocycle_C0_shape_and_range():
    daily_T_B, daily_Phi = generate_macrocycle_C0(N_DAYS, seed=42)
    assert daily_T_B.shape == (N_DAYS,)
    assert daily_Phi.shape == (N_DAYS,)
    assert daily_T_B.min() >= 0.0 and daily_T_B.max() <= 1.0
    assert daily_Phi.min() >= 0.0


def test_morning_loaded_phi_peak_at_expected_hour():
    """Phi(t) over wake hours is Gamma(k=2) shaped: t * exp(-t/tau).
    Peaks at t = tau post-wake (tau=3h default) → 10am if waking at 7am.
    """
    _, daily_phi = generate_macrocycle_C0(N_DAYS, seed=42)
    phi_arr = generate_morning_loaded_phi(daily_phi, bins_per_day=BINS_PER_DAY,
                                           seed=42)
    assert phi_arr.shape == (N_DAYS * BINS_PER_DAY,)

    # Day 0: find argmax of the wake portion.
    day0 = phi_arr[:BINS_PER_DAY]
    peak_bin = int(np.argmax(day0))
    peak_hour = peak_bin * (24.0 / BINS_PER_DAY)
    # tau_hours=3.0, wake_hour=7.0 → peak at 10am ± noise tolerance
    assert 9.0 <= peak_hour <= 11.0, \
        f"Phi peak at hour {peak_hour}, expected ~10am"


def test_morning_loaded_phi_zero_during_sleep():
    """Phi(t) must be zero during sleep hours [23, 7) every day."""
    _, daily_phi = generate_macrocycle_C0(N_DAYS, seed=42)
    phi_arr = generate_morning_loaded_phi(daily_phi, bins_per_day=BINS_PER_DAY,
                                           seed=42)
    for d in range(N_DAYS):
        for k in range(BINS_PER_DAY):
            h = k * (24.0 / BINS_PER_DAY)
            if h < 7.0 or h >= 23.0:
                assert phi_arr[d * BINS_PER_DAY + k] == 0.0, \
                    f"Phi nonzero during sleep at day {d}, bin {k}, h={h}"


def test_make_C_array_matches_global_cos():
    """C_arr[k] must equal cos(2 pi k * dt + phi) on the global grid."""
    n_bins = N_DAYS * BINS_PER_DAY
    C_arr = make_C_array(n_bins, dt_days=DT_DAYS, phi=0.0)
    assert C_arr.shape == (n_bins,)
    t = np.arange(n_bins) * DT_DAYS
    expected = np.cos(2.0 * np.pi * t)
    np.testing.assert_allclose(C_arr, expected, atol=1e-5)


def test_apply_dropout_deterministic_at_fixed_seed():
    """Same seed → same dropout mask → same surviving t_idx set."""
    obs = {
        "obs_HR": {
            "t_idx": np.arange(1000, dtype=np.int32),
            "obs_value": np.zeros(1000, dtype=np.float32),
        },
    }
    apply_dropout(obs, ["obs_HR"], rate=0.05, seed=7)
    a = obs["obs_HR"]["t_idx"].copy()

    obs2 = {
        "obs_HR": {
            "t_idx": np.arange(1000, dtype=np.int32),
            "obs_value": np.zeros(1000, dtype=np.float32),
        },
    }
    apply_dropout(obs2, ["obs_HR"], rate=0.05, seed=7)
    b = obs2["obs_HR"]["t_idx"]
    np.testing.assert_array_equal(a, b)
    # ~5% of 1000 dropped → ~950 kept (binomial — give it room)
    assert 920 <= len(a) <= 980, f"unexpected drop count: {len(a)}/1000"


@requires_public_dev
def test_high_res_fsa_model_loads_from_public_dev():
    """Sanity: the canonical model is importable via namespace package."""
    from models.fsa_high_res.simulation import HIGH_RES_FSA_MODEL, DEFAULT_PARAMS
    assert HIGH_RES_FSA_MODEL.name == "fsa_high_res"
    assert HIGH_RES_FSA_MODEL.version == "0.1"
    # 3 latent states (B, F, A)
    assert len(HIGH_RES_FSA_MODEL.states) == 3
    state_names = [s.name for s in HIGH_RES_FSA_MODEL.states]
    assert state_names == ["B", "F", "A"]
    # truth params include the 4-channel obs coefficients
    for p in ("HR_base", "kappa_B", "beta_C_HR", "k_C", "S_base", "mu_step0"):
        assert p in DEFAULT_PARAMS, f"missing truth param: {p}"


@requires_public_dev
def test_high_res_fsa_estimation_loads():
    """Sanity: the EstimationModel partner imports clean (29 estimable
    params, contracts wired)."""
    from models.fsa_high_res.estimation import HIGH_RES_FSA_ESTIMATION
    m = HIGH_RES_FSA_ESTIMATION
    assert m.n_dim == 29
    for fn in ("propagate_fn", "diffusion_fn", "obs_log_weight_fn",
               "align_obs_fn"):
        assert callable(getattr(m, fn)), f"{fn} not callable"
