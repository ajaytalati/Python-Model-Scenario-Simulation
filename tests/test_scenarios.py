"""Sanity tests on the scenario primitives."""

import numpy as np

from psim.scenarios.exogenous import (
    generate_macrocycle_C0, generate_morning_loaded_phi,
    circadian, make_C_array,
)
from psim.scenarios.missing_data import (
    apply_dropout, apply_broken_watch_gap, apply_rest_days,
)


def test_macrocycle_shapes_and_ranges():
    T_B, Phi = generate_macrocycle_C0(14, seed=42)
    assert T_B.shape == (14,) and Phi.shape == (14,)
    assert T_B.min() >= 0.05 and T_B.max() <= 0.95
    assert Phi.min() >= 0.005 and Phi.max() <= 0.25


def test_macrocycle_seed_reproducible():
    a = generate_macrocycle_C0(14, seed=42)[0]
    b = generate_macrocycle_C0(14, seed=42)[0]
    np.testing.assert_array_equal(a, b)


def test_morning_loaded_zero_during_sleep():
    daily_phi = np.array([0.1, 0.1, 0.1])
    bins = 96
    phi = generate_morning_loaded_phi(daily_phi, bins_per_day=bins, seed=42)
    # First 7am = 28 bins are sleep
    assert phi[:28].max() == 0.0
    # Last 1h before midnight (bins 92-95) is sleep
    assert phi[92:96].max() == 0.0


def test_morning_loaded_peak_post_wake():
    """Peak should be ~3h post-wake (default tau)."""
    daily_phi = np.array([0.1])
    bins = 96
    phi = generate_morning_loaded_phi(
        daily_phi, bins_per_day=bins, tau_hours=3.0, noise_frac=0.0, seed=0,
    )
    # Peak: t = wake (7) + tau (3) = 10am = bin 40
    peak_bin = phi.argmax()
    assert 36 <= peak_bin <= 44, f"peak at bin {peak_bin}, expected ~40"


def test_circadian_endpoints():
    assert abs(circadian(0.0) - 1.0) < 1e-12       # midnight
    assert abs(circadian(0.25) - 0.0) < 1e-9       # 6am
    assert abs(circadian(0.5) - (-1.0)) < 1e-12    # noon
    assert abs(circadian(0.75) - 0.0) < 1e-9       # 6pm
    assert abs(circadian(1.0) - 1.0) < 1e-12       # next midnight


def test_C_array_phase_global():
    """make_C_array should preserve global phase regardless of array length."""
    C = make_C_array(96 * 14, dt_days=1 / 96, phi=0.0)
    # bin 48 = noon
    assert abs(C[48] - (-1.0)) < 1e-5
    # bin 96*7 + 48 = noon of day 7
    assert abs(C[96 * 7 + 48] - (-1.0)) < 1e-5


def test_dropout_keeps_about_rate_fraction():
    obs = {"hr": {"t_idx": np.arange(1000, dtype=np.int32),
                   "obs_value": np.zeros(1000, dtype=np.float32)}}
    apply_dropout(obs, ["hr"], rate=0.20, seed=1)
    n = len(obs["hr"]["t_idx"])
    assert 750 <= n <= 850   # ~80% retention with statistical wiggle


def test_broken_watch_creates_gap():
    obs = {"hr": {"t_idx": np.arange(2000, dtype=np.int32),
                   "obs_value": np.zeros(2000, dtype=np.float32)}}
    apply_broken_watch_gap(
        obs, ["hr"], n_days=2000, gap_days=200,
        edge_buffer_days=400, seed=1,
    )
    assert len(obs["hr"]["t_idx"]) == 2000 - 200


def test_rest_days_masks_active_channel():
    obs = {"intensity": {"t_idx": np.arange(28, dtype=np.int32),
                          "obs_value": np.zeros(28, dtype=np.float32)}}
    apply_rest_days(obs, ["intensity"], n_days=28, bins_per_day=1,
                     rest_days_per_week=(2, 3), seed=1)
    # 4 weeks * 2-3 rest days = 8-12 days masked → 16-20 days kept
    n = len(obs["intensity"]["t_idx"])
    assert 16 <= n <= 20
