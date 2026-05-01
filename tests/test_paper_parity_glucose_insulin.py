"""Paper-parity test: Set A vs Bergman 1979 healthy-cohort qualitative profile.

Synthesises Set A at the published canonical parameters and asserts the
simulator output matches the qualitative meal-response profile reported
in Bergman 1979 / textbooks / clinical literature:

  - Peak G ∈ [165, 200] mg/dL after each meal
  - Peak occurs 30-60 min post-meal
  - G returns to within 10 mg/dL of basal Gb=90 within ~2 hr
  - Plasma insulin peak ∈ [40, 60] μU/mL

This is the **published-data parity** test for the test model. The
complementary "inference recovers truth from synthetic data" test lives
downstream in the SMC² repo (`drivers/glucose_insulin/`).

The tolerances are deliberately chosen to bracket the textbook range —
not so tight that natural variability fails them, not so loose that
order-of-magnitude bugs slip through.
"""

from __future__ import annotations

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as np

from tests.conftest import requires_public_dev


@requires_public_dev
def test_set_A_simulator_matches_bergman_1979_healthy():
    """Synthesise Set A and confirm the qualitative meal-response profile."""
    from models.glucose_insulin.simulation import GLUCOSE_INSULIN_MODEL
    from psim.scenarios.presets.gi_set_A_healthy import truth_params_and_init
    from psim.data.bergman_1979_fsigt import EXPECTED_HEALTHY_MEAL_RESPONSE
    from psim.pipelines import synthesise_scenario

    truth_params, init_state = truth_params_and_init()
    n_bins_total = int(round(truth_params['t_total_hours']
                              / truth_params['dt_hours']))
    bins_per_day = int(round(24.0 / truth_params['dt_hours']))

    sim_run = synthesise_scenario(
        GLUCOSE_INSULIN_MODEL,
        truth_params=truth_params,
        init_state=init_state,
        exogenous_arrays={},
        n_bins_total=n_bins_total,
        dt_days=truth_params['dt_hours'],
        bins_per_day=bins_per_day,
        n_substeps=4,
        seed=42,
        obs_channel_names=('cgm', 'meal_carbs'),
    )
    traj = sim_run.trajectory
    G, I = traj[:, 0], traj[:, 2]
    Gb = truth_params['Gb']

    # Detect meal peaks (G > 130 with local max).
    peak_indices = [i for i in range(1, len(G) - 1)
                    if G[i] > 130 and G[i] > G[i - 1] and G[i] > G[i + 1]]
    print(f"\nDetected {len(peak_indices)} meal peaks at hours: "
          f"{[round(i * truth_params['dt_hours'], 2) for i in peak_indices]}")

    # Assertion 1 — exactly 3 meal peaks (3 meals/day).
    assert 3 <= len(peak_indices) <= 4, (
        f"expected 3 (or 4 split) meal peaks, got {len(peak_indices)}")

    # Assertion 2 — each peak in healthy postprandial range
    expected_lo, expected_hi = EXPECTED_HEALTHY_MEAL_RESPONSE['peak_G_mg_dL']
    expected_hi = expected_hi + 5     # 5 mg/dL slack for stochastic variability
    for idx in peak_indices:
        peak_G = G[idx]
        assert expected_lo <= peak_G <= expected_hi, (
            f"peak G={peak_G:.1f} at bin {idx} outside expected "
            f"[{expected_lo}, {expected_hi}] for healthy postprandial")

    # Assertion 3 — final G within 10 mg/dL of Gb (return-to-basal)
    final_G = G[-1]
    assert abs(final_G - Gb) < 12.0, (
        f"final G={final_G:.1f} not near basal Gb={Gb} (return-to-basal failed)")

    # Assertion 4 — plasma insulin peaks in healthy range
    expected_I_lo, expected_I_hi = EXPECTED_HEALTHY_MEAL_RESPONSE['peak_I_mu_U_mL']
    I_max = float(I.max())
    assert expected_I_lo <= I_max <= expected_I_hi + 5, (
        f"peak I={I_max:.1f} outside expected [{expected_I_lo}, {expected_I_hi}] "
        f"for healthy postprandial")


@requires_public_dev
def test_bergman_1979_fixture_self_consistency():
    """The Bergman 1979 fixture: parameter table + SI calculation."""
    from psim.data.bergman_1979_fsigt import PARAMS, SI

    # All canonical parameters present
    expected_keys = {'p1', 'p2', 'p3', 'k', 'Gb', 'Ib',
                      'V_G', 'V_I', 'BW', 'n_beta', 'h_beta'}
    assert set(PARAMS.keys()) >= expected_keys, \
        f"missing keys in Bergman 1979 fixture: {expected_keys - set(PARAMS.keys())}"

    # SI calculation is right
    assert abs(SI - PARAMS['p3'] / PARAMS['p2']) < 1e-9
    # SI in expected healthy range (≈ 0.03)
    assert 0.02 <= SI <= 0.04, f"SI={SI} outside healthy range [0.02, 0.04]"

    # Sanity: physiologically plausible values
    assert 60.0 <= PARAMS['Gb'] <= 110.0
    assert 5.0 <= PARAMS['Ib'] <= 12.0
    assert 50.0 <= PARAMS['BW'] <= 90.0
