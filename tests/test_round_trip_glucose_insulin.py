"""End-to-end round-trip on a Set A glucose_insulin scenario.

Synthesise → align_obs → re-integrate propagate_fn with zero noise from
truth init → assert recovered (G, X, I) trajectory matches simulator's.
"""

from __future__ import annotations

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as np
import pytest

from tests.conftest import requires_public_dev

from psim.pipelines import synthesise_scenario
from psim.validation import round_trip_check


# Set A — 24 hours at 5-min cadence.
DT = 5.0 / 60.0
N_BINS = int(24.0 / DT)        # 288
BINS_PER_DAY = 288


@pytest.fixture(scope="module")
def gi_sim_run():
    """Synthesise a Set A glucose_insulin scenario."""
    pytest.importorskip("jax")
    from models.glucose_insulin.simulation import GLUCOSE_INSULIN_MODEL
    from psim.scenarios.presets.gi_set_A_healthy import truth_params_and_init

    truth_params, init_state = truth_params_and_init()
    return synthesise_scenario(
        GLUCOSE_INSULIN_MODEL,
        truth_params=truth_params,
        init_state=init_state,
        exogenous_arrays={},
        n_bins_total=N_BINS,
        dt_days=DT,
        bins_per_day=BINS_PER_DAY,
        n_substeps=4,
        seed=42,
        obs_channel_names=('cgm', 'meal_carbs'),
    )


@requires_public_dev
def test_gi_synthesis_in_range(gi_sim_run):
    """Sanity: trajectory shape + physiological-range checks."""
    traj = gi_sim_run.trajectory
    assert traj.shape == (N_BINS, 3), f"got {traj.shape}, expected ({N_BINS}, 3)"
    G, X, I = traj[:, 0], traj[:, 1], traj[:, 2]
    # Healthy ranges
    assert 50.0 <= G.min(), f"G_min {G.min()} below 50"
    assert G.max() <= 220.0, f"G_max {G.max()} above 220 for Set A"
    assert X.min() >= -1e-3, f"X went negative: {X.min()}"
    assert I.min() >= -1e-3, f"I went negative: {I.min()}"
    assert 'cgm' in gi_sim_run.obs_channels
    assert 'meal_carbs' in gi_sim_run.obs_channels


@requires_public_dev
def test_gi_round_trip_set_A(gi_sim_run):
    """Re-integrate propagate_fn with zero noise from truth init —
    recovered (G, X, I) matches simulator within tolerance."""
    pytest.importorskip("jax")
    from models.glucose_insulin.estimation import (
        make_glucose_insulin_estimation, PI,
    )
    from psim.scenarios.presets.gi_set_A_healthy import truth_params_and_init
    from models.glucose_insulin.simulation import _meal_schedule

    truth_params, _ = truth_params_and_init()
    # Build the same meal schedule used at synthesis time
    meal_sched = _meal_schedule(seed=0, n_days=1, meal_carbs_g=40.0)
    sir_estimation = make_glucose_insulin_estimation(
        meal_schedule=meal_sched,
        insulin_schedule=None,
    )

    est_v = np.array([truth_params[k] for k in sorted(PI, key=PI.get)],
                      dtype=np.float64)

    obs_data = dict(gi_sim_run.obs_channels)
    init_state_arr = gi_sim_run.trajectory[0]
    diffusion_diag = sir_estimation.diffusion_fn(est_v)

    res = round_trip_check(
        true_trajectory=gi_sim_run.trajectory,
        align_obs_fn=sir_estimation.align_obs_fn,
        propagate_fn=sir_estimation.propagate_fn,
        obs_data=obs_data,
        window_start=0,
        window_end=N_BINS,
        dt=DT,
        est_params_vec=est_v,
        init_state=init_state_arr,
        diffusion_diag=np.asarray(diffusion_diag),
        # Tolerance: G in [85, 195] mg/dL, propagate_fn includes Pitt-
        # Shephard CGM tilt which biases G toward the noisy CGM at every
        # step. With zero state noise the discretization drift between
        # sim (4 substeps) and est (1 step) accumulates to ~10-20 mg/dL
        # over 288 bins. atol=30 catches off-by-one bugs that would give
        # 100+ mg/dL drift.
        atol=30.0,
    )
    res.assert_pass()
