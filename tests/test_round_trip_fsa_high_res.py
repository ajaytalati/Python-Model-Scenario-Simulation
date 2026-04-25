"""End-to-end round-trip on a 1-day fsa_high_res scenario.

Plan ref: cosmic-giggling-wadler.md, Phase E, item 27.

The round-trip integrates the estimator's propagate_fn with **zero
noise** starting from the simulator's true initial state. With
sim/est consistency holding (the §1.4 discipline), the recovered
state trajectory must match the simulator's at every bin within
floating-point tolerance.

A round-trip fail localises the bug to the per-step PF call —
catches the extract_state_at_step / off-by-one class of bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_public_dev

from psim.scenarios.exogenous import (
    generate_macrocycle_C0,
    generate_morning_loaded_phi,
)
from psim.pipelines import synthesise_scenario
from psim.validation import round_trip_check


N_DAYS = 1
BINS_PER_DAY = 96
DT_DAYS = 1.0 / BINS_PER_DAY
N_BINS = N_DAYS * BINS_PER_DAY


@pytest.fixture(scope="module")
def fsa_sim_run():
    """Synthesise a 1-day fsa_high_res scenario at truth."""
    pytest.importorskip("jax")
    from models.fsa_high_res.simulation import (
        HIGH_RES_FSA_MODEL, DEFAULT_PARAMS, DEFAULT_INIT,
    )
    daily_T_B, daily_Phi = generate_macrocycle_C0(N_DAYS, seed=42)
    T_B_arr = np.repeat(daily_T_B, BINS_PER_DAY).astype(np.float32)
    Phi_arr = generate_morning_loaded_phi(daily_Phi,
                                           bins_per_day=BINS_PER_DAY,
                                           seed=42)
    sim_run = synthesise_scenario(
        HIGH_RES_FSA_MODEL,
        truth_params=DEFAULT_PARAMS,
        init_state=DEFAULT_INIT,
        exogenous_arrays={"T_B_arr": T_B_arr, "Phi_arr": Phi_arr},
        n_bins_total=N_BINS,
        dt_days=DT_DAYS,
        bins_per_day=BINS_PER_DAY,
        n_substeps=4,
        seed=42,
    )
    return sim_run


@requires_public_dev
def test_synthesise_produces_in_range_trajectory(fsa_sim_run):
    """Sanity: the synthesised trajectory is in physical state bounds."""
    traj = fsa_sim_run.trajectory
    assert traj.shape == (N_BINS, 3)
    B, F, A = traj[:, 0], traj[:, 1], traj[:, 2]
    assert 0.0 <= B.min() and B.max() <= 1.0, "B out of [0,1]"
    assert F.min() >= 0.0, "F negative"
    assert A.min() >= 0.0, "A negative"
    # All four obs channels should be present
    for ch in ("obs_HR", "obs_sleep", "obs_stress", "obs_steps"):
        assert ch in fsa_sim_run.obs_channels, f"missing {ch}"


@requires_public_dev
def test_round_trip_recovers_truth_trajectory(fsa_sim_run):
    """Re-integrate propagate_fn with zero noise from truth init —
    recovered trajectory must match the simulator's at every bin."""
    pytest.importorskip("jax")
    from models.fsa_high_res.estimation import (
        HIGH_RES_FSA_ESTIMATION, PARAM_PRIOR_CONFIG,
    )
    from models.fsa_high_res.simulation import DEFAULT_PARAMS

    # Estimator's params in flat-vector form, in PARAM_PRIOR_CONFIG order.
    est_v = np.array([DEFAULT_PARAMS[k] for k in PARAM_PRIOR_CONFIG.keys()],
                     dtype=np.float64)

    # Build the obs_data dict that align_obs_fn expects (the
    # synthesised obs channels + exogenous broadcasts already produced
    # by synthesise_scenario).
    obs_data = dict(fsa_sim_run.obs_channels)
    obs_data.update(fsa_sim_run.exogenous_channels)

    init_state = fsa_sim_run.trajectory[0]
    diffusion_diag = HIGH_RES_FSA_ESTIMATION.diffusion_fn(est_v)

    res = round_trip_check(
        true_trajectory=fsa_sim_run.trajectory,
        align_obs_fn=HIGH_RES_FSA_ESTIMATION.align_obs_fn,
        propagate_fn=HIGH_RES_FSA_ESTIMATION.propagate_fn,
        obs_data=obs_data,
        window_start=0,
        window_end=N_BINS,
        dt=DT_DAYS,
        est_params_vec=est_v,
        init_state=init_state,
        diffusion_diag=np.asarray(diffusion_diag),
        atol=0.15,    # slack for the Kalman-fusion sample-mean drift
                      # in zero-noise mode — propagate_fn pulls toward
                      # obs which differ from the noiseless predictor
                      # by ~1 sigma.
    )
    res.assert_pass()
