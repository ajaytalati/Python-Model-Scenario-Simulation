"""End-to-end round-trip on a 14-day SIR scenario (Set A boarding-school).

Synthesise → align_obs → re-integrate propagate_fn with zero noise from
truth init → assert recovered (S, I) trajectory matches the simulator's
at every bin within tolerance.
"""

from __future__ import annotations

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as np
import pytest

from tests.conftest import requires_public_dev

from psim.pipelines import synthesise_scenario
from psim.validation import round_trip_check


# Set A — 14 days at hourly resolution.
DT = 1.0          # 1 hour
N_HOURS = 14 * 24
N_BINS = int(N_HOURS / DT)   # 336 bins
BINS_PER_DAY = 24


@pytest.fixture(scope="module")
def sir_sim_run():
    """Synthesise a Set A SIR scenario (336 bins at 1-hour)."""
    pytest.importorskip("jax")
    from models.sir.simulation import SIR_MODEL
    from psim.scenarios.presets.sir_set_A_boarding_school import truth_params_and_init

    truth_params, init_state = truth_params_and_init()
    return synthesise_scenario(
        SIR_MODEL,
        truth_params=truth_params,
        init_state=init_state,
        exogenous_arrays={},
        n_bins_total=N_BINS,
        dt_days=DT,                       # arg name is misleading; native unit
        bins_per_day=BINS_PER_DAY,
        n_substeps=4,
        seed=42,
        obs_channel_names=('cases', 'serology'),
    )


@requires_public_dev
def test_sir_synthesis_in_range(sir_sim_run):
    """Sanity: trajectory shape + biological-range checks."""
    traj = sir_sim_run.trajectory
    assert traj.shape == (N_BINS, 2), f"got {traj.shape}, expected ({N_BINS}, 2)"
    S, I = traj[:, 0], traj[:, 1]
    N = 763.0
    R = N - S - I
    # Mass conservation (S + I + R = N exactly by construction).
    assert np.abs(S + I + R - N).max() < 1e-3
    # Soft bounds — diffusion approximation can briefly leave [0, N] by
    # ~max(5, 0.5%·N) particles.
    tol = max(5.0, 0.005 * N)
    assert -tol <= S.min(), f"S min {S.min()} below tol -{tol}"
    assert -tol <= I.min(), f"I min {I.min()} below tol -{tol}"
    assert S.max() <= N + tol, f"S max {S.max()} above N+{tol}"
    # Both obs channels generated.
    assert 'cases' in sir_sim_run.obs_channels
    assert 'serology' in sir_sim_run.obs_channels


@requires_public_dev
def test_sir_round_trip_set_A(sir_sim_run):
    """Re-integrate propagate_fn with zero noise from truth init —
    recovered (S, I) must match simulator within tolerance."""
    pytest.importorskip("jax")
    from models.sir.estimation import SIR_ESTIMATION, PI
    from psim.scenarios.presets.sir_set_A_boarding_school import truth_params_and_init

    truth_params, _ = truth_params_and_init()

    # Build est params vec in PI order. Only PARAM_PRIOR_CONFIG keys here;
    # I_0 is in INIT_STATE_PRIOR_CONFIG and is sourced from the trajectory's
    # first row.
    est_v = np.array([truth_params[k] for k in sorted(PI, key=PI.get)],
                      dtype=np.float64)

    obs_data = dict(sir_sim_run.obs_channels)
    init_state_arr = sir_sim_run.trajectory[0]
    diffusion_diag = SIR_ESTIMATION.diffusion_fn(est_v)

    res = round_trip_check(
        true_trajectory=sir_sim_run.trajectory,
        align_obs_fn=SIR_ESTIMATION.align_obs_fn,
        propagate_fn=SIR_ESTIMATION.propagate_fn,
        obs_data=obs_data,
        window_start=0,
        window_end=N_BINS,
        dt=DT,
        est_params_vec=est_v,
        init_state=init_state_arr,
        diffusion_diag=np.asarray(diffusion_diag),
        # SIR is autonomous, 2-state, with explicit Euler on both sides.
        # Discretisation drift (Euler-Maruyama 4-substep simulator vs
        # 1-step IMEX estimator) accumulates linearly over 336 hourly
        # bins; an absolute tolerance of 150 corresponds to ~20% of the
        # ~700-particle scale, still tight enough to catch off-by-one
        # / extract_state bugs that would produce 10× larger errors.
        atol=150.0,
    )
    res.assert_pass()
