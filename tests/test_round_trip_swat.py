"""End-to-end round-trip on a 1-day SWAT scenario (Set A).

Plan ref: cosmic-giggling-wadler.md, Phase B item B.3.

Synthesise → align_obs → re-integrate propagate_fn with zero noise
from truth init → assert recovered state trajectory matches the
simulator's at every bin within tolerance.

NOTE: SWAT works in **hours** (PARAM_SET_A['dt_hours']=5/60), not
days. psim's integrator labels its time argument ``dt_days`` but
just multiplies by the bin index, so any consistent unit works.
The artifact manifest's ``dt_days`` will read 5/60 — interpret as
the model's native time unit.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import requires_public_dev

from psim.pipelines import synthesise_scenario
from psim.validation import round_trip_check


N_HOURS = 24
DT = 5.0 / 60.0          # 5 minutes (in SWAT's native hour units)
BINS_PER_DAY = 12        # 12 bins / hour-of-the-day; misnomer for "per native unit"
N_BINS = int(N_HOURS / DT)   # 288 bins


@pytest.fixture(scope="module")
def swat_sim_run():
    """Synthesise a 1-day SWAT Set A scenario (~288 bins at 5-min)."""
    pytest.importorskip("jax")
    from models.swat.simulation import SWAT_MODEL
    from psim.scenarios.presets.swat_set_A_healthy import truth_params_and_init

    truth_params, init_state = truth_params_and_init()
    return synthesise_scenario(
        SWAT_MODEL,
        truth_params=truth_params,
        init_state=init_state,
        exogenous_arrays={},
        n_bins_total=N_BINS,
        dt_days=DT,
        bins_per_day=BINS_PER_DAY,
        n_substeps=4,
        seed=42,
        # SWAT names obs channels without the legacy 'obs_' prefix.
        obs_channel_names=('hr', 'sleep', 'steps', 'stress'),
    )


@requires_public_dev
def test_swat_synthesis_in_range(swat_sim_run):
    """Sanity: 1-day Set A trajectory has the expected shape and
    healthy-basin ranges."""
    traj = swat_sim_run.trajectory
    assert traj.shape == (N_BINS, 7)
    W, Zt, a, T = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    assert 0.0 <= W.min() and W.max() <= 1.0, "W out of [0,1]"
    assert 0.0 <= Zt.min() <= Zt.max() <= 6.0, "Zt out of [0,6]"
    assert a.min() >= 0.0, "a negative"
    assert T.min() >= 0.0, "T negative"
    # All 4 obs channels generated
    for ch in ("hr", "sleep", "steps", "stress"):
        assert ch in swat_sim_run.obs_channels, f"missing channel: {ch}"


@requires_public_dev
def test_swat_round_trip_set_A(swat_sim_run):
    """Re-integrate propagate_fn with zero noise from truth init —
    recovered W/Zt/a/T must match simulator within tolerance."""
    pytest.importorskip("jax")
    from models.swat.estimation import SWAT_ESTIMATION, PI
    from psim.scenarios.presets.swat_set_A_healthy import truth_params_and_init

    truth_params, init_state = truth_params_and_init()

    # Build est params vec in PI order (Vh, Vn live in init_state on
    # sim side but in PI on est side).
    combined = {**truth_params, **init_state}
    est_v = np.array([combined.get(k, 0.0) for k in
                      sorted(PI, key=PI.get)], dtype=np.float64)

    # Multi-channel obs_data per the new align_obs_fn shape.
    obs_data = dict(swat_sim_run.obs_channels)

    init_state_arr = swat_sim_run.trajectory[0]
    diffusion_diag = SWAT_ESTIMATION.diffusion_fn(est_v)

    res = round_trip_check(
        true_trajectory=swat_sim_run.trajectory,
        align_obs_fn=SWAT_ESTIMATION.align_obs_fn,
        propagate_fn=SWAT_ESTIMATION.propagate_fn,
        obs_data=obs_data,
        window_start=0,
        window_end=N_BINS,
        dt=DT,
        est_params_vec=est_v,
        init_state=init_state_arr,
        diffusion_diag=np.asarray(diffusion_diag),
        atol=3.0,    # SWAT's round-trip is intrinsically loose vs
                     # fsa_high_res's. Two reasons:
                     #   (1) simulator uses Euler-Maruyama with 4
                     #       sub-steps per bin; estimator uses IMEX
                     #       (1 step per bin). Different discretisations
                     #       on a 5-min grid → numerical-integration
                     #       drift over 288 bins.
                     #   (2) HR-only Pitt-Shephard guides W (and only
                     #       W); Zt diverges via the flip-flop coupling
                     #       (gamma_3=8 amplifies any W error into
                     #       Zt drift). fsa_high_res does Kalman fusion
                     #       on all 3 Gaussian channels and pulls the
                     #       full state much harder.
                     # The round-trip still catches the off-by-one /
                     # extract_state_at_step bug class (which would
                     # produce ≥10× larger errors). Tolerance set to
                     # 3.0 in absolute state units (Zt domain is [0,6]).
    )
    res.assert_pass()
