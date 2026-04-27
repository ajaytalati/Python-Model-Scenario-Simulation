"""Paper-parity test: SIR Set A vs Anderson & May 1978 boarding-school flu.

Synthesises the Set A scenario at the published truth parameters and
compares the resulting daily-prevalence trajectory to the canonical
14-day series from Anderson & May 1991 Table 6.1 (the de-facto PMCMC
benchmark used by Endo et al 2019 *Epidemics* 29).

This is the **published-data parity** test: it confirms that
"simulator + truth params" produces output consistent with the
real-world benchmark. The complementary "inference recovers truth from
synthetic data" test lives downstream in the SMC² repo (`drivers/sir/`).

The assertions match what a reviewer would want to verify:
  - Peak prevalence within ±25% of Anderson-May's 298
  - Peak day within ±2 days of Anderson-May's day 6
  - Final attack rate within ±0.1 of Anderson-May's ~0.94
  - Sum of squared deviations across 14 days, normalised by peak,
    below 0.6 (loose; SDE-simulator vs deterministic reference)

These tolerances are deliberately loose because:
  - The reference series is a single observed realisation, not a
    deterministic theory curve.
  - Our simulator includes diffusion noise (Itô SDE).
  - The data is daily prevalence (boys confined to bed at morning roll-call),
    a single integer-valued snapshot of the continuous I(t) trajectory.

The point is *order-of-magnitude* parity: if any of these checks fail
by an order of magnitude, the model is broken or the fixture is wrong.
"""

from __future__ import annotations

import os
os.environ['JAX_ENABLE_X64'] = 'True'

import numpy as np
import pytest

from tests.conftest import requires_public_dev


@requires_public_dev
def test_set_A_simulator_matches_anderson_may_1978():
    """Synthesise Set A and compare daily prevalence to Anderson-May 1991 Table 6.1."""
    from models.sir.simulation import SIR_MODEL
    from psim.scenarios.presets.sir_set_A_boarding_school import truth_params_and_init
    from psim.data.anderson_may_1978_flu import (
        ANDERSON_MAY_1978_FLU, DAILY_PREVALENCE, peak_day, peak_prevalence,
    )
    from psim.pipelines import synthesise_scenario

    truth_params, init_state = truth_params_and_init()
    n_bins_total = int(round(truth_params['t_total_hours'] / truth_params['dt_hours']))   # 336 bins
    bins_per_day = int(round(24.0 / truth_params['dt_hours']))                            # 24

    sim_run = synthesise_scenario(
        SIR_MODEL,
        truth_params=truth_params,
        init_state=init_state,
        exogenous_arrays={},
        n_bins_total=n_bins_total,
        dt_days=truth_params['dt_hours'],
        bins_per_day=bins_per_day,
        n_substeps=4,
        seed=42,
        obs_channel_names=('cases', 'serology'),
    )
    traj = sim_run.trajectory     # (336, 2) — [S, I]
    I = traj[:, 1]
    N = truth_params['N']

    # Daily prevalence: I(t) at the end of each 24-hour bin.
    daily_prev_sim = np.array([
        I[(d + 1) * bins_per_day - 1] for d in range(14)
    ])
    daily_prev_ref = np.array(DAILY_PREVALENCE)

    # Diagnostic — emit on test failure to make debugging fast.
    print(f"\nSimulator daily prevalence: {daily_prev_sim.round().astype(int).tolist()}")
    print(f"Anderson-May 1978 reference:  {daily_prev_ref.tolist()}")

    # Assertion 1 — peak prevalence within ±25%.
    sim_peak = float(daily_prev_sim.max())
    ref_peak = float(peak_prevalence())   # 298
    rel_err = abs(sim_peak - ref_peak) / ref_peak
    assert rel_err < 0.25, (
        f"Peak prevalence: simulator {sim_peak:.0f} vs Anderson-May {ref_peak} "
        f"(rel err {rel_err:.1%}, tolerance 25%)")

    # Assertion 2 — peak day within ±2 days.
    sim_peak_day = int(daily_prev_sim.argmax()) + 1   # 1-indexed
    ref_peak_day = peak_day()                          # 6
    assert abs(sim_peak_day - ref_peak_day) <= 2, (
        f"Peak day: simulator day {sim_peak_day} vs Anderson-May day {ref_peak_day}")

    # Assertion 3 — final attack rate within ±0.10 of expected ~0.94.
    final_R = N - traj[-1, 0] - traj[-1, 1]
    attack_rate = float(final_R / N)
    expected_attack = 1.0 - np.exp(
        -ANDERSON_MAY_1978_FLU['truth_params']['R_0'] * 0.94)   # final-size approx
    assert abs(attack_rate - 0.94) < 0.10, (
        f"Attack rate: simulator {attack_rate:.3f} vs Anderson-May ~0.94 "
        f"(diff {abs(attack_rate - 0.94):.3f}, tolerance 0.10)")

    # Assertion 4 — overall trajectory shape: normalised SSD < 0.6.
    ssd = float(np.sum((daily_prev_sim - daily_prev_ref) ** 2))
    norm_ssd = ssd / (ref_peak ** 2 * 14)
    assert norm_ssd < 0.6, (
        f"Overall trajectory shape: normalised SSD {norm_ssd:.3f} > tolerance 0.6")


@requires_public_dev
def test_anderson_may_fixture_self_consistency():
    """The Anderson-May fixture itself: peak day = day 6, peak value = 298."""
    from psim.data.anderson_may_1978_flu import (
        DAILY_PREVALENCE, POPULATION, DAYS, peak_day, peak_prevalence,
    )

    assert DAYS == 14
    assert POPULATION == 763
    assert peak_day() == 6, f"expected peak on day 6, got day {peak_day()}"
    assert peak_prevalence() == 298, f"expected peak prevalence 298, got {peak_prevalence()}"
    # All values are non-negative integers.
    assert all(isinstance(d, int) and d >= 0 for d in DAILY_PREVALENCE)
    # Outbreak shape: tails small, peak in the middle.
    assert DAILY_PREVALENCE[0] < 50    # day 1 small
    assert DAILY_PREVALENCE[-1] < 50   # day 14 small
    assert max(DAILY_PREVALENCE) > 100 # peak large
