#!/usr/bin/env python3
"""End-to-end reference scenario: high_res_FSA, 14 days, C0 macrocycle, recovery.

Produces a packaged scenario artifact that the SMC² repo's driver
consumes (via a thin --scenario-artifact loader). Same seed reproduces
the published 96.8% mean coverage / 27-of-27 PASS result documented
at https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/C_phase_fix_result.md
"""

from __future__ import annotations

import os
import sys

# fsa_high_res lives canonically in the public dev repo
# (Python-Model-Development-Simulation). Add its version_1/ to sys.path
# so `from models.fsa_high_res...` resolves there.
PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)
if not os.path.isdir(PUBLIC_DEV_V1):
    raise SystemExit(
        f"fsa_high_res model lives in the public dev repo at "
        f"{PUBLIC_DEV_V1}. Clone "
        "https://github.com/ajaytalati/Python-Model-Development-Simulation "
        f"there or adjust PUBLIC_DEV_V1."
    )
if PUBLIC_DEV_V1 not in sys.path:
    sys.path.insert(0, PUBLIC_DEV_V1)

import numpy as np

from models.fsa_high_res.simulation import (
    HIGH_RES_FSA_MODEL,
    DEFAULT_PARAMS,
    DEFAULT_INIT,
)

# --- psim primitives + pipelines -----------------------------------------

from psim.scenarios.exogenous import (
    generate_macrocycle_C0,
    generate_morning_loaded_phi,
    make_C_array,
)
from psim.scenarios.missing_data import apply_dropout
from psim.pipelines import (
    synthesise_scenario,
    validate_simrun,
    package_scenario,
)


# --- Scenario constants (matching the SMC² repo's recovery scenario) ----

SCENARIO_NAME = "C0_recovery_14d"
N_DAYS_TOTAL = 14
BINS_PER_DAY = 96
DT_DAYS = 1.0 / BINS_PER_DAY
N_BINS_TOTAL = N_DAYS_TOTAL * BINS_PER_DAY
N_SUBSTEPS = 4
SEED = 42
DROPOUT_RATE = 0.05


def main():
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "outputs", "fsa_high_res", SCENARIO_NAME,
    )

    print(f"=== {SCENARIO_NAME} ===")
    print(f"  bins:    {N_BINS_TOTAL}  ({N_DAYS_TOTAL}d × {BINS_PER_DAY}/d)")
    print(f"  dt:      {DT_DAYS:.5f} day = {DT_DAYS * 24 * 60:.0f} min")
    print(f"  out:     {out_dir}")

    # 1. Daily macrocycle
    print("\n[1/5] Macrocycle C0 (daily)")
    daily_T_B, daily_Phi = generate_macrocycle_C0(N_DAYS_TOTAL, seed=SEED)

    # 2. Sub-daily exogenous arrays
    print("[2/5] Sub-daily exogenous arrays (T_B, morning-loaded Phi, C)")
    T_B_arr = np.repeat(daily_T_B, BINS_PER_DAY).astype(np.float32)
    Phi_arr = generate_morning_loaded_phi(daily_Phi,
                                           bins_per_day=BINS_PER_DAY,
                                           seed=SEED)
    # NOTE: C is computed by the model's gen_C_channel during synthesise,
    # using the global t_grid — which is the C-phase-bug-safe path.

    # 3. Forward-simulate + observation generation (channel DAG)
    print("[3/5] Forward-simulate SDE + run channel DAG")
    sim_run = synthesise_scenario(
        HIGH_RES_FSA_MODEL,
        truth_params=DEFAULT_PARAMS,
        init_state=DEFAULT_INIT,
        exogenous_arrays={"T_B_arr": T_B_arr, "Phi_arr": Phi_arr},
        n_bins_total=N_BINS_TOTAL,
        dt_days=DT_DAYS,
        bins_per_day=BINS_PER_DAY,
        n_substeps=N_SUBSTEPS,
        seed=SEED,
    )
    print(f"   trajectory: {sim_run.trajectory.shape}")
    print(f"   B range: [{sim_run.trajectory[:, 0].min():.3f}, "
          f"{sim_run.trajectory[:, 0].max():.3f}]")
    print(f"   F range: [{sim_run.trajectory[:, 1].min():.3f}, "
          f"{sim_run.trajectory[:, 1].max():.3f}]")
    print(f"   A range: [{sim_run.trajectory[:, 2].min():.3f}, "
          f"{sim_run.trajectory[:, 2].max():.3f}]")
    for ch_name, ch in sim_run.obs_channels.items():
        print(f"   {ch_name}: {len(ch.get('t_idx', []))} samples")

    # 4. Apply minimal missing-data corruption
    print("[4/5] Apply 5% dropout on HR/stress/steps")
    apply_dropout(sim_run.obs_channels, ["obs_HR", "obs_stress", "obs_steps"],
                   rate=DROPOUT_RATE, seed=SEED + 200)

    # 5. Validate + package
    print("[5/5] Validate + package")
    report = validate_simrun(sim_run, HIGH_RES_FSA_MODEL)
    print(f"   physics passed: {report.physics.passed}")

    artifact_dir = package_scenario(
        sim_run, report,
        out_dir=out_dir,
        model_name="fsa_high_res",
        model_version="0.1",
        scenario_name=SCENARIO_NAME,
        require_all_passed=False,    # in v0.1.0 we package even if no consistency checks ran
        # Per-model diagnostic plot in artifact (psim #1)
        model_sim=HIGH_RES_FSA_MODEL,
        emit_diagnostic_plot=True,
    )
    print(f"\nDone. Artifact at: {artifact_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
