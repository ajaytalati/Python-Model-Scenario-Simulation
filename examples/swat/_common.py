"""Shared helper for the SWAT example scripts.

Each per-set script imports ``run_swat_scenario`` and passes its preset
factory; this module handles the public-dev path injection, model load,
synthesis, validation, packaging, and output-directory layout.
"""

from __future__ import annotations

import os
import sys


# fsa_high_res lives canonically in the public dev repo
# (Python-Model-Development-Simulation). Add its version_1/ to sys.path
# so `from models.swat...` resolves there.
PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)


def _ensure_public_dev_on_path():
    if not os.path.isdir(PUBLIC_DEV_V1):
        raise SystemExit(
            f"SWAT lives in the public dev repo at {PUBLIC_DEV_V1}. "
            f"Clone https://github.com/ajaytalati/Python-Model-Development-Simulation "
            f"there or set PUBLIC_DEV_V1."
        )
    if PUBLIC_DEV_V1 not in sys.path:
        sys.path.insert(0, PUBLIC_DEV_V1)


def run_swat_scenario(
    preset_module,
    *,
    n_days: int = 14,
    dt_hours: float = 5.0 / 60.0,
    n_substeps: int = 4,
    seed: int = 42,
    dropout_rate: float = 0.05,
):
    """End-to-end: preset → synthesise → validate → package."""
    _ensure_public_dev_on_path()

    from models.swat.simulation import SWAT_MODEL

    from psim.scenarios.missing_data import apply_dropout
    from psim.pipelines import (
        synthesise_scenario, validate_simrun, package_scenario,
    )

    truth_params, init_state = preset_module.truth_params_and_init()
    scenario_name = preset_module.SCENARIO_NAME

    bins_per_day = int(round(24.0 / dt_hours))   # 288 at 5-min
    n_bins_total = n_days * bins_per_day

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "outputs", "swat", scenario_name,
    )
    out_dir = os.path.abspath(out_dir)

    print(f"=== SWAT / {scenario_name} ===")
    print(f"  bins:   {n_bins_total}  ({n_days}d × {bins_per_day}/d)")
    print(f"  dt:     {dt_hours} h = {dt_hours*60:.0f} min "
          f"(SWAT's native time unit)")
    print(f"  out:    {out_dir}")

    print("\n[1/3] Forward-simulate SDE + run channel DAG (4 channels)")
    sim_run = synthesise_scenario(
        SWAT_MODEL,
        truth_params=truth_params,
        init_state=init_state,
        exogenous_arrays={},                # SWAT has no exogenous inputs
        n_bins_total=n_bins_total,
        dt_days=dt_hours,                   # parameter is misnamed; actually SWAT's hours
        bins_per_day=bins_per_day,
        n_substeps=n_substeps,
        seed=seed,
        obs_channel_names=('hr', 'sleep', 'steps', 'stress'),
    )
    traj = sim_run.trajectory
    print(f"   trajectory: {traj.shape}")
    print(f"   W range:  [{traj[:, 0].min():.3f}, {traj[:, 0].max():.3f}]"
          f"   mean={traj[:, 0].mean():.3f}")
    print(f"   Zt range: [{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}]"
          f"   mean={traj[:, 1].mean():.3f}")
    print(f"   T range:  [{traj[:, 3].min():.3f}, {traj[:, 3].max():.3f}]"
          f"   mean={traj[:, 3].mean():.3f}")
    for ch_name, ch in sim_run.obs_channels.items():
        print(f"   {ch_name}: {len(ch.get('t_idx', []))} samples")

    print(f"\n[2/3] Apply {dropout_rate*100:.0f}% dropout on hr/stress (sleep, "
          f"steps preserved)")
    apply_dropout(sim_run.obs_channels, ['hr', 'stress'],
                   rate=dropout_rate, seed=seed + 200)

    print("\n[3/3] Validate + package")
    report = validate_simrun(sim_run, SWAT_MODEL)
    print(f"   physics passed: {report.physics.passed}")

    artifact_dir = package_scenario(
        sim_run, report,
        out_dir=out_dir,
        model_name="swat",
        model_version="1.0",
        scenario_name=scenario_name,
        require_all_passed=False,
    )
    print(f"\nDone. Artifact at: {artifact_dir}")
    return 0
