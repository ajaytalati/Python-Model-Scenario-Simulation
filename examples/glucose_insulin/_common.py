"""Shared helper for the glucose-insulin example scripts."""

from __future__ import annotations

import os
import sys


PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)


def _ensure_public_dev_on_path():
    if not os.path.isdir(PUBLIC_DEV_V1):
        raise SystemExit(
            f"glucose_insulin lives in the public dev repo at {PUBLIC_DEV_V1}. "
            f"Clone https://github.com/ajaytalati/Python-Model-Development-Simulation "
            f"there or set PUBLIC_DEV_V1."
        )
    if PUBLIC_DEV_V1 not in sys.path:
        sys.path.insert(0, PUBLIC_DEV_V1)


def run_glucose_insulin_scenario(
    preset_module,
    *,
    seed: int = 42,
    dropout_rate: float = 0.0,
):
    """End-to-end: preset → synthesise → validate → package.

    glucose_insulin scenarios use the per-scenario t_total_hours and
    dt_hours encoded in PARAM_SET_*. dt_hours=5/60 (5-min CGM cadence),
    t_total_hours=24 (1-day trial).
    """
    _ensure_public_dev_on_path()

    from models.glucose_insulin.simulation import GLUCOSE_INSULIN_MODEL

    from psim.scenarios.missing_data import apply_dropout
    from psim.pipelines import (
        synthesise_scenario, validate_simrun, package_scenario,
    )

    truth_params, init_state = preset_module.truth_params_and_init()
    scenario_name = preset_module.SCENARIO_NAME

    dt_hours = float(truth_params['dt_hours'])
    t_total_hours = float(truth_params['t_total_hours'])
    n_days = int(round(t_total_hours / 24.0))
    bins_per_day = int(round(24.0 / dt_hours))    # 288 at dt=5min
    n_bins_total = int(round(t_total_hours / dt_hours))

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "outputs", "glucose_insulin", scenario_name,
    )
    out_dir = os.path.abspath(out_dir)

    print(f"=== glucose_insulin / {scenario_name} ===")
    print(f"  bins:   {n_bins_total} ({n_days}d × {bins_per_day}/d)")
    print(f"  dt:     {dt_hours:.4f} h ({dt_hours * 60:.0f} min)")
    print(f"  Bergman: p1={truth_params['p1']:.2f}/hr, p2={truth_params['p2']:.2f}/hr, "
          f"p3={truth_params['p3']:.2e}/(hr²·μU/mL), "
          f"SI={truth_params['p3']/truth_params['p2']:.4f}/(hr·μU/mL)")
    print(f"  Ib={truth_params['Ib']:.1f} μU/mL, n_β={truth_params['n_beta']:.1f}, "
          f"insulin_schedule={'on' if truth_params.get('insulin_schedule_active') else 'off'}")
    print(f"  out:    {out_dir}")

    print("\n[1/3] Forward-simulate Bergman SDE + run channel DAG (CGM + meal carbs)")
    sim_run = synthesise_scenario(
        GLUCOSE_INSULIN_MODEL,
        truth_params=truth_params,
        init_state=init_state,
        exogenous_arrays={},
        n_bins_total=n_bins_total,
        dt_days=dt_hours,             # arg name misnomer; native unit is hours
        bins_per_day=bins_per_day,
        n_substeps=4,
        seed=seed,
        obs_channel_names=('cgm', 'meal_carbs'),
    )
    traj = sim_run.trajectory
    print(f"   trajectory: {traj.shape}")
    print(f"   G range:  [{traj[:, 0].min():.1f}, {traj[:, 0].max():.1f}]   "
          f"mean={traj[:, 0].mean():.1f}")
    print(f"   X range:  [{traj[:, 1].min():.3f}, {traj[:, 1].max():.3f}]   "
          f"peak={traj[:, 1].max():.3f}")
    print(f"   I range:  [{traj[:, 2].min():.2f}, {traj[:, 2].max():.2f}]   "
          f"peak={traj[:, 2].max():.2f}")
    for ch_name, ch in sim_run.obs_channels.items():
        n_obs = len(ch.get('t_idx', []))
        print(f"   {ch_name}: {n_obs} samples")

    if dropout_rate > 0:
        print(f"\n[2/3] Apply {dropout_rate*100:.0f}% CGM dropout")
        apply_dropout(sim_run.obs_channels, ['cgm'],
                       rate=dropout_rate, seed=seed + 200)
    else:
        print("\n[2/3] No dropout applied")

    print("\n[3/3] Validate + package")
    report = validate_simrun(sim_run, GLUCOSE_INSULIN_MODEL)
    print(f"   physics passed: {report.physics.passed}")

    artifact_dir = package_scenario(
        sim_run, report,
        out_dir=out_dir,
        model_name="glucose_insulin",
        model_version="1.0",
        scenario_name=scenario_name,
        require_all_passed=False,
        model_sim=GLUCOSE_INSULIN_MODEL,
        emit_diagnostic_plot=True,
    )
    print(f"\nDone. Artifact at: {artifact_dir}")
    return 0
