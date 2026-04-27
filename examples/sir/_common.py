"""Shared helper for the SIR example scripts.

Each per-set script imports ``run_sir_scenario`` and passes its preset
factory; this module handles the public-dev path injection, model load,
synthesis, validation, packaging, and output-directory layout.
"""

from __future__ import annotations

import os
import sys


# SIR (and its companions) live canonically in the public dev repo
# (Python-Model-Development-Simulation).
PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)


def _ensure_public_dev_on_path():
    if not os.path.isdir(PUBLIC_DEV_V1):
        raise SystemExit(
            f"SIR lives in the public dev repo at {PUBLIC_DEV_V1}. "
            f"Clone https://github.com/ajaytalati/Python-Model-Development-Simulation "
            f"there or set PUBLIC_DEV_V1."
        )
    if PUBLIC_DEV_V1 not in sys.path:
        sys.path.insert(0, PUBLIC_DEV_V1)


def run_sir_scenario(
    preset_module,
    *,
    seed: int = 42,
    dropout_rate: float = 0.0,
):
    """End-to-end: preset → synthesise → validate → package.

    SIR scenarios use the per-scenario t_total_hours and dt_hours
    encoded in PARAM_SET_*; this helper just reads them off the
    truth-params dict.
    """
    _ensure_public_dev_on_path()

    from models.sir.simulation import SIR_MODEL

    from psim.scenarios.missing_data import apply_dropout
    from psim.pipelines import (
        synthesise_scenario, validate_simrun, package_scenario,
    )

    truth_params, init_state = preset_module.truth_params_and_init()
    scenario_name = preset_module.SCENARIO_NAME

    dt_hours = float(truth_params['dt_hours'])
    t_total_hours = float(truth_params['t_total_hours'])
    n_days = int(round(t_total_hours / 24.0))
    bins_per_day = int(round(24.0 / dt_hours))     # 24 at dt=1h
    n_bins_total = int(round(t_total_hours / dt_hours))

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "outputs", "sir", scenario_name,
    )
    out_dir = os.path.abspath(out_dir)

    print(f"=== SIR / {scenario_name} ===")
    print(f"  bins:   {n_bins_total}  ({n_days}d × {bins_per_day}/d)")
    print(f"  dt:     {dt_hours} h")
    print(f"  N:      {truth_params['N']}, β/day = {truth_params['beta']*24:.2f}, "
          f"γ/day = {truth_params['gamma']*24:.2f}, "
          f"R₀ = {truth_params['beta']/truth_params['gamma']:.2f}")
    print(f"  out:    {out_dir}")

    print("\n[1/3] Forward-simulate SDE + run channel DAG (cases + serology)")
    sim_run = synthesise_scenario(
        SIR_MODEL,
        truth_params=truth_params,
        init_state=init_state,
        exogenous_arrays={},
        n_bins_total=n_bins_total,
        dt_days=dt_hours,             # arg is in the SDE's native time unit
        bins_per_day=bins_per_day,
        n_substeps=4,
        seed=seed,
        obs_channel_names=('cases', 'serology'),
    )
    traj = sim_run.trajectory
    N = truth_params['N']
    print(f"   trajectory: {traj.shape}")
    print(f"   S range:  [{traj[:, 0].min():.1f}, {traj[:, 0].max():.1f}]"
          f"   final={traj[-1, 0]:.1f}")
    print(f"   I range:  [{traj[:, 1].min():.1f}, {traj[:, 1].max():.1f}]"
          f"   peak={traj[:, 1].max():.1f}")
    print(f"   R final:  {N - traj[-1, 0] - traj[-1, 1]:.1f}  "
          f"(attack rate = {(N - traj[-1, 0] - traj[-1, 1]) / N:.3f})")
    for ch_name, ch in sim_run.obs_channels.items():
        n_obs = len(ch.get('t_idx', []))
        print(f"   {ch_name}: {n_obs} samples")

    if dropout_rate > 0:
        print(f"\n[2/3] Apply {dropout_rate*100:.0f}% dropout on serology")
        apply_dropout(sim_run.obs_channels, ['serology'],
                       rate=dropout_rate, seed=seed + 200)
    else:
        print("\n[2/3] No dropout applied (full reporting)")

    print("\n[3/3] Validate + package")
    report = validate_simrun(sim_run, SIR_MODEL)
    print(f"   physics passed: {report.physics.passed}")

    artifact_dir = package_scenario(
        sim_run, report,
        out_dir=out_dir,
        model_name="sir",
        model_version="1.0",
        scenario_name=scenario_name,
        require_all_passed=False,
        model_sim=SIR_MODEL,
        emit_diagnostic_plot=True,
    )
    print(f"\nDone. Artifact at: {artifact_dir}")
    return 0
