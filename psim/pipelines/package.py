"""Bundle a SimRun + ValidationReport into the canonical artifact format."""

from __future__ import annotations

import os

import numpy as np

from psim.io.format import Manifest, write_artifact
from psim.pipelines.synthesise import SimRun
from psim.pipelines.validate import ValidationReport


def package_scenario(
    sim_run: SimRun,
    validation_report: ValidationReport,
    *,
    out_dir: str,
    model_name: str,
    model_version: str = "",
    scenario_name: str = "",
    require_all_passed: bool = True,
    model_sim=None,
    emit_diagnostic_plot: bool = True,
) -> str:
    """Write the canonical scenario artifact directory.

    Parameters
    ----------
    sim_run : SimRun
    validation_report : ValidationReport
    out_dir : str — directory to create
    model_name : str — e.g. ``'fsa_high_res'``
    model_version : str — e.g. ``'0.1'``
    scenario_name : str — e.g. ``'C0_recovery_14d'``
    require_all_passed : bool — if True, raise on any failed validation
        check before writing. Set False during development to inspect
        partial artifacts.
    model_sim : SDEModel-like, optional — the model object. When
        provided AND ``emit_diagnostic_plot=True`` AND ``model_sim``
        has a non-None ``plot_fn``, package writes a per-model
        diagnostic plot (the model's own plot_fn output) into the
        artifact directory. Lets a human reviewer inspect the
        simulated truth visually before downstream consumers (SMC²,
        analysts) pick up the artifact.
    emit_diagnostic_plot : bool — default True. Set False for headless
        / fast runs.

    Returns
    -------
    out_dir : str (absolute path written)
    """
    if require_all_passed and not validation_report.all_passed:
        raise RuntimeError(
            f"Refusing to package: validation failed "
            f"({validation_report.n_failed} of "
            f"{validation_report.n_passed + validation_report.n_failed} checks). "
            f"Pass require_all_passed=False to inspect."
        )

    manifest = Manifest(
        model_name=model_name,
        model_version=model_version,
        scenario_name=scenario_name,
        n_bins_total=sim_run.n_bins_total,
        dt_days=sim_run.dt_days,
        bins_per_day=sim_run.bins_per_day,
        seed=sim_run.seed,
        state_names=sim_run.state_names,
        obs_channels=sorted(sim_run.obs_channels.keys()),
        exogenous_channels=sorted(sim_run.exogenous_channels.keys()),
        truth_params={k: float(v) for k, v in sim_run.truth_params.items()},
        validation_summary=dict(
            all_passed=validation_report.all_passed,
            n_passed=validation_report.n_passed,
            n_failed=validation_report.n_failed,
            n_warnings=(
                len(validation_report.physics.warnings)
                if validation_report.physics else 0
            ),
        ),
    )

    write_artifact(
        out_dir,
        manifest=manifest,
        trajectory=sim_run.trajectory,
        obs_channels=sim_run.obs_channels,
        exogenous_channels=sim_run.exogenous_channels,
        validation_report=validation_report.to_dict(),
    )

    # Per-model diagnostic plot (psim #1). Lets a human reviewer
    # inspect the simulated truth at packaging time, catching
    # model-tuning issues that the §1.4 consistency tests can't see
    # (those test sim/est consistency, not biological realism).
    if (emit_diagnostic_plot
            and model_sim is not None
            and getattr(model_sim, "plot_fn", None) is not None):
        try:
            t_grid = (np.arange(sim_run.n_bins_total, dtype=np.float32)
                      * sim_run.dt_days)
            channel_outputs = {**sim_run.obs_channels,
                               **sim_run.exogenous_channels}
            model_sim.plot_fn(
                sim_run.trajectory, t_grid, channel_outputs,
                sim_run.truth_params, out_dir,
            )
        except Exception as e:
            # Non-fatal — packaging proceeds without the plot.
            print(f"  warning: model.plot_fn failed: {e!r}")

    return os.path.abspath(out_dir)


__all__ = ["package_scenario"]
