"""Run all validation checks against a SimRun + return a structured report.

The validation pipeline runs the §1.4 mandatory checks plus optional
visual diagnostics, aggregates them into a single dict that lands in
the artifact's ``validation/report.json``, and gates packaging on all
mandatory checks passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from psim.pipelines.synthesise import SimRun
from psim.validation import (
    ConsistencyResult,
    PhysicsResult,
    check_physics,
)


@dataclass
class ValidationReport:
    """Aggregated validation result over a SimRun."""

    all_passed: bool = False
    n_passed: int = 0
    n_failed: int = 0
    physics: PhysicsResult | None = None
    consistency: List[ConsistencyResult] = field(default_factory=list)
    plot_paths: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            all_passed=self.all_passed,
            n_passed=self.n_passed,
            n_failed=self.n_failed,
            n_warnings=len(self.physics.warnings) if self.physics else 0,
            physics=(
                None if self.physics is None
                else dict(passed=self.physics.passed,
                          failed_checks=self.physics.failed_checks,
                          warnings=self.physics.warnings,
                          raw_report=self.physics.raw_report)
            ),
            consistency=[
                dict(name=c.name, passed=c.passed,
                     max_abs_err=c.max_abs_err,
                     max_rel_err=c.max_rel_err,
                     details=c.details)
                for c in self.consistency
            ],
            plot_paths=self.plot_paths,
            notes=self.notes,
        )


def validate_simrun(
    sim_run: SimRun,
    model_sim,
    *,
    consistency_checks: List[ConsistencyResult] | None = None,
    plot_paths: List[str] | None = None,
    treat_realism_as_gate: bool = False,
    print_warnings: bool = True,
) -> ValidationReport:
    """Aggregate physics + consistency checks into a single report.

    Surfaces realism warnings (``*_realistic: 'no'`` flags from the
    model's ``verify_physics_fn``) to the console. When
    ``treat_realism_as_gate=True``, realism failures also fail the
    physics gate (refuse packaging).

    Note: caller is responsible for running the consistency checks
    (because they need both the SDEModel and EstimationModel) and
    passing the results in. This pipeline orchestrator records and
    aggregates them; it doesn't duplicate them.
    """
    physics_result = check_physics(
        model_sim,
        sim_run.trajectory,
        np.arange(sim_run.n_bins_total) * sim_run.dt_days,
        sim_run.truth_params,
        treat_realism_as_gate=treat_realism_as_gate,
    )

    # Surface realism warnings to console — these are the
    # biological-realism flags that indicate a model-tuning concern
    # (the simulator's truth doesn't look like real wearable data).
    # See the workflow-gate observation on public-dev #5.
    if print_warnings and physics_result.warnings:
        print(f"   physics WARNINGS ({len(physics_result.warnings)} realism flag(s)):")
        for w in physics_result.warnings:
            metric_str = ""
            if "metric_value" in w:
                v = w["metric_value"]
                metric_str = (
                    f" ({w['metric_key']} = {v:.4g})"
                    if isinstance(v, float)
                    else f" ({w['metric_key']} = {v!r})"
                )
            print(f"     WARN: {w['key']} = {w['value']!r}{metric_str}")

    consistency_checks = list(consistency_checks or [])
    plot_paths = list(plot_paths or [])

    n_passed = int(physics_result.passed) + sum(c.passed for c in consistency_checks)
    n_total = 1 + len(consistency_checks)
    n_failed = n_total - n_passed

    report = ValidationReport(
        all_passed=(n_failed == 0),
        n_passed=n_passed,
        n_failed=n_failed,
        physics=physics_result,
        consistency=consistency_checks,
        plot_paths=plot_paths,
    )
    return report


__all__ = ["ValidationReport", "validate_simrun"]
