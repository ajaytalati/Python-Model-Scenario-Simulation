"""Wraps `SDEModel.verify_physics_fn` into a structured assertion report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PhysicsResult:
    passed: bool
    failed_checks: list = field(default_factory=list)
    raw_report: Dict[str, Any] = field(default_factory=dict)

    def assert_pass(self):
        if not self.passed:
            raise AssertionError(
                f"Physics check FAILED: {self.failed_checks}\n"
                f"raw report: {self.raw_report}"
            )


_DEFAULT_REQUIRED_BOOLEANS = ("all_finite",)


def check_physics(
    model_sim,
    trajectory,
    t_grid,
    params,
    *,
    required_booleans: tuple = _DEFAULT_REQUIRED_BOOLEANS,
) -> PhysicsResult:
    """Run the model's ``verify_physics_fn`` and gate on a configurable
    subset of its boolean keys.

    Models often return a mix of:
      - **Hard pass/fail booleans** (e.g. ``'all_finite': True``) that
        must be True for the trajectory to be usable.
      - **Informational booleans** (e.g. ``'mu_crosses_zero'``,
        ``'A_activated'``) that are True/False depending on the
        scenario's intent and shouldn't gate packaging.

    Pass ``required_booleans`` to opt specific keys into the gate. By
    default only ``all_finite`` is gated. All other booleans (and all
    non-bool diagnostics) are still recorded in ``raw_report``.
    """
    if model_sim.verify_physics_fn is None:
        return PhysicsResult(
            passed=True,
            failed_checks=[],
            raw_report={"status": "no verify_physics_fn defined"},
        )

    raw = model_sim.verify_physics_fn(trajectory, t_grid, params)
    failed = [
        k for k in required_booleans
        if k in raw and isinstance(raw[k], bool) and not raw[k]
    ]
    return PhysicsResult(
        passed=(len(failed) == 0),
        failed_checks=failed,
        raw_report=dict(raw),
    )


__all__ = ["PhysicsResult", "check_physics"]
