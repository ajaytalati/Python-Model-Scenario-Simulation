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


def check_physics(model_sim, trajectory, t_grid, params) -> PhysicsResult:
    """Run the model's ``verify_physics_fn`` and report any failed
    boolean checks.

    The simulator's ``verify_physics_fn`` returns a dict where
    boolean-valued keys are pass/fail flags (e.g. ``'all_finite': True``,
    ``'mu_crosses_zero': False``). Non-boolean entries are diagnostic
    values that we surface but don't gate on.
    """
    if model_sim.verify_physics_fn is None:
        return PhysicsResult(
            passed=True,
            failed_checks=[],
            raw_report={"status": "no verify_physics_fn defined"},
        )

    raw = model_sim.verify_physics_fn(trajectory, t_grid, params)
    failed = [k for k, v in raw.items() if isinstance(v, bool) and not v]
    return PhysicsResult(
        passed=(len(failed) == 0),
        failed_checks=failed,
        raw_report=dict(raw),
    )


__all__ = ["PhysicsResult", "check_physics"]
