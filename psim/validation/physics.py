"""Wraps `SDEModel.verify_physics_fn` into a structured assertion report.

Three classes of return value from a model's verify_physics_fn:

  - **Hard pass/fail booleans** (e.g. ``'all_finite': True``) that must
    be True for the trajectory to be usable. Configurable via
    ``required_booleans`` (defaults to ``('all_finite',)``).

  - **Informational booleans** (e.g. ``'mu_crosses_zero'``,
    ``'A_activated'``) that are True/False depending on scenario intent
    and shouldn't gate packaging. Recorded in ``raw_report``.

  - **Realism warnings** (NEW): keys ending in ``_realistic`` with
    string value ``"yes"`` or ``"no"``. ``"no"`` is surfaced as a
    structured warning (in ``warnings`` list and console output), but
    does NOT gate packaging unless ``treat_realism_as_gate=True``.

The realism convention was added so that biological-realism issues
(e.g. SWAT Set A's Zt amplitude limit, see public-dev #5) surface at
the psim packaging stage rather than only at the SMC² downstream
inference stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class PhysicsResult:
    passed: bool
    failed_checks: list = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    raw_report: Dict[str, Any] = field(default_factory=dict)

    def assert_pass(self):
        if not self.passed:
            raise AssertionError(
                f"Physics check FAILED: {self.failed_checks}\n"
                f"warnings: {self.warnings}\n"
                f"raw report: {self.raw_report}"
            )


_DEFAULT_REQUIRED_BOOLEANS = ("all_finite",)
_REALISM_SUFFIX = "_realistic"


def _extract_realism_warnings(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find keys ending in ``_realistic`` whose value is ``'no'`` (or
    False), and pair each with its companion metric value if present.

    For ``'sleep_fraction_realistic': 'no'``, we also look up
    ``'sleep_fraction'`` so the warning carries the actual value
    that triggered it.
    """
    warnings: List[Dict[str, Any]] = []
    for k, v in raw.items():
        if not k.endswith(_REALISM_SUFFIX):
            continue
        if v in ("no", False):
            metric_key = k[:-len(_REALISM_SUFFIX)].rstrip("_")
            entry = {"key": k, "value": v}
            if metric_key in raw:
                entry["metric_key"] = metric_key
                entry["metric_value"] = raw[metric_key]
            warnings.append(entry)
    return warnings


def check_physics(
    model_sim,
    trajectory,
    t_grid,
    params,
    *,
    required_booleans: tuple = _DEFAULT_REQUIRED_BOOLEANS,
    treat_realism_as_gate: bool = False,
) -> PhysicsResult:
    """Run the model's ``verify_physics_fn`` and gate on a configurable
    subset of its boolean keys.

    Parameters
    ----------
    required_booleans : tuple of str
        Keys whose boolean value must be True for the gate to pass.
        Defaults to ``('all_finite',)``.
    treat_realism_as_gate : bool
        When True, any ``*_realistic: 'no'`` flag adds the key to
        ``failed_checks`` and refuses packaging. Defaults to False
        (warnings only — surfaced in console + report but not gating).

    Returns
    -------
    PhysicsResult with ``failed_checks`` (gating violations),
    ``warnings`` (realism-flag misses), ``raw_report`` (everything
    the model returned).
    """
    if model_sim.verify_physics_fn is None:
        return PhysicsResult(
            passed=True,
            failed_checks=[],
            warnings=[],
            raw_report={"status": "no verify_physics_fn defined"},
        )

    raw = model_sim.verify_physics_fn(trajectory, t_grid, params)
    failed = [
        k for k in required_booleans
        if k in raw and isinstance(raw[k], bool) and not raw[k]
    ]
    warnings = _extract_realism_warnings(raw)
    if treat_realism_as_gate:
        failed.extend(w["key"] for w in warnings)
    return PhysicsResult(
        passed=(len(failed) == 0),
        failed_checks=failed,
        warnings=warnings,
        raw_report=dict(raw),
    )


__all__ = ["PhysicsResult", "check_physics"]
