"""SWAT Set A — healthy basin.

Truth parameters: re-exported from
``models.swat.simulation.PARAM_SET_A`` and ``INIT_STATE_A`` in the
public dev repo (Python-Model-Development-Simulation).

Reference:
- TESTING.md §4.1 (`models/swat/TESTING.md` in public dev) describes
  the expected behaviour: T equilibrates near
  T* = sqrt(mu/eta) ≈ 0.55 over a 14-day window with the entrainment
  quality E_dyn ≈ 0.55, mu(E) > 0.
- Doc: SWAT_Basic_Documentation.md §6 (basin classification).

Imported lazily — the consumer must have the public dev repo on
sys.path (psim's conftest.py and example scripts handle this).
"""

from __future__ import annotations


SCENARIO_NAME = "set_A_healthy_14d"


def truth_params_and_init():
    """Returns (truth_params: dict, init_state: dict).

    Lazy import so this module is itself import-clean even when the
    public dev repo isn't on sys.path; downstream callers (examples,
    tests gated by ``requires_public_dev``) trigger the resolution.
    """
    from models.swat.simulation import PARAM_SET_A, INIT_STATE_A
    return dict(PARAM_SET_A), dict(INIT_STATE_A)
