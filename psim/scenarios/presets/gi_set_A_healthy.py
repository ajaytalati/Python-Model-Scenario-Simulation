"""Glucose-insulin Set A — healthy adult (Bergman 1979 paper-parity).

Truth parameters re-exported from
``models.glucose_insulin.simulation.PARAM_SET_A`` and ``INIT_STATE_A``
in the public dev repo. These values are the canonical Bergman 1979
healthy-cohort means tabulated in ``psim.data.bergman_1979_fsigt``.

24-hour trial, 3 meals/day at 08:00 / 13:00 / 19:00. Postprandial
peak G ≈ 175-185 mg/dL, return to basal Gb=90 within ~1.5 hr; plasma
insulin peaks 45-55 μU/mL.

Imported lazily — the consumer must have the public dev repo on
sys.path (psim's conftest.py and example scripts handle this).
"""

from __future__ import annotations


SCENARIO_NAME = "set_A_healthy_24h"


def truth_params_and_init():
    """Returns (truth_params: dict, init_state: dict)."""
    from models.glucose_insulin.simulation import PARAM_SET_A, INIT_STATE_A
    return dict(PARAM_SET_A), dict(INIT_STATE_A)
