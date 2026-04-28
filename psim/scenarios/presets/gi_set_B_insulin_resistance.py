"""Glucose-insulin Set B — insulin resistance (pre-T2D).

`p₃` halved relative to Set A → SI = p₃/p₂ drops 50%. Tissues respond
less to insulin; meal peaks slightly higher (185-205 mg/dL); β-cells
still functional so insulin response remains in normal range.

24-hour trial, same meal schedule as Set A.
"""

from __future__ import annotations


SCENARIO_NAME = "set_B_insulin_resistance_24h"


def truth_params_and_init():
    from models.glucose_insulin.simulation import PARAM_SET_B, INIT_STATE_B
    return dict(PARAM_SET_B), dict(INIT_STATE_B)
