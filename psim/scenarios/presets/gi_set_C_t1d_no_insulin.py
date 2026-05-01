"""Glucose-insulin Set C — T1D no-control.

Ib=0, n_β=0 → β-cells destroyed, no endogenous insulin secretion, no
exogenous insulin doses. Glucose climbs into mild hyperglycemia
(G_max ≈ 210 mg/dL); plasma insulin remains essentially zero. The
1-day trial captures missed-dose-style elevation rather than full DKA.

24-hour trial, same meal schedule as Set A.
"""

from __future__ import annotations


SCENARIO_NAME = "set_C_t1d_no_insulin_24h"


def truth_params_and_init():
    from models.glucose_insulin.simulation import PARAM_SET_C, INIT_STATE_C
    return dict(PARAM_SET_C), dict(INIT_STATE_C)
