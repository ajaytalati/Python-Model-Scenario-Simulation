"""Glucose-insulin Set D — T1D with open-loop insulin schedule.

Ib=0, n_β=0 (T1D); but exogenous insulin delivered via a fixed bolus +
basal schedule: 1 U per 10 g carbs (typical I:C ratio) bolus at each
meal time, plus 0.5 U/hr basal. Peak G ≈ 175-180 mg/dL (back to near-
normal post-meal); plasma insulin peaks 25-35 μU/mL after each bolus.

This sets up the **closed-loop control benchmark** — the next-stage
follow-up replaces the open-loop schedule with an MPC controller that
takes the inferred posterior from rolling SMC² and proposes optimal
insulin doses minimising time-out-of-range.

24-hour trial, same meal schedule as Set A.
"""

from __future__ import annotations


SCENARIO_NAME = "set_D_t1d_open_loop_24h"


def truth_params_and_init():
    from models.glucose_insulin.simulation import PARAM_SET_D, INIT_STATE_D
    return dict(PARAM_SET_D), dict(INIT_STATE_D)
