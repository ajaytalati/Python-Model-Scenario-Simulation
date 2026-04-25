"""SWAT Set C — recovery (T rises from a near-zero start).

Truth parameters: same as Set A but ``init_state['T_0']=0.05`` —
starts near the 0-flatline. Healthy V_h, V_n, V_c. Expect T to rise
toward the equilibrium T* ≈ 1.0 over ≈4·tau_T = 192 h.

Reference:
- TESTING.md §4.3 (public dev): expected T(end day 14) ≈ 0.9–1.0;
  basin label = healthy (recovery from the flatline basin).
- SWAT_Basic_Documentation.md §6.C.

Lazy import — see ``swat_set_A_healthy.py`` for rationale.
"""

from __future__ import annotations


SCENARIO_NAME = "set_C_recovery_14d"


def truth_params_and_init():
    from models.swat.simulation import PARAM_SET_C, INIT_STATE_C
    return dict(PARAM_SET_C), dict(INIT_STATE_C)
