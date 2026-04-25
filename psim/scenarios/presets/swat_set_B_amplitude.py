"""SWAT Set B — amplitude collapse (severe insomnia / hyperarousal).

Truth parameters: same dynamical params as Set A; differs in
``init_state['Vh']=0.2, Vn=3.5``. Drives E_dyn → ~0.035, mu(E) → ~-0.45,
T decays from 0.5 → ~0.13 over 14 days (the hypogonadal-flatline
basin).

Reference:
- TESTING.md §4.2 (public dev): expected T(end day 14) ≈ 0.13;
  basin label = flatline.
- SWAT_Basic_Documentation.md §6.B.

Lazy import — see ``swat_set_A_healthy.py`` for rationale.
"""

from __future__ import annotations


SCENARIO_NAME = "set_B_amplitude_collapse_14d"


def truth_params_and_init():
    from models.swat.simulation import PARAM_SET_B, INIT_STATE_B
    return dict(PARAM_SET_B), dict(INIT_STATE_B)
