"""SWAT Set D — phase-shift pathology (chronic shift work / jet lag).

Truth parameters: healthy V_h=1.0, V_n=0.3 but ``V_c=6.0`` h
(subject's rhythm is 6 hours delayed relative to external light).
Both W and Zt swing with full amplitude, but their peak timing is
mis-aligned with C(t). The entrainment-quality phase-correlation
term drops toward 0, driving E below E_crit and collapsing T from
0.5 → ~0.11.

This is the fourth failure mode the (V_h, V_n) potentials alone
cannot produce: phase misalignment with healthy amplitude.

Reference:
- TESTING.md §4.4 (public dev): expected T(end day 14) ≈ 0.11;
  basin label = flatline (phase-shift pathology).
- SWAT_Basic_Documentation.md §6.D and SWAT_Clinical_Specification.md
  §H7 (V_c distinguishability claim).

Lazy import — see ``swat_set_A_healthy.py`` for rationale.
"""

from __future__ import annotations


SCENARIO_NAME = "set_D_phase_shift_14d"


def truth_params_and_init():
    from models.swat.simulation import PARAM_SET_D, INIT_STATE_D
    return dict(PARAM_SET_D), dict(INIT_STATE_D)
