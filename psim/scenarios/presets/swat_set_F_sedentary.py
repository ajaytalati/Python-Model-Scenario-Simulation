"""SWAT Set F — sedentary (V_h = 0, V_n = 0).

Both vitality and chronic-load potentials at zero — no anabolic drive,
no chronic stress.

Per swat_entrainment_docs/03_corner_cases.md §3.1 / §3.5, V_h = 0 is a
HARD ZERO through the entrainment amplitude factors:
  A_W = lambda_amp_W · 0 = 0
  A_Z = lambda_amp_Z · 0 = 0
  amp_W = sigma(B_W + 0) - sigma(B_W - 0) = 0  exactly
  amp_Z = sigma(B_Z + 0) - sigma(B_Z - 0) = 0
  E_dyn = damp · 0 · 0 · phase = 0
  mu = mu_0 = -0.5
  T* = 0

Expected behaviour: testosterone collapses to zero amplitude regardless
of how rested the patient is. The model says zero vitality means no
rhythm at all, even with no chronic stress — clinically this is the
de-conditioned sedentary patient who has lost the capacity to
generate a healthy testosterone pulse.

Lazy import — see ``swat_set_A_healthy.py`` for rationale.
"""

from __future__ import annotations


SCENARIO_NAME = "set_F_sedentary_14d"


def truth_params_and_init():
    from models.swat.simulation import PARAM_SET_F, INIT_STATE_F
    return dict(PARAM_SET_F), dict(INIT_STATE_F)
