"""SWAT Set E — over-trained athlete (V_h = 1, V_n = 1).

Both vitality AND chronic-load potentials simultaneously high — analogous
to the B-and-F-both-high regime in the FSA model. Vitality keeps the
entrainment amplitude bands wide (amp_W, amp_Z near saturation), but
the chronic-load damper exp(-V_n / V_n_scale) = exp(-0.5) ≈ 0.61
attenuates E_dyn by 39%.

Reference state at a=0.5, T=0.85 (per swat_entrainment_docs/04_worked_examples.md):
  B_W = 1 - 0.5 + 0.255 = 0.755
  B_Z = -1 + 4·0.5 = 1.0
  A_W = 5,  A_Z = 8
  amp_W = sigma(5.755) - sigma(-4.245) ≈ 0.997 - 0.014 ≈ 0.983
  amp_Z = sigma(9) - sigma(-7) ≈ 1.000 - 0.001 ≈ 0.999
  damp  = exp(-1/2) ≈ 0.607
  phase = 1.0
  E_dyn ≈ 0.595 → mu ≈ +0.095 → T* ≈ 0.44

Expected behaviour: marginally super-critical — T sustains at a lower
equilibrium than Set A (T*≈0.44 vs ≈0.98), demonstrating the narrow
margin against testosterone collapse that the over-trained athlete
sits in. A small perturbation (e.g. extra V_n) would tip into Set B.

Lazy import — see ``swat_set_A_healthy.py`` for rationale.
"""

from __future__ import annotations


SCENARIO_NAME = "set_E_overtrained_14d"


def truth_params_and_init():
    from models.swat.simulation import PARAM_SET_E, INIT_STATE_E
    return dict(PARAM_SET_E), dict(INIT_STATE_E)
