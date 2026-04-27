"""SIR Set B — small community outbreak.

Synthetic baseline: R₀ = 2.5, N=10000 community, 60 days, 50% case
detection. Smooth full-cycle outbreak, attack rate ~89%.

Imported lazily — see other SIR presets for the pattern.
"""

from __future__ import annotations


SCENARIO_NAME = "set_B_small_outbreak_60d"


def truth_params_and_init():
    from models.sir.simulation import PARAM_SET_B, INIT_STATE_B
    return dict(PARAM_SET_B), dict(INIT_STATE_B)
