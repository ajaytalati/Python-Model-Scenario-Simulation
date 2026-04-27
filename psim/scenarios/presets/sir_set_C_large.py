"""SIR Set C — large community outbreak.

Synthetic high-R₀: R₀ = 4.0, N=10000 community, 90 days, 50% case
detection. Near-total infection, attack rate ~98%.

Imported lazily — see other SIR presets for the pattern.
"""

from __future__ import annotations


SCENARIO_NAME = "set_C_large_outbreak_90d"


def truth_params_and_init():
    from models.sir.simulation import PARAM_SET_C, INIT_STATE_C
    return dict(PARAM_SET_C), dict(INIT_STATE_C)
