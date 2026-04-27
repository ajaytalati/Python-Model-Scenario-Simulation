"""SIR Set D — vaccination intervention.

Sets up the closed-loop control benchmark: R₀ = 3.0, N=10000, 90 days,
sustained vaccination v(t) = 0.02/day (≈ 8.3e-4/hr) from t = 0.

The vaccination rate is encoded as a constant in PARAM_SET_D (frozen
on the estimation side via DEFAULT_FROZEN_PARAMS override). Future
work (Phase 5 of the SMC² rollout) will add piecewise-time v(t)
schedules and closed-loop optimal-control synthesis on top of the
inferred posterior.

Imported lazily — see other SIR presets for the pattern.
"""

from __future__ import annotations


SCENARIO_NAME = "set_D_vaccination_90d"


def truth_params_and_init():
    from models.sir.simulation import PARAM_SET_D, INIT_STATE_D
    return dict(PARAM_SET_D), dict(INIT_STATE_D)
