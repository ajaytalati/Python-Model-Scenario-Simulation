"""SIR Set A — Anderson & May 1978 boarding-school flu (paper-parity).

The canonical PMCMC tutorial benchmark. N=763 students, 14-day flu
outbreak, R₀ ≈ 3.32. Used by Endo et al 2019 Epidemics 29 as the
reference state-space inference target.

Truth parameters re-exported from
``models.sir.simulation.PARAM_SET_A`` and ``INIT_STATE_A`` in the
public dev repo (Python-Model-Development-Simulation).

Reference daily prevalence series (for paper-parity tests):
``psim.data.anderson_may_1978_flu.DAILY_PREVALENCE``.

Imported lazily — the consumer must have the public dev repo on
sys.path (psim's conftest.py and example scripts handle this).
"""

from __future__ import annotations


SCENARIO_NAME = "set_A_boarding_school_14d"


def truth_params_and_init():
    """Returns (truth_params: dict, init_state: dict)."""
    from models.sir.simulation import PARAM_SET_A, INIT_STATE_A
    return dict(PARAM_SET_A), dict(INIT_STATE_A)
