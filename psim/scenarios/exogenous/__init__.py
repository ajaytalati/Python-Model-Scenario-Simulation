"""Exogenous-input scenario generators."""

from psim.scenarios.exogenous.macrocycle import generate_macrocycle_C0
from psim.scenarios.exogenous.morning_loaded import generate_morning_loaded_phi
from psim.scenarios.exogenous.circadian import circadian, make_C_array

__all__ = [
    "generate_macrocycle_C0",
    "generate_morning_loaded_phi",
    "circadian",
    "make_C_array",
]
