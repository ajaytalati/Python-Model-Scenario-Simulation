"""Mandatory pre-SMC² validation discipline (executable)."""

from psim.validation.consistency import (
    ConsistencyResult,
    check_drift_parity,
    check_obs_prediction_parity,
    check_cold_start_coverage,
)
from psim.validation.data_flow import plot_covariate_alignment, plot_obs_alignment
from psim.validation.round_trip import RoundTripResult, round_trip_check
from psim.validation.physics import PhysicsResult, check_physics

__all__ = [
    "ConsistencyResult",
    "check_drift_parity",
    "check_obs_prediction_parity",
    "check_cold_start_coverage",
    "plot_covariate_alignment",
    "plot_obs_alignment",
    "RoundTripResult",
    "round_trip_check",
    "PhysicsResult",
    "check_physics",
]
