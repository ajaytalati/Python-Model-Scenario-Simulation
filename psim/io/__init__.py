"""Canonical scenario-artifact I/O — the stable interface to SMC²."""

from psim.io.format import (
    SCENARIO_SCHEMA_VERSION,
    Manifest,
    write_artifact,
    read_artifact,
)

__all__ = [
    "SCENARIO_SCHEMA_VERSION",
    "Manifest",
    "write_artifact",
    "read_artifact",
]
