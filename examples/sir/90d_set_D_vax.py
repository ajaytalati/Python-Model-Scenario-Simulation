#!/usr/bin/env python3
"""SIR Set D — vaccination intervention. R₀ = 3.0, N=10000, 90 days, v=0.02/day.

Sets up the closed-loop control benchmark for the next-stage SMC²
Phase 5 work (out of scope for this initial pipeline rollout).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_sir_scenario   # noqa: E402

from psim.scenarios.presets import sir_set_D_vax   # noqa: E402


def main():
    return run_sir_scenario(sir_set_D_vax)


if __name__ == "__main__":
    sys.exit(main())
