#!/usr/bin/env python3
"""SIR Set B — small community outbreak. R₀ = 2.5, N=10000, 60 days."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_sir_scenario   # noqa: E402

from psim.scenarios.presets import sir_set_B_small   # noqa: E402


def main():
    return run_sir_scenario(sir_set_B_small)


if __name__ == "__main__":
    sys.exit(main())
