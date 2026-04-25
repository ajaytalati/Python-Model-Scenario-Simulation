#!/usr/bin/env python3
"""SWAT Set A — healthy basin: 14-day scenario.

Truth: V_h=1.0, V_n=0.3, V_c=0.0 h, T_0=0.5. Expected behaviour:
T equilibrates near T* = sqrt(mu/eta) ≈ 0.55 over 14 days.
"""

from __future__ import annotations

import os
import sys

# Make examples/swat/_common importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402

from psim.scenarios.presets import swat_set_A_healthy   # noqa: E402


def main():
    return run_swat_scenario(swat_set_A_healthy)


if __name__ == "__main__":
    sys.exit(main())
