#!/usr/bin/env python3
"""SWAT Set C — recovery: 14-day scenario.

Truth: V_h=1.0, V_n=0.3, V_c=0.0 h, T_0=0.05. Expected behaviour:
T rises from 0.05 toward T* ≈ 1.0 over ~4·tau_T = 192 h
(recovery from the flatline basin to the healthy basin).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402

from psim.scenarios.presets import swat_set_C_recovery   # noqa: E402


def main():
    return run_swat_scenario(swat_set_C_recovery)


if __name__ == "__main__":
    sys.exit(main())
