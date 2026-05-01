#!/usr/bin/env python3
"""SWAT Set F — sedentary: 14-day scenario.

Truth: V_h=0.0, V_n=0.0, V_c=0.0 h, T_0=0.5. Both potentials at zero.
Expected behaviour: V_h = 0 is a HARD ZERO through the entrainment
amplitude factors (per swat_entrainment_docs/03_corner_cases.md §3.1).
E_dyn = 0 → mu = -0.5 → T collapses to zero amplitude.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402

from psim.scenarios.presets import swat_set_F_sedentary   # noqa: E402


def main():
    return run_swat_scenario(swat_set_F_sedentary)


if __name__ == "__main__":
    sys.exit(main())
