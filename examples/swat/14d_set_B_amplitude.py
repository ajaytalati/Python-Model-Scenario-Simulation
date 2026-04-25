#!/usr/bin/env python3
"""SWAT Set B — amplitude collapse: 14-day scenario.

Truth: V_h=0.2, V_n=3.5, V_c=0.0 h, T_0=0.5. Expected behaviour:
E_dyn → ~0.035, mu(E) → ~-0.45, T decays from 0.5 to ~0.13
(hypogonadal-flatline basin).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402

from psim.scenarios.presets import swat_set_B_amplitude   # noqa: E402


def main():
    return run_swat_scenario(swat_set_B_amplitude)


if __name__ == "__main__":
    sys.exit(main())
