#!/usr/bin/env python3
"""SWAT Set D — phase-shift pathology (chronic shift work / jet lag).

Truth: V_h=1.0, V_n=0.3, V_c=6.0 h, T_0=0.5. Expected behaviour:
healthy potentials but rhythm 6h delayed → phase quality drops →
E_dyn collapses → T → ~0.11 over 14 days. The fourth failure mode
that (V_h, V_n) alone cannot produce.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402

from psim.scenarios.presets import swat_set_D_phase_shift   # noqa: E402


def main():
    return run_swat_scenario(swat_set_D_phase_shift)


if __name__ == "__main__":
    sys.exit(main())
