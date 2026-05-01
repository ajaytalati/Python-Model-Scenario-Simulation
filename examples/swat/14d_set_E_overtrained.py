#!/usr/bin/env python3
"""SWAT Set E — over-trained athlete: 14-day scenario.

Truth: V_h=1.0, V_n=1.0, V_c=0.0 h, T_0=0.5. Both potentials high.
Expected behaviour: the V_n damper attenuates E_dyn from ≈ 0.85 (Set A)
to ≈ 0.60, marginally super-critical. T sustains at a lower equilibrium
than Set A (T* ≈ 0.44 vs ≈ 0.98), demonstrating the narrow margin
the over-trained athlete sits in.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_swat_scenario   # noqa: E402

from psim.scenarios.presets import swat_set_E_overtrained   # noqa: E402


def main():
    return run_swat_scenario(swat_set_E_overtrained)


if __name__ == "__main__":
    sys.exit(main())
