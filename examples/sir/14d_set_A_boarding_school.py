#!/usr/bin/env python3
"""SIR Set A — Anderson & May 1978 boarding-school flu (paper-parity).

The canonical PMCMC tutorial benchmark: N=763, R₀ ≈ 3.32, 14 days.
Reference daily-prevalence series in
``psim.data.anderson_may_1978_flu``.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_sir_scenario   # noqa: E402

from psim.scenarios.presets import sir_set_A_boarding_school   # noqa: E402


def main():
    return run_sir_scenario(sir_set_A_boarding_school)


if __name__ == "__main__":
    sys.exit(main())
