#!/usr/bin/env python3
"""glucose_insulin Set D — T1D with open-loop insulin schedule.

Sets up the closed-loop control benchmark for Phase 5 follow-up.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_glucose_insulin_scenario   # noqa: E402

from psim.scenarios.presets import gi_set_D_t1d_open_loop   # noqa: E402


def main():
    return run_glucose_insulin_scenario(gi_set_D_t1d_open_loop)


if __name__ == "__main__":
    sys.exit(main())
