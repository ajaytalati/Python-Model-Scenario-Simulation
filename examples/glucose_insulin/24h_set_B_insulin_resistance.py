#!/usr/bin/env python3
"""glucose_insulin Set B — insulin resistance (pre-T2D)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import run_glucose_insulin_scenario   # noqa: E402

from psim.scenarios.presets import gi_set_B_insulin_resistance   # noqa: E402


def main():
    return run_glucose_insulin_scenario(gi_set_B_insulin_resistance)


if __name__ == "__main__":
    sys.exit(main())
