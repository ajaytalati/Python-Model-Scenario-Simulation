"""Tiny CLI: `psim` entrypoint.

Currently a thin shell that exposes the pipelines to the command line.
Each subcommand expects a Python module path that defines a callable
``build_scenario()`` returning a ``(model_sim, kwargs)`` tuple ready
for ``psim.pipelines.synthesise_scenario``.

For v0.1.0 the recommended workflow is to call the Python API directly
from a small example script (see ``examples/fsa_high_res/``); the CLI
is a future convenience.
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="psim", description="Scenario simulation CLI")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("version")

    args = p.parse_args(argv)

    if args.cmd == "version":
        from psim import __version__
        print(__version__)
        return 0

    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
