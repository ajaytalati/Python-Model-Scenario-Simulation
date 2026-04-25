# Vendored from Python-Model-Development-Simulation

Source: https://github.com/ajaytalati/Python-Model-Development-Simulation
Vendored at commit: `51544049f120` (main branch as of 2026-04-25)

Files:
- `simulator/sde_model.py` — SDEModel, StateSpec, ChannelSpec dataclasses
- `simulator/sde_observations.py` — channel DAG dispatcher
- `simulator/sde_solver_scipy.py` — Euler-Maruyama integrator (numpy)
- `simulator/__init__.py`
- `estimation_model.py` — EstimationModel contract
- `_likelihood_constants.py` — HALF_LOG_2PI

## Why vendored?

The upstream public dev repo currently has no `pyproject.toml` /
`setup.py`, so `pip install git+https://...` is not yet possible.
Vendoring keeps THIS repo self-contained and pip-installable.

## Tech debt: when to unvendor

Once the upstream repo gains a `pyproject.toml` (separate small PR),
delete this directory and replace with a runtime dep in the new repo's
`pyproject.toml`:

    Python-Model-Development-Simulation @ git+https://github.com/ajaytalati/Python-Model-Development-Simulation.git@<tag>

Then change all `from psim._vendored.simulator.X` imports to
`from simulator.X`. Trivial sed pass.

## Refresh procedure

To pull a newer upstream snapshot, re-run
`scripts/refresh_vendored.sh` (TODO: add this script when needed).
