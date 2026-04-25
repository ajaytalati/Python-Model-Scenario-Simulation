# Bridge to the SMC² repo

How the `smc2-blackjax-rolling` driver consumes a packaged scenario
artifact. The bridge is a ~40-line adapter; no invasive change to the
existing rolling-window driver is required.

## Design

- The artifact format ([SCENARIO_FORMAT.md](SCENARIO_FORMAT.md)) is
  the only contract. Both repos pin
  `SCENARIO_SCHEMA_VERSION = "1.0"`.
- The SMC² driver gains an optional `--scenario-artifact <dir>` flag.
  When set, the inline data-generation code path is bypassed and the
  driver consumes the artifact instead.
- The original inline path stays as-is for backward compatibility. A
  follow-up consolidation can deprecate it once enough scenarios have
  migrated.

## The adapter

Drop this into `smc2-blackjax-rolling/drivers/_artifact_loader.py`:

```python
"""Load a Python-Model-Scenario-Simulation scenario artifact.

Mirrors the in-memory shape that the existing rolling-window driver
expects from its inline data-generation step.
"""
from __future__ import annotations

import os
import sys

# Until psim is pip-installable from PyPI, point sys.path at the
# locally-cloned Python-Model-Scenario-Simulation checkout.
_PSIM_ROOT = os.path.expanduser("~/Repos/Python-Model-Scenario-Simulation")
if _PSIM_ROOT not in sys.path:
    sys.path.insert(0, _PSIM_ROOT)

from psim.io.format import read_artifact   # noqa: E402


def load_scenario(artifact_dir: str) -> dict:
    """Return the bundle the rolling-window driver expects.

    Keys: trajectory, obs_channels, exogenous_channels, truth_params,
          n_bins_total, dt_days, bins_per_day, seed.
    """
    a = read_artifact(artifact_dir)
    m = a["manifest"]
    return dict(
        trajectory=a["trajectory"],
        obs_channels=a["obs_channels"],
        exogenous_channels=a["exogenous_channels"],
        truth_params=m.truth_params,
        n_bins_total=m.n_bins_total,
        dt_days=m.dt_days,
        bins_per_day=m.bins_per_day,
        seed=m.seed,
        state_names=m.state_names,
        scenario_name=m.scenario_name,
        model_name=m.model_name,
        model_version=m.model_version,
    )
```

## Wiring it into the driver

In e.g. `drivers/fsa_high_res_rolling.py`, near the top:

```python
import argparse
from drivers._artifact_loader import load_scenario

parser = argparse.ArgumentParser()
parser.add_argument("--scenario-artifact", default=None,
                    help="Path to a Python-Model-Scenario-Simulation artifact dir")
# ... existing flags ...
args = parser.parse_args()
```

Then where the driver currently generates data inline (something like
`bundle = generate_observations(model_sim, ...)`):

```python
if args.scenario_artifact:
    bundle = load_scenario(args.scenario_artifact)
    print(f"[scenario] loaded {bundle['model_name']}/{bundle['scenario_name']} "
          f"({bundle['n_bins_total']} bins, seed={bundle['seed']})")
else:
    bundle = generate_observations(model_sim, ...)   # original inline path
```

Everything downstream — the rolling-window loop, the inner GK-DPF, the
outer tempered SMC, the diagnostics — sees the same in-memory bundle
shape as before. No code change required there.

## Verifying the round-trip

The acceptance test for the bridge is reproduction of the published
result. From a clean clone:

```bash
# 1. In the scenario repo: produce the canonical artifact
cd ~/Repos/Python-Model-Scenario-Simulation
python examples/fsa_high_res/14d_C0_recovery.py
# → outputs/fsa_high_res/C0_recovery_14d/

# 2. In the SMC² repo: consume the artifact
cd ~/Repos/smc2_blackjax_framework
python drivers/fsa_high_res_rolling.py \
    --scenario-artifact ~/Repos/Python-Model-Scenario-Simulation/outputs/fsa_high_res/C0_recovery_14d \
    --seed 42

# 3. Confirm the result file matches the C-fix reference
diff outputs/fsa_high_res_rolling/C_phase_fix_result.md \
     outputs/fsa_high_res_rolling/<new_run_dir>/result.md
```

The expected result is **96.8% mean coverage / 27 of 27 PASS** at
seed 42 — bit-identical to the C-fix reference run.

## Schema-version handshake

`read_artifact` raises `ValueError` if the manifest's
`schema_version` doesn't match the `psim` version pinned in the SMC²
checkout. To upgrade across a schema bump:

1. Bump `psim` in the SMC² environment to the new version.
2. Re-run the artifact producer (or use a pre-built artifact at the
   matching version).
3. Update the compatibility row in
   [ARCHITECTURE.md](ARCHITECTURE.md).

This means a stale artifact never silently feeds a newer driver and a
newer artifact never silently feeds an older driver — both fail loud.

## Why not import the SMC² driver from this repo?

Asymmetric coupling is intentional. The SMC² repo can depend on this
repo (it imports `psim.io.format`); this repo cannot depend on the
SMC² repo (it stays public, the SMC² repo stays private). The
artifact format is the only thing crossing the boundary and the only
thing that needs to be versioned.

## Future: SMC² as an importable runner

For [check C — cold-start coverage](VALIDATION_DISCIPLINE.md), the
SMC² repo would expose an importable
`run_one_window(artifact_dir, n_smc, prior_sigma_scale) -> dict` API.
That removes the v0.1.0 stub and lets `pytest tests/` exercise the
end-to-end discipline without leaving this repo. Tracked as a
follow-up in the SMC² repo's HANDOFF; not in scope for `psim` v0.1.0.
