# Scenario artifact format

The on-disk format that this repo writes and the SMC² repo reads. It
is the **only contract** between the two repos.

Schema version: **`1.0`** (see `SCENARIO_SCHEMA_VERSION` in
[psim/io/format.py](../psim/io/format.py)).

## Directory layout

```
artifact_dir/
├── manifest.json              # schema version, model + scenario metadata,
│                              # truth params, run config, validation summary
├── trajectory.npz             # latent state arrays (n_bins × n_state)
├── obs/
│   ├── obs_HR.npz             # per-channel: keys = t_idx + value(s)
│   ├── obs_sleep.npz
│   ├── obs_stress.npz
│   └── obs_steps.npz
├── exogenous/
│   ├── T_B.npz                # per-channel: keys = t_idx + <name>_value
│   ├── Phi.npz
│   └── C.npz
└── validation/
    ├── report.json            # ValidationReport.to_dict()
    └── *.png                  # optional diagnostic plots
```

Writers and readers live in [psim/io/format.py](../psim/io/format.py)
and don't import any model-specific code, so the artifact is truly
the stable interface.

## `manifest.json`

```json
{
  "schema_version": "1.0",
  "model_name": "fsa_high_res",
  "model_version": "0.1",
  "scenario_name": "C0_recovery_14d",
  "n_bins_total": 1344,
  "dt_days": 0.01041666...,
  "bins_per_day": 96,
  "seed": 42,
  "state_names": ["B", "F", "A"],
  "obs_channels": ["obs_HR", "obs_sleep", "obs_steps", "obs_stress"],
  "exogenous_channels": ["C", "Phi", "T_B"],
  "truth_params": { "...": 0.0 },
  "validation_summary": {
    "all_passed": true,
    "n_passed": 5,
    "n_failed": 0
  }
}
```

Field reference:

| Field | Type | Meaning |
|-------|------|---------|
| `schema_version` | str | **Mandatory.** Reader rejects mismatch. |
| `model_name` | str | Stable model identifier (e.g. `fsa_high_res`). |
| `model_version` | str | Per-model version string. Bump when drift / channels change in a way that breaks artifact compatibility. |
| `scenario_name` | str | The named scenario (e.g. `C0_recovery_14d`). |
| `n_bins_total` | int | Number of global time bins. |
| `dt_days` | float | Bin width in days (e.g. `1/96` ≈ 15 min). |
| `bins_per_day` | int | Convenience: `1/dt_days`. |
| `seed` | int | Top-level RNG seed used during synthesis. Reproducibility is gated on this seed plus `model_name` + `model_version`. |
| `state_names` | list[str] | Latent-state names, in the column order of `trajectory.npz`. |
| `obs_channels` | list[str] | Sorted observation-channel names. |
| `exogenous_channels` | list[str] | Sorted exogenous-channel names. |
| `truth_params` | dict[str, float] | Truth parameters in the simulator's keyword form. |
| `validation_summary` | dict | High-level validation outcome (full report under `validation/report.json`). |

## `trajectory.npz`

Single key:

| Key | Dtype | Shape | Meaning |
|-----|-------|-------|---------|
| `trajectory` | float32 | `(n_bins_total, n_state)` | Latent state at every global bin. Columns indexed by `manifest.state_names`. |

## `obs/<channel>.npz`

Per-channel structure. The exact keys depend on the channel's
generator function in the model's `simulation.py`, but at minimum:

| Key | Dtype | Shape | Meaning |
|-----|-------|-------|---------|
| `t_idx` | int32 | `(n_obs,)` | Global bin indices where this channel produced an observation. After missing-data corruption, dropped bins are absent. |

Plus one or more value arrays. For the high_res_FSA channels:

| Channel | Value keys | Notes |
|---------|------------|-------|
| `obs_HR` | `obs_HR_value: float32` | Sleep-gated; `t_idx` only includes sleep bins. |
| `obs_sleep` | `obs_sleep_value: int32` | Bernoulli {0, 1}. |
| `obs_stress` | `obs_stress_value: float32` | Wake-gated. |
| `obs_steps` | `obs_steps_value: float32` | log-counts; per-day or per-bin per the channel definition. |

Missing-data corruption (`apply_dropout`, `apply_broken_watch_gap`,
`apply_rest_days` from [psim/scenarios/missing_data/](../psim/scenarios/missing_data/))
removes entries from `t_idx` and the value arrays in lockstep, so the
arrays remain index-aligned.

## `exogenous/<name>.npz`

One file per exogenous channel (T_B, Phi, C, ...). Same structure as
obs files:

| Key | Dtype | Shape | Meaning |
|-----|-------|-------|---------|
| `t_idx` | int32 | `(n_bins_total,)` | All global bin indices (no gating). |
| `<name>_value` | float32 | `(n_bins_total,)` | The exogenous signal at every bin. |

The C array is stored on the **global time grid** (the C-phase-bug-safe
representation). The SMC² adapter is expected to slice the global
array per window, **not** to recompute C from window-local time. See
[VALIDATION_DISCIPLINE.md](VALIDATION_DISCIPLINE.md).

## `validation/report.json`

Serialised [`ValidationReport`](../psim/pipelines/validate.py):

```json
{
  "all_passed": true,
  "n_passed": 5,
  "n_failed": 0,
  "physics": {
    "passed": true,
    "failed_checks": [],
    "raw_report": { "all_finite": true, "mu_crosses_zero": false, "...": "..." }
  },
  "consistency": [
    { "name": "drift_parity", "passed": true, "max_abs_err": 1.2e-9, "max_rel_err": 3e-12, "details": { "...": "..." } },
    { "name": "obs_prediction_parity::obs_HR", "passed": true, "max_abs_err": 0.0, "max_rel_err": 0.0, "details": { "...": "..." } },
    "..."
  ],
  "round_trip": { "passed": true, "max_abs_err": 0.0, "details": { "...": "..." } }
}
```

The full schema is whatever `ValidationReport.to_dict()` produces; do
not write parsers that assume a fixed key set beyond `all_passed`,
`n_passed`, and `n_failed`. Treat the rest as audit metadata.

## Versioning rules

- `SCENARIO_SCHEMA_VERSION` is bumped whenever a backwards-incompatible
  change is made to **any** of the files above (manifest fields,
  trajectory dtype, obs key layout, exogenous representation).
- Both writer and reader pin the version. The reader raises on
  mismatch.
- The compatibility row in [ARCHITECTURE.md](ARCHITECTURE.md) tracks
  which `psim` version emits which `SCENARIO_SCHEMA_VERSION` and which
  SMC² commit consumes it.
- Backwards-compatible additions (e.g. extra optional manifest fields
  with sensible defaults) are allowed within a major schema version.

## Reading an artifact

```python
from psim.io.format import read_artifact

a = read_artifact("outputs/fsa_high_res/C0_recovery_14d")
manifest          = a["manifest"]               # Manifest dataclass
trajectory        = a["trajectory"]             # (n_bins, n_state) float32
obs_channels      = a["obs_channels"]           # dict[name -> dict]
exogenous_channels = a["exogenous_channels"]    # dict[name -> dict]
validation_report = a["validation_report"]      # dict or None
```

Self-contained example: see [examples/fsa_high_res/14d_C0_recovery.py](../examples/fsa_high_res/14d_C0_recovery.py)
for the writer side and [BRIDGE_TO_SMC2.md](BRIDGE_TO_SMC2.md) for the
reader side as used by the SMC² driver.
