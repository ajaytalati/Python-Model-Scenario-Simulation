# Architecture

Three repositories, one workflow. This doc explains the responsibility
split, the data that flows between them, and the rule for deciding
where new code belongs.

## Three-repo diagram

```
┌──────────────────────────────────────────────┐
│  Python-Model-Development-Simulation         │   public
│  (model definitions only)                    │
│                                              │
│  version_1/                                  │
│  ├── simulator/        SDEModel,             │
│  │                     EstimationModel,      │
│  │                     contracts             │
│  └── models/<name>/    simulation.py         │
│                        estimation.py         │
│                        sim_plots.py          │
└──────────────────┬───────────────────────────┘
                   │  pip install (vendored
                   │  in psim/_vendored/ until
                   │  upstream gains pyproject)
                   ▼
┌──────────────────────────────────────────────┐
│  Python-Model-Scenario-Simulation  (THIS)    │   public
│                                              │
│  psim/                                       │
│  ├── scenarios/   exogenous + missing-data   │
│  ├── validation/  drift / obs / round-trip   │
│  ├── pipelines/   synthesise + validate +    │
│  │                package                    │
│  ├── io/          canonical artifact format  │
│  └── cli/         psim {synthesise|...}      │
└──────────────────┬───────────────────────────┘
                   │  scenario artifact
                   │  (directory on disk;
                   │  format frozen at
                   │  SCENARIO_SCHEMA_VERSION)
                   ▼
┌──────────────────────────────────────────────┐
│  smc2-blackjax-rolling                       │   private
│  (SMC² estimation)                           │
│                                              │
│  drivers/<model>_rolling.py                  │
│      --scenario-artifact <dir>               │
│  smc2bj/                                     │
│  ├── filter/      inner GK-DPF particle      │
│  │                filter                     │
│  └── pipeline/    outer tempered SMC,        │
│                   rolling-window driver      │
└──────────────────────────────────────────────┘
```

## Responsibilities

| Repo | What it owns | What it does NOT own |
|------|--------------|----------------------|
| Public dev repo | `SDEModel` + `EstimationModel` contracts; per-model `simulation.py` / `estimation.py` / `sim_plots.py` | scenario generation, validation orchestration, CI infra |
| **THIS repo** | reusable scenario primitives (training-load profiles, circadian forcings, missing-data patterns); the **mandatory** sim-est consistency discipline; end-to-end pipelines; the canonical artifact format | inner-PF / SMC² estimation; the rolling-window driver |
| SMC² repo | inner GK-DPF, outer SMC², rolling-window driver, posterior diagnostics | scenario generation (consumes via artifact) |

## Data flow

1. **Definition.** A model lands in the public dev repo following the
   `how_to_add_a_new_model/` checklist. The 3-file convention
   guarantees both an `SDEModel` (for the simulator) and an
   `EstimationModel` (for the estimator) exist, sharing a parameter
   vector layout.

2. **Validation.** In **this** repo, scenario primitives are composed
   into a full scenario, the SDE is forward-integrated, observation
   channels are generated through the model's channel DAG, and the
   three §1.4 consistency checks run against the truth parameters.
   See [VALIDATION_DISCIPLINE.md](VALIDATION_DISCIPLINE.md).

3. **Packaging.** The validated `SimRun` + `ValidationReport` are
   written to disk in the canonical artifact format. See
   [SCENARIO_FORMAT.md](SCENARIO_FORMAT.md).

4. **Estimation.** The SMC² driver is invoked with
   `--scenario-artifact <dir>`; a tiny adapter (≈40 lines) loads the
   artifact and feeds it to the existing rolling-window driver
   unchanged. See [BRIDGE_TO_SMC2.md](BRIDGE_TO_SMC2.md).

The **scenario artifact is the only contract** between this repo and
the SMC² repo. Both sides pin `SCENARIO_SCHEMA_VERSION` (currently
`"1.0"`); the loader rejects mismatched versions.

## Where does new code belong?

A useful rule of thumb:

| If you are adding... | It belongs in... |
|----------------------|------------------|
| a new `SDEModel` definition (drift, diffusion, channels) | public dev repo |
| a new estimator front-end for an existing model | public dev repo |
| a new scenario primitive (a kind of training schedule, a missing-data pattern) | this repo, under `psim/scenarios/` |
| a new validation check (catches a class of sim-est bug) | this repo, under `psim/validation/` |
| a per-model preset / scenario recipe | this repo, under `psim/scenarios/presets/` or `examples/<model>/` |
| an SMC² hyperparameter tuner | SMC² repo |
| a posterior diagnostic | SMC² repo |
| a change to the artifact format | this repo (`psim/io/format.py`) **and** bump `SCENARIO_SCHEMA_VERSION` and update the SMC² adapter |

## Why a middle repo at all?

The historical pattern (high_res_FSA followed it inadvertently and
paid ≈15 hours of GPU + analyst time):

1. Sketch a model in the public dev repo.
2. Drop it into the SMC² repo with quick scenario glue.
3. Run SMC², get bad coverage, spend a week chasing the bug in the
   posterior-diagnostic surface.

The full case study is the
[POSTMORTEM_three_bugs](https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md)
in the SMC² repo. Three sim/est consistency bugs (a sign-flipped
`mu_0`, an off-by-one in `extract_state_at_step`, and a phase-misaligned
`C(t)` covariate) presented in the posterior as a phantom "bridge
cascade" problem and prompted two whole research-track plans before
the user spotted the actual cause by eye on a parameter-tracking plot.

The **discipline that would have caught all three bugs in 30 minutes**
is now codified as runnable code in `psim/validation/`. The middle
repo exists so future models cannot skip it.

## Vendoring of the framework

`psim/_vendored/` contains the `simulator/` package + `EstimationModel`
from the public dev repo at the commit recorded in
[psim/\_vendored/PROVENANCE.md](../psim/_vendored/PROVENANCE.md). This
is a **temporary** measure: the public dev repo currently has no
`pyproject.toml`, so it cannot be `pip install`-ed via `git+https`.

When the public dev repo gains a `pyproject.toml`, the unvendor
procedure is documented in `PROVENANCE.md` — delete `_vendored/`,
add a `git+https://github.com/ajaytalati/Python-Model-Development-Simulation.git@<tag>`
dependency to `pyproject.toml`, and rewrite `from psim._vendored.simulator`
imports back to `from simulator`.

## Compatibility

| psim version | scenario schema | tested public-dev commit | tested SMC² commit |
|--------------|-----------------|---------------------------|---------------------|
| `0.1.0` | `1.0` | `51544049` | (the C-fix commit on `main`) |

When bumping `SCENARIO_SCHEMA_VERSION`, add a row here and update both
the SMC² adapter and the public dev repo's compatibility note.
