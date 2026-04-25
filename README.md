# Python-Model-Scenario-Simulation

**The mandatory pre-SMC² stage of the modelling workflow.** Bridges the
gap between [Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation)
(model definitions) and the SMC² estimation framework
(`smc2-blackjax-rolling`, private).

## Why this repo exists

Three sim/est consistency bugs cost ~15 hours of GPU + analyst time
during the high_res_FSA development (full case study:
[POSTMORTEM_three_bugs](https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md)
in the SMC² repo). Each bug would have been caught in <30 minutes by
end-to-end data-flow tests at the simulator-vs-estimator level.

This repo turns those tests into **mandatory, automated, machine-checkable
pre-conditions** for any model entering the SMC² pipeline. No model
should be ported to SMC² without first passing the validation discipline
defined here.

## Three-repo workflow

```
Python-Model-Development-Simulation       (public, defines models)
              │
              ▼
Python-Model-Scenario-Simulation          (THIS repo: scenarios, validation, packaging)
              │
              ▼
smc2-blackjax-rolling                     (private, SMC² estimation)
```

| Repo | Owns |
|------|------|
| Public dev repo | model definitions (3-file convention: `simulation.py` / `estimation.py` / `sim_plots.py`); `SDEModel` + `EstimationModel` contracts |
| **THIS repo** | **scenario primitives, sim-est consistency checks, end-to-end pipelines, packaged scenario artifacts** |
| SMC² repo | inner PF, outer tempered SMC, rolling-window driver; consumes scenario artifacts via a thin loader |

## Quickstart

```bash
git clone https://github.com/ajaytalati/Python-Model-Scenario-Simulation
cd Python-Model-Scenario-Simulation
pip install -e ".[dev]"

# Run the FSA high-res reference scenario end-to-end
python examples/fsa_high_res/14d_C0_recovery.py

# Run the validation discipline
pytest tests/
```

## Workflow gate for new SMC² models

**Every new model added to the SMC² repo MUST first pass through this
repo.** Including SWAT (planned). The discipline:

1. Add the model to the public dev repo (3-file convention).
2. **In this repo**: add scenario primitives → write tests → produce a
   validated scenario artifact via `psim/pipelines/`. All three
   §1.4 consistency checks (drift parity, obs-prediction parity,
   cold-start coverage) must pass.
3. **Then** port to the SMC² repo as a thin adapter that reads the
   validated artifact.

See [`docs/ADDING_A_MODEL.md`](docs/ADDING_A_MODEL.md) for the
checklist and [`docs/VALIDATION_DISCIPLINE.md`](docs/VALIDATION_DISCIPLINE.md)
for what the checks do.

## Repo layout

```
psim/                            # the importable package
├── scenarios/                   # reusable scenario primitives
│   ├── exogenous/               # macrocycle, morning-loaded Phi, circadian C(t)
│   ├── missing_data/            # dropout, broken-watch, rest-day patterns
│   └── presets/                 # named full-scenario configs per model
├── validation/                  # mandatory pre-SMC² checks (executable)
├── pipelines/                   # synthesise + validate + package
├── io/                          # canonical SMC²-bound artifact format
├── cli/                         # `psim synthesise|validate|package`
└── _vendored/                   # framework code from public dev repo (see ARCHITECTURE)
tests/                           # pytest suite
examples/                        # working per-model scripts
docs/                            # architecture + porting + format + bridge
```

## Status

v0.1.0 (initial release):
- Reference implementation: high_res_FSA model from the SMC² repo.
- Validates → packages a 14-day rolling scenario that the SMC² driver
  consumes to reproduce the published 96.8% mean coverage / 27-of-27
  PASS result.

## Provenance

Three-repo design proposed by user 2026-04-25 in response to the
high_res_FSA postmortem. Built on the model conventions from the public
dev repo and the SMC² framework lessons documented in
`docs/PORTING_GUIDE.md` of the SMC² repo.
