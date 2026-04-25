# Adding a model

Step-by-step checklist for adding a new model to the modelling
workflow. This is the workflow gate: every model that ends up in the
SMC² repo must pass through this checklist first.

## Prerequisites

The model must already exist in the public dev repo
([Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation))
under `version_1/models/<model_name>/`, with the 3-file convention in
place:

- `simulation.py` — exports an `SDEModel` instance plus
  `DEFAULT_PARAMS` / `DEFAULT_INIT`.
- `estimation.py` — exports an `EstimationModel` instance with
  `propagate_fn`, `align_obs_fn`, and observation predictors that share
  the parameter-vector layout used by `simulation.py`.
- `sim_plots.py` — diagnostic plotters.

If the model is missing the 3-file structure, do that first using the
public dev repo's `how_to_add_a_new_model/` checklist. **Stop and fix
that before starting here.**

## Checklist

### 1. Scenario primitives — `psim/scenarios/`

- [ ] If the model needs a new training-load profile, add it under
  `psim/scenarios/exogenous/<name>.py`. Patterns to imitate:
  [macrocycle.py](../psim/scenarios/exogenous/macrocycle.py),
  [morning_loaded.py](../psim/scenarios/exogenous/morning_loaded.py),
  [circadian.py](../psim/scenarios/exogenous/circadian.py).
- [ ] If the model has new missing-data semantics (e.g. a watch model
  that drops data on rest days), add a new module under
  [psim/scenarios/missing_data/](../psim/scenarios/missing_data/).
- [ ] If the scenario is named (e.g. "C0_recovery_14d"), add a preset
  under `psim/scenarios/presets/<model>_<scenario>.py`.

### 2. Synthesis test — `tests/test_scenario_<model>.py`

- [ ] Add a small (1-day) scenario fixture.
- [ ] Assert the per-channel observation arrays have the expected
  shapes and units.
- [ ] Assert exogenous arrays (T_B, Phi, C, ...) have the expected
  daily structure (e.g. Phi peaks at the configured wake hour).

### 3. Validation discipline — `tests/test_consistency_<model>.py`

This is the **non-negotiable** step. All three checks below must pass
on this model before it goes anywhere near the SMC² repo. The full
discipline is documented in [VALIDATION_DISCIPLINE.md](VALIDATION_DISCIPLINE.md).

- [ ] **Drift parity** —
  [`check_drift_parity`](../psim/validation/consistency.py).
  Compare `model_sim.drift_fn` vs the deterministic part of
  `model_est.propagate_fn` at the truth parameters and a representative
  state. Catches sign flips, missing terms, wrong parameter indexing.
- [ ] **Obs-prediction parity per Gaussian channel** —
  [`check_obs_prediction_parity`](../psim/validation/consistency.py).
  For every Gaussian observation channel, compare the simulator's
  noiseless mean against the estimator's predictor at the same
  `(state, time, k)` and the truth parameters. Catches the
  C-phase / covariate-misalignment bug class — the one that masqueraded
  as a "bridge cascade" problem in high_res_FSA.
- [ ] **Cold-start coverage** —
  [`check_cold_start_coverage`](../psim/validation/consistency.py).
  Single-window cold-start with truth-tight prior should yield
  ≥95% coverage on every estimable scalar. End-to-end check; if A and
  B pass but this fails, the bug is in the SMC² stack not the data
  alignment. (Currently a stub for v0.1.0; exercise via the SMC²
  driver — see [BRIDGE_TO_SMC2.md](BRIDGE_TO_SMC2.md).)

### 4. Round-trip test — `tests/test_round_trip_<model>.py`

- [ ] Use [`round_trip_check`](../psim/validation/round_trip.py) on a
  short scenario: synthesise → extract one window → re-integrate the
  estimator's `propagate_fn` with zero noise from the truth initial
  state → assert the recovered trajectory matches the simulator's at
  every grid point.
- [ ] Catches the `extract_state_at_step` / off-by-one class of bug
  (where the estimator believes it is at bin `k` but the simulator
  produced bin `k±1`).

### 5. Working example — `examples/<model>/<scenario>.py`

- [ ] End-to-end script that runs `synthesise_scenario` →
  `validate_simrun` → `package_scenario`, producing a complete
  artifact under `outputs/<model>/<scenario>/`. Pattern to imitate:
  [examples/fsa_high_res/14d_C0_recovery.py](../examples/fsa_high_res/14d_C0_recovery.py).
- [ ] Add a one-page README under `examples/<model>/README.md`
  explaining what the scenario is, how to run it, and which downstream
  results the artifact reproduces.

### 6. CI

- [ ] Confirm `pytest tests/` is green locally.
- [ ] Push and confirm CI is green on Python 3.10/3.11/3.12.

### 7. SMC² port (in the **other** repo)

Only after steps 1-6 are green:

- [ ] In the SMC² repo, add `drivers/<model>_rolling.py` that loads the
  scenario artifact via `psim.io.format.read_artifact` and feeds it to
  the existing rolling-window SMC² driver. The reference adapter is in
  [BRIDGE_TO_SMC2.md](BRIDGE_TO_SMC2.md).
- [ ] Run the SMC² driver against the artifact and confirm the
  expected coverage. If coverage is bad, **don't** start
  hyperparameter-tuning the SMC² stack; revisit consistency checks 3A
  and 3B first — that's where every published bug to date has lived.

## What this gate prevents

Three bug classes that have actually shipped:

- **Sign-flipped drift term** (the `mu_0` bug in high_res_FSA) →
  caught by check 3A in <1 minute.
- **Phase-misaligned covariate across windows** (the `C(t)` bug) →
  caught by check 3B in <1 minute. This one cost 12 hours of GPU and
  prompted two phantom research-track plans before the user spotted it
  by eye.
- **Window-local vs global indexing mismatch** (the
  `extract_state_at_step` bug) → caught by step 4 (round-trip check).

The full case study is in
[POSTMORTEM_three_bugs](https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md).

## Reference: the high_res_FSA model

[examples/fsa_high_res/](../examples/fsa_high_res/) is the canonical
worked example. When in doubt, read it end-to-end before adding your
own model — every step in this checklist has a concrete realisation
there.
