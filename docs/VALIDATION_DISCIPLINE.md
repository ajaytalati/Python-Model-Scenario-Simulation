# Validation discipline

The mandatory pre-SMC² discipline. Three checks, each a few minutes of
compute, that catch the bug classes that have actually shipped.

This doc ports the prose from the SMC² repo's
`docs/PORTING_GUIDE.md` § 1.4 and adds direct references to the
runnable code in [psim/validation/](../psim/validation/).

## Why this exists

Three sim/est consistency bugs cost ≈15 hours of GPU + analyst time
during the high_res_FSA development. Full case study:
[POSTMORTEM_three_bugs](https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md).

The bugs:

| # | Bug | Class | Caught by |
|---|-----|-------|-----------|
| 1 | `mu_0` sign-flipped in estimator's drift | sim-est drift mismatch | check A (drift parity) |
| 2 | `extract_state_at_step` indexing off by one window-size | window-local vs global indexing | step 4 (round-trip) |
| 3 | `C(t)` covariate phase-misaligned across rolling windows | sim-est obs-prediction mismatch | check B (obs-prediction parity) |

Bug 3 was the costliest. It presented in the posterior as narrow
credible intervals locked off-truth on every β_C_* coefficient, looking
like a "bridge cascade" / particle-degeneracy problem. Two whole
research-track plans were drafted around the phantom problem before
the user spotted the actual cause by eye on a parameter-tracking plot.
A 1-line obs-prediction-parity test would have caught it in 60
seconds.

The lesson: **the simulation/scenario generation MUST be thoroughly
tested before any GPU time is spent on SMC²**. This discipline
operationalises that.

## The three checks

### A. Drift parity — [`check_drift_parity`](../psim/validation/consistency.py)

Compare `model_sim.drift_fn(t, y, sim_params, aux)` against the
deterministic part of `model_est.propagate_fn` at the same state, time,
and parameters.

```python
from psim.validation import check_drift_parity

res = check_drift_parity(
    sim_drift_fn=model_sim.drift_fn,
    est_drift_at_state=lambda state, t, dt, p, g, k: (
        model_est.propagate_fn(state, t, dt, p, g, k, noise_off=True) - state
    ) / dt,
    state=truth_state, t=truth_t,
    sim_params=DEFAULT_PARAMS,
    est_params_vec=truth_params_vec,
    aux=aux, grid_obs=grid_obs, k=0,
)
res.assert_pass()
```

Catches: sign flips on individual drift terms, missing terms, wrong
parameter index in `est_params_vec`, wrong state index, accidentally
using `+= ` where `=` was intended. Runs in <1 second.

### B. Observation-prediction parity per Gaussian channel — [`check_obs_prediction_parity`](../psim/validation/consistency.py)

For every Gaussian observation channel, compare the simulator's
noiseless mean against the estimator's predictor at the same `(state,
time, k)` and the truth parameters.

```python
from psim.validation import check_obs_prediction_parity

for ch_name, ch in zip(channel_names, channels):
    res = check_obs_prediction_parity(
        sim_obs_predictor=ch.sim_predictor,        # uses global C[k]
        est_obs_predictor=ch.est_predictor,        # uses grid_obs['C'][k]
        channel_name=ch_name,
        state=truth_state, t=truth_t,
        sim_params=DEFAULT_PARAMS,
        est_params_vec=truth_params_vec,
        grid_obs=grid_obs, k=k,
    )
    res.assert_pass()
```

Catches the **C-phase-bug class**: the simulator computes a covariate
on global time `t`, the estimator (often inadvertently) computes it on
window-local time. At window 1, bin 0 of the window happens to be bin
W of global time, with a different `cos(2πt + φ)` value. The drift
checks pass; everything looks fine until you stare at the posterior
plot. Runs in <1 second per channel.

### C. Cold-start coverage — [`check_cold_start_coverage`](../psim/validation/consistency.py)

End-to-end check: a single window of cold-start SMC² with a tight
truth-centred prior should give ≥95% coverage on every estimable
scalar. If A and B pass but C fails, the bug is in the SMC² stack
(inner PF or outer SMC) not the data alignment.

```python
from psim.validation import check_cold_start_coverage

res = check_cold_start_coverage(
    smc_runner_callable=run_one_window,            # from SMC² repo
    scenario_artifact_dir="outputs/<model>/<scenario>",
    n_smc=256, prior_sigma_scale=0.1,
    coverage_threshold=0.95,
)
res.assert_pass()
```

Currently a stub for v0.1.0 — until the SMC² repo exposes
`run_one_window` as an importable API, callers exercise this discipline
by running the full SMC² rollout themselves and confirming W1
coverage ≥95%.

## Supporting checks

### Round-trip — [`round_trip_check`](../psim/validation/round_trip.py)

Re-integrate `model_est.propagate_fn` with **zero noise** from the
truth initial state across one window. Assert the recovered trajectory
matches the simulator's trajectory at every grid point. Catches
indexing bugs in `extract_state_at_step` and friends.

### Data-flow plots — [`plot_covariate_alignment`](../psim/validation/data_flow.py)

Side-by-side per-covariate plots: simulator's array vs estimator's
window-local copy, every window. The plot that would have caught the
C-phase bug visually in 30 seconds. Use it any time the consistency
checks have a near-pass that you want to investigate.

### Physics — [`check_physics`](../psim/validation/physics.py)

Thin wrapper around `model_sim.verify_physics_fn`. Gates only on the
`required_booleans` keys (default `("all_finite",)`); informational
booleans like `mu_crosses_zero` are recorded in the report but don't
fail packaging. Per-model models can opt additional keys into the
gate.

## How the checks compose

Run order for a new model:

1. **A and B first.** They are cheap and catch the costly bugs. If A
   or B fails, **stop** — there is no point running anything
   downstream until the data alignment is fixed.
2. **Round-trip.** Catches the off-by-one class.
3. **Physics.** Mostly defensive; cheap; catches NaN/Inf early.
4. **Cold-start coverage (when the SMC² API is wired up).** End-to-end
   confirmation; if A/B/round-trip/physics pass and this fails, the
   bug is in the SMC² stack and you save your hyperparameter-tuning
   time for the right surface.

The pipeline runner [`validate_simrun`](../psim/pipelines/validate.py)
orchestrates all of these and emits a single
[`ValidationReport`](../psim/pipelines/validate.py) with a
`.all_passed` summary.

## What this gate prevents

Concretely:

- The C-phase / covariate-misalignment bug class becomes literally
  impossible to ship: `tests/test_consistency_*.py` fails on
  `check_obs_prediction_parity` if any window's predicted obs differs
  from the simulator's at the same global bin.
- The mu_0-style sign mismatch is caught by `check_drift_parity`
  before the model ever sees a particle filter.
- The `extract_state_at_step`-style indexing bug is caught by
  `round_trip_check`.

The historical pattern was "ship to SMC², debug for a week"; the new
pattern is "fail in `pytest tests/` in 30 seconds, fix in 5 minutes".

## When the checks themselves are wrong

These tests can be wrong too. Symptoms and fixes:

- **Drift parity passes but cold-start coverage is bad.** Either the
  diffusion is mis-specified (run with smaller σ to confirm) or the
  obs-prediction parity needs more channels covered.
- **Drift parity flags a small `max_abs_err` of the order of `1e-7`.**
  Likely a floating-point rounding difference between the simulator's
  64-bit drift and the estimator's mixed-precision propagate. Increase
  `atol` to `1e-5` and confirm it doesn't blow up at the truth.
- **Obs-prediction parity fails on one channel only.** This is the
  most common real failure mode. Side-by-side compare the two
  predictor closures and look for a covariate that's indexed
  differently.

## References

- [POSTMORTEM_three_bugs](https://github.com/ajaytalati/smc2-blackjax-rolling/blob/main/outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md)
  — the case study these checks were extracted from.
- [psim/validation/consistency.py](../psim/validation/consistency.py)
  — the three checks.
- [psim/validation/round_trip.py](../psim/validation/round_trip.py)
  — the round-trip helper.
- [psim/validation/data_flow.py](../psim/validation/data_flow.py)
  — diagnostic plotters.
- [psim/validation/physics.py](../psim/validation/physics.py)
  — physics-check wrapper.
