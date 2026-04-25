# SWAT reference scenarios

End-to-end demonstration of the psim pipeline on the SWAT (Sleep-Wake-
Adenosine-Testosterone) model — psim's second canonical model after
fsa_high_res.

## Running

From the psim repo root:

```bash
python examples/swat/14d_set_A_healthy.py
python examples/swat/14d_set_B_amplitude.py
python examples/swat/14d_set_C_recovery.py
python examples/swat/14d_set_D_phase_shift.py
```

Each produces a packaged scenario artifact at
`outputs/swat/set_X_<name>_14d/` containing:

- `manifest.json` — model + scenario metadata + truth params
- `trajectory.npz` — 14-day W/Zt/a/T/C/Vh/Vn trajectory (4032 bins × 7 states at 5-min)
- `obs/{hr, sleep, steps, stress}.npz`
- `validation/report.json`

## The four scenarios

Per SWAT's `TESTING.md` §4 in the public dev repo:

| Set | V_h | V_n | V_c | T_0 | Expected end-of-trial T | Behaviour |
|-----|-----|-----|-----|-----|-------------------------|-----------|
| **A** | 1.0 | 0.3 | 0.0 h | 0.5 | ~0.55 (T*) | Healthy basin equilibrium |
| **B** | 0.2 | 3.5 | 0.0 h | 0.5 | ~0.13 | Amplitude collapse → hypogonadal flatline |
| **C** | 1.0 | 0.3 | 0.0 h | 0.05 | ~1.0 | Recovery: rise from flatline → healthy basin |
| **D** | 1.0 | 0.3 | 6.0 h | 0.5 | ~0.11 | Phase-shift pathology (shift work / jet lag) |

Sets B and D test two **independent** failure modes (amplitude vs
phase). Set C is the time-reversed Set B. Together these exercise
the Stuart-Landau bifurcation structure of the testosterone amplitude
state and the entrainment-quality coupling.

## Mixed-likelihood obs model

SWAT's 4 channels exercise the full mixed-likelihood discipline (the
generalisation of fsa_high_res's all-Gaussian channels):

| Channel | Likelihood | Time grid | Drives |
|---------|------------|-----------|--------|
| `hr` | Gaussian | dense (every 5 min) | W (via Pitt-Shephard guidance in propagate_fn) |
| `sleep` | 3-level ordinal {0=wake, 1=light+REM, 2=deep} | dense | Zt (via obs_log_weight_fn) |
| `steps` | Poisson | sparse (every 15 min) | W (via obs_log_weight_fn) |
| `stress` | Gaussian | dense | W, V_n (via obs_log_weight_fn; helps disambiguate V_n from V_h) |

The §1.4 consistency tests
([`tests/test_consistency_swat.py`](../../tests/test_consistency_swat.py))
verify that every channel's predictor matches between sim and est sides.

## Model location

SWAT lives canonically in the public dev repo:
[Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation)
under `version_1/models/swat/`. The scripts expect that repo cloned at
`~/Repos/Python-Model-Development-Simulation` (set `PUBLIC_DEV_V1` in
`_common.py` if you've put it elsewhere).

## SMC² consumption

After psim ships these artifacts, the SMC² repo's adapter
(`drivers/swat_rolling.py` — separate follow-up plan) consumes them
via the existing `--scenario-artifact` flag. No psim-side change
required to switch from fsa_high_res's rolling driver to SWAT's; the
artifact format is the only contract.
