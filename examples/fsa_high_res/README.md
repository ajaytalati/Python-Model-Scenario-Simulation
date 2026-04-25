# FSA high-res reference scenario

End-to-end demonstration of the psim pipeline on the high_res_FSA model.

## Run

```bash
python examples/fsa_high_res/14d_C0_recovery.py
```

Produces a packaged scenario artifact at
`outputs/fsa_high_res/C0_recovery_14d/` containing:

- `manifest.json` — model + scenario metadata + truth params
- `trajectory.npz` — 14-day B/F/A latent trajectory (1344 bins × 3 states)
- `obs/{obs_HR, obs_sleep, obs_stress, obs_steps}.npz`
- `exogenous/{T_B, Phi, C}.npz`
- `validation/report.json`

## SMC² consumption

The artifact is consumed by the SMC² repo's driver via the
`--scenario-artifact` flag (see [`docs/BRIDGE_TO_SMC2.md`](../../docs/BRIDGE_TO_SMC2.md)).
Same seed reproduces the published 96.8% mean coverage / 27-of-27 PASS
result from the SMC² repo's `outputs/fsa_high_res_rolling/C_phase_fix_result.md`.

## Model location (temporary)

Until high_res_FSA lands in the public dev repo
([Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation)),
this example imports the model from the user's local SMC² repo at
`~/Repos/smc2_blackjax_framework` via sys.path injection. After the
public-repo PR merges, the import will switch to:

```python
from version_1.models.fsa_high_res.simulation import HIGH_RES_FSA_MODEL
```
