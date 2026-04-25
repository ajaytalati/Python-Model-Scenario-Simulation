"""Canonical scenario-artifact format (writer + reader).

Spec
----
A "scenario artifact" is a directory with this layout:

    artifact_dir/
    ├── manifest.json              # SCENARIO_SCHEMA_VERSION, model, scenario,
    │                              # truth params, run config, validation summary
    ├── trajectory.npz             # latent state arrays (B/F/A or whatever)
    ├── obs/                       # one .npz per observation channel
    │   └── <channel>.npz          #   keys: t_idx + per-channel value(s)
    ├── exogenous/                 # one .npz per exogenous channel
    │   └── <name>.npz             #   keys: t_idx + <name>_value
    └── validation/                # validation_report.json + diagnostic plots

Schema version is mandatory in manifest. Reader checks compatibility.

Writers and readers below; they don't import any model code, so the
artifact format is the **stable** interface between this repo and the
SMC² repo.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


SCENARIO_SCHEMA_VERSION = "1.0"


@dataclass
class Manifest:
    """Top-level metadata for a scenario artifact."""

    schema_version: str = SCENARIO_SCHEMA_VERSION
    model_name: str = ""
    model_version: str = ""
    scenario_name: str = ""
    n_bins_total: int = 0
    dt_days: float = 0.0
    bins_per_day: int = 0
    seed: int = 0
    state_names: List[str] = field(default_factory=list)
    obs_channels: List[str] = field(default_factory=list)
    exogenous_channels: List[str] = field(default_factory=list)
    truth_params: Dict[str, float] = field(default_factory=dict)
    validation_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(
            schema_version=self.schema_version,
            model_name=self.model_name,
            model_version=self.model_version,
            scenario_name=self.scenario_name,
            n_bins_total=self.n_bins_total,
            dt_days=self.dt_days,
            bins_per_day=self.bins_per_day,
            seed=self.seed,
            state_names=self.state_names,
            obs_channels=self.obs_channels,
            exogenous_channels=self.exogenous_channels,
            truth_params=self.truth_params,
            validation_summary=self.validation_summary,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        return cls(**d)


def write_artifact(
    out_dir: str,
    *,
    manifest: Manifest,
    trajectory: np.ndarray,
    obs_channels: Dict[str, dict],
    exogenous_channels: Dict[str, dict],
    validation_report: dict | None = None,
):
    """Write all artifact files atomically.

    Parameters
    ----------
    out_dir : str
    manifest : Manifest
    trajectory : (n_bins, n_state) array
    obs_channels : dict[name -> per-channel dict]
        Each per-channel dict has at minimum 't_idx' (int32 array) and
        one or more value arrays. Stored as .npz with all keys preserved.
    exogenous_channels : dict[name -> per-channel dict]
        Same structure; typically T_B, Phi, C.
    validation_report : optional dict — written as validation/report.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "obs"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "exogenous"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "validation"), exist_ok=True)

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    np.savez_compressed(
        os.path.join(out_dir, "trajectory.npz"),
        trajectory=np.asarray(trajectory, dtype=np.float32),
    )

    for name, d in obs_channels.items():
        np.savez_compressed(os.path.join(out_dir, "obs", f"{name}.npz"),
                             **{k: np.asarray(v) for k, v in d.items()})

    for name, d in exogenous_channels.items():
        np.savez_compressed(os.path.join(out_dir, "exogenous", f"{name}.npz"),
                             **{k: np.asarray(v) for k, v in d.items()})

    if validation_report is not None:
        with open(os.path.join(out_dir, "validation", "report.json"), "w") as f:
            json.dump(validation_report, f, indent=2)


def read_artifact(in_dir: str) -> dict:
    """Load all artifact files into a single dict for downstream consumption.

    Returns
    -------
    dict with keys: manifest (Manifest), trajectory (ndarray),
                    obs_channels (dict[str, dict]),
                    exogenous_channels (dict[str, dict]),
                    validation_report (dict or None).

    Raises if SCENARIO_SCHEMA_VERSION doesn't match.
    """
    with open(os.path.join(in_dir, "manifest.json")) as f:
        m_dict = json.load(f)
    if m_dict.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        raise ValueError(
            f"Artifact schema version mismatch: file has "
            f"{m_dict.get('schema_version')!r}, expected "
            f"{SCENARIO_SCHEMA_VERSION!r}"
        )
    manifest = Manifest.from_dict(m_dict)

    with np.load(os.path.join(in_dir, "trajectory.npz")) as f:
        trajectory = np.asarray(f["trajectory"])

    obs_channels = {}
    obs_dir = os.path.join(in_dir, "obs")
    for fname in sorted(os.listdir(obs_dir)):
        if not fname.endswith(".npz"):
            continue
        name = fname[:-4]
        with np.load(os.path.join(obs_dir, fname)) as f:
            obs_channels[name] = {k: np.asarray(f[k]) for k in f.files}

    exo_channels = {}
    exo_dir = os.path.join(in_dir, "exogenous")
    if os.path.isdir(exo_dir):
        for fname in sorted(os.listdir(exo_dir)):
            if not fname.endswith(".npz"):
                continue
            name = fname[:-4]
            with np.load(os.path.join(exo_dir, fname)) as f:
                exo_channels[name] = {k: np.asarray(f[k]) for k in f.files}

    validation_report = None
    val_path = os.path.join(in_dir, "validation", "report.json")
    if os.path.exists(val_path):
        with open(val_path) as f:
            validation_report = json.load(f)

    return dict(
        manifest=manifest,
        trajectory=trajectory,
        obs_channels=obs_channels,
        exogenous_channels=exo_channels,
        validation_report=validation_report,
    )


__all__ = [
    "SCENARIO_SCHEMA_VERSION",
    "Manifest",
    "write_artifact",
    "read_artifact",
]
