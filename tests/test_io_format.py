"""Round-trip tests on the canonical scenario-artifact format."""

import json
import os
import tempfile

import numpy as np
import pytest

from psim.io import Manifest, SCENARIO_SCHEMA_VERSION, write_artifact, read_artifact


def _toy_artifact_inputs():
    m = Manifest(
        model_name="toy", model_version="0.0", scenario_name="test",
        n_bins_total=10, dt_days=1.0, bins_per_day=1, seed=42,
        state_names=["x"], obs_channels=["y"], exogenous_channels=["T_B"],
        truth_params={"a": 1.5, "b": 2.5},
        validation_summary={"all_passed": True, "n_passed": 3, "n_failed": 0},
    )
    traj = np.arange(10).reshape(10, 1).astype(np.float32)
    obs = {"y": {"t_idx": np.arange(10, dtype=np.int32),
                 "obs_value": np.random.RandomState(0).randn(10).astype(np.float32)}}
    exo = {"T_B": {"t_idx": np.arange(10, dtype=np.int32),
                    "T_B_value": np.full(10, 0.6, dtype=np.float32)}}
    val = {"all_passed": True, "n_passed": 3, "n_failed": 0}
    return m, traj, obs, exo, val


def test_round_trip_preserves_all_fields():
    with tempfile.TemporaryDirectory() as d:
        m, traj, obs, exo, val = _toy_artifact_inputs()
        write_artifact(d, manifest=m, trajectory=traj,
                        obs_channels=obs, exogenous_channels=exo,
                        validation_report=val)
        r = read_artifact(d)
        assert r["manifest"].model_name == "toy"
        assert r["manifest"].schema_version == SCENARIO_SCHEMA_VERSION
        np.testing.assert_array_equal(r["trajectory"], traj)
        np.testing.assert_array_equal(r["obs_channels"]["y"]["t_idx"], obs["y"]["t_idx"])
        np.testing.assert_allclose(r["obs_channels"]["y"]["obs_value"], obs["y"]["obs_value"])
        np.testing.assert_array_equal(r["exogenous_channels"]["T_B"]["T_B_value"],
                                       exo["T_B"]["T_B_value"])
        assert r["validation_report"]["all_passed"] is True


def test_schema_version_mismatch_raises():
    with tempfile.TemporaryDirectory() as d:
        m, traj, obs, exo, val = _toy_artifact_inputs()
        write_artifact(d, manifest=m, trajectory=traj,
                        obs_channels=obs, exogenous_channels=exo,
                        validation_report=val)
        # corrupt the manifest's schema version
        with open(os.path.join(d, "manifest.json")) as f:
            cp = json.load(f)
        cp["schema_version"] = "999.0"
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(cp, f)

        with pytest.raises(ValueError, match="schema version mismatch"):
            read_artifact(d)


def test_directory_structure():
    with tempfile.TemporaryDirectory() as d:
        m, traj, obs, exo, val = _toy_artifact_inputs()
        write_artifact(d, manifest=m, trajectory=traj,
                        obs_channels=obs, exogenous_channels=exo,
                        validation_report=val)
        assert os.path.exists(os.path.join(d, "manifest.json"))
        assert os.path.exists(os.path.join(d, "trajectory.npz"))
        assert os.path.exists(os.path.join(d, "obs", "y.npz"))
        assert os.path.exists(os.path.join(d, "exogenous", "T_B.npz"))
        assert os.path.exists(os.path.join(d, "validation", "report.json"))
