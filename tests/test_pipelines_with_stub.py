"""End-to-end pipeline tests using a stub model (CI-runnable)."""

import tempfile

import numpy as np

from psim.pipelines import (
    synthesise_scenario, validate_simrun, package_scenario,
)
from psim.io import read_artifact


def test_synthesise_with_stub_model(stub_sde_model, stub_truth, stub_init):
    n_bins = 30
    sim_run = synthesise_scenario(
        stub_sde_model,
        truth_params=stub_truth,
        init_state=stub_init,
        exogenous_arrays={},
        n_bins_total=n_bins,
        dt_days=0.1,
        bins_per_day=10,
        n_substeps=4,
        seed=0,
    )
    assert sim_run.trajectory.shape == (n_bins, 1)
    # OU should converge toward mu = 1.0
    assert sim_run.trajectory[-1, 0] > 0.5

    assert "obs_y" in sim_run.obs_channels
    assert "T_B" in sim_run.exogenous_channels


def test_validate_then_package(stub_sde_model, stub_truth, stub_init):
    sim_run = synthesise_scenario(
        stub_sde_model,
        truth_params=stub_truth,
        init_state=stub_init,
        exogenous_arrays={},
        n_bins_total=30,
        dt_days=0.1,
        bins_per_day=10,
        seed=0,
    )
    report = validate_simrun(sim_run, stub_sde_model)
    # Stub model has no verify_physics_fn so physics check passes by default
    assert report.physics is not None
    assert report.physics.passed is True

    with tempfile.TemporaryDirectory() as d:
        path = package_scenario(
            sim_run, report, out_dir=d,
            model_name="stub_ou", scenario_name="ou_test",
            require_all_passed=True,
        )
        loaded = read_artifact(path)
        assert loaded["manifest"].model_name == "stub_ou"
        assert loaded["manifest"].scenario_name == "ou_test"
        np.testing.assert_array_equal(loaded["trajectory"], sim_run.trajectory.astype(np.float32))
