"""Shared pytest fixtures.

Defines a tiny 1-state OU-like stub model so tests can exercise the
psim pipelines without depending on the external SMC² repo or the
not-yet-public-dev-repo high_res_FSA model.
"""

from __future__ import annotations

import numpy as np
import pytest

from psim._vendored.simulator.sde_model import SDEModel, StateSpec, ChannelSpec


def _stub_drift(t, y, params, aux):
    """OU drift: dy = -theta * (y - mu)."""
    del aux
    return np.array([-params["theta"] * (y[0] - params["mu"])])


def _stub_diffusion(params):
    return np.array([params.get("sigma", 0.1)])


def _stub_noise_scale(y, params):
    del y, params
    return np.array([1.0])


def _stub_make_aux(params, init_state, t_grid, exogenous):
    del params, init_state, t_grid
    return (exogenous,)


def _stub_make_y0(init_dict, params):
    del params
    return np.array([init_dict.get("y_0", 0.0)])


def _stub_gen_obs_y(trajectory, t_grid, params, aux, prior, seed):
    rng = np.random.default_rng(seed)
    y = trajectory[:, 0]
    obs = y + rng.normal(0, params.get("sigma_obs", 0.1), size=len(t_grid))
    return {
        "t_idx": np.arange(len(t_grid), dtype=np.int32),
        "obs_value": obs.astype(np.float32),
    }


def _stub_gen_T_B(trajectory, t_grid, params, aux, prior, seed):
    return {
        "t_idx": np.arange(len(t_grid), dtype=np.int32),
        "T_B_value": np.full(len(t_grid), 0.5, dtype=np.float32),
    }


@pytest.fixture
def stub_sde_model():
    """A 1-state OU-like SDEModel for pipeline tests."""
    return SDEModel(
        name="stub_ou",
        version="0.0.1",
        states=(StateSpec("y", -10.0, 10.0),),
        drift_fn=_stub_drift,
        diffusion_fn=_stub_diffusion,
        noise_scale_fn=_stub_noise_scale,
        make_aux_fn=_stub_make_aux,
        make_y0_fn=_stub_make_y0,
        channels=(
            ChannelSpec("obs_y", depends_on=(), generate_fn=_stub_gen_obs_y),
            ChannelSpec("T_B", depends_on=(), generate_fn=_stub_gen_T_B),
        ),
        param_sets={"default": {"theta": 0.5, "mu": 1.0, "sigma": 0.1, "sigma_obs": 0.1}},
        init_states={"default": {"y_0": 0.0}},
    )


@pytest.fixture
def stub_truth():
    return {"theta": 0.5, "mu": 1.0, "sigma": 0.1, "sigma_obs": 0.1}


@pytest.fixture
def stub_init():
    return {"y_0": 0.0}
