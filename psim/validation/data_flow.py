"""Data-flow visual diagnostics.

Plots that compare what the simulator USED to generate the data with
what the estimator SEES per-window after extract_window + align_obs_fn.
For exogenous covariates (T_B, Phi, C, ...) these arrays should be
**pointwise identical** (within float32 precision).

The C-phase bug that ate ~12 hours of high_res_FSA development would
have been caught instantly by this kind of plot for window 2 (estimator's
C inverted relative to simulator's). See the postmortem in the SMC²
repo.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np


def _use_agg():
    import matplotlib
    matplotlib.use("Agg")


def plot_covariate_alignment(
    sim_array: np.ndarray,
    est_array: np.ndarray,
    *,
    name: str,
    save_dir: str,
    window_start_bin: int = 0,
    bins_per_unit: int = 1,
    unit_label: str = "bin",
):
    """Side-by-side line plot of the simulator's vs estimator's array
    for one covariate over one window.

    Pointwise identity is the expected pass condition. The plot shows
    sim (solid blue), est (dashed orange), and the absolute difference
    (red, secondary axis) so misalignments are visually obvious.

    Parameters
    ----------
    sim_array : (T,) — what the simulator used at the global bins
                       [window_start_bin, window_start_bin + T)
    est_array : (T,) — what the estimator's align_obs_fn produced for
                       the same window (window-local index 0..T-1)
    name : str — covariate name (for filename and title)
    save_dir : str
    window_start_bin : int — for the x-axis label, defaults to 0
    bins_per_unit : int — e.g. 96 for 15-min bins per day
    unit_label : str — axis label for the inverse, e.g. 'day' or 'hour'
    """
    _use_agg()
    import matplotlib.pyplot as plt

    sim = np.asarray(sim_array)
    est = np.asarray(est_array)
    assert sim.shape == est.shape, (
        f"sim and est array shapes don't match: {sim.shape} vs {est.shape}"
    )
    n = len(sim)
    x = (np.arange(n) + window_start_bin) / bins_per_unit
    diff = sim - est

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x, sim, "b-", lw=1.5, label=f"sim ({name})")
    ax1.plot(x, est, "orange", linestyle="--", lw=1.0, label=f"est ({name})")
    ax1.set_xlabel(unit_label)
    ax1.set_ylabel(name)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, diff, "r:", lw=0.8, alpha=0.6, label="sim − est")
    ax2.set_ylabel("difference (red)", color="red")
    ax2.tick_params(axis="y", colors="red")

    max_abs = float(np.abs(diff).max())
    title = (
        f"Covariate alignment: {name} "
        f"(max|sim-est| = {max_abs:.3e}, "
        f"window starts at bin {window_start_bin})"
    )
    if max_abs > 1e-3:
        title += "  ← MISALIGNMENT DETECTED"
    plt.title(title)

    os.makedirs(save_dir, exist_ok=True)
    fname = f"covariate_alignment_{name}.png"
    path = os.path.join(save_dir, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


def plot_obs_alignment(
    sim_t_idx: np.ndarray,
    sim_obs: np.ndarray,
    est_t_idx: np.ndarray,
    est_obs: np.ndarray,
    *,
    channel: str,
    save_dir: str,
):
    """Compare observation arrays produced by simulator vs received by
    estimator (after extract_window + align_obs_fn). For sparse channels
    (HR sleep-gated, etc.) the two should be identical after window
    re-indexing.
    """
    _use_agg()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(sim_t_idx, sim_obs, s=8, c="blue", label="sim", alpha=0.6)
    ax.scatter(est_t_idx, est_obs, s=4, c="orange", label="est", alpha=0.8)
    ax.set_xlabel("window-local bin")
    ax.set_ylabel(channel)
    ax.set_title(f"Observation alignment: {channel} "
                 f"(sim n={len(sim_t_idx)}, est n={len(est_t_idx)})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"obs_alignment_{channel}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return path


__all__ = ["plot_covariate_alignment", "plot_obs_alignment"]
