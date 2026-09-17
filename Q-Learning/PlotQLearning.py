"""Plot diagnostics from a completed QLearning.py experiment."""


from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

if __name__ == "__main__":
    print("Initializing plotting libraries", flush=True)

# macOS font discovery invokes system_profiler without a timeout. It can hang
# while building a fresh cache; these plots only need Matplotlib's bundled fonts.
# Respect an explicit user setting (an empty value enables system font discovery).
if sys.platform == "darwin":
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")

import matplotlib

# Saving plots is the default workflow.  On headless macOS/Python setups the
# implicit GUI backend can abort the process before Python can report an error.
# Keep an explicitly requested backend or the interactive ``--show`` behavior.
if "MPLBACKEND" not in os.environ and "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Each run is written to its own parameter-labeled folder in
# q_learning_plots/Saved_Plots, next to the legacy plots archived there.  The
# standalone control-correction filename is still generated below so an old
# image cannot be mistaken for output from the currently loaded run.

QLEARNING_PATH = Path(__file__).resolve().with_name("QLearning.py")


def _load_qlearning(path: str | Path = QLEARNING_PATH) -> ModuleType:
    """Load the implementation module from its non-package directory."""
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Q-learning implementation not found: {source_path}")

    module_name = "_lorenz_qlearning_implementation"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


qlearning = _load_qlearning()
steps_to_lyapunov_times = qlearning.steps_to_lyapunov_times

_EPISODE_HISTORY_KEYS = (
    "episode_rewards",
    "episode_lengths",
    "epsilons",
    "diverged",
    "first_target_steps",
    "target_steps",
    "average_rewards",
    "rolling_mean_rewards",
)
_SAMPLED_HISTORY_KEYS = (
    "sampled_episodes",
    "sampled_trajectories",
    "sampled_control_values",
)
_CHECKPOINT_SERIES_KEYS = (
    "episodes",
    "mean_rewards",
    "reward_standard_deviations",
    "divergence_rates",
    "mean_control_efforts",
)


def _run_directory(
    root: Path, settings: Mapping[str, Any], run_time: datetime
) -> Path:
    """Name a per-run archive folder after the parameters that define the run."""
    state_bins = "x".join(str(count) for count in np.atleast_1d(settings["state_bins"]))
    label = (
        f"{run_time:%Y%m%d-%H%M%S}"
        f"_ep{settings['episodes']}"
        f"_lr{settings['learning_rate']:g}"
        f"_gamma{settings['discount_factor']:g}"
        f"_eps{settings['epsilon']:g}-{settings['epsilon_decay']:g}"
        f"_bins{state_bins}"
        f"_actions{settings['action_bins']}"
    )
    return root / "Saved_Plots" / label


def _run_time(run: Mapping[str, Any], run_data_path: Path) -> datetime:
    """Use stable run metadata, falling back for archives created before it existed."""
    created_at = run.get("created_at")
    if created_at is None:
        return datetime.fromtimestamp(run_data_path.stat().st_mtime)
    if not isinstance(created_at, str):
        raise ValueError("run data 'created_at' must be an ISO-8601 string")
    try:
        return datetime.fromisoformat(created_at)
    except ValueError as error:
        raise ValueError(
            "run data 'created_at' must be an ISO-8601 timestamp"
        ) from error


def _require_local_run_data(path: Path) -> None:
    """Avoid blocking on reads of macOS cloud placeholders."""
    if sys.platform == "darwin" and path.stat().st_flags & 0x40000000:  # SF_DATALESS
        raise ValueError(
            f"run data is stored only in iCloud: {path}; "
            "select Download Now or Keep Downloaded for this file in Finder, "
            "then rerun PlotQLearning.py"
        )


def _history_through_episode(
    history: Mapping[str, Sequence[Any]], episode: int
) -> dict[str, Sequence[Any]]:
    """Return the portion of training history available at one checkpoint."""
    snapshot = dict(history)
    for key in _EPISODE_HISTORY_KEYS:
        if key in history:
            snapshot[key] = history[key][:episode]

    sampled_episodes = np.asarray(history["sampled_episodes"], dtype=np.int64)
    sampled_count = int(np.count_nonzero(sampled_episodes <= episode))
    for key in _SAMPLED_HISTORY_KEYS:
        snapshot[key] = history[key][:sampled_count]
    return snapshot


def _checkpoint_history_through_index(
    checkpoints: Mapping[str, Sequence[Any]], checkpoint_index: int
) -> dict[str, Sequence[Any]]:
    """Return scalar checkpoint metrics through the selected checkpoint."""
    count = checkpoint_index + 1
    return {
        key: checkpoints[key][:count]
        for key in _CHECKPOINT_SERIES_KEYS
    }


def _lyapunov_time_axis(number_of_steps: int) -> np.ndarray:
    """Return plotting coordinates using QLearning.py's time conversion."""
    return np.asarray(
        [steps_to_lyapunov_times(step) for step in range(number_of_steps)],
        dtype=np.float64,
    )


def _finish_figure(
    figure: Figure,
    output_path: Path | None,
    *,
    show: bool,
    dpi: int,
    tight_layout: bool = True,
) -> None:
    """Save and/or display a completed Matplotlib figure."""
    if tight_layout:
        figure.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {output_path}", flush=True)
    if show:
        figure.show()
    else:
        plt.close(figure)


def plot_training_diagnostics(
    history: Mapping[str, Sequence[Any]],
    output_path: Path | None,
    *,
    show: bool = False,
    dpi: int = 160,
) -> None:
    """Plot learning progress returned by ``train_q_learning``."""
    rewards = np.asarray(history["episode_rewards"], dtype=np.float64)
    rolling_rewards = np.asarray(history["average_rewards"], dtype=np.float64)
    lengths = np.asarray(history["episode_lengths"], dtype=np.int64)
    epsilons = np.asarray(history["epsilons"], dtype=np.float64)
    diverged = np.asarray(history["diverged"], dtype=bool)
    episode_count = rewards.size
    if rewards.ndim != 1 or episode_count == 0:
        raise ValueError("Training history must contain at least one episode")
    for name, values in (
        ("average_rewards", rolling_rewards),
        ("episode_lengths", lengths),
        ("epsilons", epsilons),
        ("diverged", diverged),
    ):
        if values.shape != rewards.shape:
            raise ValueError(
                f"history[{name!r}] must contain {episode_count} values; "
                f"received shape {values.shape}"
            )
    episodes = np.arange(1, rewards.size + 1)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8))
    figure.suptitle("Tabular Q-learning training diagnostics", fontsize=15)

    reward_axis = axes[0, 0]
    reward_axis.plot(episodes, rewards, color="tab:blue", alpha=0.28, label="Episode")
    reward_axis.plot(
        episodes,
        rolling_rewards,
        color="navy",
        linewidth=2.0,
        label="Rolling mean from QLearning.py",
    )
    reward_axis.set_xlabel("Episode")
    reward_axis.set_ylabel("Total reward")
    reward_axis.set_title("Reward history")
    reward_axis.grid(alpha=0.25)
    reward_axis.legend()

    epsilon_axis = axes[0, 1]
    epsilon_axis.plot(episodes, epsilons, color="tab:orange", linewidth=2.0)
    epsilon_axis.set_xlabel("Episode")
    epsilon_axis.set_ylabel("Epsilon")
    epsilon_axis.set_title("Exploration schedule")
    epsilon_axis.grid(alpha=0.25)

    length_axis = axes[1, 0]
    length_axis.plot(episodes, lengths, color="tab:green", linewidth=1.3)
    if np.any(diverged):
        length_axis.scatter(
            episodes[diverged],
            lengths[diverged],
            color="tab:red",
            marker="x",
            s=45,
            label="Diverged",
            zorder=3,
        )
        length_axis.legend()
    length_axis.set_xlabel("Episode")
    length_axis.set_ylabel("Steps")
    length_axis.set_title("Episode length")
    length_axis.grid(alpha=0.25)

    distribution_axis = axes[1, 1]
    bin_count = min(40, max(5, int(np.sqrt(rewards.size))))
    distribution_axis.hist(
        rewards,
        bins=bin_count,
        color="tab:purple",
        alpha=0.78,
        edgecolor="white",
    )
    distribution_axis.axvline(
        float(np.mean(rewards)),
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"Mean: {np.mean(rewards):.2f}",
    )
    distribution_axis.set_xlabel("Total reward")
    distribution_axis.set_ylabel("Episodes")
    distribution_axis.set_title("Reward distribution")
    distribution_axis.grid(axis="y", alpha=0.25)
    distribution_axis.legend()

    _finish_figure(figure, output_path, show=show, dpi=dpi)


def plot_training_rollouts(
    history: Mapping[str, Sequence[Any]],
    output_dir: Path | None,
    *,
    show: bool = False,
    dpi: int = 160,
) -> None:
    """Plot state and control histories sampled throughout training."""
    required = {
        "sampled_episodes",
        "sampled_trajectories",
        "sampled_control_values",
    }
    missing = required.difference(history)
    if missing:
        raise ValueError(
            "Run QLearning.py again to record training rollout data; missing "
            + ", ".join(sorted(missing))
        )

    episode_numbers = np.asarray(history["sampled_episodes"], dtype=np.int64)
    trajectories = [
        np.asarray(values, dtype=np.float64)
        for values in history["sampled_trajectories"]
    ]
    controls = [
        np.asarray(values, dtype=np.float64)
        for values in history["sampled_control_values"]
    ]
    if episode_numbers.ndim != 1 or episode_numbers.size == 0:
        raise ValueError("Training history contains no sampled episodes")
    if len(trajectories) != episode_numbers.size or len(controls) != episode_numbers.size:
        raise ValueError(
            "Training history must contain one trajectory and control sequence "
            "per sampled episode"
        )
    for episode, trajectory, control_values in zip(
        episode_numbers, trajectories, controls
    ):
        if trajectory.ndim != 2 or trajectory.shape[1] != 3:
            raise ValueError(
                f"Sampled training episode {episode} trajectory must have shape (n, 3)"
            )
        if control_values.ndim != 1 or control_values.size != trajectory.shape[0] - 1:
            raise ValueError(
                f"Sampled training episode {episode} must have one control per transition"
            )

    # Diverged episodes can be shorter. Restrict averages to the common interval
    # so every plotted point has the same number of contributing episodes.
    common_state_steps = min(trajectory.shape[0] for trajectory in trajectories)
    common_control_steps = common_state_steps - 1
    if common_control_steps < 1:
        raise ValueError("Sampled training rollouts are too short to plot")
    state_stack = np.stack(
        [trajectory[:common_state_steps] for trajectory in trajectories]
    )
    control_stack = np.stack(
        [values[:common_control_steps] for values in controls]
    )
    mean_state = np.mean(state_stack, axis=0)
    state_std = np.std(state_stack, axis=0)
    mean_control = np.mean(control_stack, axis=0)
    control_std = np.std(control_stack, axis=0)
    state_times = _lyapunov_time_axis(common_state_steps)
    control_times = state_times[:-1]

    def output_path(filename: str) -> Path | None:
        return None if output_dir is None else output_dir / filename

    figure, (state_axis, control_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": (1.15, 1.0), "hspace": 0.14},
    )
    figure.suptitle(
        f"Training rollouts: mean of {episode_numbers.size} sampled episodes",
        fontsize=15,
    )
    for coordinate, (label, color) in enumerate(
        zip(("x", "y", "z"), ("tab:blue", "tab:orange", "tab:green"))
    ):
        state_axis.plot(
            state_times,
            mean_state[:, coordinate],
            color=color,
            linewidth=1.0,
            label=f"mean {label}",
        )
        state_axis.fill_between(
            state_times,
            mean_state[:, coordinate] - state_std[:, coordinate],
            mean_state[:, coordinate] + state_std[:, coordinate],
            color=color,
            alpha=0.12,
            linewidth=0.0,
        )
    state_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    state_axis.set_ylabel("State")
    state_axis.set_title("Mean controlled state (bands: +/- 1 standard deviation)")
    state_axis.grid(alpha=0.25)
    state_axis.legend(loc="upper right", ncol=3)

    control_axis.step(
        control_times,
        mean_control,
        where="post",
        color="crimson",
        linewidth=0.9,
        label="mean control",
    )
    control_axis.fill_between(
        control_times,
        mean_control - control_std,
        mean_control + control_std,
        step="post",
        color="crimson",
        alpha=0.16,
        linewidth=0.0,
        label="+/- 1 standard deviation",
    )
    control_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    control_axis.set_xlim(0.0, float(state_times[-1]))
    control_axis.set_xlabel("Lyapunov times (t / τ)")
    control_axis.set_ylabel("Control correction u")
    control_axis.set_title("Mean correction applied to the x dynamics")
    control_axis.grid(alpha=0.25)
    control_axis.legend(loc="upper right")
    figure.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.08,
        top=0.91,
        hspace=0.20,
    )
    _finish_figure(
        figure,
        output_path("training_corrected_trajectory.png"),
        show=show,
        dpi=dpi,
        tight_layout=False,
    )

    snapshot_count = min(4, episode_numbers.size)
    snapshot_indices = np.rint(
        np.linspace(0, episode_numbers.size - 1, snapshot_count)
    ).astype(int)
    trajectory_figure, trajectory_axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
        subplot_kw={"projection": "3d"},
    )
    trajectory_figure.suptitle(
        "Controlled trajectory snapshots across training",
        fontsize=15,
    )
    episode_min = int(np.min(episode_numbers))
    episode_max = int(np.max(episode_numbers))
    color_norm = Normalize(episode_min, max(episode_min + 1, episode_max))
    color_map = plt.cm.viridis
    flat_axes = trajectory_axes.ravel()
    for axis, snapshot_index in zip(flat_axes, snapshot_indices):
        episode = int(episode_numbers[snapshot_index])
        trajectory = trajectories[snapshot_index]
        axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color_map(color_norm(episode)),
            linewidth=0.55,
            alpha=0.85,
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
        axis.set_title(f"Training episode {episode}")
    for unused_axis in flat_axes[snapshot_count:]:
        unused_axis.set_visible(False)
    trajectory_figure.subplots_adjust(
        left=0.03,
        right=0.97,
        bottom=0.04,
        top=0.91,
        wspace=0.08,
        hspace=0.16,
    )
    _finish_figure(
        trajectory_figure,
        output_path("training_trajectory.png"),
        show=show,
        dpi=dpi,
        tight_layout=False,
    )


def _rolling_nanmean(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Rolling mean over a trailing window, ignoring episodes without a value."""
    means = np.full(values.size, np.nan, dtype=np.float64)
    for end in range(1, values.size + 1):
        measured = values[max(0, end - window):end]
        measured = measured[np.isfinite(measured)]
        if measured.size:
            means[end - 1] = float(np.mean(measured))
    return means


def _steps_to_times(steps: np.ndarray) -> np.ndarray:
    """Convert step counts to Lyapunov times, propagating missing values."""
    return np.asarray(
        [
            np.nan if not np.isfinite(value) else steps_to_lyapunov_times(int(value))
            for value in np.atleast_1d(steps)
        ],
        dtype=np.float64,
    )


def plot_target_acquisition(
    history: Mapping[str, Sequence[Any]],
    output_path: Path | None,
    *,
    show: bool = False,
    dpi: int = 160,
) -> None:
    """Plot when each training episode reaches the target and how long it holds it."""
    lengths = np.asarray(history["episode_lengths"], dtype=np.int64)
    dwell_steps = np.asarray(history["target_steps"], dtype=np.int64)
    raw_first_steps = list(history["first_target_steps"])
    episode_count = lengths.size
    if lengths.ndim != 1 or episode_count == 0:
        raise ValueError("Training history must contain at least one episode")
    if dwell_steps.shape != lengths.shape:
        raise ValueError(
            f"history['target_steps'] must contain {episode_count} values; "
            f"received shape {dwell_steps.shape}"
        )
    if len(raw_first_steps) != episode_count:
        raise ValueError(
            f"history['first_target_steps'] must contain {episode_count} values; "
            f"received {len(raw_first_steps)}"
        )

    episodes = np.arange(1, episode_count + 1)
    first_steps = np.asarray(
        [np.nan if step is None else float(step) for step in raw_first_steps],
        dtype=np.float64,
    )
    reached = np.isfinite(first_steps)
    acquisition_times = _steps_to_times(first_steps)
    episode_times = _steps_to_times(lengths.astype(np.float64))
    dwell_times = _steps_to_times(dwell_steps.astype(np.float64))
    held = dwell_steps > 0

    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    figure.suptitle(
        "Target fixed point acquisition during training "
        f"(tolerance {qlearning.FIXED_POINT_TOLERANCE:g})",
        fontsize=15,
    )

    acquisition_axis = axes[0]
    if np.any(reached):
        acquisition_axis.scatter(
            episodes[reached],
            acquisition_times[reached],
            color="tab:blue",
            s=12,
            alpha=0.45,
            label=f"First arrival ({int(np.count_nonzero(reached))} episodes)",
        )
        acquisition_axis.plot(
            episodes,
            _rolling_nanmean(acquisition_times),
            color="navy",
            linewidth=2.0,
            label="Rolling mean (50 episodes that arrived)",
        )
    if np.any(~reached):
        acquisition_axis.scatter(
            episodes[~reached],
            episode_times[~reached],
            facecolors="none",
            edgecolors="tab:red",
            marker="^",
            s=28,
            linewidths=0.9,
            label=(
                f"Never reached ({int(np.count_nonzero(~reached))} episodes, "
                "plotted at episode end)"
            ),
        )
    acquisition_axis.set_ylabel("Lyapunov times (t / τ)")
    acquisition_axis.set_title("Time until the target fixed point is first reached")
    # Leave headroom so the legend clears the row of never-reached markers.
    acquisition_axis.set_ylim(0.0, 1.35 * float(np.nanmax(episode_times)))
    acquisition_axis.grid(alpha=0.25)
    acquisition_axis.legend(loc="upper right", fontsize=9, framealpha=0.95)

    dwell_axis = axes[1]
    dwell_axis.plot(
        episodes,
        dwell_times,
        color="tab:green",
        linewidth=0.8,
        alpha=0.45,
        label="Episode",
    )
    dwell_axis.plot(
        episodes,
        _rolling_nanmean(dwell_times),
        color="darkgreen",
        linewidth=2.0,
        label="Rolling mean (50 episodes)",
    )
    dwell_axis.set_xlabel("Episode")
    dwell_axis.set_ylabel("Lyapunov times (t / τ)")
    dwell_axis.set_title(
        "Time held at the target fixed point "
        f"(reached in {int(np.count_nonzero(held))}/{episode_count} episodes; "
        f"longest hold {np.nanmax(dwell_times):.2f})"
    )
    dwell_axis.set_ylim(0.0, 1.25 * max(float(np.nanmax(dwell_times)), 1e-3))
    dwell_axis.grid(alpha=0.25)
    dwell_axis.legend(loc="upper right", fontsize=9, framealpha=0.95)

    _finish_figure(figure, output_path, show=show, dpi=dpi)


def plot_checkpoint_evaluations(
    checkpoints: Mapping[str, Sequence[Any]],
    output_path: Path | None,
    *,
    show: bool = False,
    dpi: int = 160,
) -> None:
    """Plot greedy policy quality measured during continuous training."""
    episodes = np.asarray(checkpoints["episodes"], dtype=np.int64)
    mean_rewards = np.asarray(checkpoints["mean_rewards"], dtype=np.float64)
    reward_stds = np.asarray(
        checkpoints["reward_standard_deviations"], dtype=np.float64
    )
    divergence_rates = np.asarray(
        checkpoints["divergence_rates"], dtype=np.float64
    )
    mean_efforts = np.asarray(
        checkpoints["mean_control_efforts"], dtype=np.float64
    )
    if episodes.ndim != 1 or episodes.size == 0:
        raise ValueError("Checkpoint history must contain at least one evaluation")
    for name, values in (
        ("mean_rewards", mean_rewards),
        ("reward_standard_deviations", reward_stds),
        ("divergence_rates", divergence_rates),
        ("mean_control_efforts", mean_efforts),
    ):
        if values.shape != episodes.shape:
            raise ValueError(
                f"checkpoints[{name!r}] must contain {episodes.size} values"
            )

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    figure.suptitle("Greedy policy evaluation during training", fontsize=15)

    lower = mean_rewards - reward_stds
    upper = mean_rewards + reward_stds
    axes[0].plot(episodes, mean_rewards, marker="o", color="tab:blue")
    axes[0].fill_between(episodes, lower, upper, color="tab:blue", alpha=0.18)
    axes[0].set_ylabel("Mean reward")
    axes[0].set_title("Greedy evaluation reward (band: +/- 1 standard deviation)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(episodes, divergence_rates, marker="o", color="tab:red")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("Divergence rate")
    axes[1].set_title("Fraction of evaluation trials that diverged")
    axes[1].grid(alpha=0.25)

    axes[2].plot(episodes, mean_efforts, marker="o", color="tab:purple")
    axes[2].set_xlabel("Completed training episode")
    axes[2].set_ylabel("Mean control effort")
    axes[2].set_title("Normalized greedy control effort")
    axes[2].grid(alpha=0.25)

    _finish_figure(figure, output_path, show=show, dpi=dpi)


def save_learning_progress_gif(
    checkpoints: Mapping[str, Sequence[Any]],
    state_bounds: Sequence[Sequence[float]],
    target_state: Sequence[float],
    output_path: Path,
    *,
    fps: float = 2.0,
    dpi: int = 100,
    max_trajectory_points: int = 2_000,
) -> None:
    """Animate representative greedy trajectories across training checkpoints."""
    episodes = np.asarray(checkpoints["episodes"], dtype=np.int64)
    mean_rewards = np.asarray(checkpoints["mean_rewards"], dtype=np.float64)
    reward_stds = np.asarray(
        checkpoints["reward_standard_deviations"], dtype=np.float64
    )
    divergence_rates = np.asarray(
        checkpoints["divergence_rates"], dtype=np.float64
    )
    evaluations = checkpoints.get("evaluations")

    if episodes.ndim != 1 or episodes.size == 0:
        raise ValueError("Checkpoint history must contain at least one evaluation")
    if evaluations is None:
        raise ValueError("Checkpoint history does not contain evaluation snapshots")
    if len(evaluations) != episodes.size:
        raise ValueError(
            "Checkpoint evaluations must contain one snapshot per episode"
        )
    for name, values in (
        ("mean_rewards", mean_rewards),
        ("reward_standard_deviations", reward_stds),
        ("divergence_rates", divergence_rates),
    ):
        if values.shape != episodes.shape or not np.isfinite(values).all():
            raise ValueError(
                f"checkpoints[{name!r}] must contain {episodes.size} finite values"
            )
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if dpi < 1:
        raise ValueError("dpi must be at least 1")
    if max_trajectory_points < 2:
        raise ValueError("max_trajectory_points must be at least 2")
    if not animation.PillowWriter.isAvailable():
        raise RuntimeError("Saving learning_progress.gif requires Pillow")

    bounds = np.asarray(state_bounds, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("state_bounds must have shape (3, 2) with finite values")
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target_state must contain three finite values")

    trajectories: list[np.ndarray] = []
    for checkpoint_index, evaluation in enumerate(evaluations):
        checkpoint_trajectories = evaluation.get("trajectories", [])
        if not checkpoint_trajectories:
            raise ValueError(
                f"Checkpoint {checkpoint_index + 1} contains no evaluation trajectory"
            )
        trajectory = np.asarray(checkpoint_trajectories[0], dtype=np.float64)
        if trajectory.ndim != 2 or trajectory.shape[1] != 3 or trajectory.shape[0] < 2:
            raise ValueError("Each GIF trajectory must have shape (n, 3), n >= 2")
        finite_rows = np.isfinite(trajectory).all(axis=1)
        trajectory = trajectory[finite_rows]
        if trajectory.shape[0] < 2:
            raise ValueError("Each GIF trajectory must contain two finite states")
        if trajectory.shape[0] > max_trajectory_points:
            sample_indices = np.linspace(
                0,
                trajectory.shape[0] - 1,
                max_trajectory_points,
                dtype=np.int64,
            )
            trajectory = trajectory[sample_indices]
        trajectories.append(trajectory)

    reward_lower = mean_rewards - reward_stds
    reward_upper = mean_rewards + reward_stds
    reward_min = float(np.min(reward_lower))
    reward_max = float(np.max(reward_upper))
    reward_padding = max(0.05 * (reward_max - reward_min), 1e-6)
    final_episode = max(int(episodes[-1]), 1)

    figure = plt.figure(figsize=(12, 7))
    grid = figure.add_gridspec(2, 2, width_ratios=(1.45, 1.0), hspace=0.34)
    trajectory_axis = figure.add_subplot(grid[:, 0], projection="3d")
    reward_axis = figure.add_subplot(grid[0, 1])
    divergence_axis = figure.add_subplot(grid[1, 1], sharex=reward_axis)
    frame_colors = plt.cm.viridis(np.linspace(0.12, 0.9, episodes.size))

    def draw_frame(frame_index: int) -> tuple[Any, ...]:
        trajectory_axis.clear()
        reward_axis.clear()
        divergence_axis.clear()

        trajectory = trajectories[frame_index]
        color = frame_colors[frame_index]
        trajectory_axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color,
            linewidth=0.9,
            alpha=0.9,
        )
        trajectory_axis.scatter(
            *trajectory[0], color="black", s=32, marker="o", label="Start"
        )
        trajectory_axis.scatter(
            *trajectory[-1], color="crimson", s=48, marker="X", label="Agent"
        )
        trajectory_axis.scatter(
            *target, color="gold", edgecolor="black", s=100, marker="*", label="Target"
        )
        trajectory_axis.set_xlim(*bounds[0])
        trajectory_axis.set_ylim(*bounds[1])
        trajectory_axis.set_zlim(*bounds[2])
        trajectory_axis.set_xlabel("x")
        trajectory_axis.set_ylabel("y")
        trajectory_axis.set_zlabel("z")
        trajectory_axis.set_title(
            f"Greedy agent trajectory after episode {int(episodes[frame_index])}"
        )
        trajectory_axis.legend(loc="upper left", fontsize=8)

        visible = slice(0, frame_index + 1)
        reward_axis.plot(
            episodes[visible],
            mean_rewards[visible],
            marker="o",
            color="tab:blue",
        )
        reward_axis.fill_between(
            episodes[visible],
            reward_lower[visible],
            reward_upper[visible],
            color="tab:blue",
            alpha=0.18,
        )
        reward_axis.set_xlim(0, final_episode)
        reward_axis.set_ylim(
            reward_min - reward_padding, reward_max + reward_padding
        )
        reward_axis.set_ylabel("Mean reward")
        reward_axis.set_title("Greedy evaluation reward")
        reward_axis.grid(alpha=0.25)

        divergence_axis.plot(
            episodes[visible],
            divergence_rates[visible],
            marker="o",
            color="tab:red",
        )
        divergence_axis.set_xlim(0, final_episode)
        divergence_axis.set_ylim(-0.02, 1.02)
        divergence_axis.set_xlabel("Completed training episode")
        divergence_axis.set_ylabel("Divergence rate")
        divergence_axis.set_title("Evaluation stability")
        divergence_axis.grid(alpha=0.25)

        figure.suptitle(
            "Q-learning agent progress\n"
            f"Checkpoint {frame_index + 1}/{episodes.size} | "
            f"reward {mean_rewards[frame_index]:.4f} | "
            f"divergence {divergence_rates[frame_index]:.1%}",
            fontsize=14,
        )
        return ()

    movie = animation.FuncAnimation(
        figure,
        draw_frame,
        frames=episodes.size,
        interval=1_000.0 / fps,
        repeat=True,
        blit=False,
        cache_frame_data=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {episodes.size} GIF frames to {output_path}", flush=True)
    try:
        movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(figure)
    print(f"Saved {output_path}", flush=True)


def save_evaluation_trajectory_animation(
    evaluation: Mapping[str, Sequence[Any]],
    state_bounds: Sequence[Sequence[float]],
    target_state: Sequence[float],
    output_path: Path,
    *,
    episode_index: int = 0,
    forcing_scale: float = qlearning.U_REF,
    fps: float = 25.0,
    duration: float = 20.0,
    dpi: int = 90,
) -> None:
    """Animate one greedy evaluation rollout, colored by forcing magnitude."""
    trajectories = evaluation.get("trajectories", [])
    controls = evaluation.get("control_values", [])
    if len(trajectories) == 0:
        raise ValueError("Evaluation results contain no trajectories")
    if not isinstance(episode_index, (int, np.integer)) or isinstance(
        episode_index, (bool, np.bool_)
    ):
        raise TypeError("episode_index must be an integer")
    if episode_index < 0 or episode_index >= len(trajectories):
        raise ValueError(
            f"episode_index must be in [0, {len(trajectories) - 1}]"
        )
    if len(controls) != len(trajectories):
        raise ValueError(
            "Evaluation results must contain one control sequence per trajectory"
        )
    if not np.isfinite(forcing_scale) or forcing_scale <= 0.0:
        raise ValueError("forcing_scale must be finite and positive")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")
    if dpi < 1:
        raise ValueError("dpi must be at least 1")
    if not animation.PillowWriter.isAvailable():
        raise RuntimeError("Saving the evaluation animation requires Pillow")

    trajectory = np.asarray(trajectories[episode_index], dtype=np.float64)
    control_values = np.asarray(controls[episode_index], dtype=np.float64)
    bounds = np.asarray(state_bounds, dtype=np.float64)
    target = np.asarray(target_state, dtype=np.float64)
    if (
        trajectory.ndim != 2
        or trajectory.shape[1] != 3
        or trajectory.shape[0] < 2
        or not np.isfinite(trajectory).all()
    ):
        raise ValueError("The selected trajectory must have shape (n, 3), n >= 2")
    if (
        control_values.ndim != 1
        or control_values.size != trajectory.shape[0] - 1
        or not np.isfinite(control_values).all()
    ):
        raise ValueError("The selected episode must have one finite control per transition")
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
        raise ValueError("state_bounds must have shape (3, 2) with finite values")
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target_state must contain three finite values")

    segments = np.stack((trajectory[:-1], trajectory[1:]), axis=1)
    forcing_magnitudes = np.abs(control_values)
    frame_count = min(
        segments.shape[0],
        max(2, int(np.rint(fps * duration))),
    )
    frame_ends = np.unique(
        np.rint(np.linspace(1, segments.shape[0], frame_count)).astype(np.int64)
    )
    times = _lyapunov_time_axis(trajectory.shape[0])

    color_map = LinearSegmentedColormap.from_list(
        "signed_forcing",
        ("#174ea6", "#8da9d6", "#f7f7f7", "#ef9a9a", "#b2182b"),
    )
    color_norm = TwoSlopeNorm(
        vmin=-forcing_scale,
        vcenter=0.0,
        vmax=forcing_scale,
    )

    figure = plt.figure(figsize=(8.6, 7.2))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="0.72",
        linewidth=0.45,
        alpha=0.28,
        label="Full evaluation path",
    )
    colored_path = Line3DCollection(
        segments[:1], cmap=color_map, norm=color_norm, linewidth=2.1, alpha=0.96
    )
    colored_path.set_array(control_values[:1])
    axis.add_collection3d(colored_path)
    position_marker = axis.scatter(
        *trajectory[0],
        color=color_map(color_norm(control_values[0])),
        edgecolor="white",
        linewidth=0.9,
        s=62,
        depthshade=False,
        label="Current position",
        zorder=5,
    )
    axis.scatter(
        *target,
        color="gold",
        edgecolor="black",
        linewidth=0.7,
        marker="*",
        s=145,
        label="Target fixed point",
        zorder=4,
    )
    axis.set_xlim(*bounds[0])
    axis.set_ylim(*bounds[1])
    axis.set_zlim(*bounds[2])
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_title(f"Greedy evaluation episode {episode_index + 1}", pad=18)
    axis.view_init(elev=24, azim=-58)
    axis.legend(loc="upper left", fontsize=8)

    color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=color_norm, cmap=color_map),
        ax=axis,
        pad=0.08,
        shrink=0.72,
    )
    color_bar.set_label("Signed forcing u (blue = negative, white = zero, red = positive)")
    status = figure.text(0.5, 0.025, "", ha="center", fontsize=10)
    figure.suptitle(
        "Q-learning motion along the Lorenz attractor",
        fontsize=15,
        y=0.97,
    )
    figure.subplots_adjust(left=0.02, right=0.88, bottom=0.08, top=0.90)

    def draw_frame(frame_index: int) -> tuple[Any, ...]:
        end = int(frame_ends[frame_index])
        colored_path.set_segments(segments[:end])
        colored_path.set_array(control_values[:end])
        current_state = trajectory[end]
        position_marker._offsets3d = (
            np.asarray([current_state[0]]),
            np.asarray([current_state[1]]),
            np.asarray([current_state[2]]),
        )
        current_forcing = control_values[end - 1]
        position_marker.set_facecolor(color_map(color_norm(current_forcing)))
        status.set_text(
            f"t / τ = {times[end]:.2f}   |   "
            f"u = {current_forcing:+.1f}   |   "
            f"|u| = {forcing_magnitudes[end - 1]:.1f}"
        )
        return colored_path, position_marker, status

    movie = animation.FuncAnimation(
        figure,
        draw_frame,
        frames=frame_ends.size,
        interval=1_000.0 / fps,
        repeat=True,
        blit=False,
        cache_frame_data=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Rendering {frame_ends.size} evaluation-animation frames to {output_path}",
        flush=True,
    )
    try:
        movie.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    finally:
        plt.close(figure)
    print(f"Saved {output_path}", flush=True)


def plot_evaluation_diagnostics(
    evaluation: Mapping[str, Sequence[Any]],
    output_dir: Path | None,
    *,
    show: bool = False,
    dpi: int = 160,
) -> None:
    """Create separate plots from greedy ``evaluate_q_learning`` results."""
    trajectories = evaluation["trajectories"]
    controls = evaluation["control_values"]
    rewards = np.asarray(evaluation["episode_rewards"], dtype=np.float64)
    diverged = np.asarray(evaluation["diverged"], dtype=bool)
    if len(trajectories) == 0:
        raise ValueError("Evaluation results contain no trajectories")
    if len(controls) == 0:
        raise ValueError("Evaluation results contain no control sequences")
    if rewards.ndim != 1 or rewards.size == 0:
        raise ValueError("Evaluation results contain no episode rewards")
    if len(trajectories) != rewards.size or len(controls) != rewards.size:
        raise ValueError(
            "Evaluation results must contain one trajectory and control sequence "
            "per episode reward"
        )
    if diverged.shape != rewards.shape:
        raise ValueError(
            "evaluation['diverged'] must have one value per episode reward"
        )

    trajectory_arrays = [
        np.asarray(trajectory, dtype=np.float64) for trajectory in trajectories
    ]
    for trajectory in trajectory_arrays:
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 3
            or trajectory.shape[0] == 0
        ):
            raise ValueError("Each evaluation trajectory must have shape (n, 3)")
    reference_condition_label = np.array2string(
        trajectory_arrays[0][0],
        precision=3,
        separator=", ",
    )
    initial_states = np.asarray(
        [trajectory[0] for trajectory in trajectory_arrays], dtype=np.float64
    )
    has_perturbed_initial_states = bool(
        np.any(~np.isclose(initial_states, initial_states[0], rtol=0.0, atol=0.0))
    )

    control_arrays = [np.asarray(values, dtype=np.float64) for values in controls]
    for trial, (trajectory, control_values) in enumerate(
        zip(trajectory_arrays, control_arrays), start=1
    ):
        if control_values.ndim != 1:
            raise ValueError(
                "Each evaluation control sequence must be one-dimensional"
            )
        expected_controls = trajectory.shape[0] - 1
        if control_values.size != expected_controls:
            raise ValueError(
                f"Evaluation trial {trial} must contain one control value "
                f"per transition; expected {expected_controls}, received "
                f"{control_values.size}"
            )

    state_time_axes = [
        _lyapunov_time_axis(trajectory.shape[0]) for trajectory in trajectory_arrays
    ]
    control_time_axes = [times[:-1] for times in state_time_axes]
    actual_time_end = max(float(times[-1]) for times in state_time_axes)
    rounded_time_end = float(np.rint(actual_time_end))
    display_time_end = (
        rounded_time_end
        if rounded_time_end >= 1.0
        and np.isclose(actual_time_end, rounded_time_end, atol=0.02)
        else actual_time_end
    )

    def output_path(filename: str) -> Path | None:
        return None if output_dir is None else output_dir / filename

    representative_trajectory = trajectory_arrays[0]
    representative_controls = control_arrays[0]
    representative_times = state_time_axes[0]
    representative_control_times = control_time_axes[0]

    corrected_figure, (corrected_state_axis, corrected_control_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": (1.15, 1.0), "hspace": 0.14},
    )
    corrected_figure.suptitle(
        "Closed-loop Lorenz trajectory with greedy Q-learning correction",
        fontsize=15,
    )
    for coordinate, label in enumerate(("x", "y", "z")):
        corrected_state_axis.plot(
            representative_times,
            representative_trajectory[:, coordinate],
            linewidth=0.8,
            label=label,
        )
    corrected_state_axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    corrected_state_axis.set_ylabel("State")
    corrected_state_axis.set_title("Controlled Lorenz state")
    corrected_state_axis.grid(alpha=0.25)
    corrected_state_axis.legend(loc="upper right", ncol=3)

    corrected_control_axis.step(
        representative_control_times,
        representative_controls,
        where="post",
        color="crimson",
        linewidth=0.75,
    )
    corrected_control_axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    corrected_control_axis.set_xlim(0.0, display_time_end)
    corrected_control_axis.set_xlabel("Lyapunov times (t / τ)")
    corrected_control_axis.set_ylabel("Control correction u")
    corrected_control_axis.set_title("Correction applied to the x dynamics")
    corrected_control_axis.grid(alpha=0.25)
    corrected_figure.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.08,
        top=0.91,
        hspace=0.20,
    )
    _finish_figure(
        corrected_figure,
        output_path("evaluation_corrected_trajectory.png"),
        show=show,
        dpi=dpi,
        tight_layout=False,
    )

    control_figure, control_axis = plt.subplots(figsize=(14, 5.5))
    control_axis.step(
        representative_control_times,
        representative_controls,
        where="post",
        color="crimson",
        linewidth=0.75,
    )
    control_axis.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
    )
    control_axis.set_xlim(0.0, display_time_end)
    control_axis.set_xlabel("Lyapunov times (t / τ)")
    control_axis.set_ylabel("Control correction u")
    control_axis.set_title(
        "Greedy Q-learning control correction: final evaluation trial 1"
    )
    control_axis.grid(alpha=0.25)
    _finish_figure(
        control_figure,
        output_path("evaluation_control_correction.png"),
        show=show,
        dpi=dpi,
    )

    trajectory_figure = plt.figure(figsize=(9, 7))
    trajectory_axis = trajectory_figure.add_subplot(111, projection="3d")
    trajectory_colors = plt.cm.viridis(
        np.linspace(0.08, 0.92, len(trajectory_arrays))
    )
    label_trials = len(trajectory_arrays) <= 10
    for trial, (trajectory, color) in enumerate(
        zip(trajectory_arrays, trajectory_colors), start=1
    ):
        trajectory_axis.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color,
            linewidth=0.65,
            alpha=0.85,
            label=f"Trial {trial}" if label_trials else None,
        )
    trajectory_axis.set_xlabel("x")
    trajectory_axis.set_ylabel("y")
    trajectory_axis.set_zlabel("z")
    if has_perturbed_initial_states:
        trajectory_title = (
            "Evaluation trajectories near reference initial condition "
            f"{reference_condition_label}"
        )
    else:
        trajectory_title = (
            "Evaluation trajectory from fixed initial condition "
            f"{reference_condition_label}"
        )
    trajectory_axis.set_title(trajectory_title)
    if label_trials:
        trajectory_axis.legend(title="Evaluation trial")
    _finish_figure(
        trajectory_figure,
        output_path("evaluation_trajectory.png"),
        show=show,
        dpi=dpi,
    )

    state_figure, state_axes = plt.subplots(
        3,
        1,
        figsize=(14, 10),
        sharex=True,
    )
    initial_state_description = (
        "perturbed initial conditions"
        if has_perturbed_initial_states
        else "a fixed initial condition"
    )
    state_figure.suptitle(
        f"State responses near reference initial condition {reference_condition_label}\n"
        f"Greedy Q-table evaluation with {initial_state_description}"
    )
    for coordinate, (state_axis, coordinate_label) in enumerate(
        zip(state_axes, ("x", "y", "z"))
    ):
        for trial, (trajectory, times, color) in enumerate(
            zip(trajectory_arrays, state_time_axes, trajectory_colors), start=1
        ):
            state_axis.plot(
                times,
                trajectory[:, coordinate],
                color=color,
                linewidth=0.55,
                alpha=0.62,
                label=f"Trial {trial}" if label_trials else None,
            )
        state_axis.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        state_axis.set_xlim(0.0, display_time_end)
        state_axis.set_ylabel(coordinate_label)
        state_axis.grid(alpha=0.25)
    state_axes[-1].set_xlabel("Lyapunov times (t / τ)")
    if label_trials:
        state_axes[0].legend(
            loc="upper right",
            ncol=len(trajectory_arrays),
            title="Initial-condition trial",
        )
    _finish_figure(
        state_figure,
        output_path("evaluation_state_coordinates.png"),
        show=show,
        dpi=dpi,
    )


def plot_uncontrolled_evaluation_reference(
    evaluation: Mapping[str, Sequence[Any]],
    output_dir: Path | None,
    *,
    show: bool = False,
    dpi: int = 160,
) -> None:
    """Integrate and plot the ``u = 0`` reference for evaluation trial 1."""
    trajectories = evaluation.get("trajectories", [])
    if not trajectories:
        raise ValueError("Evaluation results contain no trajectory for a reference run")
    controlled_trajectory = np.asarray(trajectories[0], dtype=np.float64)
    if (
        controlled_trajectory.ndim != 2
        or controlled_trajectory.shape[1] != 3
        or controlled_trajectory.shape[0] < 2
    ):
        raise ValueError("Evaluation trial 1 trajectory must have shape (n, 3), n >= 2")

    number_of_steps = controlled_trajectory.shape[0] - 1
    uncontrolled_trajectory = qlearning.integrate_uncontrolled_lorenz(
        controlled_trajectory[0],
        number_of_steps,
    )
    times = _lyapunov_time_axis(uncontrolled_trajectory.shape[0])
    time_end = float(times[-1])
    rounded_time_end = float(np.rint(time_end))
    displayed_time_end = (
        rounded_time_end
        if rounded_time_end >= 1.0
        and np.isclose(time_end, rounded_time_end, atol=0.02)
        else time_end
    )

    def output_path(filename: str) -> Path | None:
        return None if output_dir is None else output_dir / filename

    lorenz_figure = plt.figure(figsize=(9, 8))
    lorenz_axis = lorenz_figure.add_subplot(111, projection="3d")
    lorenz_axis.plot(
        uncontrolled_trajectory[:, 0],
        uncontrolled_trajectory[:, 1],
        uncontrolled_trajectory[:, 2],
        color="tab:blue",
        linewidth=0.55,
    )
    lorenz_axis.scatter(
        *uncontrolled_trajectory[0],
        color="tab:green",
        s=42,
        label="Start",
        zorder=3,
    )
    lorenz_axis.scatter(
        *uncontrolled_trajectory[-1],
        color="tab:red",
        s=42,
        label="End",
        zorder=3,
    )
    lorenz_axis.set_xlabel("x")
    lorenz_axis.set_ylabel("y")
    lorenz_axis.set_zlabel("z")
    lorenz_axis.set_title(
        f"Uncontrolled Lorenz trajectory (u = 0, {displayed_time_end:g} Lyapunov times)"
    )
    lorenz_axis.legend()
    _finish_figure(
        lorenz_figure,
        output_path("evaluation_uncontrolled_lorenz.png"),
        show=show,
        dpi=dpi,
    )

    state_figure, state_axis = plt.subplots(figsize=(14, 5.5))
    for coordinate, label in enumerate(("x", "y", "z")):
        state_axis.plot(
            times,
            uncontrolled_trajectory[:, coordinate],
            linewidth=0.7,
            label=label,
        )
    state_axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    state_axis.set_xlim(0.0, displayed_time_end)
    state_axis.set_xlabel("Lyapunov times (t / τ)")
    state_axis.set_ylabel("State")
    state_axis.set_title(
        "Uncontrolled Lorenz reference: evaluation trial 1 initial condition, u = 0"
    )
    state_axis.grid(alpha=0.25)
    state_axis.legend(loc="upper right", ncol=3)
    _finish_figure(
        state_figure,
        output_path("evaluation_uncontrolled_state_path.png"),
        show=show,
        dpi=dpi,
    )

def _validated_q_table(
    q_table: np.ndarray,
    actions: Sequence[float],
    state_bounds: Sequence[Sequence[float]],
    reference_state: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Validate persisted table data and locate the reference state's bin."""
    q_table = np.asarray(q_table, dtype=np.float64)
    if q_table.ndim != 4:
        raise ValueError(f"Expected a 4-D Q-table, received shape {q_table.shape}")
    if any(size == 0 for size in q_table.shape):
        raise ValueError(f"Q-table dimensions must be nonzero, received {q_table.shape}")

    action_values = np.asarray(actions, dtype=np.float64)
    if action_values.ndim != 1 or action_values.size != q_table.shape[-1]:
        raise ValueError(
            "actions must be one-dimensional with one value per Q-table action; "
            f"expected {q_table.shape[-1]}, received shape {action_values.shape}"
        )

    bounds = np.asarray(state_bounds, dtype=np.float64)
    if bounds.shape != (3, 2):
        raise ValueError("state_bounds must have shape (3, 2)")
    state = np.asarray(reference_state, dtype=np.float64)
    widths = (bounds[:, 1] - bounds[:, 0]) / np.asarray(q_table.shape[:3])
    reference_bin = tuple(
        np.clip(
            np.floor((np.clip(state, bounds[:, 0], bounds[:, 1]) - bounds[:, 0]) / widths),
            0,
            np.asarray(q_table.shape[:3]) - 1,
        ).astype(int)
    )
    return q_table, action_values, bounds, reference_bin


def _bin_centers(low: float, high: float, bin_count: int) -> np.ndarray:
    """Return the state value at the middle of each discretizer bin."""
    edges = np.linspace(low, high, bin_count + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def plot_q_table_diagnostics(
    q_table: np.ndarray,
    actions: Sequence[float],
    state_bounds: Sequence[Sequence[float]],
    reference_state: Sequence[float],
    output_path: Path | None,
    *,
    show: bool = False,
    dpi: int = 160,
) -> None:
    """Visualize a learned table using only persisted run data."""
    q_table, action_values, bounds, reference_bin = _validated_q_table(
        q_table, actions, state_bounds, reference_state
    )
    z_index = reference_bin[2]
    extent = [bounds[1, 0], bounds[1, 1], bounds[0, 0], bounds[0, 1]]
    max_values = np.max(q_table[:, :, z_index, :], axis=-1)
    mean_values = np.mean(q_table[:, :, z_index, :], axis=-1)
    update_fraction = np.count_nonzero(q_table, axis=(0, 1, 3)) / (
        q_table.shape[0] * q_table.shape[1] * q_table.shape[3]
    )
    reference_values = q_table[reference_bin]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    figure.suptitle("Learned Q-table diagnostics", fontsize=15)

    max_axis = axes[0, 0]
    max_image = max_axis.imshow(
        max_values,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    max_axis.set_xlabel("y")
    max_axis.set_ylabel("x")
    max_axis.set_title(f"Maximum action value at z-bin {z_index}")
    figure.colorbar(max_image, ax=max_axis, label="max Q")

    mean_axis = axes[0, 1]
    mean_image = mean_axis.imshow(
        mean_values,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
    )
    mean_axis.set_xlabel("y")
    mean_axis.set_ylabel("x")
    mean_axis.set_title(f"Mean action value at z-bin {z_index}")
    figure.colorbar(mean_image, ax=mean_axis, label="mean Q")

    coverage_axis = axes[1, 0]
    coverage_axis.plot(np.arange(q_table.shape[2]), update_fraction, marker="o")
    coverage_axis.axvline(z_index, color="tab:red", linestyle="--", label="Reference bin")
    coverage_axis.set_xlabel("z-bin")
    coverage_axis.set_ylabel("Fraction of nonzero entries")
    coverage_axis.set_ylim(-0.02, 1.02)
    coverage_axis.set_title("Table coverage by z-bin")
    coverage_axis.grid(alpha=0.25)
    coverage_axis.legend()

    values_axis = axes[1, 1]
    values_axis.plot(action_values, reference_values, marker="o", color="tab:purple")
    values_axis.axhline(0.0, color="black", linewidth=0.8)
    values_axis.set_xlabel("Discrete control value")
    values_axis.set_ylabel("Q value")
    values_axis.set_title(f"Action values at state bin {reference_bin}")
    values_axis.grid(alpha=0.25)

    _finish_figure(figure, output_path, show=show, dpi=dpi)


def plot_q_table_action_map(
    q_table: np.ndarray,
    actions: Sequence[float],
    state_bounds: Sequence[Sequence[float]],
    reference_state: Sequence[float],
    output_path: Path | None,
    *,
    z_index: int | None = None,
    show: bool = False,
    dpi: int = 160,
) -> None:
    """Show which control each state bin prefers, one panel per discrete action.

    This follows the pythonprogramming.net Q-learning analysis chart: a state
    space scatter per action, green where that action is greedy and faded red
    where it is not.  That tutorial has a two-dimensional state, so this slices
    the Lorenz table at ``z_index``, which defaults to the z-bin holding the
    reference initial condition.  Bins
    the agent never updated stay grey; their argmax is an artifact of the
    zero-initialized table rather than a learned preference.
    """
    q_table, action_values, bounds, reference_bin = _validated_q_table(
        q_table, actions, state_bounds, reference_state
    )
    if z_index is None:
        z_index = int(reference_bin[2])
    elif not 0 <= z_index < q_table.shape[2]:
        raise ValueError(
            f"z_index must be in [0, {q_table.shape[2] - 1}]; received {z_index}"
        )
    slice_values = q_table[:, :, z_index, :]
    x_bins, y_bins, action_count = slice_values.shape

    best_values = np.max(slice_values, axis=-1, keepdims=True)
    # Match the reference chart's ``value == max(vals)`` test so ties stay green.
    is_best = slice_values == best_values
    visited = np.any(slice_values != 0.0, axis=-1)

    x_centers = _bin_centers(bounds[0, 0], bounds[0, 1], x_bins)
    y_centers = _bin_centers(bounds[1, 0], bounds[1, 1], y_bins)
    grid_x, grid_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    marker_size = float(np.clip((0.55 * 300.0 / max(x_bins, y_bins)) ** 2, 4.0, 200.0))

    columns = int(np.ceil(np.sqrt(action_count)))
    rows = int(np.ceil(action_count / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.1 * columns, 3.7 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    z_low = bounds[2, 0] + z_index * (bounds[2, 1] - bounds[2, 0]) / q_table.shape[2]
    z_high = z_low + (bounds[2, 1] - bounds[2, 0]) / q_table.shape[2]
    figure.suptitle(
        "Greedy action map by state bin "
        f"(z-bin {z_index}: {z_low:.1f} ≤ z < {z_high:.1f})",
        fontsize=15,
    )

    for index, axis in enumerate(axes.ravel()):
        if index >= action_count:
            axis.axis("off")
            continue
        greedy_here = visited & is_best[:, :, index]
        other_here = visited & ~is_best[:, :, index]
        axis.scatter(
            grid_x[~visited],
            grid_y[~visited],
            color="0.78",
            marker="o",
            s=marker_size,
            alpha=0.35,
            linewidths=0.0,
        )
        axis.scatter(
            grid_x[other_here],
            grid_y[other_here],
            color="tab:red",
            marker="o",
            s=marker_size,
            alpha=0.3,
            linewidths=0.0,
        )
        axis.scatter(
            grid_x[greedy_here],
            grid_y[greedy_here],
            color="tab:green",
            marker="o",
            s=marker_size,
            alpha=1.0,
            linewidths=0.0,
        )
        axis.set_title(
            f"Action {index}: u = {action_values[index]:+.1f}\n"
            f"greedy in {int(np.count_nonzero(greedy_here))} visited bins",
            fontsize=10,
        )
        axis.grid(alpha=0.2)
        if index // columns == rows - 1 or index + columns >= action_count:
            axis.set_xlabel("x")
        if index % columns == 0:
            axis.set_ylabel("y")

    visited_count = int(np.count_nonzero(visited))
    legend_handles = [
        Line2D([], [], marker="o", linestyle="none", color="tab:green",
               label="Greedy action for this bin"),
        Line2D([], [], marker="o", linestyle="none", color="tab:red", alpha=0.3,
               label="Visited, but not greedy"),
        Line2D([], [], marker="o", linestyle="none", color="0.78", alpha=0.6,
               label=f"Never updated ({x_bins * y_bins - visited_count} of "
                     f"{x_bins * y_bins} bins)"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    _finish_figure(figure, output_path, show=show, dpi=dpi)


def plot_run_figures(
    history: Mapping[str, Sequence[Any]],
    checkpoints: Mapping[str, Sequence[Any]],
    evaluation: Mapping[str, Sequence[Any]],
    q_table: np.ndarray,
    actions: Sequence[float],
    state_bounds: Sequence[Sequence[float]],
    reference_state: Sequence[float],
    output_dir: Path | None,
    *,
    action_map_z_bin: int | None = None,
    show: bool = False,
    dpi: int = 160,
) -> None:
    """Render the complete, existing plot set for one training checkpoint."""

    def output_path(filename: str) -> Path | None:
        return None if output_dir is None else output_dir / filename

    plot_training_diagnostics(
        history,
        output_path("training_diagnostics.png"),
        show=show,
        dpi=dpi,
    )
    plot_training_rollouts(history, output_dir, show=show, dpi=dpi)
    plot_target_acquisition(
        history,
        output_path("target_acquisition.png"),
        show=show,
        dpi=dpi,
    )
    plot_checkpoint_evaluations(
        checkpoints,
        output_path("checkpoint_evaluations.png"),
        show=show,
        dpi=dpi,
    )
    plot_evaluation_diagnostics(evaluation, output_dir, show=show, dpi=dpi)
    plot_uncontrolled_evaluation_reference(
        evaluation, output_dir, show=show, dpi=dpi
    )
    plot_q_table_diagnostics(
        q_table,
        actions,
        state_bounds,
        reference_state,
        output_path("q_table_diagnostics.png"),
        show=show,
        dpi=dpi,
    )
    plot_q_table_action_map(
        q_table,
        actions,
        state_bounds,
        reference_state,
        output_path("q_table_action_map.png"),
        z_index=action_map_z_bin,
        show=show,
        dpi=dpi,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-data",
        type=Path,
        default=qlearning.RUN_OUTPUT_PATH,
        help="Data file produced by QLearning.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("q_learning_plots"),
        help="Plot root; each run is written to Saved_Plots/<parameter label> inside it",
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=2.0,
        help="Frames per second for learning_progress.gif (default: 2)",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip GIF animations while retaining the static plots",
    )
    parser.add_argument(
        "--evaluation-episode",
        type=int,
        default=1,
        help="One-based final evaluation episode to animate (default: 1)",
    )
    parser.add_argument(
        "--evaluation-animation-fps",
        type=float,
        default=25.0,
        help="Frames per second for evaluation_trajectory_forcing.gif (default: 25)",
    )
    parser.add_argument(
        "--evaluation-animation-seconds",
        type=float,
        default=20.0,
        help="Approximate duration of the evaluation animation (default: 20)",
    )
    parser.add_argument(
        "--evaluation-animation-dpi",
        type=int,
        default=90,
        help="Resolution of the evaluation animation (default: 90 dpi)",
    )
    parser.add_argument(
        "--action-map-z-bin",
        type=int,
        default=None,
        help="z-bin for the greedy action map (default: the reference state's bin)",
    )
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not write plot or GIF files"
    )
    return parser


def main() -> None:
    """Load one completed run and render its plots."""
    parser = _build_parser()
    args = parser.parse_args()
    if args.dpi < 1:
        parser.error("--dpi must be at least 1")
    if not np.isfinite(args.gif_fps) or args.gif_fps <= 0.0:
        parser.error("--gif-fps must be finite and positive")
    if args.evaluation_episode < 1:
        parser.error("--evaluation-episode must be at least 1")
    if (
        not np.isfinite(args.evaluation_animation_fps)
        or args.evaluation_animation_fps <= 0.0
    ):
        parser.error("--evaluation-animation-fps must be finite and positive")
    if (
        not np.isfinite(args.evaluation_animation_seconds)
        or args.evaluation_animation_seconds <= 0.0
    ):
        parser.error("--evaluation-animation-seconds must be finite and positive")
    if args.evaluation_animation_dpi < 1:
        parser.error("--evaluation-animation-dpi must be at least 1")
    print(f"Loading run data from {args.run_data}", flush=True)
    try:
        _require_local_run_data(args.run_data)
        run = qlearning.load_q_learning_run(args.run_data)
    except FileNotFoundError:
        parser.error(
            f"run data not found: {args.run_data}; run QLearning.py first"
        )
    except ValueError as error:
        parser.error(str(error))

    required = {
        "history", "checkpoints", "evaluation", "q_table", "actions",
        "state_bounds", "reference_state", "final_epsilon",
    }
    missing = required.difference(run)
    if missing:
        parser.error(f"run data is missing: {', '.join(sorted(missing))}")

    history = run["history"]
    checkpoint_history = run["checkpoints"]
    evaluation = run["evaluation"]
    evaluation_count = len(evaluation.get("trajectories", []))
    if evaluation_count == 0:
        parser.error("evaluation contains no trajectories to animate")
    if args.evaluation_episode > evaluation_count:
        parser.error(
            f"--evaluation-episode must be in [1, {evaluation_count}]"
        )

    q_table_shape = np.asarray(run["q_table"]).shape
    if args.action_map_z_bin is not None and len(q_table_shape) == 4 and not (
        0 <= args.action_map_z_bin < q_table_shape[2]
    ):
        parser.error(
            f"--action-map-z-bin must be in [0, {q_table_shape[2] - 1}]"
        )

    settings = run.get("settings")
    if settings is None:
        settings = qlearning.EXPERIMENT_DEFAULTS.as_dict()
        print("Run data predates parameter recording; assuming QLearning.py defaults.")

    output_dir = None
    if not args.no_save:
        output_dir = _run_directory(
            args.output_dir.expanduser().resolve(),
            settings,
            _run_time(run, args.run_data),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "parameters.json").write_text(
            json.dumps(settings, indent=2, sort_keys=True, default=str) + "\n"
        )

    # Render the final-run plots at the root of the labeled run archive.
    print("Rendering final-run plots", flush=True)
    plot_run_figures(
        history,
        checkpoint_history,
        evaluation,
        run["q_table"],
        run["actions"],
        run["state_bounds"],
        run["reference_state"],
        output_dir,
        action_map_z_bin=args.action_map_z_bin,
        show=args.show,
        dpi=args.dpi,
    )
    if output_dir is not None and not args.no_gif:
        save_evaluation_trajectory_animation(
            evaluation,
            run["state_bounds"],
            qlearning.TARGET_FIXED_POINT,
            output_dir / "evaluation_trajectory_forcing.gif",
            episode_index=args.evaluation_episode - 1,
            forcing_scale=float(settings.get("u_ref", qlearning.U_REF)),
            fps=args.evaluation_animation_fps,
            duration=args.evaluation_animation_seconds,
            dpi=args.evaluation_animation_dpi,
        )

    checkpoint_episodes = list(checkpoint_history["episodes"])
    checkpoint_evaluations = checkpoint_history.get("evaluations")
    checkpoint_q_tables = checkpoint_history.get("q_tables")
    if checkpoint_evaluations is None or checkpoint_q_tables is None:
        print(
            "Run data predates checkpoint snapshots; generated final plots only. "
            "Run QLearning.py again to create plots every evaluation interval."
        )
    elif not (
        len(checkpoint_evaluations)
        == len(checkpoint_q_tables)
        == len(checkpoint_episodes)
    ):
        parser.error(
            "checkpoint evaluations, Q-tables, and episode numbers must have "
            "matching lengths"
        )
    else:
        for checkpoint_index, episode in enumerate(checkpoint_episodes):
            print(
                f"Rendering checkpoint {checkpoint_index + 1}/"
                f"{len(checkpoint_episodes)} (episode {episode})",
                flush=True,
            )
            snapshot_dir = (
                None if output_dir is None else output_dir / f"episode_{episode:04d}"
            )
            plot_run_figures(
                _history_through_episode(history, episode),
                _checkpoint_history_through_index(
                    checkpoint_history, checkpoint_index
                ),
                checkpoint_evaluations[checkpoint_index],
                checkpoint_q_tables[checkpoint_index],
                run["actions"],
                run["state_bounds"],
                run["reference_state"],
                snapshot_dir,
                action_map_z_bin=args.action_map_z_bin,
                show=args.show,
                dpi=args.dpi,
            )
            if snapshot_dir is not None:
                print(f"Saved episode {episode} learning snapshot to {snapshot_dir}")

        if output_dir is not None and not args.no_gif:
            save_learning_progress_gif(
                checkpoint_history,
                run["state_bounds"],
                qlearning.TARGET_FIXED_POINT,
                output_dir / "learning_progress.gif",
                fps=args.gif_fps,
                dpi=args.dpi,
            )

    final_window = min(50, len(history["episode_rewards"]))
    final_rewards = np.asarray(history["episode_rewards"][-final_window:], dtype=float)
    evaluation_rewards = np.asarray(evaluation["episode_rewards"], dtype=float)
    q_table = np.asarray(run["q_table"])
    print(f"Final epsilon: {run['final_epsilon']:.4f}")
    print(f"Mean reward over final {final_window} episodes: {np.mean(final_rewards):.4f}")
    print(f"Mean greedy evaluation reward: {np.mean(evaluation_rewards):.4f}")
    print(
        f"Evaluation divergences: {sum(evaluation['diverged'])}/"
        f"{len(evaluation['diverged'])}"
    )
    print(f"Nonzero Q-table entries: {np.count_nonzero(q_table)}/{q_table.size}")
    if output_dir is not None:
        print(f"Run saved to {output_dir}")

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
