"""Regression tests for Q-learning plot and animation outputs."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).with_name("PlotQLearning.py")
SPEC = importlib.util.spec_from_file_location("plot_qlearning_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup guard
    raise ImportError(f"Could not load {MODULE_PATH}")
plot_qlearning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_qlearning)


class PlotQLearningRegressionTests(unittest.TestCase):
    def test_evaluation_trajectories_use_separate_panels_with_common_limits(self) -> None:
        trajectories = [
            np.array([[index, 1.0, 2.0], [index + 1, 3.0, 4.0]])
            for index in range(7)
        ]
        with patch.object(plot_qlearning, "_finish_figure") as finish:
            plot_qlearning.plot_evaluation_trajectories(trajectories, None)
        figure = finish.call_args.args[0]
        try:
            self.assertEqual(len(figure.axes), len(trajectories))
            for trial, (axis, trajectory) in enumerate(zip(figure.axes, trajectories), start=1):
                self.assertEqual(axis.get_title(), f"Trial {trial}")
                self.assertEqual(len(axis.lines), 1)
                np.testing.assert_allclose(np.array(axis.lines[0].get_data_3d()).T, trajectory)
                for getter in ("get_xlim", "get_ylim", "get_zlim"):
                    self.assertEqual(getattr(axis, getter)(), getattr(figure.axes[0], getter)())
        finally:
            plot_qlearning.plt.close(figure)

    def test_state_coordinate_grid_keeps_trials_and_time_axes_separate(self) -> None:
        trajectories = [np.arange(length * 3).reshape(length, 3) for length in (2, 4, 3, 5, 2)]
        with patch.object(plot_qlearning, "_finish_figure") as finish:
            plot_qlearning.plot_evaluation_state_coordinates(trajectories, None)
        figure = finish.call_args.args[0]
        try:
            self.assertEqual(len(figure.axes), 3 * len(trajectories))
            for trial, trajectory in enumerate(trajectories):
                for coordinate in range(3):
                    axis = figure.axes[3 * trial + coordinate]
                    np.testing.assert_allclose(axis.lines[0].get_ydata(), trajectory[:, coordinate])
                    np.testing.assert_allclose(axis.lines[0].get_xdata(), plot_qlearning._lyapunov_time_axis(len(trajectory)))
                    self.assertEqual(axis.get_ylim(), figure.axes[coordinate].get_ylim())
                    self.assertEqual(axis.get_xlim(), figure.axes[0].get_xlim())
                self.assertEqual(figure.axes[3 * trial].get_title(), f"Trial {trial + 1}")
        finally:
            plot_qlearning.plt.close(figure)

    def test_trial_occupancy_aligns_each_bar_with_its_tick_and_initial_condition(self) -> None:
        trajectories = [np.array([[-1.0, 2.0, 3.0], [-2.0, 3.0, 4.0]]),
                        np.array([[4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])]
        evaluation = {"trajectories": trajectories, "episode_negative_lobe_percentages": [51.0, 49.0]}
        with patch.object(plot_qlearning, "_finish_figure") as finish:
            plot_qlearning.plot_evaluation_trial_lobe_occupancy(evaluation, None)
        figure = finish.call_args.args[0]
        try:
            axis = figure.axes[0]
            np.testing.assert_array_equal(axis.get_xticks(), [1, 2])
            np.testing.assert_allclose([bar.get_x() + bar.get_width() / 2 for bar in axis.patches], axis.get_xticks())
            np.testing.assert_allclose([bar.get_height() for bar in axis.patches], [51.0, 49.0])
            self.assertEqual(axis.get_xticklabels()[0].get_text(), "1 | IC (-1.000, 2.000, 3.000)")
            self.assertEqual(axis.get_xticklabels()[1].get_text(), "2 | IC (4.000, 5.000, 6.000)")
            self.assertNotEqual(axis.patches[0].get_facecolor(), axis.patches[1].get_facecolor())
            np.testing.assert_allclose(axis.lines[0].get_ydata(), [51, 51])
        finally:
            plot_qlearning.plt.close(figure)
        # Older tabular archives derive per-trial percentages from trajectories.
        with patch.object(plot_qlearning, "_finish_figure") as finish:
            plot_qlearning.plot_evaluation_trial_lobe_occupancy({"trajectories": trajectories}, None)
        figure = finish.call_args.args[0]
        try:
            np.testing.assert_allclose([bar.get_height() for bar in figure.axes[0].patches], [100, 0])
        finally:
            plot_qlearning.plt.close(figure)

    def test_evaluation_lobe_occupancy_bar_chart_is_saved(self) -> None:
        evaluation = {
            "negative_lobe_percentage": 51.0,
            "positive_lobe_percentage": 49.0,
        }

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "evaluation_lobe_occupancy.png"
            plot_qlearning.plot_evaluation_lobe_occupancy(
                evaluation, output_path, dpi=40
            )

            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as chart:
                self.assertEqual(chart.format, "PNG")

    def test_saved_settings_are_strict_json(self) -> None:
        settings = plot_qlearning.qlearning.EXPERIMENT_DEFAULTS.as_dict()

        encoded = json.dumps(
            plot_qlearning._json_safe_settings(settings), allow_nan=False
        )

        self.assertIsNone(json.loads(encoded)["divergence_threshold"])

    def test_cloud_placeholder_is_rejected_before_loading(self) -> None:
        with (
            patch.object(plot_qlearning.sys, "platform", "darwin"),
            patch.object(Path, "stat", return_value=SimpleNamespace(st_flags=0x40000000)),
        ):
            with self.assertRaisesRegex(ValueError, "Download Now or Keep Downloaded"):
                plot_qlearning._require_local_run_data(Path("cloud-run.pkl.gz"))

    def test_local_run_data_is_accepted(self) -> None:
        with (
            patch.object(plot_qlearning.sys, "platform", "darwin"),
            patch.object(Path, "stat", return_value=SimpleNamespace(st_flags=0)),
        ):
            plot_qlearning._require_local_run_data(Path("local-run.pkl.gz"))

    @unittest.skipUnless(sys.platform == "darwin", "macOS font-discovery regression")
    def test_cli_starts_with_a_fresh_font_cache(self) -> None:
        environment = os.environ.copy()
        environment.pop("MPL_IGNORE_SYSTEM_FONTS", None)
        with tempfile.TemporaryDirectory() as directory:
            environment["MPLCONFIGDIR"] = directory
            # Fail immediately if startup attempts the blocking system queries.
            script = """
import runpy
import subprocess
import sys
original_check_output = subprocess.check_output
def check_output(command, *args, **kwargs):
    if command[0] in {"system_profiler", "fc-list"}:
        raise RuntimeError("Startup queried system fonts")
    return original_check_output(command, *args, **kwargs)
subprocess.check_output = check_output
sys.argv = [sys.argv[1], "--help"]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
            result = subprocess.run(
                [sys.executable, "-c", script, str(MODULE_PATH.resolve())],
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--run-data", result.stdout)

    def test_run_directory_uses_parameter_label(self) -> None:
        settings = {
            "episodes": 1000, "learning_rate": 0.1, "discount_factor": 0.99,
            "control_cost": 0.01, "u_ref": 2.5, "epsilon": 0.99,
            "epsilon_decay": 0.995, "state_bins": (20, 20, 20), "action_bins": 9,
        }
        path = plot_qlearning._run_directory(
            Path("plots"), settings, datetime(2026, 9, 3, 19, 34, 46)
        )

        self.assertEqual(path.parent, Path("plots/Saved_Plots"))
        self.assertEqual(
            path.name,
            "20260903-193446_ep1000_lr0.1_gamma0.99_"
            "alpha0.01_forcing2.5_eps0.99-0.995_bins20x20x20_actions9",
        )

    def test_run_time_prefers_stable_archive_metadata(self) -> None:
        expected = datetime.fromisoformat("2026-09-03T19:34:46-07:00")
        actual = plot_qlearning._run_time(
            {"created_at": expected.isoformat()}, Path("does-not-need-to-exist.pkl")
        )

        self.assertEqual(actual, expected)

    def test_learning_progress_gif_contains_one_frame_per_checkpoint(self) -> None:
        first_trajectory = np.array(
            [[0.0, 1.0, 1.05], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        )
        second_trajectory = np.array(
            [[0.0, 1.0, 1.05], [-1.0, -2.0, 5.0], [-3.0, -4.0, 8.0]]
        )
        checkpoints = {
            "episodes": [5, 10],
            "mean_rewards": [-2.0, -1.0],
            "reward_standard_deviations": [0.2, 0.1],
            "divergence_rates": [0.5, 0.0],
            "evaluations": [
                {"trajectories": [first_trajectory]},
                {"trajectories": [second_trajectory]},
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "learning_progress.gif"
            plot_qlearning.save_learning_progress_gif(
                checkpoints,
                ((-30.0, 30.0), (-30.0, 30.0), (0.0, 60.0)),
                plot_qlearning.qlearning.TARGET_FIXED_POINT,
                output_path,
                fps=2.0,
                dpi=40,
            )

            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as movie:
                self.assertEqual(movie.format, "GIF")
                self.assertEqual(movie.n_frames, 2)

    def test_evaluation_animation_is_smooth_and_uses_forcing_colors(self) -> None:
        trajectory = np.array(
            [
                [0.0, 1.0, 1.05],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [1.0, 4.0, 5.0],
                [0.0, 2.0, 6.0],
            ]
        )
        evaluation = {
            "trajectories": [trajectory],
            "control_values": [np.array([0.0, 20.0, -40.0, 60.0])],
        }

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "evaluation_trajectory_forcing.gif"
            plot_qlearning.save_evaluation_trajectory_animation(
                evaluation,
                ((-30.0, 30.0), (-30.0, 30.0), (0.0, 60.0)),
                plot_qlearning.qlearning.TARGET_FIXED_POINT,
                output_path,
                fps=4.0,
                duration=1.0,
                dpi=40,
            )

            self.assertTrue(output_path.is_file())
            with Image.open(output_path) as movie:
                self.assertEqual(movie.format, "GIF")
                self.assertEqual(movie.n_frames, 4)
                movie.seek(0)
                first_frame = np.asarray(movie.convert("RGB"))
                movie.seek(3)
                last_frame = np.asarray(movie.convert("RGB"))
                self.assertFalse(np.array_equal(first_frame, last_frame))


if __name__ == "__main__":
    unittest.main()
