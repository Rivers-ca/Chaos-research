"""Regression tests for Q-learning plot and animation outputs."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = Path(__file__).with_name("PlotQLearning.py")
SPEC = importlib.util.spec_from_file_location("plot_qlearning_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup guard
    raise ImportError(f"Could not load {MODULE_PATH}")
plot_qlearning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot_qlearning)


class PlotQLearningRegressionTests(unittest.TestCase):
    def test_run_directory_uses_parameter_label(self) -> None:
        settings = plot_qlearning.qlearning.EXPERIMENT_DEFAULTS.as_dict()
        path = plot_qlearning._run_directory(
            Path("plots"), settings, datetime(2026, 9, 3, 19, 34, 46)
        )

        self.assertEqual(path.parent, Path("plots/Saved_Plots"))
        self.assertEqual(
            path.name,
            "20260903-193446_ep1000_lr0.01_gamma0.99_"
            "eps0.99-0.995_bins20x20x20_actions9",
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


if __name__ == "__main__":
    unittest.main()
