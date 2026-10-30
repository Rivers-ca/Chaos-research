"""Focused regression and smoke tests for the Lorenz Double DQN controller."""

from __future__ import annotations

from dataclasses import replace
import random
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import DeepQLearning as deep
import QLearning as qlearning


def tiny_settings(**changes: object) -> deep.DeepExperimentDefaults:
    settings = replace(
        deep.DEEP_EXPERIMENT_DEFAULTS,
        episodes=3,
        eval_episodes=4,
        evaluation_interval=2,
        max_steps=8,
        eval_max_steps=8,
        print_every=100,
        training_plot_samples=3,
        use_attractor_initial_state_ensemble=False,
        hidden_layers=(16, 16),
        batch_size=4,
        replay_buffer_size=64,
        minimum_replay_size=4,
        target_update_interval=3,
        device_mode="cpu",
    )
    return replace(settings, **changes)


class DeepQLearningTests(unittest.TestCase):
    def setUp(self) -> None:
        deep.seed_everything(0)

    def test_network_uses_continuous_three_coordinate_input(self) -> None:
        network = deep.QNetwork(9, (128, 128))
        output = network(torch.zeros((5, 3)))

        self.assertEqual(tuple(output.shape), (5, 9))
        linear = [module for module in network.modules() if isinstance(module, torch.nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linear], [(3, 128), (128, 128), (128, 9)])

    def test_activation_is_restored_for_new_and_legacy_runs(self) -> None:
        for activation, expected in (("tanh", torch.nn.Tanh), ("relu", torch.nn.ReLU)):
            settings = tiny_settings(activation=activation)
            agent = deep.DoubleDQNAgent(settings, torch.device("cpu"))
            restored = deep.restore_agent(agent.checkpoint(1), settings)
            self.assertIsInstance(restored.online_network.layers[1], expected)
            self.assertIsInstance(restored.target_network.layers[3], expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / deep.DEEP_RUN_FILENAME
            legacy = {"controller": "Double DQN", "settings": {}}
            deep.save_deep_q_learning_run(legacy, path)
            self.assertEqual(deep.load_deep_q_learning_run(path)["settings"]["activation"], "relu")

    def test_state_normalizer_clips_bounds_and_rejects_nonfinite_values(self) -> None:
        normalizer = deep.StateNormalizer(qlearning.DEFAULT_STATE_BOUNDS)

        np.testing.assert_allclose(normalizer.normalize([0.0, 0.0, 30.0]), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(normalizer.normalize([100.0, -100.0, 100.0]), [1.0, -1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "finite"):
            normalizer.normalize([0.0, np.inf, 1.0])

    def test_equal_q_values_prefer_zero_forcing_action(self) -> None:
        settings = tiny_settings(epsilon_start=0.0, epsilon_end=0.0)
        agent = deep.DoubleDQNAgent(settings, torch.device("cpu"))
        for parameter in agent.online_network.parameters():
            parameter.data.zero_()

        action = agent.select_action(np.array([0.0, 1.0, 1.05]), training=False)

        self.assertEqual(action, settings.action_bins // 2)

    def test_double_dqn_online_selects_and_target_evaluates(self) -> None:
        settings = tiny_settings(
            action_bins=3,
            batch_size=1,
            minimum_replay_size=1,
            discount_factor=0.5,
            network_learning_rate=1e-6,
        )
        agent = deep.DoubleDQNAgent(settings, torch.device("cpu"))
        for network in (agent.online_network, agent.target_network):
            for parameter in network.parameters():
                parameter.data.zero_()
        agent.online_network.layers[-1].bias.data.copy_(torch.tensor([0.0, 2.0, 1.0]))
        agent.target_network.layers[-1].bias.data.copy_(torch.tensor([10.0, 20.0, 30.0]))
        state = np.array([0.0, 1.0, 1.05])
        agent.observe(state, 0, 1.0, state, False)

        stats = agent.train_step()

        self.assertIsNotNone(stats)
        # Target = 1 + .5 * target(next, argmax(online(next))) = 11.
        self.assertAlmostEqual(stats["loss"], 10.5, places=5)

    def test_terminal_transition_does_not_bootstrap(self) -> None:
        settings = tiny_settings(
            action_bins=3,
            batch_size=1,
            minimum_replay_size=1,
            discount_factor=0.5,
            network_learning_rate=1e-6,
        )
        agent = deep.DoubleDQNAgent(settings, torch.device("cpu"))
        for network in (agent.online_network, agent.target_network):
            for parameter in network.parameters():
                parameter.data.zero_()
        agent.target_network.layers[-1].bias.data.fill_(100.0)
        state = np.array([0.0, 1.0, 1.05])
        agent.observe(state, 0, 1.0, state, True)

        stats = agent.train_step()

        self.assertIsNotNone(stats)
        self.assertAlmostEqual(stats["loss"], 0.5, places=5)

    def test_replay_waits_for_a_complete_batch(self) -> None:
        settings = tiny_settings(batch_size=8, minimum_replay_size=1)
        agent = deep.DoubleDQNAgent(settings, torch.device("cpu"))
        state = np.array([0.0, 1.0, 1.05])
        for _ in range(7):
            agent.observe(state, 0, 0.0, state, False)
        self.assertIsNone(agent.train_step())
        agent.observe(state, 0, 0.0, state, False)
        self.assertIsNotNone(agent.train_step())
        with self.assertRaisesRegex(ValueError, "batch_size"):
            tiny_settings(batch_size=65)

    def test_nonfinite_terminal_state_is_safe_but_nonterminal_is_rejected(self) -> None:
        settings = tiny_settings(batch_size=1, minimum_replay_size=1)
        agent = deep.DoubleDQNAgent(settings, torch.device("cpu"))
        state = np.array([0.0, 1.0, 1.05])
        agent.observe(state, 0, -1.0, np.full(3, np.inf), True)
        np.testing.assert_array_equal(agent.replay.next_states[0], np.zeros(3))
        self.assertTrue(np.isfinite(agent.train_step()["loss"]))
        with self.assertRaisesRegex(ValueError, "finite"):
            agent.observe(state, 0, -1.0, np.full(3, np.inf), False)

    def test_target_updates_on_schedule_without_target_gradients(self) -> None:
        settings = tiny_settings(batch_size=1, minimum_replay_size=1, target_update_interval=2)
        agent = deep.DoubleDQNAgent(settings, torch.device("cpu"))
        state = np.array([0.0, 1.0, 1.05])
        original = {name: value.clone() for name, value in agent.target_network.state_dict().items()}
        agent.observe(state, 0, 1.0, state, True)
        agent.train_step()
        for name, value in agent.target_network.state_dict().items():
            torch.testing.assert_close(value, original[name])
        agent.observe(state, 0, 1.0, state, True)
        agent.train_step()
        for online, target in zip(agent.online_network.parameters(), agent.target_network.parameters()):
            torch.testing.assert_close(online, target)
            self.assertIsNone(target.grad)

    def test_ties_choose_smallest_actual_forcing_for_asymmetric_actions(self) -> None:
        settings = tiny_settings(action_low=-1.0, action_high=7.0)
        agent = deep.DoubleDQNAgent(settings, torch.device("cpu"))
        for parameter in agent.online_network.parameters():
            parameter.data.zero_()
        self.assertEqual(agent.select_action(np.zeros(3), training=False), 1)

    def test_replay_wraparound_and_epsilon_schedule(self) -> None:
        replay = deep.ReplayBuffer(3, 0)
        for index in range(5):
            replay.add(np.full(3, index), 0, float(index), np.full(3, index + 1), False)
        self.assertEqual(len(replay), 3)
        self.assertEqual(set(replay.sample(3)[2]), {2.0, 3.0, 4.0})
        agent = deep.DoubleDQNAgent(tiny_settings(epsilon_decay_steps=10), torch.device("cpu"))
        self.assertEqual(agent.epsilon, 1.0)
        agent.global_step = 5
        self.assertAlmostEqual(agent.epsilon, 0.51)
        agent.global_step = 20
        self.assertAlmostEqual(agent.epsilon, 0.02)

    def test_checkpoint_eligibility_and_ranking_follow_specification(self) -> None:
        base = {
            "divergence_count": 0,
            "successful_trial_count": 21,
            "mean_absolute_forcing": 1.0,
            "rms_forcing": 1.2,
            "zero_forcing_percentage": 20.0,
        }
        lower_forcing = {**base, "mean_absolute_forcing": 0.9, "successful_trial_count": 40}
        divergent = {**base, "divergence_count": 1}

        self.assertTrue(deep.checkpoint_is_eligible(base))
        self.assertFalse(deep.checkpoint_is_eligible(divergent))
        self.assertLess(deep.checkpoint_rank(lower_forcing), deep.checkpoint_rank(base))

    def test_checkpoint_restore_includes_all_random_generators(self) -> None:
        settings = tiny_settings()
        agent = deep.DoubleDQNAgent(settings, torch.device("cpu"))
        checkpoint = agent.checkpoint(7)
        expected = (random.random(), np.random.random(), torch.rand(1).item())

        deep.restore_agent(checkpoint, settings)
        actual = (random.random(), np.random.random(), torch.rand(1).item())

        self.assertEqual(actual, expected)

    def test_smoke_training_serialization_restore_and_plotting(self) -> None:
        settings = tiny_settings()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            run = deep.run_deep_q_learning(settings, output_directory=output)
            run_path = deep.save_deep_q_learning_run(run, output / deep.DEEP_RUN_FILENAME)
            loaded = deep.load_deep_q_learning_run(run_path)
            restored = deep.restore_agent(loaded["final_checkpoint"], settings)
            plots = deep.plot_deep_q_learning(loaded, output)

            self.assertEqual(run["checkpoints"]["episodes"], [2, 3])
            self.assertEqual(len(run["checkpoints"]["evaluations"]), 2)
            self.assertEqual(len(run["evaluation"]["episode_negative_lobe_percentages"]), 4)
            self.assertEqual(run["evaluation_initial_states"].shape, (4, 3))
            self.assertEqual(restored.global_step, run["final_checkpoint"]["global_step"])
            self.assertTrue((output / "checkpoint_episode_0002.pt").is_file())
            self.assertTrue((output / "checkpoint_episode_0003.pt").is_file())
            self.assertTrue((output / "final_model.pt").is_file())
            self.assertFalse((output / "best_model.pt").exists())
            self.assertGreater(len(run["history"]["losses"]), 0)
            self.assertEqual(len(plots), 19)
            self.assertTrue(all(path.is_file() for path in plots))
            for path in plots:
                with Image.open(path) as chart:
                    self.assertEqual(chart.format, "PNG")
                    chart.verify()
            action = restored.select_action(np.array([0.0, 1.0, 1.05]), training=False)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, settings.action_bins)

    def test_plot_values_match_run_and_control_covers_last_interval(self) -> None:
        run = deep.run_deep_q_learning(tiny_settings(episodes=1, max_steps=1, eval_max_steps=1))
        figures = {}
        with tempfile.TemporaryDirectory() as directory:
            def capture(fig, path, **kwargs):
                figures[path.name] = fig

            with (
                patch.object(deep, "_save_figure", side_effect=capture),
                patch.object(deep.shared_plots, "_finish_figure", side_effect=capture),
            ):
                deep.plot_deep_q_learning(run, directory)
            try:
                bars = figures["evaluation_lobe_occupancy.png"].axes[0].patches
                np.testing.assert_allclose([bar.get_height() for bar in bars], [run["evaluation"]["negative_lobe_percentage"], run["evaluation"]["positive_lobe_percentage"]])
                reward_lines = figures["training_diagnostics.png"].axes[0].lines
                np.testing.assert_allclose(reward_lines[0].get_ydata(), run["history"]["episode_rewards"])
                np.testing.assert_allclose(reward_lines[1].get_ydata(), run["history"]["rolling_mean_rewards"])
                control_axis = figures["evaluation_control_correction.png"].axes[0]
                control = control_axis.lines[0]
                end_time = qlearning.steps_to_lyapunov_times(1)
                np.testing.assert_allclose(control.get_xdata(), [0, end_time])
                np.testing.assert_allclose(control.get_ydata(), np.repeat(run["evaluation"]["control_values"][0], 2))
                self.assertEqual(control_axis.get_xlabel(), "Lyapunov times (t / τ)")
                state_axes = figures["evaluation_state_coordinates.png"].axes
                self.assertEqual(len(state_axes), 3 * len(run["evaluation"]["trajectories"]))
                for trial, trajectory in enumerate(run["evaluation"]["trajectories"]):
                    for coordinate in range(3):
                        axis = state_axes[3 * trial + coordinate]
                        np.testing.assert_allclose(axis.lines[0].get_xdata(), [0, end_time])
                        np.testing.assert_allclose(axis.lines[0].get_ydata(), trajectory[:, coordinate])
                    self.assertEqual(state_axes[3 * trial].get_title(), f"Trial {trial + 1}")
                    self.assertEqual(state_axes[3 * trial + 2].get_xlabel(), "Lyapunov times (t / τ)")
                training_axes = figures["training_diagnostics.png"].axes
                self.assertIn("Double DQN", figures["training_diagnostics.png"]._suptitle.get_text())
                np.testing.assert_allclose(training_axes[2].lines[0].get_ydata(), [end_time])
                checkpoint_axes = figures["checkpoint_evaluations.png"].axes
                checkpoint = run["checkpoints"]["evaluations"][0]
                np.testing.assert_allclose(checkpoint_axes[0].lines[0].get_ydata(), [np.mean(checkpoint["episode_rewards"])])
                np.testing.assert_allclose(checkpoint_axes[2].lines[0].get_ydata(), [checkpoint["metrics"]["normalized_control_effort"]])
                for figure in figures.values():
                    for axis in figure.axes:
                        self.assertNotIn("step", axis.get_xlabel().lower())
                self.assertIn("Final model (episode 1)", figures["evaluation_trajectory.png"]._suptitle.get_text())
            finally:
                for figure in figures.values():
                    deep.plt.close(figure)

    def test_cli_smoke_saves_without_a_tabular_run_and_can_replot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cli"
            with patch.object(deep, "compare_with_tabular_episode") as compare:
                self.assertEqual(deep.main(["--smoke", "--device", "cpu", "--output-directory", str(output)]), 0)
                compare.assert_not_called()
            archive = next(output.iterdir())
            self.assertIn("tanh", archive.name)
            self.assertEqual(json.loads((archive / "parameters.json").read_text())["activation"], "tanh")
            original = {path.name: path.read_bytes() for path in archive.glob("*.png")}
            self.assertEqual(deep.main(["--plot-run", str(archive / deep.DEEP_RUN_FILENAME), "--output-directory", str(output)]), 0)
            archives = list(output.iterdir())
            self.assertEqual(len(archives), 2)
            replot = next(path for path in archives if path != archive)
            self.assertEqual(len(list(replot.glob("*.png"))), 19)
            self.assertTrue((replot / "parameters.json").is_file())
            self.assertEqual(original, {path.name: path.read_bytes() for path in archive.glob("*.png")})

    def test_repeated_cpu_smoke_runs_are_reproducible(self) -> None:
        settings = tiny_settings(episodes=2, evaluation_interval=2)

        first = deep.run_deep_q_learning(settings)
        second = deep.run_deep_q_learning(settings)

        self.assertEqual(first["history"]["episode_rewards"], second["history"]["episode_rewards"])
        self.assertEqual(first["history"]["losses"], second["history"]["losses"])
        np.testing.assert_array_equal(
            first["evaluation"]["actions"][0], second["evaluation"]["actions"][0]
        )


if __name__ == "__main__":
    unittest.main()
