"""Regression tests for the tabular Lorenz Q-learning workflow."""

from __future__ import annotations

import importlib.util
import dataclasses
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("QLearning.py")
SPEC = importlib.util.spec_from_file_location("qlearning_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup guard
    raise ImportError(f"Could not load {MODULE_PATH}")
qlearning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qlearning)


def experiment_settings(**changes: object):
    """Create an explicit experiment variant for an environment test."""
    return dataclasses.replace(qlearning.EXPERIMENT_DEFAULTS, **changes)


class QLearningRegressionTests(unittest.TestCase):
    def test_canonical_run_contains_all_plotting_data(self) -> None:
        settings = dataclasses.replace(
            qlearning.EXPERIMENT_DEFAULTS,
            episodes=3,
            evaluation_interval=2,
            eval_episodes=2,
            training_lyapunov_times=0.02,
            evaluation_lyapunov_times=0.02,
        )
        run = qlearning.run_q_learning(settings)

        self.assertTrue(
            {
                "history",
                "checkpoints",
                "evaluation",
                "uncontrolled_trajectory",
                "q_table",
                "actions",
                "state_bounds",
                "reference_state",
                "final_epsilon",
            }.issubset(run)
        )
        self.assertEqual(run["checkpoints"]["episodes"], [2, 3])
        self.assertEqual(len(run["checkpoints"]["mean_control_efforts"]), 2)
        self.assertEqual(len(run["history"]["episode_rewards"]), 3)
        self.assertEqual(np.asarray(run["q_table"]).ndim, 4)
        self.assertEqual(np.asarray(run["uncontrolled_trajectory"]).shape, (3, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.pkl"
            qlearning.save_q_learning_run(run, path)
            loaded = qlearning.load_q_learning_run(path)
        np.testing.assert_array_equal(loaded["q_table"], run["q_table"])

    def test_lyapunov_time_conversions_are_consistent(self) -> None:
        steps = 123
        lyapunov_times = qlearning.steps_to_lyapunov_times(steps)

        self.assertEqual(qlearning.lyapunov_times_to_steps(lyapunov_times), steps)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            qlearning.steps_to_lyapunov_times(-1)

    def test_default_state_cost_uses_phi(self) -> None:
        self.assertAlmostEqual(
            qlearning.default_state_cost_fn(2.0), float(qlearning.phi(2.0))
        )

    def test_full_episode_return_is_finite(self) -> None:
        env = qlearning.LorenzEnvEuler(
            experiment_settings(
                evaluation_lyapunov_times=0.05,
                action_bins=3,
                regularized=True,
            )
        )
        state = env.reset()
        total_reward = 0.0

        while True:
            state, reward, done, _ = env.step(1)
            total_reward += reward
            if done:
                break

        self.assertTrue(np.isfinite(total_reward))
        self.assertLessEqual(total_reward, 0.0)

    def test_nonfinite_next_state_is_divergence(self) -> None:
        env = qlearning.LorenzEnvEuler(
            qlearning.EXPERIMENT_DEFAULTS, controlled=False
        )
        env.reset(np.array([1e308, 0.0, 0.0]))

        with np.errstate(over="ignore", invalid="ignore"):
            _, reward, done, info = env.step(0.0)

        self.assertTrue(done)
        self.assertTrue(info["diverged"])
        self.assertTrue(np.isfinite(reward))
        self.assertAlmostEqual(reward, -1.0)

    def test_divergence_does_not_poison_training_or_evaluation(self) -> None:
        settings = experiment_settings(
            ic=(1e308, 0.0, 0.0),
            training_ic_perturbation=0.0,
            action_bins=1,
        )
        env = qlearning.LorenzEnvEuler(
            settings,
            training=True,
        )
        agent = qlearning.QLearningAgent(n_actions=1, epsilon=0.0, epsilon_min=0.0)

        with np.errstate(over="ignore", invalid="ignore"):
            history = qlearning.train_q_learning(env, agent, num_episodes=1)
            evaluation = qlearning.evaluate_q_learning(env, agent, num_episodes=1)

        self.assertEqual(history["diverged"], [True])
        self.assertTrue(np.isfinite(agent.q_table).all())
        self.assertTrue(np.isfinite(evaluation["episode_rewards"][0]))
        self.assertEqual(evaluation["diverged"], [True])

    def test_step_after_terminal_transition_requires_reset(self) -> None:
        env = qlearning.LorenzEnvEuler(
            experiment_settings(
                evaluation_lyapunov_times=qlearning.LYAPUNOV_EXP * qlearning.DT
            ),
            controlled=False,
        )
        env.reset()
        _, _, done, _ = env.step(0.0)

        self.assertTrue(done)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(0.0)

    def test_reset_rejects_nonfinite_state(self) -> None:
        env = qlearning.LorenzEnvEuler(
            qlearning.EXPERIMENT_DEFAULTS, controlled=False
        )
        with self.assertRaisesRegex(ValueError, "must be finite"):
            env.reset(np.array([np.nan, 0.0, 0.0]))

    def test_discretizer_rejects_infinite_state(self) -> None:
        discretizer = qlearning.StateDiscretizer()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            discretizer.discretize([np.inf, 0.0, 0.0])

    def test_step_requires_reset_even_with_assertions_disabled(self) -> None:
        env = qlearning.LorenzEnvEuler(
            qlearning.EXPERIMENT_DEFAULTS, controlled=False
        )
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(0.0)

    def test_episode_must_contain_an_integration_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "one integration step"):
            qlearning.LorenzEnvEuler(
                experiment_settings(evaluation_lyapunov_times=0.001),
                controlled=False,
            )

    def test_q_update_bootstraps_only_nonterminal_transitions(self) -> None:
        agent = qlearning.QLearningAgent(
            n_actions=2,
            learning_rate=0.5,
            discount_factor=0.9,
            epsilon=0.0,
            epsilon_min=0.0,
            state_bins=2,
        )
        state = np.array([-20.0, -20.0, 10.0])
        next_state = np.array([20.0, 20.0, 50.0])
        next_index = agent.discretize_state(next_state)
        agent.q_table[next_index] = [2.0, 4.0]

        td_error = agent.update(state, 1, -1.0, next_state, done=False)
        self.assertAlmostEqual(td_error, 2.6)
        self.assertAlmostEqual(agent.q_table[agent.discretize_state(state) + (1,)], 1.3)

        td_error = agent.update(state, 1, -1.0, [np.nan, 0.0, 0.0], done=True)
        self.assertAlmostEqual(td_error, -2.3)
        self.assertAlmostEqual(agent.q_table[agent.discretize_state(state) + (1,)], 0.15)

    def test_q_update_rejects_nonfinite_reward(self) -> None:
        agent = qlearning.QLearningAgent(n_actions=2)
        with self.assertRaisesRegex(ValueError, "reward must be finite"):
            agent.update([0.0, 0.0, 1.0], 0, np.nan, [0.0, 0.0, 1.0], False)

    def test_agent_rejects_nonfinite_hyperparameters(self) -> None:
        with self.assertRaises(ValueError):
            qlearning.QLearningAgent(n_actions=2, learning_rate=np.nan)
        with self.assertRaises(ValueError):
            qlearning.QLearningAgent(n_actions=2, discount_factor=np.nan)
        with self.assertRaises(ValueError):
            qlearning.QLearningAgent(n_actions=2, epsilon_decay=np.nan)

    def test_integer_configuration_does_not_silently_truncate(self) -> None:
        with self.assertRaises(TypeError):
            qlearning.QLearningAgent(n_actions=2.5)
        with self.assertRaises(ValueError):
            qlearning.StateDiscretizer(bins=2.5)

    def test_evaluation_trials_use_reproducible_distinct_initial_states(self) -> None:
        starts = qlearning.make_evaluation_initial_states(
            [0.0, 1.0, 1.05], 4, perturbation=0.01, random_seed=7
        )
        repeated = qlearning.make_evaluation_initial_states(
            [0.0, 1.0, 1.05], 4, perturbation=0.01, random_seed=7
        )

        np.testing.assert_array_equal(starts, repeated)
        reference = np.array([0.0, 1.0, 1.05])
        self.assertTrue(np.all(np.abs(starts - reference) <= 0.01))
        self.assertEqual(np.unique(starts, axis=0).shape[0], 4)

        env = qlearning.LorenzEnvEuler(
            experiment_settings(evaluation_lyapunov_times=0.02, action_bins=3)
        )
        agent = qlearning.QLearningAgent(
            n_actions=3, epsilon=0.0, epsilon_min=0.0
        )
        evaluation = qlearning.evaluate_q_learning(
            env, agent, num_episodes=4, initial_states=starts
        )
        actual_starts = np.asarray(
            [trajectory[0] for trajectory in evaluation["trajectories"]]
        )
        np.testing.assert_array_equal(actual_starts, starts)

    def test_training_sampler_randomizes_every_trial_reproducibly(self) -> None:
        sampler = qlearning.make_random_initial_state_sampler(
            [0.0, 1.0, 1.05], perturbation=0.01, random_seed=7
        )
        repeated_sampler = qlearning.make_random_initial_state_sampler(
            [0.0, 1.0, 1.05], perturbation=0.01, random_seed=7
        )
        starts = np.asarray([sampler() for _ in range(4)])
        repeated = np.asarray([repeated_sampler() for _ in range(4)])

        np.testing.assert_array_equal(starts, repeated)
        self.assertEqual(np.unique(starts, axis=0).shape[0], 4)
        self.assertTrue(
            np.all(np.abs(starts - np.array([0.0, 1.0, 1.05])) <= 0.01)
        )

    def test_checkpoint_training_preserves_continuous_history(self) -> None:
        settings = experiment_settings(
            training_lyapunov_times=0.02,
            evaluation_lyapunov_times=0.02,
            action_bins=3,
        )
        training_env = qlearning.LorenzEnvEuler(
            settings,
            training=True,
        )
        evaluation_env = qlearning.LorenzEnvEuler(settings)
        agent = qlearning.QLearningAgent(
            n_actions=3,
            epsilon=0.8,
            epsilon_decay=0.5,
            epsilon_min=0.0,
            random_seed=4,
        )
        starts = qlearning.make_evaluation_initial_states(
            [0.0, 1.0, 1.05], 2, perturbation=0.01, random_seed=9
        )

        history, checkpoints, final_evaluation = (
            qlearning.train_q_learning_with_evaluation(
                training_env,
                evaluation_env,
                agent,
                num_episodes=5,
                evaluation_interval=2,
                evaluation_episodes=2,
                evaluation_initial_states=starts,
            )
        )

        self.assertEqual(len(history["episode_rewards"]), 5)
        self.assertEqual(len(history["rolling_mean_rewards"]), 5)
        self.assertEqual(checkpoints["episodes"], [2, 4, 5])
        self.assertEqual(len(checkpoints["mean_rewards"]), 3)
        self.assertAlmostEqual(agent.epsilon, 0.8 * 0.5**5)
        actual_starts = np.asarray(
            [trajectory[0] for trajectory in final_evaluation["trajectories"]]
        )
        np.testing.assert_array_equal(actual_starts, starts)

    def test_training_callback_runs_at_intervals_and_final_episode(self) -> None:
        env = qlearning.LorenzEnvEuler(
            experiment_settings(training_lyapunov_times=0.02, action_bins=3),
            training=True,
        )
        agent = qlearning.QLearningAgent(
            n_actions=3,
            epsilon=0.8,
            epsilon_decay=0.5,
            epsilon_min=0.0,
        )
        evaluations = []

        history = qlearning.train_q_learning(
            env,
            agent,
            num_episodes=5,
            print_every=10,
            evaluation_interval=2,
            on_evaluation=lambda episode: evaluations.append(
                (episode, agent.epsilon)
            ),
        )

        self.assertEqual(
            evaluations,
            [(2, 0.8 * 0.5**2), (4, 0.8 * 0.5**4), (5, 0.8 * 0.5**5)],
        )
        self.assertEqual(len(history["episode_rewards"]), 5)

    def test_training_interval_requires_callback(self) -> None:
        env = qlearning.LorenzEnvEuler(
            experiment_settings(training_lyapunov_times=0.02, action_bins=3),
            training=True,
        )
        agent = qlearning.QLearningAgent(n_actions=3)

        with self.assertRaisesRegex(ValueError, "on_evaluation"):
            qlearning.train_q_learning(
                env,
                agent,
                num_episodes=2,
                evaluation_interval=1,
            )
        self.assertEqual(agent.epsilon, qlearning.EXPERIMENT_DEFAULTS.epsilon)

    def test_checkpoint_training_validates_interval_before_training(self) -> None:
        env = qlearning.LorenzEnvEuler(
            experiment_settings(evaluation_lyapunov_times=0.02, action_bins=3)
        )
        agent = qlearning.QLearningAgent(n_actions=3)

        with self.assertRaisesRegex(ValueError, "evaluation_interval"):
            qlearning.train_q_learning_with_evaluation(
                env,
                env,
                agent,
                num_episodes=2,
                evaluation_interval=0,
                evaluation_episodes=1,
            )
        self.assertEqual(agent.epsilon, qlearning.EXPERIMENT_DEFAULTS.epsilon)


if __name__ == "__main__":
    unittest.main()
