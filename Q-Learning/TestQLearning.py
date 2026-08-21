"""Regression tests for the tabular Lorenz Q-learning workflow."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("QLearning.py")
SPEC = importlib.util.spec_from_file_location("qlearning_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup guard
    raise ImportError(f"Could not load {MODULE_PATH}")
qlearning = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qlearning)


class QLearningRegressionTests(unittest.TestCase):
    def test_state_loss_matches_gradient_reference_phi(self) -> None:
        points = np.array([[-2.0, 0.0, 0.0], [0.0, 1.0, 1.0], [2.0, 0.0, 0.0]])
        expected = float(np.mean(qlearning.phi(points[:, 0])))

        self.assertAlmostEqual(qlearning.calculate_loss(points), expected)
        self.assertAlmostEqual(
            qlearning.default_state_cost_fn(2.0), float(qlearning.phi(2.0))
        )

    def test_full_episode_return_matches_reference_objective(self) -> None:
        env = qlearning.LorenzEnvEuler(
            lyapunov_times=0.05,
            action_type="discrete",
            n_action_bins=3,
            regularized=True,
        )
        state = env.reset()
        states = [state]
        controls = []
        total_reward = 0.0

        while True:
            state, reward, done, _ = env.step(1)
            states.append(state)
            controls.append(float(env.actions[1]))
            total_reward += reward
            if done:
                break

        objective = qlearning.reference_objective(
            np.asarray(states),
            controls,
            lam=env.alpha,
            regularized=True,
        )
        self.assertAlmostEqual(total_reward, -objective)

    def test_nonfinite_next_state_is_divergence(self) -> None:
        env = qlearning.LorenzEnvEuler(action_type="continuous")
        env.reset(np.array([1e308, 0.0, 0.0]))

        with np.errstate(over="ignore", invalid="ignore"):
            _, reward, done, info = env.step(0.0)

        self.assertTrue(done)
        self.assertTrue(info["diverged"])
        self.assertTrue(np.isfinite(reward))
        self.assertAlmostEqual(reward, -1.0)

    def test_divergence_does_not_poison_training_or_evaluation(self) -> None:
        env = qlearning.LorenzEnvEuler(
            action_type="discrete",
            n_action_bins=1,
            ic_dist=lambda: np.array([1e308, 0.0, 0.0]),
        )
        agent = qlearning.QLearningAgent(n_actions=1, epsilon=0.0, epsilon_min=0.0)

        with np.errstate(over="ignore", invalid="ignore"):
            history = qlearning.train_q_learning(env, agent, num_episodes=1)
            evaluation = qlearning.control_q_learning(env, agent, num_episodes=1)

        self.assertEqual(history["diverged"], [True])
        self.assertTrue(np.isfinite(agent.q_table).all())
        self.assertTrue(np.isfinite(evaluation["episode_rewards"][0]))
        self.assertEqual(evaluation["objectives"], [float("inf")])

    def test_step_after_terminal_transition_requires_reset(self) -> None:
        env = qlearning.LorenzEnvEuler(
            lyapunov_times=qlearning.LYAPUNOV_EXP * qlearning.DT,
            action_type="continuous",
        )
        env.reset()
        _, _, done, _ = env.step(0.0)

        self.assertTrue(done)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(0.0)

    def test_reset_rejects_nonfinite_state(self) -> None:
        env = qlearning.LorenzEnvEuler(action_type="continuous")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            env.reset(np.array([np.nan, 0.0, 0.0]))

    def test_discretizer_rejects_infinite_state(self) -> None:
        discretizer = qlearning.StateDiscretizer()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            discretizer.discretize([np.inf, 0.0, 0.0])

    def test_step_requires_reset_even_with_assertions_disabled(self) -> None:
        env = qlearning.LorenzEnvEuler(action_type="continuous")
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step(0.0)

    def test_episode_must_contain_an_integration_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "one integration step"):
            qlearning.LorenzEnvEuler(
                lyapunov_times=0.001,
                action_type="continuous",
                regularized=True,
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
        np.testing.assert_array_equal(starts[0], [0.0, 1.0, 1.05])
        self.assertTrue(np.all(np.abs(starts[1:] - starts[0]) <= 0.01))
        self.assertEqual(np.unique(starts, axis=0).shape[0], 4)

        env = qlearning.LorenzEnvEuler(
            lyapunov_times=0.02,
            action_type="discrete",
            n_action_bins=3,
        )
        agent = qlearning.QLearningAgent(
            n_actions=3, epsilon=0.0, epsilon_min=0.0
        )
        evaluation = qlearning.control_q_learning(
            env, agent, num_episodes=4, initial_states=starts
        )
        actual_starts = np.asarray(
            [trajectory[0] for trajectory in evaluation["trajectories"]]
        )
        np.testing.assert_array_equal(actual_starts, starts)


if __name__ == "__main__":
    unittest.main()
