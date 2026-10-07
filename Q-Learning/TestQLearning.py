"""Regression tests for the tabular Lorenz Q-learning workflow."""

from __future__ import annotations

import importlib.util
import dataclasses
import io
import tempfile
import unittest
from contextlib import redirect_stdout
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
                "q_table",
                "actions",
                "state_bounds",
                "reference_state",
                "final_epsilon",
                "settings",
                "created_at",
            }.issubset(run)
        )
        self.assertEqual(run["checkpoints"]["episodes"], [2, 3])
        self.assertEqual(len(run["checkpoints"]["mean_control_efforts"]), 2)
        self.assertEqual(len(run["checkpoints"]["target_acquisition_rates"]), 2)
        self.assertEqual(len(run["checkpoints"]["mean_target_occupancies"]), 2)
        self.assertEqual(len(run["checkpoints"]["evaluations"]), 2)
        self.assertEqual(len(run["checkpoints"]["q_tables"]), 2)
        self.assertEqual(len(run["checkpoints"]["epsilons"]), 2)
        self.assertEqual(len(run["history"]["episode_rewards"]), 3)
        self.assertEqual(run["history"]["sampled_episodes"], [1, 2, 3])
        self.assertEqual(len(run["history"]["sampled_trajectories"]), 3)
        self.assertEqual(len(run["history"]["sampled_control_values"]), 3)
        for trajectory, controls in zip(
            run["history"]["sampled_trajectories"],
            run["history"]["sampled_control_values"],
        ):
            self.assertEqual(trajectory.shape[0], controls.shape[0] + 1)
            self.assertEqual(trajectory.shape[1], 3)
        self.assertEqual(np.asarray(run["q_table"]).ndim, 4)
        self.assertEqual(len(run["evaluation"]["first_target_steps"]), 2)
        self.assertEqual(len(run["evaluation"]["target_steps"]), 2)
        self.assertIn("negative_lobe_percentage", run["evaluation"])
        self.assertIn("positive_lobe_percentage", run["evaluation"])
        self.assertEqual(
            len(run["evaluation"]["episode_negative_lobe_percentages"]), 2
        )
        self.assertGreaterEqual(run["evaluation"]["negative_lobe_success_rate"], 0.0)
        self.assertLessEqual(run["evaluation"]["negative_lobe_success_rate"], 100.0)
        self.assertAlmostEqual(
            run["evaluation"]["negative_lobe_percentage"]
            + run["evaluation"]["positive_lobe_percentage"],
            100.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            compressed_path = Path(directory) / "run.pkl.gz"
            qlearning.save_q_learning_run(run, compressed_path)
            compressed = qlearning.load_q_learning_run(compressed_path)

            legacy_path = Path(directory) / "run.pkl"
            qlearning.save_q_learning_run(run, legacy_path)
            legacy = qlearning.load_q_learning_run(legacy_path)

        self.assertEqual(qlearning.RUN_OUTPUT_PATH.suffixes[-2:], [".pkl", ".gz"])
        np.testing.assert_array_equal(compressed["q_table"], run["q_table"])
        np.testing.assert_array_equal(legacy["q_table"], run["q_table"])

    def test_lyapunov_time_conversions_are_consistent(self) -> None:
        steps = 123
        lyapunov_times = qlearning.steps_to_lyapunov_times(steps)

        self.assertEqual(qlearning.lyapunov_times_to_steps(lyapunov_times), steps)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            qlearning.steps_to_lyapunov_times(-1)

    def test_uncontrolled_integration_uses_zero_control_euler_steps(self) -> None:
        initial_state = np.array([0.0, 1.0, 1.05])
        trajectory = qlearning.integrate_uncontrolled_lorenz(initial_state, 4)

        self.assertEqual(trajectory.shape, (5, 3))
        np.testing.assert_array_equal(trajectory[0], initial_state)
        np.testing.assert_allclose(
            trajectory[1],
            qlearning.LorenzEnvEuler._euler_step(initial_state, 0.0),
        )
        self.assertTrue(np.isfinite(trajectory).all())

    def test_default_state_cost_uses_phi(self) -> None:
        self.assertAlmostEqual(
            qlearning.default_state_cost_fn(2.0), float(qlearning.phi(2.0))
        )

    def test_lobe_state_cost_only_rewards_the_negative_target(self) -> None:
        self.assertEqual(qlearning.lobe_state_cost(qlearning.TARGET_FIXED_POINT), 0.0)
        self.assertEqual(
            qlearning.lobe_state_cost(qlearning.POSITIVE_FIXED_POINT),
            qlearning.POSITIVE_LOBE_PENALTY,
        )

        negative_boundary = qlearning.TARGET_FIXED_POINT + np.array(
            [qlearning.FIXED_POINT_TOLERANCE, 0.0, 0.0]
        )
        positive_boundary = qlearning.POSITIVE_FIXED_POINT + np.array(
            [qlearning.FIXED_POINT_TOLERANCE, 0.0, 0.0]
        )
        self.assertEqual(qlearning.lobe_state_cost(negative_boundary), 0.0)
        self.assertEqual(
            qlearning.lobe_state_cost(positive_boundary),
            qlearning.POSITIVE_LOBE_PENALTY,
        )

    def test_reward_penalizes_positive_lobe_and_minimizes_forcing_at_target(self) -> None:
        settings = experiment_settings(
            evaluation_lyapunov_times=qlearning.LYAPUNOV_EXP * qlearning.DT,
            action_bins=3,
            regularized=True,
        )

        negative_env = qlearning.LorenzEnvEuler(settings)
        negative_env.reset(qlearning.TARGET_FIXED_POINT)
        _, zero_forcing_reward, _, _ = negative_env.step(1)

        forced_env = qlearning.LorenzEnvEuler(settings)
        forced_env.reset(qlearning.TARGET_FIXED_POINT)
        _, forced_reward, _, _ = forced_env.step(2)

        positive_env = qlearning.LorenzEnvEuler(settings)
        positive_env.reset(qlearning.POSITIVE_FIXED_POINT)
        _, positive_lobe_reward, _, _ = positive_env.step(1)

        self.assertEqual(zero_forcing_reward, 0.0)
        self.assertLess(forced_reward, zero_forcing_reward)
        self.assertAlmostEqual(positive_lobe_reward, -qlearning.POSITIVE_LOBE_PENALTY)

    def test_training_reports_target_fixed_point_occupancy(self) -> None:
        settings = experiment_settings(
            ic=tuple(qlearning.TARGET_FIXED_POINT),
            training_ic_perturbation=0.0,
            training_lyapunov_times=3 * qlearning.LYAPUNOV_EXP * qlearning.DT,
            action_bins=3,
        )
        env = qlearning.LorenzEnvEuler(settings, training=True)
        agent = qlearning.QLearningAgent(
            n_actions=3,
            epsilon=0.0,
            epsilon_min=0.0,
        )
        agent.q_table[..., 1] = 1.0  # Select zero control at the fixed point.

        output = io.StringIO()
        with redirect_stdout(output):
            history = qlearning.train_q_learning(
                env, agent, num_episodes=1, print_every=2
            )

        self.assertEqual(history["first_target_steps"], [1])
        self.assertEqual(history["target_steps"], [3])
        self.assertIn(
            "Episode 1 reached the target fixed point at step 1 and held it for 3/3 steps.",
            output.getvalue(),
        )

    def test_lobe_time_percentages_aggregate_all_evaluation_steps(self) -> None:
        trajectories = [
            np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[3.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]]),
        ]

        negative, positive = qlearning.lobe_time_percentages(trajectories)

        self.assertEqual(negative, 75.0)
        self.assertEqual(positive, 25.0)

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
            trace_lambda=0.0,
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

    def test_watkins_q_lambda_assigns_credit_to_recent_greedy_actions(self) -> None:
        agent = qlearning.QLearningAgent(
            n_actions=1,
            learning_rate=1.0,
            discount_factor=0.9,
            trace_lambda=0.8,
            epsilon=0.0,
            epsilon_min=0.0,
            state_bins=3,
        )
        first_state = np.array([-20.0, -20.0, 10.0])
        second_state = np.array([0.0, 0.0, 30.0])
        terminal_state = np.array([20.0, 20.0, 50.0])

        agent.update(first_state, 0, 0.0, second_state, done=False)
        agent.update(second_state, 0, 1.0, terminal_state, done=True)

        self.assertAlmostEqual(agent.q_table[agent.discretize_state(second_state) + (0,)], 1.0)
        self.assertAlmostEqual(agent.q_table[agent.discretize_state(first_state) + (0,)], 0.72)
        self.assertTrue(np.all(agent.eligibility_traces == 0.0))

    def test_unseen_states_prefer_neutral_center_action(self) -> None:
        agent = qlearning.QLearningAgent(
            n_actions=9,
            epsilon=0.0,
            epsilon_min=0.0,
        )
        state = np.array([0.0, 1.0, 1.05])

        self.assertEqual(agent.select_action(state, training=False), 4)

        state_index = agent.discretize_state(state)
        agent.q_table[state_index + (1,)] = 1.0
        agent.q_table[state_index + (4,)] = 1.0
        self.assertEqual(agent.select_action(state, training=False), 4)

    def test_watkins_q_lambda_clears_traces_after_nongreedy_action(self) -> None:
        agent = qlearning.QLearningAgent(
            n_actions=2,
            learning_rate=1.0,
            discount_factor=0.9,
            trace_lambda=0.8,
            epsilon=0.0,
            epsilon_min=0.0,
            state_bins=3,
        )
        first_state = np.array([-20.0, -20.0, 10.0])
        second_state = np.array([0.0, 0.0, 30.0])
        terminal_state = np.array([20.0, 20.0, 50.0])

        agent.update(
            first_state,
            0,
            0.0,
            second_state,
            done=False,
            next_action_is_greedy=False,
        )
        agent.update(second_state, 0, 1.0, terminal_state, done=True)

        self.assertEqual(agent.q_table[agent.discretize_state(first_state) + (0,)], 0.0)
        self.assertEqual(agent.q_table[agent.discretize_state(second_state) + (0,)], 1.0)

    def test_potential_shaping_rewards_progress_toward_negative_target(self) -> None:
        farther_state = np.array([0.0, 1.0, 1.05])
        closer_state = qlearning.TARGET_FIXED_POINT.copy()
        shaped_reward = qlearning.potential_shaped_reward(
            0.0,
            farther_state,
            closer_state,
            discount_factor=0.99,
            shaping_weight=0.05,
            done=False,
        )
        self.assertGreater(shaped_reward, 0.0)
        self.assertAlmostEqual(
            qlearning.potential_shaped_reward(
                -0.25,
                farther_state,
                closer_state,
                discount_factor=0.99,
                shaping_weight=0.0,
                done=False,
            ),
            -0.25,
        )

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
        with self.assertRaises(ValueError):
            qlearning.QLearningAgent(n_actions=2, trace_lambda=np.nan)

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

    def test_attractor_ensemble_is_balanced_distinct_and_reproducible(self) -> None:
        ensemble = qlearning.make_attractor_initial_states(
            [0.0, 1.0, 1.05],
            states_per_lobe=10,
            burn_in_steps=500,
            sample_spacing=10,
        )
        repeated = qlearning.make_attractor_initial_states(
            [0.0, 1.0, 1.05],
            states_per_lobe=10,
            burn_in_steps=500,
            sample_spacing=10,
        )

        self.assertEqual(ensemble.shape, (20, 3))
        np.testing.assert_array_equal(ensemble, repeated)
        self.assertEqual(np.count_nonzero(ensemble[:, 0] < 0.0), 10)
        self.assertEqual(np.count_nonzero(ensemble[:, 0] >= 0.0), 10)
        self.assertEqual(np.unique(ensemble, axis=0).shape[0], 20)

    def test_ensemble_sampler_uses_every_state_before_repeating(self) -> None:
        ensemble = qlearning.make_attractor_initial_states(
            [0.0, 1.0, 1.05],
            states_per_lobe=3,
            burn_in_steps=500,
            sample_spacing=10,
        )
        sampler = qlearning.make_ensemble_initial_state_sampler(ensemble, random_seed=4)
        sampled = np.asarray([sampler() for _ in range(len(ensemble))])

        self.assertEqual(np.unique(sampled, axis=0).shape[0], len(ensemble))
        self.assertEqual(np.count_nonzero(sampled[:, 0] < 0.0), 3)
        self.assertEqual(np.count_nonzero(sampled[:, 0] >= 0.0), 3)

    def test_default_evaluation_ensemble_alternates_lobes(self) -> None:
        starts = qlearning.EXPERIMENT_DEFAULTS.make_evaluation_initial_states()

        self.assertEqual(starts.shape, (40, 3))
        self.assertEqual(np.count_nonzero(starts[:, 0] < 0.0), 20)
        self.assertEqual(np.count_nonzero(starts[:, 0] >= 0.0), 20)
        self.assertTrue(np.all(starts[0::2, 0] < 0.0))
        self.assertTrue(np.all(starts[1::2, 0] >= 0.0))

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
        self.assertEqual(len(checkpoints["evaluations"]), 3)
        self.assertEqual(len(checkpoints["q_tables"]), 3)
        self.assertEqual(len(checkpoints["epsilons"]), 3)
        self.assertFalse(
            np.shares_memory(checkpoints["q_tables"][0], agent.q_table)
        )
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
            rollout_samples=1,
        )

        self.assertEqual(
            evaluations,
            [(2, 0.8 * 0.5**2), (4, 0.8 * 0.5**4), (5, 0.8 * 0.5**5)],
        )
        self.assertEqual(len(history["episode_rewards"]), 5)
        self.assertEqual(history["sampled_episodes"], [1, 2, 4, 5])

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
