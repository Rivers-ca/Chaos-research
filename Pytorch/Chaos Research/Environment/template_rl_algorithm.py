"""
Template for implementing RL algorithms with LorenzEnv.

This file shows the expected structure for DQN, SAC, PPO, etc.
Do NOT run this directly — copy and modify for your algorithm.

Key constraints:
  1. Environment interface must not change
  2. All configurable parameters (alpha, horizon, IC) go in __main__
  3. Policy class should inherit from env.Policy (or implement its interface)
"""

import numpy as np
from env import LorenzEnv, Policy


# ============================================================================
# 1. Policy class (reusable across environments)
# ============================================================================

class MyRLPolicy(Policy):
    """
    Your RL algorithm's policy.
    Must implement: select_action(state) and update(...).
    """

    def __init__(self, env: LorenzEnv):
        """
        Initialize policy.

        Args:
            env: LorenzEnv instance (used to inspect action/state spaces)
        """
        self.env = env

        # Store action space info
        if env.action_type == "discrete":
            self.n_actions = env.n_action_bins
        else:
            self.action_low = env.action_low
            self.action_high = env.action_high

        # TODO: Initialize your algorithm here (networks, buffers, etc.)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Select action given state.

        Args:
            state: [x, y, z]

        Returns:
            int (for discrete) or float (for continuous)
        """
        # TODO: Implement your algorithm's action selection
        if self.env.action_type == "discrete":
            return np.random.randint(0, self.n_actions)
        else:
            return np.random.uniform(self.action_low, self.action_high)

    def update(self, batch):
        """
        Update policy from experience.

        Args:
            batch: Your algorithm-specific batch (transitions, trajectories, etc.)
        """
        # TODO: Implement your algorithm's learning update
        pass


# ============================================================================
# 2. Training loop structure
# ============================================================================

def train_episode(policy: Policy, env: LorenzEnv) -> dict:
    """
    Run one training episode.

    Returns:
        Episode summary dict with keys like "reward", "steps", "terminated"
    """
    state = env.reset()
    episode_reward = 0.0
    episode_length = 0
    terminated = False

    while True:
        # Select and execute action
        action = policy.select_action(state)
        next_state, reward, done, info = env.step(action)

        episode_reward += reward
        episode_length += 1

        # TODO: Store transition in replay buffer / trajectory
        # experience = (state, action, reward, next_state, done)
        # memory.push(experience)

        if done:
            terminated = info.get("diverged", False)
            break

        state = next_state

    return {
        "reward": episode_reward,
        "length": episode_length,
        "terminated": terminated,
    }


def train_sweep(
    config: dict,
    n_seeds: int = 3,
) -> dict:
    """
    Train across a sweep of configurations.

    This is the structure you'll use for final results.

    Args:
        config: {
            "alphas": [0.01, 0.1, 1.0],
            "horizons": [100, 200],
            "ic_dist": None,  # or Callable
            "n_episodes": 100,
            "action_type": "continuous",
        }
        n_seeds: Number of random seeds to average over

    Returns:
        results dict with shape {alpha: {horizon: [seed results...]}}
    """
    results = {}

    for alpha in config["alphas"]:
        results[alpha] = {}

        for horizon in config["horizons"]:
            seed_results = []

            for seed in range(n_seeds):
                np.random.seed(seed)

                # Create environment with this config
                env = LorenzEnv(
                    alpha=alpha,
                    horizon=horizon,
                    action_type=config["action_type"],
                    ic_dist=config.get("ic_dist"),
                )

                # Create policy
                policy = MyRLPolicy(env)

                # Train for N episodes
                episode_rewards = []
                for episode in range(config["n_episodes"]):
                    summary = train_episode(policy, env)
                    episode_rewards.append(summary["reward"])

                    # TODO: policy.update(...)

                    if (episode + 1) % 10 == 0:
                        print(
                            f"  Seed {seed}, α={alpha}, h={horizon}, "
                            f"ep {episode+1}: reward={np.mean(episode_rewards[-10:]):.4f}"
                        )

                seed_results.append(episode_rewards)

            results[alpha][horizon] = seed_results

    return results


# ============================================================================
# 3. Example usage
# ============================================================================

def example_basic_training():
    """
    Minimal example: train a policy on one configuration.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Basic training loop")
    print("=" * 70 + "\n")

    # Create environment
    env = LorenzEnv(
        alpha=0.1,
        horizon=200,
        action_type="continuous",
        action_bounds=(-1.0, 1.0),
    )

    # Create policy
    policy = MyRLPolicy(env)

    # Train for a few episodes
    for episode in range(3):
        state = env.reset()
        episode_reward = 0.0

        for step in range(env.horizon):
            action = policy.select_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}: reward={episode_reward:.4f}")


def example_sweep():
    """
    Example: train across a configuration sweep.
    This demonstrates the interface for producing paper results.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Sweep training (alpha × horizon)")
    print("=" * 70 + "\n")

    config = {
        "alphas": [0.01, 0.1, 1.0],
        "horizons": [100, 200],
        "ic_dist": None,  # Use default
        "n_episodes": 5,  # Use 5 for demo; use 100+ for real results
        "action_type": "continuous",
    }

    results = train_sweep(config, n_seeds=2)

    # Summarize results
    print("\nSummary (mean reward over final 5 episodes):")
    print(f"\n{'α \\ h':<8}" + "".join(f"{h:>12}" for h in config["horizons"]))
    print("-" * (8 + 12 * len(config["horizons"])))

    for alpha in config["alphas"]:
        row = []
        for horizon in config["horizons"]:
            seed_results = results[alpha][horizon]
            # Average across seeds, take final episode per seed
            final_rewards = [r[-1] for r in seed_results]
            mean_final = np.mean(final_rewards)
            row.append(mean_final)

        print(f"{alpha:<8.2f}" + "".join(f"{r:>12.4f}" for r in row))


if __name__ == "__main__":
    # Uncomment to test structure:
    # example_basic_training()
    # example_sweep()

    print("\nThis is a TEMPLATE. Copy and modify for your RL algorithm.")
    print("\nKey steps:")
    print("  1. Replace MyRLPolicy with your algorithm (e.g., DQN, SAC)")
    print("  2. Implement select_action() and update()")
    print("  3. Configure config dict with your sweep parameters")
    print("  4. Run train_sweep() to generate results")
    print("\nThe environment interface (reset, step) does not change.")
