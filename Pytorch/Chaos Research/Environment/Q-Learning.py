import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env import LorenzEnv
import os
import json
from datetime import datetime


CONTROL_GOAL = np.array([-8.49, -8.49, 27.0])
POSITIVE_EQUILIBRIUM = np.array([8.49, 8.49, 27.0])

CONFIG = {
    "control_objective": "Minimize velocity (damping)",
    "objective_type": "damping",
    "n_episodes": 1000,
    "train_steps_per_episode": 1000,
    "eval_steps_per_episode": 10000,

    "learning_rate": 0.1,
    "gamma": 0.5,
    "epsilon_start": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay_per_episode": 0.9975,

    "reward_velocity_weight": 5.0,
    "reward_control_weight": 0.01,
    "reward_goal_weight": 5.0,

    "alpha": 1.0,
    "horizon": 10000,

    "n_x_bins": 25,
    "n_y_bins": 25,
    "n_z_bins": 25,
    "x_range": (-25.0, 25.0),
    "y_range": (-35.0, 35.0),
    "z_range": (0.0, 55.0),

    "n_action_bins": 11,
    "action_bounds": (-1.0, 1.0),

    "eval_episodes": 10,
    "random_seed": 42,
}


def discretize_state(state, config):
    """
    Map continuous Lorenz state [x, y, z] to discrete bin indices.

    Uses linear binning within specified ranges. States outside ranges
    are clipped to nearest bin.

    Args:
        state: np.array [x, y, z]
        config: Configuration dict with ranges and bin counts

    Returns:
        Tuple (x_bin, y_bin, z_bin) as integers
    """
    x, y, z = state

    x_bin = np.clip(
        int(np.floor((x - config["x_range"][0]) /
                    (config["x_range"][1] - config["x_range"][0]) *
                    config["n_x_bins"])),
        0, config["n_x_bins"] - 1
    )

    y_bin = np.clip(
        int(np.floor((y - config["y_range"][0]) /
                    (config["y_range"][1] - config["y_range"][0]) *
                    config["n_y_bins"])),
        0, config["n_y_bins"] - 1
    )

    z_bin = np.clip(
        int(np.floor((z - config["z_range"][0]) /
                    (config["z_range"][1] - config["z_range"][0]) *
                    config["n_z_bins"])),
        0, config["n_z_bins"] - 1
    )

    return (x_bin, y_bin, z_bin)


def compute_reward(state, action_value, config):
    """
    Custom reward function supporting multiple control objectives.

    Objectives:
    - 'damping': Minimize velocity (||v|| small)
    - 'negative_eq': Stabilize at negative equilibrium (-8.49, -8.49, 27)
    - 'positive_eq': Stabilize at positive equilibrium (8.49, 8.49, 27)

    Args:
        state: [x, y, z] continuous state
        action_value: The actual control input u applied
        config: Configuration dict with reward weights and objective_type

    Returns:
        Scalar reward
    """
    control_penalty = -config["reward_control_weight"] * (action_value ** 2)

    objective = config.get("objective_type", "damping")

    if objective == "damping":
        velocity_magnitude = np.linalg.norm(state)
        velocity_reward = -config["reward_velocity_weight"] * velocity_magnitude
        return velocity_reward + control_penalty

    elif objective == "negative_eq":
        distance_to_goal = np.linalg.norm(state - CONTROL_GOAL)
        goal_reward = -config["reward_goal_weight"] * distance_to_goal
        return goal_reward + control_penalty

    elif objective == "positive_eq":
        distance_to_goal = np.linalg.norm(state - POSITIVE_EQUILIBRIUM)
        goal_reward = -config["reward_goal_weight"] * distance_to_goal
        return goal_reward + control_penalty

    else:
        raise ValueError(f"Unknown objective_type: {objective}")



class QLearningAgent:
    def __init__(self, config):
        self.config = config

        self.Q = np.zeros((
            config["n_x_bins"],
            config["n_y_bins"],
            config["n_z_bins"],
            config["n_action_bins"]
        ))

        self.epsilon = config["epsilon_start"]
        self.steps_taken = 0
        self.episodes_trained = 0

    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.

        Args:
            state: [x, y, z]
            training: If True, use epsilon-greedy; else greedy.

        Returns:
            Action index (0 to n_action_bins-1)
        """
        s_bin = discretize_state(state, self.config)

        if training and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.config["n_action_bins"])
        else:
            return np.argmax(self.Q[s_bin])

    def update(self, state, action, reward, next_state, done):
        """
        Standard Q-learning update:
        Q(s,a) ← Q(s,a) + lr * (r + γ * max_a' Q(s',a') - Q(s,a))

        When done, future term is 0.
        """
        s_bin = discretize_state(state, self.config)
        s_next_bin = discretize_state(next_state, self.config)

        if done:
            future_value = 0.0
        else:
            future_value = np.max(self.Q[s_next_bin])

        current_q = self.Q[s_bin + (action,)]
        td_target = reward + self.config["gamma"] * future_value
        td_error = td_target - current_q

        self.Q[s_bin + (action,)] += self.config["learning_rate"] * td_error

        self.steps_taken += 1

    def decay_epsilon(self):
        """Decay epsilon at the end of an episode."""
        self.epsilon = max(
            self.config["epsilon_min"],
            self.epsilon * self.config["epsilon_decay_per_episode"]
        )
        self.episodes_trained += 1


def train(agent, config):
    """
    Train Q-learning agent on LorenzEnv with custom reward.

    Returns:
        metrics: Dict with per-episode data
    """
    env = LorenzEnv(
        alpha=config["alpha"],
        horizon=config["horizon"],
        action_type="discrete",
        action_bounds=config["action_bounds"],
        n_action_bins=config["n_action_bins"],
    )

    metrics = {
        "episode_reward": [],
        "episode_length": [],
        "epsilon": [],
        "mean_control": [],
        "termination_reason": [],
    }

    for episode in range(config["n_episodes"]):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        controls = []
        term_reason = "timeout"

        for _ in range(config["train_steps_per_episode"]):
            action = agent.select_action(state, training=True)
            u = env.actions[action]
            controls.append(abs(u))

            next_state, _, done, info = env.step(action)

            reward = compute_reward(state, u, config)
            agent.update(state, action, reward, next_state, done)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                if info.get("diverged"):
                    term_reason = "diverged"
                elif info.get("horizon_reached"):
                    term_reason = "horizon"
                break

        agent.decay_epsilon()

        metrics["episode_reward"].append(episode_reward)
        metrics["episode_length"].append(episode_length)
        metrics["epsilon"].append(agent.epsilon)
        metrics["mean_control"].append(np.mean(controls) if controls else 0.0)
        metrics["termination_reason"].append(term_reason)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(metrics["episode_reward"][-100:])
            print(f"Episode {episode + 1}/{config['n_episodes']}, "
                  f"Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    env.close()
    return metrics


def evaluate_policy(agent, config, n_episodes, seed_states=None):
    """
    Run learned greedy policy with custom reward and fixed eval length.

    Args:
        agent: Trained QLearningAgent
        config: Configuration dict
        n_episodes: Number of evaluation episodes
        seed_states: Optional list of fixed initial states to test

    Returns:
        trajectories: List of (trajectory, total_reward, controls, mean_reward_per_step)
    """
    env = LorenzEnv(
        alpha=config["alpha"],
        horizon=config["horizon"],
        action_type="discrete",
        action_bounds=config["action_bounds"],
        n_action_bins=config["n_action_bins"],
    )

    trajectories = []

    if seed_states is None:
        seed_states = [None] * n_episodes

    for x0 in seed_states[:n_episodes]:
        state = env.reset(x0=x0)
        traj = [state.copy()]
        controls = []
        total_reward = 0.0

        for _ in range(config["eval_steps_per_episode"]):
            action = agent.select_action(state, training=False)
            u = env.actions[action]
            controls.append(u)

            next_state, _, done, info = env.step(action)
            reward = compute_reward(state, u, config)
            total_reward += reward
            traj.append(next_state.copy())
            state = next_state

            if done:
                break

        traj = np.array(traj)
        mean_reward = total_reward / len(traj) if len(traj) > 0 else 0.0
        trajectories.append((traj, total_reward, np.array(controls), mean_reward))

    env.close()
    return trajectories


def uncontrolled_rollout(config, n_episodes, seed_states=None):
    """
    Run with zero control (u=0) for comparison using custom reward.
    """
    env = LorenzEnv(
        alpha=config["alpha"],
        horizon=config["horizon"],
        action_type="discrete",
        action_bounds=config["action_bounds"],
        n_action_bins=config["n_action_bins"],
    )

    trajectories = []

    if seed_states is None:
        seed_states = [None] * n_episodes

    for x0 in seed_states[:n_episodes]:
        state = env.reset(x0=x0)
        traj = [state.copy()]
        total_reward = 0.0

        for _ in range(config["eval_steps_per_episode"]):
            action = config["n_action_bins"] // 2
            u = env.actions[action]
            next_state, _, done, _ = env.step(action)
            reward = compute_reward(state, u, config)
            total_reward += reward
            traj.append(next_state.copy())
            state = next_state

            if done:
                break

        traj = np.array(traj)
        mean_reward = total_reward / len(traj) if len(traj) > 0 else 0.0
        trajectories.append((traj, total_reward, None, mean_reward))

    env.close()
    return trajectories


def plot_training_curves(metrics, output_dir="results"):
    """Plot training reward, epsilon, and control input vs episode."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(metrics["episode_reward"], alpha=0.6, label="Total Reward")
    window = 50
    if len(metrics["episode_reward"]) >= window:
        avg = np.convolve(metrics["episode_reward"],
                          np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(metrics["episode_reward"])), avg,
                label=f"{window}-ep avg", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Total Reward per Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(metrics["epsilon"], linewidth=2, color='orange')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate Decay")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(metrics["episode_length"], alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(metrics["mean_control"], alpha=0.6, color='green')
    ax.set_xlabel("Episode")
    ax.set_ylabel("|u|")
    ax.set_title("Mean Absolute Control Input")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=150)
    print(f"Saved training curves to {output_dir}/training_curves.png")


def plot_trajectory_comparison(learned_trajs, uncontrolled_trajs,
                              output_dir="results"):
    """
    Compare learned control vs uncontrolled trajectories.

    Plots x, y, z components and 3D phase space.
    """
    os.makedirs(output_dir, exist_ok=True)

    learned_traj = learned_trajs[0][0]
    uncontrolled_traj = uncontrolled_trajs[0][0]

    t_learned = np.arange(len(learned_traj))
    t_uncontrolled = np.arange(len(uncontrolled_traj))

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    for i, var in enumerate(['x', 'y', 'z']):
        ax = axes[i]
        ax.plot(t_learned, learned_traj[:, i], label="Learned", linewidth=2)
        ax.plot(t_uncontrolled, uncontrolled_traj[:, i],
                label="Uncontrolled", linewidth=2, linestyle="--", alpha=0.7)
        ax.axhline(y=CONTROL_GOAL[i], color='red', linestyle=':', alpha=0.5, label='Goal')
        ax.set_ylabel(var)
        ax.set_title(f"Lorenz {var} Component")
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trajectory_components.png", dpi=150)
    print(f"Saved trajectory comparison to {output_dir}/trajectory_components.png")

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(learned_traj[:, 0], learned_traj[:, 1], learned_traj[:, 2],
            label="Learned", linewidth=1.5)
    ax1.scatter(learned_traj[0, 0], learned_traj[0, 1], learned_traj[0, 2],
               color='green', s=100, marker='o', label='Start')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.set_title("Learned Policy (3D Phase Space)")
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(uncontrolled_traj[:, 0], uncontrolled_traj[:, 1],
            uncontrolled_traj[:, 2], label="Uncontrolled", linewidth=1.5)
    ax2.scatter(uncontrolled_traj[0, 0], uncontrolled_traj[0, 1],
               uncontrolled_traj[0, 2], color='green', s=100, marker='o',
               label='Start')
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("Uncontrolled (3D Phase Space)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/phase_space_3d.png", dpi=150)
    print(f"Saved 3D phase space to {output_dir}/phase_space_3d.png")


def plot_control_input(learned_trajs, output_dir="results"):
    """Plot learned control input over time."""
    os.makedirs(output_dir, exist_ok=True)

    _, ax = plt.subplots(figsize=(10, 5))

    for i, (_, _, controls, _) in enumerate(learned_trajs[:3]):
        if controls is not None:
            t = np.arange(len(controls))
            ax.plot(t, controls, label=f"Eval {i+1}", alpha=0.7, linewidth=1.5)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Control Input u(t)")
    ax.set_title("Learned Control Inputs (First 3 Eval Episodes)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/control_input.png", dpi=150)
    print(f"Saved control input plot to {output_dir}/control_input.png")


def plot_reward_comparison(learned_trajs, uncontrolled_trajs, output_dir="results"):
    """Plot evaluation reward comparison: learned vs uncontrolled."""
    os.makedirs(output_dir, exist_ok=True)

    learned_rewards = [r for _, r, _, _ in learned_trajs]
    learned_mean_rewards = [mr for _, _, _, mr in learned_trajs]
    uncontrolled_rewards = [r for _, r, _, _ in uncontrolled_trajs]
    uncontrolled_mean_rewards = [mr for _, _, _, mr in uncontrolled_trajs]

    x_pos = np.arange(2)
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x_pos[0], np.mean(learned_rewards), width, label='Learned',
            color='steelblue', yerr=np.std(learned_rewards), capsize=10)
    ax1.bar(x_pos[1], np.mean(uncontrolled_rewards), width, label='Uncontrolled',
            color='orange', yerr=np.std(uncontrolled_rewards), capsize=10)
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Total Reward Comparison (Evaluation)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Learned', 'Uncontrolled'])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    ax2.bar(x_pos[0], np.mean(learned_mean_rewards), width, label='Learned',
            color='steelblue', yerr=np.std(learned_mean_rewards), capsize=10)
    ax2.bar(x_pos[1], np.mean(uncontrolled_mean_rewards), width, label='Uncontrolled',
            color='orange', yerr=np.std(uncontrolled_mean_rewards), capsize=10)
    ax2.set_ylabel("Mean Reward per Time Step")
    ax2.set_title("Mean Reward per Time Step (Evaluation)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['Learned', 'Uncontrolled'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_comparison.png", dpi=150)
    print(f"Saved reward comparison to {output_dir}/reward_comparison.png")


def generate_seed_states(n_states, config, seed=42):
    """Generate fixed initial states for consistent evaluation."""
    np.random.seed(seed)
    return [np.array([0.0, 1.0, 1.05]) + np.random.randn(3) * 0.1
            for _ in range(n_states)]


def main():
    print("=" * 80)
    print("Q-LEARNING FOR LORENZ SYSTEM CONTROL")
    print("=" * 80)
    print(f"\nControl Objective: {CONFIG['control_objective']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"qlearning_results_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(CONFIG["random_seed"])

    seed_states = generate_seed_states(CONFIG["eval_episodes"], CONFIG,
                                       seed=CONFIG["random_seed"])

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print("\nInitializing Q-Learning Agent...")
    agent = QLearningAgent(CONFIG)
    print(f"Q-table shape: {agent.Q.shape}")
    print(f"Total Q-table entries: {np.prod(agent.Q.shape):,}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Gamma (discount): {CONFIG['gamma']}")
    print(f"Epsilon schedule: {CONFIG['epsilon_start']} → {CONFIG['epsilon_min']}")

    print(f"\nTraining for {CONFIG['n_episodes']} episodes ({CONFIG['train_steps_per_episode']} steps/episode)...")
    metrics = train(agent, CONFIG)
    print(f"Training complete. Final epsilon: {agent.epsilon:.4f}")

    print(f"\nEvaluating learned policy ({CONFIG['eval_steps_per_episode']} steps/eval)...")
    learned_trajs = evaluate_policy(agent, CONFIG, CONFIG["eval_episodes"],
                                    seed_states=seed_states)
    learned_rewards = [r for _, r, _, _ in learned_trajs]
    learned_mean_rewards = [mr for _, _, _, mr in learned_trajs]

    print(f"Learned policy:")
    print(f"  Total reward:     {np.mean(learned_rewards):8.2f} ± {np.std(learned_rewards):6.2f}")
    print(f"  Mean reward/step: {np.mean(learned_mean_rewards):8.4f} ± {np.std(learned_mean_rewards):6.4f}")

    print(f"\nEvaluating uncontrolled policy (zero control)...")
    uncontrolled_trajs = uncontrolled_rollout(CONFIG, CONFIG["eval_episodes"],
                                             seed_states=seed_states)
    uncontrolled_rewards = [r for _, r, _, _ in uncontrolled_trajs]
    uncontrolled_mean_rewards = [mr for _, _, _, mr in uncontrolled_trajs]

    print(f"Uncontrolled policy:")
    print(f"  Total reward:     {np.mean(uncontrolled_rewards):8.2f} ± {np.std(uncontrolled_rewards):6.2f}")
    print(f"  Mean reward/step: {np.mean(uncontrolled_mean_rewards):8.4f} ± {np.std(uncontrolled_mean_rewards):6.4f}")

    improvement_total = np.mean(learned_rewards) - np.mean(uncontrolled_rewards)
    improvement_mean = np.mean(learned_mean_rewards) - np.mean(uncontrolled_mean_rewards)
    beats_baseline = improvement_mean > 0

    print(f"\nImprovement (learned vs uncontrolled):")
    print(f"  Total reward:     {improvement_total:8.2f}")
    print(f"  Mean reward/step: {improvement_mean:8.4f}")
    print(f"  Beats baseline:   {'YES ✓' if beats_baseline else 'NO ✗'}")

    print("\nSaving outputs...")
    np.savez(
        f"{output_dir}/qtable.npz",
        Q=agent.Q,
    )
    print(f"  ✓ Q-table saved")

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"  ✓ Config saved")

    np.savez(
        f"{output_dir}/metrics.npz",
        episode_reward=np.array(metrics["episode_reward"]),
        episode_length=np.array(metrics["episode_length"]),
        epsilon=np.array(metrics["epsilon"]),
        mean_control=np.array(metrics["mean_control"]),
        learned_rewards=np.array(learned_rewards),
        learned_mean_rewards=np.array(learned_mean_rewards),
        uncontrolled_rewards=np.array(uncontrolled_rewards),
        uncontrolled_mean_rewards=np.array(uncontrolled_mean_rewards),
    )
    print(f"  ✓ Metrics saved")

    print("\nGenerating plots...")
    print("  Generating training curves...")
    plot_training_curves(metrics, output_dir)
    print("  Generating trajectory comparison...")
    plot_trajectory_comparison(learned_trajs, uncontrolled_trajs, output_dir)
    print("  Generating control input plot...")
    plot_control_input(learned_trajs, output_dir)
    print("  Generating reward comparison...")
    plot_reward_comparison(learned_trajs, uncontrolled_trajs, output_dir)

    summary = f"""Q-LEARNING CONTROL OF LORENZ SYSTEM
{'='*60}

CONTROL OBJECTIVE:
  {CONFIG['control_objective']}

TRAINING CONFIGURATION:
  Episodes:         {CONFIG['n_episodes']}
  Train steps/ep:   {CONFIG['train_steps_per_episode']}
  Eval steps/ep:    {CONFIG['eval_steps_per_episode']}
  Learning rate:    {CONFIG['learning_rate']}
  Discount (γ):     {CONFIG['gamma']}
  Epsilon schedule: {CONFIG['epsilon_start']} → {CONFIG['epsilon_min']} (decay: {CONFIG['epsilon_decay_per_episode']})

DISCRETIZATION:
  State space: {CONFIG['n_x_bins']}×{CONFIG['n_y_bins']}×{CONFIG['n_z_bins']} = {np.prod([CONFIG['n_x_bins'], CONFIG['n_y_bins'], CONFIG['n_z_bins']])} bins
  Action space: {CONFIG['n_action_bins']} discrete actions

CONTROL OBJECTIVE:
  Type:           {CONFIG['objective_type']}
  Description:    {CONFIG['control_objective']}

REWARD FUNCTION WEIGHTS:
  Velocity weight: {CONFIG.get('reward_velocity_weight', 'N/A')} (for damping)
  Goal weight:     {CONFIG.get('reward_goal_weight', 'N/A')} (for equilibrium)
  Control weight:  {CONFIG['reward_control_weight']} (control energy penalty - REDUCED)

RESULTS (Mean ± Std over {CONFIG['eval_episodes']} eval episodes):

Learned Policy (with control):
  Total Reward:      {np.mean(learned_rewards):8.2f} ± {np.std(learned_rewards):6.2f}
  Mean Reward/Step:  {np.mean(learned_mean_rewards):8.4f} ± {np.std(learned_mean_rewards):6.4f}

Uncontrolled Policy (u=0):
  Total Reward:      {np.mean(uncontrolled_rewards):8.2f} ± {np.std(uncontrolled_rewards):6.2f}
  Mean Reward/Step:  {np.mean(uncontrolled_mean_rewards):8.4f} ± {np.std(uncontrolled_mean_rewards):6.4f}

COMPARISON:
  Improvement (Total):           {improvement_total:8.2f}
  Improvement (Mean/Step):       {improvement_mean:8.4f}
  Beats Baseline (Mean/Step):    {'YES ✓' if beats_baseline else 'NO ✗'}

OUTPUT FILES:
  - qlearning_results_{timestamp}/
    ├── training_curves.png         (training metrics over episodes)
    ├── trajectory_components.png   (learned vs uncontrolled states)
    ├── phase_space_3d.png          (3D Lorenz attractor comparison)
    ├── control_input.png           (control signals used)
    ├── reward_comparison.png       (evaluation reward comparison)
    ├── qtable.npz                  (learned Q-table)
    ├── metrics.npz                 (all numerical results)
    ├── config.json                 (experiment configuration)
    └── summary.txt                 (this file)

INTERPRETATION:
  The mean reward per time step is the key metric because it accounts for
  different trajectory lengths. A positive improvement indicates the learned
  controller successfully reduces the distance to the control goal while
  managing control energy.
"""

    with open(f"{output_dir}/summary.txt", "w") as f:
        f.write(summary)

    print(summary)
    print(f"\n{'='*80}")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}")

    print("\n" + "="*80)
    print("DISPLAYING PLOTS")
    print("="*80)
    print("\nShowing 5 plots (close each window to continue)...")
    print("1. Training curves (reward, epsilon, episode length, control)")
    print("2. Trajectory components (x, y, z over time)")
    print("3. 3D phase space (learned vs uncontrolled)")
    print("4. Control input signals")
    print("5. Reward comparison (learned vs baseline)")

    try:
        import matplotlib.pyplot as plt_display
        plt_display.show()
    except Exception as e:
        print(f"\nNote: Could not display plots interactively: {e}")
        print(f"Plots are saved to: {output_dir}")

    print(f"\n✓ Experiment complete!")
    print(f"✓ All plots saved and displayed")
    print(f"✓ Results directory: {output_dir}")

    plt.show()


if __name__ == "__main__":
    main()
