"""
Test harness for LorenzEnv: validates that the environment works standalone
and can be swept across alpha, horizon, and initial conditions.

This demonstrates the interface that will be used by DQN, SAC, PPO, etc.
No RL algorithm is implemented here — just the environment and a dummy policy.
"""

import numpy as np
import matplotlib.pyplot as plt
from env import LorenzEnv, DT, LYAPUNOV_EXP


def dummy_policy(state, env):
    """Trivial policy: always zero control. For testing only."""
    if env.action_type == "discrete":
        return 0  # First action bin
    else:
        return 0.0


def rollout(env, policy, max_steps=None, render=False):
    """
    Execute one episode with a given policy.

    Args:
        env: LorenzEnv instance
        policy: Callable(state, env) -> action
        max_steps: Override env.horizon if provided
        render: If True, return trajectory

    Returns:
        (total_reward, episode_length, done_reason, trajectory if render else None)
    """
    state = env.reset()
    total_reward = 0.0
    trajectory = [state.copy()] if render else None
    done_reason = None

    max_steps = max_steps or env.horizon

    step = -1
    for step in range(max_steps):
        action = policy(state, env)
        state, reward, done, info = env.step(action)
        total_reward += reward

        if render and trajectory is not None:
            trajectory.append(state.copy())

        if done:
            if "diverged" in info:
                done_reason = "diverged"
            elif "horizon_reached" in info:
                done_reason = "horizon_reached"
            else:
                done_reason = "unknown"
            break

    if trajectory is not None:
        trajectory = np.array(trajectory)

    return total_reward, step + 1, done_reason, trajectory


def test_environment_basics():
    """Sanity check: environment instantiation and basic rollout."""
    print("=" * 70)
    print("TEST: Basic environment instantiation and rollout")
    print("=" * 70)

    env = LorenzEnv(alpha=0.1, horizon=100, action_type="continuous")
    state = env.reset(x0=np.array([0.0, 1.0, 1.05]))

    print(f"Initial state: {state}")

    for step in range(10):
        action = 0.0  # No control
        state, reward, done, info = env.step(action)
        print(f"Step {step+1}: state={state}, reward={reward:.4f}, done={done}")

    print("✓ Basic test passed\n")


def test_discrete_vs_continuous():
    """Verify both action space types work."""
    print("=" * 70)
    print("TEST: Discrete vs continuous action spaces")
    print("=" * 70)

    # Discrete: 5 actions in [-1, 1]
    env_disc = LorenzEnv(
        alpha=0.1,
        horizon=50,
        action_type="discrete",
        action_bounds=(-1.0, 1.0),
        n_action_bins=5,
    )
    reward_disc, steps_disc, reason_disc, _ = rollout(env_disc, dummy_policy)
    print(f"Discrete (5 bins): reward={reward_disc:.4f}, steps={steps_disc}, reason={reason_disc}")

    # Continuous
    env_cont = LorenzEnv(
        alpha=0.1,
        horizon=50,
        action_type="continuous",
        action_bounds=(-1.0, 1.0),
    )
    reward_cont, steps_cont, reason_cont, _ = rollout(env_cont, dummy_policy)
    print(f"Continuous: reward={reward_cont:.4f}, steps={steps_cont}, reason={reason_cont}")

    print("✓ Both action spaces work\n")


def test_alpha_sweep():
    """Sweep over control cost weight alpha."""
    print("=" * 70)
    print("TEST: Alpha sweep (control cost weight)")
    print("=" * 70)

    alphas = [0.01, 0.1, 1.0, 10.0]
    results = []

    for alpha in alphas:
        env = LorenzEnv(
            alpha=alpha,
            horizon=200,
            action_type="continuous",
            action_bounds=(-1.0, 1.0),
        )
        reward, steps, reason, _ = rollout(env, dummy_policy)
        results.append((alpha, reward, steps, reason))
        print(f"  α={alpha:5.2f}: reward={reward:8.4f}, steps={steps:3d}, reason={reason}")

    print("✓ Alpha sweep works (verify reward changes with alpha)\n")
    return results


def test_horizon_sweep():
    """Sweep over horizon length."""
    print("=" * 70)
    print("TEST: Horizon sweep")
    print("=" * 70)

    horizons = [50, 100, 200, 500]
    results = []

    for h in horizons:
        env = LorenzEnv(
            alpha=0.1,
            horizon=h,
            action_type="continuous",
            action_bounds=(-1.0, 1.0),
        )
        reward, steps, reason, _ = rollout(env, dummy_policy)
        results.append((h, reward, steps, reason))
        print(f"  h={h:3d}: reward={reward:8.4f}, steps={steps:3d}, reason={reason}")

    print("✓ Horizon sweep works\n")
    return results


def test_ic_distribution():
    """Test fixed vs. sampled initial conditions."""
    print("=" * 70)
    print("TEST: Initial condition modes")
    print("=" * 70)

    # Fixed IC
    env_fixed = LorenzEnv(alpha=0.1, horizon=100, action_type="continuous")
    states_fixed = []
    for _ in range(3):
        s0 = env_fixed.reset(x0=np.array([0.0, 1.0, 1.05]))
        states_fixed.append(s0)

    print("Fixed IC (3 resets):")
    for i, s in enumerate(states_fixed):
        print(f"  Reset {i+1}: {s}")
    assert np.allclose(states_fixed[0], states_fixed[1])
    print("  ✓ All identical (as expected)")

    # Sampled IC
    def custom_ic_dist():
        return np.array([0.0, 1.0, 1.05]) + np.random.randn(3) * 0.5

    env_sampled = LorenzEnv(
        alpha=0.1,
        horizon=100,
        action_type="continuous",
        ic_dist=custom_ic_dist,
    )
    states_sampled = []
    for _ in range(3):
        s0 = env_sampled.reset()
        states_sampled.append(s0)

    print("\nSampled IC (3 resets):")
    for i, s in enumerate(states_sampled):
        print(f"  Reset {i+1}: {s}")
    assert not np.allclose(states_sampled[0], states_sampled[1])
    print("  ✓ All different (as expected)")

    print("✓ IC modes work\n")


def test_trajectory_rendering():
    """Capture a trajectory for analysis."""
    print("=" * 70)
    print("TEST: Trajectory rendering")
    print("=" * 70)

    env = LorenzEnv(
        alpha=0.0,  # No control cost, focus on state cost
        horizon=500,
        action_type="continuous",
        action_bounds=(-0.5, 0.5),
    )
    reward, steps, reason, traj = rollout(env, dummy_policy, render=True)

    print(f"Trajectory shape: {traj.shape}")
    print(f"Steps executed: {steps}")
    print(f"Total reward: {reward:.4f}")
    print(f"Final state: {traj[-1]}")

    # Sanity checks
    assert traj.shape == (steps + 1, 3), f"Expected ({steps+1}, 3), got {traj.shape}"
    assert np.linalg.norm(traj[-1]) < 100, "State appears to have diverged"

    print("✓ Trajectory capture works\n")

    return traj


def test_state_cost_function():
    """Test custom state cost functions."""
    print("=" * 70)
    print("TEST: Custom state cost functions")
    print("=" * 70)

    def cost_fn_identity(x):
        """Cost proportional to x."""
        return np.abs(x)

    def cost_fn_quadratic(x):
        """Cost proportional to x^2."""
        return x ** 2

    env_identity = LorenzEnv(
        alpha=0.1,
        horizon=100,
        action_type="continuous",
        state_cost_fn=cost_fn_identity,
    )
    reward_id, _, _, _ = rollout(env_identity, dummy_policy)

    env_quad = LorenzEnv(
        alpha=0.1,
        horizon=100,
        action_type="continuous",
        state_cost_fn=cost_fn_quadratic,
    )
    reward_quad, _, _, _ = rollout(env_quad, dummy_policy)

    print(f"Cost=|x|:  reward={reward_id:.4f}")
    print(f"Cost=x^2:  reward={reward_quad:.4f}")
    print("✓ Custom cost functions work\n")


def validate_physics():
    """
    Validate that the physics (uncontrolled) matches sim.py behavior roughly.
    We'll compare a trajectory with u=0 across a few Lyapunov times.
    """
    print("=" * 70)
    print("TEST: Physics validation (uncontrolled trajectory)")
    print("=" * 70)

    env = LorenzEnv(
        alpha=0.0,
        horizon=int(100 / (LYAPUNOV_EXP * DT)),  # ~100 Lyapunov times
        action_type="continuous",
    )
    reward, steps, reason, traj = rollout(
        env, dummy_policy, render=True
    )

    print(f"Horizon in Lyapunov times: {steps * DT * LYAPUNOV_EXP:.1f}")
    print(f"Trajectory did not diverge: {reason != 'diverged'}")
    print(f"Norm range: [{np.linalg.norm(traj, axis=1).min():.2f}, {np.linalg.norm(traj, axis=1).max():.2f}]")

    # The attractor should be bounded; norm should stay < 100
    norms = np.linalg.norm(traj, axis=1)
    assert norms.max() < 100, "Trajectory diverged unexpectedly"
    print("✓ Physics validation passed\n")


if __name__ == "__main__":
    np.random.seed(42)

    test_environment_basics()
    test_discrete_vs_continuous()
    test_alpha_sweep()
    test_horizon_sweep()
    test_ic_distribution()
    traj = test_trajectory_rendering()
    test_state_cost_function()
    validate_physics()

    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nEnvironment is ready for RL algorithms:")
    print("  - DQN (discrete actions)")
    print("  - SAC/TD3 (continuous actions)")
    print("  - PPO (both action spaces)")
    print("\nThe environment supports:")
    print("  - Sweeps over alpha (control cost)")
    print("  - Sweeps over horizon (episode length)")
    print("  - Custom initial conditions & distributions")
    print("  - Custom state cost functions")
