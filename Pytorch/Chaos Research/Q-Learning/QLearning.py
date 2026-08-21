"""Controlled Lorenz environment and a tabular Q-learning workflow."""

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, TypedDict, cast

class TrainingHistory(TypedDict):
    episode_rewards: List[float]
    episode_lengths: List[int]
    epsilons: List[float]
    diverged: List[bool]
    rolling_mean_rewards: List[float]

import numpy as np


"""Patrick's parameters for the Lorenz system."""
RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
DT = 0.01
LYAPUNOV_EXP = 0.9056
EPS = 2.0
U_REF = 60.0
LAMBDA = 0.007

INITIAL_STATE = np.array([0.0, 1.0, 1.05], dtype=np.float64)

DEFAULT_STATE_BOUNDS: Tuple[Tuple[float, float], ...] = (
    (-30.0, 30.0),
    (-30.0, 30.0),
    (0.0, 60.0),
)


@dataclass(frozen=True)
class ExperimentDefaults:
    """Shared runnable experiment settings used by both Q-learning scripts.

    Edit this block when changing experiment parameters.  ``QLearning.py`` and
    ``PlotQLearning.py`` both read the same values, while command-line options
    can still override them for a single run.
    """

    # Episode settings
    episodes: int = 600
    eval_episodes: int = 5
    max_steps: Optional[int] = None
    eval_max_steps: Optional[int] = None

    # Initial state and optional exploration reproducibility
    ic: Tuple[float, float, float] = (0.0, 1.0, 1.05)
    exploration_seed: Optional[int] = None
    evaluation_seed: Optional[int] = 0
    evaluation_ic_perturbation: float = 0.01

    # Simulation duration
    training_lyapunov_times: float = 10.0
    evaluation_lyapunov_times: float = 50.0

    # Control settings
    control_cost: float = LAMBDA
    regularized: bool = True
    action_low: float = -U_REF
    action_high: float = U_REF
    action_bins: int = 9

    # State discretization
    state_bins: Tuple[int, int, int] = (30, 30, 30)

    # Q-learning hyperparameters
    learning_rate: float = 0.01
    discount_factor: float = 1.0
    epsilon: float = 0.99
    epsilon_decay: float = 0.5
    epsilon_min: float = 0.05

EXPERIMENT_DEFAULTS = ExperimentDefaults()


def lyapunov_times_to_steps(lyapunov_times: float) -> int:
    """Convert a duration in Lyapunov times to Patrick's step convention."""
    return round(lyapunov_times / (LYAPUNOV_EXP * DT))


def steps_to_lyapunov_times(steps: int) -> float:
    """Convert a step count to a duration in Lyapunov times."""
    return steps * LYAPUNOV_EXP * DT


def phi(x: Union[float, np.ndarray], eps: float = EPS) -> Union[float, np.ndarray]:
    """Smooth indicator used by the gradient-based reference objective."""
    return 0.5 * (1.0 + np.tanh(x / eps))


def default_state_cost_fn(x: float) -> float:
    """Typed default state cost function for the environment."""
    return float(phi(float(x)))


def calculate_loss(points: np.ndarray, eps: float = EPS) -> float:
    """Reference state loss: the trajectory mean of ``phi(x)``."""
    trajectory = np.asarray(points, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3 or trajectory.shape[0] < 1:
        raise ValueError("points must have nonempty shape (n, 3)")
    return float(np.mean(phi(trajectory[:, 0], eps)))


def control_effort(control_values: Sequence[float], u_ref: float = U_REF) -> float:
    """Reference normalized mean-square control effort."""
    if u_ref <= 0.0:
        raise ValueError("u_ref must be positive")
    controls = np.asarray(control_values, dtype=np.float64)
    if controls.ndim != 1:
        raise ValueError("control_values must be one-dimensional")
    if controls.size == 0:
        return 0.0
    return float(np.mean((controls / u_ref) ** 2))


def reference_objective(
    points: np.ndarray,
    control_values: Sequence[float],
    lam: float = LAMBDA,
    regularized: bool = False,
) -> float:
    """Evaluate the same objective minimized by ``optimize_gradient``."""
    task = calculate_loss(points)
    if not regularized:
        return task
    return task + lam * control_effort(control_values)


class LorenzEnvEuler:
    """Controlled Lorenz system simulated with forward Euler integration."""

    def __init__(
        self,
        alpha: float = LAMBDA,
        lyapunov_times: float = 1.0,
        action_type: str = "continuous",
        action_bounds: Optional[Tuple[float, float]] = None,
        n_action_bins: Optional[int] = None,
        ic_dist: Optional[Callable[[], np.ndarray]] = None,
        state_cost_fn: Optional[Callable[[float], float]] = None,
        divergence_threshold: float = np.inf,
        regularized: bool = False,
        u_ref: float = U_REF,
    ):
        """
        Args:
            alpha: Control-cost weight corresponding to ``LAMBDA``.
            lyapunov_times: Episode length in Lyapunov times.
            action_type: ``"continuous"`` or ``"discrete"``.
            action_bounds: Inclusive ``(low, high)`` bounds for the control.
            n_action_bins: Number of controls when actions are discrete.
            ic_dist: Callable returning ``[x0, y0, z0]``.
            state_cost_fn: Callable mapping x to its state cost.
            divergence_threshold: End an episode above this state norm.
            regularized: Include the reference control-effort penalty.
            u_ref: Control normalization corresponding to ``U_REF``.
        """
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError("alpha must be finite and nonnegative")
        if not np.isfinite(lyapunov_times) or lyapunov_times <= 0.0:
            raise ValueError("lyapunov_times must be finite and positive")
        if not np.isfinite(u_ref) or u_ref <= 0.0:
            raise ValueError("u_ref must be finite and positive")
        if np.isnan(divergence_threshold) or divergence_threshold <= 0.0:
            raise ValueError("divergence_threshold must be positive")
        self.alpha = alpha
        self.lyapunov_times = lyapunov_times
        self.horizon = lyapunov_times_to_steps(lyapunov_times)
        if self.horizon < 1:
            raise ValueError(
                "lyapunov_times is too short to contain one integration step"
            )
        self.divergence_threshold = divergence_threshold
        self.regularized = regularized
        self.u_ref = u_ref

        self.action_type = action_type
        if action_type == "continuous":
            action_bounds = (-U_REF, U_REF) if action_bounds is None else action_bounds
            self.action_low, self.action_high = action_bounds
        elif action_type == "discrete":
            if n_action_bins is None:
                raise ValueError("n_action_bins required for discrete action_type")
            if not isinstance(n_action_bins, (int, np.integer)) or isinstance(
                n_action_bins, (bool, np.bool_)
            ):
                raise TypeError("n_action_bins must be an integer")
            if n_action_bins < 1:
                raise ValueError("n_action_bins must be at least 1")
            action_bounds = (-U_REF, U_REF) if action_bounds is None else action_bounds
            self.action_low, self.action_high = action_bounds
            self.n_action_bins = int(n_action_bins)
            self.actions = np.linspace(
                self.action_low, self.action_high, self.n_action_bins, dtype=np.float64
            )
        else:
            raise ValueError(f"Unknown action_type: {action_type}")
        if not np.isfinite([self.action_low, self.action_high]).all():
            raise ValueError("action bounds must be finite")
        if self.action_low >= self.action_high:
            raise ValueError("the lower action bound must be below the upper bound")

        self.ic_dist = ic_dist or (lambda: INITIAL_STATE.copy())
        # The reward specifies the control objective; it is separate from both
        # the numerical integrator and the Q-learning update below.
        self.state_cost_fn = state_cost_fn or default_state_cost_fn

        self.state: Optional[np.ndarray] = None
        self.step_count = 0
        self._initial_state_cost = 0.0
        self._terminated = False

    def reset(self, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """Start an episode and return a copy of its initial state."""
        initial_state = x0 if x0 is not None else self.ic_dist()
        self.state = np.asarray(initial_state, dtype=np.float64).copy()
        if self.state.shape != (3,):
            raise ValueError("An initial Lorenz state must contain exactly x, y, z")
        if not np.isfinite(self.state).all():
            raise ValueError("The initial Lorenz state must be finite")
        self.step_count = 0
        self._initial_state_cost = float(self.state_cost_fn(self.state[0]))
        if not np.isfinite(self._initial_state_cost):
            raise ValueError("state_cost_fn must return a finite value")
        self._terminated = False
        return self.state.copy()

    def step(
        self, action: Union[int, float]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, bool]]:
        """Advance the Lorenz dynamics by one Euler step."""
        if self.state is None:
            raise RuntimeError("Must call reset() before step()")
        if self._terminated:
            raise RuntimeError("Episode is done; call reset() before step()")

        if self.action_type == "discrete":
            if not isinstance(action, (int, np.integer)):
                raise TypeError("A discrete action must be an integer index")
            action_index = int(action)
            if action_index < 0 or action_index >= self.n_action_bins:
                raise IndexError("Discrete action index is out of range")
            u = float(self.actions[action_index])
        else:
            u = float(action)
            if not np.isfinite(u):
                raise ValueError("A continuous action must be finite")
            u = float(np.clip(u, self.action_low, self.action_high))

        self.state = self._euler_step(self.state, u)
        self.step_count += 1

        state_is_finite = bool(np.isfinite(self.state).all())
        diverged = not state_is_finite
        if state_is_finite and np.isfinite(self.divergence_threshold):
            # Compare the Euclidean norm without overflowing while squaring a
            # very large, but still finite, state. Skip this work when the
            # divergence threshold is infinite (the default).
            state_scale = float(np.max(np.abs(self.state)))
            if state_scale == 0.0:
                diverged = False
            else:
                scaled_norm = float(np.linalg.norm(self.state / state_scale))
                diverged = bool(
                    state_scale > self.divergence_threshold / scaled_norm
                )

        # Scaling each contribution by the reference mean's denominator makes
        # a full episode's return exactly the negative reference objective.
        # The fixed initial-state cost is assigned to the first transition.
        # A divergent transition absorbs the maximum default cost for every
        # unvisited state in the episode.  This both keeps non-finite dynamics
        # from poisoning the Q-table with NaNs and prevents early failure from
        # looking attractive merely because it avoids future negative rewards.
        raw_state_cost = (
            1.0 if diverged else float(self.state_cost_fn(self.state[0]))
        )
        if not np.isfinite(raw_state_cost):
            raise ValueError("state_cost_fn must return a finite value")
        remaining_state_terms = (
            self.horizon - self.step_count + 1 if diverged else 1
        )
        state_cost = raw_state_cost * remaining_state_terms / (self.horizon + 1)
        if self.step_count == 1:
            state_cost += self._initial_state_cost / (self.horizon + 1)
        control_cost = 0.0
        if self.regularized:
            control_cost = self.alpha * (u / self.u_ref) ** 2 / self.horizon
        reward = -float(state_cost + control_cost)
        done = diverged
        info: Dict[str, bool] = {}
        if diverged:
            info["diverged"] = True
        if self.step_count >= self.horizon:
            done, info["horizon_reached"] = True, True

        self._terminated = done

        return self.state.copy(), reward, done, info

    @staticmethod
    def _euler_step(state: np.ndarray, u: float) -> np.ndarray:
        # Euler integration simulates the controlled Lorenz dynamics.
        x, y, z = state
        dx = PRANDTL * (y - x) + u
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z
        return state + np.array([dx, dy, dz]) * DT


class StateDiscretizer:
    """Map continuous ``[x, y, z]`` states to clipped, uniform bin indices.

    Discretization makes tabular Q-learning feasible, but it is necessarily
    coarse: every continuous state in one bin shares the same action values.
    """

    def __init__(
        self,
        bins: Union[int, Sequence[int]] = 15,
        bounds: Sequence[Tuple[float, float]] = DEFAULT_STATE_BOUNDS,
    ):
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if bounds_array.shape != (3, 2):
            raise ValueError("bounds must contain (low, high) for x, y, and z")
        if not np.isfinite(bounds_array).all():
            raise ValueError("state bounds must be finite")
        if np.any(bounds_array[:, 0] >= bounds_array[:, 1]):
            raise ValueError("each lower state bound must be below its upper bound")

        raw_bins = np.asarray([bins] * 3 if np.isscalar(bins) else bins)
        if raw_bins.shape != (3,):
            raise ValueError("bins must be an integer or three bin counts")
        if any(
            not isinstance(value, (int, np.integer))
            or isinstance(value, (bool, np.bool_))
            for value in raw_bins.tolist()
        ):
            raise ValueError("bin counts must be positive integers")
        bins_array = raw_bins.astype(np.int64)
        if np.any(bins_array < 1):
            raise ValueError("bin counts must be positive integers")

        self.bounds = bounds_array
        self.bins: Tuple[int, int, int] = tuple(  # type: ignore[assignment]
            int(value) for value in bins_array
        )
        self._lower = bounds_array[:, 0]
        self._upper = bounds_array[:, 1]
        self._widths = (self._upper - self._lower) / bins_array

    def discretize(self, state: Sequence[float]) -> Tuple[int, int, int]:
        """Return the x/y/z bin tuple, clipping values to valid edge bins."""
        state_array = np.asarray(state, dtype=np.float64)
        if state_array.shape != (3,):
            raise ValueError("state must contain exactly x, y, and z")
        if not np.isfinite(state_array).all():
            raise ValueError("cannot discretize a non-finite state")

        clipped = np.clip(state_array, self._lower, self._upper)
        indices = np.floor((clipped - self._lower) / self._widths).astype(np.int64)
        indices = np.clip(indices, 0, np.asarray(self.bins) - 1)
        return cast(Tuple[int, int, int], tuple(int(index) for index in indices))


class QLearningAgent:
    """A reproducible tabular Q-learning agent for discrete controls."""

    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 1.0,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        state_bins: Union[int, Sequence[int]] = 15,
        state_bounds: Sequence[Tuple[float, float]] = DEFAULT_STATE_BOUNDS,
        random_seed: Optional[int] = None,
    ):
        """Create a tabular Q-learning agent."""
        if not isinstance(n_actions, (int, np.integer)) or isinstance(
            n_actions, (bool, np.bool_)
        ):
            raise TypeError("n_actions must be an integer")
        if n_actions < 1:
            raise ValueError("n_actions must be at least 1")

        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        if (
            not np.isfinite(self.learning_rate)
            or not 0.0 < self.learning_rate <= 1.0
        ):
            raise ValueError("learning_rate must be in (0, 1]")
        if (
            not np.isfinite(self.discount_factor)
            or not 0.0 <= self.discount_factor <= 1.0
        ):
            raise ValueError("discount_factor must be in [0, 1]")
        if (
            not np.isfinite([epsilon_min, epsilon, epsilon_decay]).all()
            or not 0.0 <= epsilon_min <= epsilon <= 1.0
        ):
            raise ValueError("require 0 <= epsilon_min <= epsilon <= 1")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.discretizer = StateDiscretizer(state_bins, state_bounds)
        self.rng = np.random.default_rng(random_seed)
        self.q_table = np.zeros(
            self.discretizer.bins + (self.n_actions,), dtype=np.float64
        )

    def discretize_state(self, state: Sequence[float]) -> Tuple[int, int, int]:
        """Map a continuous Lorenz state to the agent's state-bin tuple."""
        return self.discretizer.discretize(state)

    def select_action(self, state: Sequence[float], training: bool = True) -> int:
        """Select an epsilon-greedy training action or a greedy evaluation action."""
        state_index = self.discretize_state(state)
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state_index]))

    def update(
        self,
        state: Sequence[float],
        action: int,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> float:
        """Apply one standard off-policy tabular Q-learning update.

        Returns the temporal-difference error.  Terminal transitions use the
        reward alone and therefore never bootstrap from ``next_state``.
        """
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer index")
        action_index = int(action)
        if action_index < 0 or action_index >= self.n_actions:
            raise IndexError("action index is out of range")
        if not isinstance(done, (bool, np.bool_)):
            raise TypeError("done must be a boolean")

        reward_value = float(reward)
        if not np.isfinite(reward_value):
            raise ValueError("reward must be finite")

        state_index = self.discretize_state(state)
        table_index = state_index + (action_index,)
        current_value = self.q_table[table_index]

        if done:
            target = reward_value
        else:
            next_index = self.discretize_state(next_state)
            target = reward_value + self.discount_factor * float(
                np.max(self.q_table[next_index])
            )
        td_error = target - current_value
        self.q_table[table_index] += self.learning_rate * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        """Decay epsilon without allowing it to fall below ``epsilon_min``."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon


EvaluationResults = Dict[str, Union[List[float], List[int], List[bool], List[np.ndarray]]]


def _validate_discrete_pair(env: LorenzEnvEuler, agent: QLearningAgent) -> None:
    if env.action_type != "discrete":
        raise ValueError("Tabular Q-learning requires env.action_type='discrete'")
    if len(env.actions) != agent.n_actions:
        raise ValueError("agent.n_actions must match the environment's action bins")


def _rolling_mean(values: List[float], window: int = 100) -> List[float]:
    means: List[float] = []
    for end in range(1, len(values) + 1):
        start = max(0, end - window)
        means.append(float(np.mean(values[start:end])))
    return means


def train_q_learning(
    env: LorenzEnvEuler,
    agent: QLearningAgent,
    num_episodes: int,
    max_steps: Optional[int] = None,
) -> TrainingHistory:
    """Train ``agent`` against ``env`` and return episode-level history."""
    _validate_discrete_pair(env, agent)
    if not isinstance(num_episodes, (int, np.integer)) or isinstance(
        num_episodes, (bool, np.bool_)
    ):
        raise TypeError("num_episodes must be an integer")
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")
    if max_steps is not None:
        if not isinstance(max_steps, (int, np.integer)) or isinstance(
            max_steps, (bool, np.bool_)
        ):
            raise TypeError("max_steps must be an integer when supplied")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1 when supplied")

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    epsilons: List[float] = []
    diverged: List[bool] = []

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        episode_diverged = False

        while not done and (max_steps is None or steps < max_steps):
            action = agent.select_action(state.tolist(), training=True)
            next_state, reward, done, info = env.step(action)
            # ``max_steps`` is a time-limit truncation, not a terminal Lorenz
            # state, so only the environment's terminal flag stops bootstrap.
            agent.update(state.tolist(), action, reward, next_state.tolist(), done)
            state = next_state
            total_reward += reward
            steps += 1
            episode_diverged = episode_diverged or bool(info.get("diverged", False))

        agent.decay_epsilon()
        episode_rewards.append(float(total_reward))
        episode_lengths.append(steps)
        epsilons.append(agent.epsilon)
        diverged.append(episode_diverged)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "epsilons": epsilons,
        "diverged": diverged,
        "rolling_mean_rewards": _rolling_mean(episode_rewards),
    }

#Creates control function for the Q-learning agent.
#The control function is used to evaluate the performance of the agent after training.
#It runs the agent in the environment without exploration and without updating the Q-table,
# and collects statistics about the episodes.
#It is an evaluation/control rollout function, not a training function.

def control_q_learning(
    env: LorenzEnvEuler,
    agent: QLearningAgent,
    num_episodes: int,
    max_steps: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
) -> EvaluationResults:
    """Evaluate greedily without exploration or Q-table updates.

    Each trajectory includes its initial state.  ``actions`` contains discrete
    action indices; ``control_values`` contains their corresponding controls.
    """
    _validate_discrete_pair(env, agent)
    if not isinstance(num_episodes, (int, np.integer)) or isinstance(
        num_episodes, (bool, np.bool_)
    ):
        raise TypeError("num_episodes must be an integer")
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")
    if max_steps is not None:
        if not isinstance(max_steps, (int, np.integer)) or isinstance(
            max_steps, (bool, np.bool_)
        ):
            raise TypeError("max_steps must be an integer when supplied")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1 when supplied")

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    all_actions: List[np.ndarray] = []
    all_controls: List[np.ndarray] = []
    trajectories: List[np.ndarray] = []
    diverged: List[bool] = []
    task_losses: List[float] = []
    control_efforts: List[float] = []
    objectives: List[float] = []

    for _ in range(num_episodes):
        state = env.reset(x0=x0)
        states = [state.copy()]
        actions: List[int] = []
        controls: List[float] = []
        total_reward = 0.0
        steps = 0
        done = False
        episode_diverged = False

        while not done and (max_steps is None or steps < max_steps):
            action = agent.select_action(state.tolist(), training=False)
            next_state, reward, done, info = env.step(action)
            actions.append(action)
            controls.append(float(env.actions[action]))
            states.append(next_state.copy())
            state = next_state
            total_reward += reward
            steps += 1
            episode_diverged = episode_diverged or bool(info.get("diverged", False))

        trajectory = np.asarray(states, dtype=np.float64)
        control_array = np.asarray(controls, dtype=np.float64)
        episode_rewards.append(float(total_reward))
        episode_lengths.append(steps)
        all_actions.append(np.asarray(actions, dtype=np.int64))
        all_controls.append(control_array)
        trajectories.append(trajectory)
        diverged.append(episode_diverged)
        effort = control_effort(control_array.tolist(), env.u_ref)
        control_efforts.append(effort)
        if episode_diverged:
            # Non-finite trajectories have no meaningful reference objective;
            # report them as failures instead of leaking NaNs into summaries.
            task_losses.append(float("inf"))
            objectives.append(float("inf"))
        else:
            task_loss = calculate_loss(trajectory)
            task_losses.append(task_loss)
            objectives.append(
                reference_objective(
                    trajectory,
                    control_array.tolist(),
                    lam=env.alpha,
                    regularized=env.regularized,
                )
            )

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "actions": all_actions,
        "control_values": all_controls,
        "trajectories": trajectories,
        "diverged": diverged,
        "task_losses": task_losses,
        "control_efforts": control_efforts,
        "objectives": objectives,
    }


def main() -> None:
    """Run a configurable tabular Q-learning example."""
    defaults = EXPERIMENT_DEFAULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=defaults.episodes)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--eval-max-steps", type=int, default=defaults.eval_max_steps)
    parser.add_argument(
        "--initial-condition",
        type=float,
        nargs=3,
        default=list(defaults.ic),
        metavar=("X0", "Y0", "Z0"),
    )
    parser.add_argument(
        "--exploration-seed",
        type=int,
        default=defaults.exploration_seed,
        help="optional seed for epsilon-greedy exploration only",
    )
    parser.add_argument(
        "--lyapunov-times",
        type=float,
        default=defaults.training_lyapunov_times,
    )
    parser.add_argument(
        "--eval-lyapunov-times",
        type=float,
        default=defaults.evaluation_lyapunov_times,
    )
    parser.add_argument("--control-cost", type=float, default=defaults.control_cost)
    parser.add_argument(
        "--regularized",
        action=argparse.BooleanOptionalAction,
        default=defaults.regularized,
        help="include LAMBDA times normalized control effort",
    )
    parser.add_argument("--action-low", type=float, default=defaults.action_low)
    parser.add_argument("--action-high", type=float, default=defaults.action_high)
    parser.add_argument("--action-bins", type=int, default=defaults.action_bins)
    parser.add_argument(
        "--state-bins",
        type=int,
        nargs=3,
        default=list(defaults.state_bins),
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--discount-factor", type=float, default=defaults.discount_factor
    )
    parser.add_argument("--epsilon", type=float, default=defaults.epsilon)
    parser.add_argument("--epsilon-decay", type=float, default=defaults.epsilon_decay)
    parser.add_argument("--epsilon-min", type=float, default=defaults.epsilon_min)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.eval_episodes < 1:
        parser.error("--eval-episodes must be at least 1")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be at least 1")
    if args.eval_max_steps is not None and args.eval_max_steps < 1:
        parser.error("--eval-max-steps must be at least 1")
    if args.lyapunov_times <= 0.0 or args.eval_lyapunov_times <= 0.0:
        parser.error("Lyapunov-time horizons must be positive")
    ic = np.asarray(args.initial_condition, dtype=np.float64)
    if ic.shape != (3,) or not np.isfinite(ic).all():
        parser.error("--initial-condition must contain three finite values")

    def fixed_initial_state() -> np.ndarray:
        return ic.copy()

    training_env = LorenzEnvEuler(
        alpha=args.control_cost,
        lyapunov_times=args.lyapunov_times,
        action_type="discrete",
        action_bounds=(args.action_low, args.action_high),
        n_action_bins=args.action_bins,
        ic_dist=fixed_initial_state,
        regularized=args.regularized,
        u_ref=U_REF,
    )
    agent = QLearningAgent(
        n_actions=len(training_env.actions),
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        state_bins=tuple(args.state_bins),
        random_seed=args.exploration_seed,
    )

    history = train_q_learning(
        training_env,
        agent,
        args.episodes,
        args.max_steps,
    )
    evaluation_env = LorenzEnvEuler(
        alpha=args.control_cost,
        lyapunov_times=args.eval_lyapunov_times,
        action_type="discrete",
        action_bounds=(args.action_low, args.action_high),
        n_action_bins=args.action_bins,
        ic_dist=fixed_initial_state,
        regularized=args.regularized,
        u_ref=U_REF,
    )
    evaluation = control_q_learning(
        evaluation_env,
        agent,
        args.eval_episodes,
        args.eval_max_steps,
        x0=ic,
    )

    window = min(50, len(history["episode_rewards"]))
    final_training_rewards = history["episode_rewards"][-window:]
    sample_bin = agent.discretize_state(ic.tolist())
    sample_q_values = agent.q_table[sample_bin]
    first_actions = cast(np.ndarray, evaluation["actions"][0])[:10]

    print(
        "learned Q-feedback law: Q", sample_bin, "=",
        np.array2string(sample_q_values, precision=3),
        "; greedy actions =", first_actions.tolist(),
        "; epsilon =", f"{agent.epsilon:.4f}",
        "; final", window, "episode reward =", f"{np.mean(final_training_rewards):.4f}",
        "; objective =", f"{np.mean(evaluation['objectives']):.4f}",
        "; task loss =", f"{np.mean(evaluation['task_losses']):.4f}",
        "; control effort =", f"{np.mean(evaluation['control_efforts']):.4f}",
        "; diverged =", any(evaluation["diverged"]),
    )


if __name__ == "__main__":
    main()
