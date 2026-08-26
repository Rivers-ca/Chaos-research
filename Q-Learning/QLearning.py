from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, TypedDict, cast

class TrainingHistory(TypedDict):
    episode_rewards: List[float]
    episode_lengths: List[int]
    epsilons: List[float]
    diverged: List[bool]
    rolling_mean_rewards: List[float]


class CheckpointEvaluationHistory(TypedDict):
    episodes: List[int]
    mean_rewards: List[float]
    reward_standard_deviations: List[float]
    divergence_rates: List[float]
    mean_control_efforts: List[float]

import numpy as np

RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
DT = 0.01
LYAPUNOV_EXP = 0.9056
EPS = 2.0
U_REF = 60.0
LAMBDA = 0.007
EPISODES = 600
PRINT_EVERY = 25

INITIAL_STATE = np.array([0.0, 1.0, 1.05], dtype=np.float64)

DEFAULT_STATE_BOUNDS: Tuple[Tuple[float, float], ...] = (
    (-30.0, 30.0),
    (-30.0, 30.0),
    (0.0, 60.0),
)


@dataclass(frozen=True)
class ExperimentDefaults:
    """Update these defaults to change the behavior of the main() function and the default training/evaluation settings, 
    this will also be refle
    """

    # Episode settings
    episodes: int = EPISODES
    eval_episodes: int = 5
    evaluation_interval: int = 50
    max_steps: Optional[int] = None
    eval_max_steps: Optional[int] = None

    # Initial state and optional exploration reproducibility
    ic: Tuple[float, float, float] = (0.0, 1.0, 1.05)
    training_ic_seed: Optional[int] = 0
    training_ic_perturbation: float = 0.01
    exploration_seed: Optional[int] = None
    evaluation_seed: Optional[int] = 0
    evaluation_ic_perturbation: float = 0.01

    # Simulation duration
    training_lyapunov_times: float = 50.0
    evaluation_lyapunov_times: float = 50.0

    # Control settings
    control_cost: float = LAMBDA
    regularized: bool = True
    action_low: float = -U_REF
    action_high: float = U_REF
    action_bins: int = 9

    # State discretization
    state_bins: Tuple[int, int, int] = (15, 15, 15)

    # Q-learning hyperparameters
    learning_rate: float = 0.01
    discount_factor: float = 1.0
    epsilon: float = 0.99
    epsilon_decay: float = 0.01
    epsilon_min: float = 0.05

EXPERIMENT_DEFAULTS = ExperimentDefaults()


def lyapunov_times_to_steps(lyapunov_times: float) -> int:
    return round(lyapunov_times / (LYAPUNOV_EXP * DT))


def steps_to_lyapunov_times(steps: int) -> float:
    return steps * LYAPUNOV_EXP * DT


def make_evaluation_initial_states(
    reference_state: Sequence[float],
    num_episodes: int,
    perturbation: float = 0.01,
    random_seed: Optional[int] = 0,
) -> np.ndarray:
    reference = np.asarray(reference_state, dtype=np.float64)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("reference_state must contain three finite values")
    if not isinstance(num_episodes, (int, np.integer)) or isinstance(
        num_episodes, (bool, np.bool_)
    ):
        raise TypeError("num_episodes must be an integer")
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")
    if not np.isfinite(perturbation) or perturbation < 0.0:
        raise ValueError("perturbation must be finite and nonnegative")

    starts = np.repeat(reference[None, :], int(num_episodes), axis=0)
    if perturbation > 0.0:
        rng = np.random.default_rng(random_seed)
        starts += rng.uniform(
            -perturbation,
            perturbation,
            size=(num_episodes, 3),
        )
    return starts

#Check back on this later
def make_random_initial_state_sampler(
    reference_state: Sequence[float],
    perturbation: float = 0.01,
    random_seed: Optional[int] = None,
) -> Callable[[], np.ndarray]:
    """Return a callable that draws a new perturbed state for every reset."""
    reference = np.asarray(reference_state, dtype=np.float64)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("reference_state must contain three finite values")
    if not np.isfinite(perturbation) or perturbation < 0.0:
        raise ValueError("perturbation must be finite and nonnegative")
    rng = np.random.default_rng(random_seed)

    def sample() -> np.ndarray:
        return reference + rng.uniform(-perturbation, perturbation, size=3)

    return sample


def phi(x: Union[float, np.ndarray], eps: float = EPS) -> Union[float, np.ndarray]:
    return 0.5 * (1.0 + np.tanh(x / eps))


def default_state_cost_fn(x: float) -> float:
    return float(phi(float(x)))


#check back on this later
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


class LorenzEnvEuler:
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
        self.state_cost_fn = state_cost_fn or default_state_cost_fn

        self.state: Optional[np.ndarray] = None
        self.step_count = 0
        self._initial_state_cost = 0.0
        self._terminated = False

    def reset(self, x0: Optional[np.ndarray] = None) -> np.ndarray:
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

        # Scale state costs by the episode length so returns are comparable.
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
        x, y, z = state
        dx = PRANDTL * (y - x) + u
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z
        return state + np.array([dx, dy, dz]) * DT


class StateDiscretizer:

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
    def __init__(
        self,
        n_actions: int,
        learning_rate: float = EXPERIMENT_DEFAULTS.learning_rate,
        discount_factor: float = EXPERIMENT_DEFAULTS.discount_factor,
        epsilon: float = EXPERIMENT_DEFAULTS.epsilon,
        epsilon_decay: float = EXPERIMENT_DEFAULTS.epsilon_decay,
        epsilon_min: float = EXPERIMENT_DEFAULTS.epsilon_min,
        state_bins: Union[int, Sequence[int]] = EXPERIMENT_DEFAULTS.state_bins,
        state_bounds: Sequence[Tuple[float, float]] = DEFAULT_STATE_BOUNDS,
        random_seed: Optional[int] = EXPERIMENT_DEFAULTS.exploration_seed,
    ):

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
        return self.discretizer.discretize(state)

    def select_action(self, state: Sequence[float], training: bool = True) -> int:
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


def _run_training_episode(
    env: LorenzEnvEuler,
    agent: QLearningAgent,
    max_steps: Optional[int],
) -> Tuple[float, int, bool]:
    """Run one exploratory episode and update the shared Q-table."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    episode_diverged = False

    while not done and (max_steps is None or steps < max_steps):
        action = agent.select_action(state.tolist(), training=True)
        next_state, reward, done, info = env.step(action)
        agent.update(state.tolist(), action, reward, next_state.tolist(), done)
        state = next_state
        total_reward += reward
        steps += 1
        episode_diverged = episode_diverged or bool(info.get("diverged", False))

    agent.decay_epsilon()
    return float(total_reward), steps, episode_diverged


def train_q_learning(
    env: LorenzEnvEuler,
    agent: QLearningAgent,
    num_episodes: int,
    max_steps: Optional[int] = None,
) -> TrainingHistory:
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

    for episode in range(1, num_episodes + 1):
        total_reward, steps, episode_diverged = _run_training_episode(
            env, agent, max_steps
        )
        episode_rewards.append(float(total_reward))
        episode_lengths.append(steps)
        epsilons.append(agent.epsilon)
        diverged.append(episode_diverged)

        if episode % PRINT_EVERY == 0:
            recent_rewards = episode_rewards[-PRINT_EVERY:]
            nonzero_entries = np.count_nonzero(agent.q_table)
            print(
                f"Episode {episode}/{num_episodes} | "
                f"average reward: {np.mean(recent_rewards):.4f} | "
                f"steps: {steps} | "
                f"epsilon: {agent.epsilon:.4f} | "
                f"nonzero Q-values: {nonzero_entries}/{agent.q_table.size} | "
                f"Q range: [{agent.q_table.min():.4f}, "
                f"{agent.q_table.max():.4f}]"
            )
        
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "epsilons": epsilons,
        "diverged": diverged,
        "rolling_mean_rewards": _rolling_mean(episode_rewards),
    }



def evaluate_q_learning(
    env: LorenzEnvEuler,
    agent: QLearningAgent,
    num_episodes: int,
    max_steps: Optional[int] = None,
    x0: Optional[np.ndarray] = None,
    initial_states: Optional[Sequence[Sequence[float]]] = None,
) -> EvaluationResults:
    _validate_discrete_pair(env, agent)
    if not isinstance(num_episodes, (int, np.integer)) or isinstance(
        num_episodes, (bool, np.bool_)
    ):
        raise TypeError("num_episodes must be an integer")
    if num_episodes < 1:
        raise ValueError("num_episodes must be at least 1")
    if x0 is not None and initial_states is not None:
        raise ValueError("provide either x0 or initial_states, not both")
    episode_initial_states: Optional[np.ndarray] = None
    if initial_states is not None:
        episode_initial_states = np.asarray(initial_states, dtype=np.float64)
        if episode_initial_states.shape != (num_episodes, 3):
            raise ValueError("initial_states must have shape (num_episodes, 3)")
        if not np.isfinite(episode_initial_states).all():
            raise ValueError("initial_states must contain only finite values")
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
    control_efforts: List[float] = []

    for episode_index in range(num_episodes):
        episode_x0 = (
            episode_initial_states[episode_index]
            if episode_initial_states is not None
            else x0
        )
        state = env.reset(x0=episode_x0)
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

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "actions": all_actions,
        "control_values": all_controls,
        "trajectories": trajectories,
        "diverged": diverged,
        "control_efforts": control_efforts,
    }


# Backward-compatible name retained for callers that used the original API.
control_q_learning = evaluate_q_learning


def train_q_learning_with_evaluation(
    training_env: LorenzEnvEuler,
    evaluation_env: LorenzEnvEuler,
    agent: QLearningAgent,
    num_episodes: int,
    evaluation_interval: int,
    evaluation_episodes: int,
    max_steps: Optional[int] = None,
    evaluation_max_steps: Optional[int] = None,
    evaluation_initial_states: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[TrainingHistory, CheckpointEvaluationHistory, EvaluationResults]:
    """Train continuously and greedily evaluate at fixed checkpoints.

    Training history spans all episodes without being restarted. Evaluation
    never updates the Q-table or epsilon. The final episode is always a
    checkpoint, even when it is not divisible by ``evaluation_interval``.
    """
    _validate_discrete_pair(training_env, agent)
    _validate_discrete_pair(evaluation_env, agent)
    for name, value in (
        ("num_episodes", num_episodes),
        ("evaluation_interval", evaluation_interval),
        ("evaluation_episodes", evaluation_episodes),
    ):
        if not isinstance(value, (int, np.integer)) or isinstance(
            value, (bool, np.bool_)
        ):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    for name, value in (
        ("max_steps", max_steps),
        ("evaluation_max_steps", evaluation_max_steps),
    ):
        if value is not None:
            if not isinstance(value, (int, np.integer)) or isinstance(
                value, (bool, np.bool_)
            ):
                raise TypeError(f"{name} must be an integer when supplied")
            if value < 1:
                raise ValueError(f"{name} must be at least 1 when supplied")

    checkpoint_initial_states: Optional[np.ndarray] = None
    if evaluation_initial_states is not None:
        checkpoint_initial_states = np.asarray(
            evaluation_initial_states, dtype=np.float64
        )
        if checkpoint_initial_states.shape != (evaluation_episodes, 3):
            raise ValueError(
                "evaluation_initial_states must have shape "
                "(evaluation_episodes, 3)"
            )
        if not np.isfinite(checkpoint_initial_states).all():
            raise ValueError(
                "evaluation_initial_states must contain only finite values"
            )

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    epsilons: List[float] = []
    training_diverged: List[bool] = []
    checkpoint_episodes: List[int] = []
    checkpoint_mean_rewards: List[float] = []
    checkpoint_reward_standard_deviations: List[float] = []
    checkpoint_divergence_rates: List[float] = []
    checkpoint_mean_control_efforts: List[float] = []
    final_evaluation: Optional[EvaluationResults] = None

    for episode in range(1, num_episodes + 1):
        total_reward, steps, episode_diverged = _run_training_episode(
            training_env, agent, max_steps
        )
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        epsilons.append(agent.epsilon)
        training_diverged.append(episode_diverged)

        if episode % PRINT_EVERY == 0:
            recent_rewards = episode_rewards[-PRINT_EVERY:]
            nonzero_entries = np.count_nonzero(agent.q_table)
            print(
                f"Episode {episode}/{num_episodes} | "
                f"average reward: {np.mean(recent_rewards):.4f} | "
                f"steps: {steps} | epsilon: {agent.epsilon:.4f} | "
                f"nonzero Q-values: {nonzero_entries}/{agent.q_table.size} | "
                f"Q range: [{agent.q_table.min():.4f}, "
                f"{agent.q_table.max():.4f}]"
            )

        if episode % evaluation_interval == 0 or episode == num_episodes:
            table_before_evaluation = agent.q_table.copy()
            epsilon_before_evaluation = agent.epsilon
            evaluation = evaluate_q_learning(
                evaluation_env,
                agent,
                evaluation_episodes,
                evaluation_max_steps,
                initial_states=checkpoint_initial_states,
            )
            if not np.array_equal(table_before_evaluation, agent.q_table):
                raise RuntimeError("Checkpoint evaluation unexpectedly changed the Q-table")
            if agent.epsilon != epsilon_before_evaluation:
                raise RuntimeError("Checkpoint evaluation unexpectedly changed epsilon")

            rewards = np.asarray(evaluation["episode_rewards"], dtype=np.float64)
            divergences = np.asarray(evaluation["diverged"], dtype=np.float64)
            efforts = np.asarray(evaluation["control_efforts"], dtype=np.float64)
            checkpoint_episodes.append(episode)
            checkpoint_mean_rewards.append(float(np.mean(rewards)))
            checkpoint_reward_standard_deviations.append(float(np.std(rewards)))
            checkpoint_divergence_rates.append(float(np.mean(divergences)))
            checkpoint_mean_control_efforts.append(float(np.mean(efforts)))
            final_evaluation = evaluation
            print(
                f"Checkpoint {episode}/{num_episodes} | "
                f"greedy reward: {np.mean(rewards):.4f} +/- {np.std(rewards):.4f} | "
                f"divergences: {int(np.sum(divergences))}/{evaluation_episodes}"
            )

    if final_evaluation is None:  # Defensive guard; num_episodes is positive.
        raise RuntimeError("Training completed without a final evaluation checkpoint")

    training_history: TrainingHistory = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "epsilons": epsilons,
        "diverged": training_diverged,
        "rolling_mean_rewards": _rolling_mean(episode_rewards),
    }
    checkpoint_history: CheckpointEvaluationHistory = {
        "episodes": checkpoint_episodes,
        "mean_rewards": checkpoint_mean_rewards,
        "reward_standard_deviations": checkpoint_reward_standard_deviations,
        "divergence_rates": checkpoint_divergence_rates,
        "mean_control_efforts": checkpoint_mean_control_efforts,
    }
    return training_history, checkpoint_history, final_evaluation


def main() -> None:
    settings = EXPERIMENT_DEFAULTS
    env = LorenzEnvEuler(
        alpha=settings.control_cost,
        lyapunov_times=settings.training_lyapunov_times,
        action_type="discrete",
        action_bounds=(settings.action_low, settings.action_high),
        n_action_bins=settings.action_bins,
        regularized=settings.regularized,
        u_ref=U_REF,
    )
    agent = QLearningAgent(n_actions=len(env.actions))
    history = train_q_learning(env, agent, num_episodes=EPISODES)

    window = min(50, len(history["episode_rewards"]))
    final_training_rewards = history["episode_rewards"][-window:]
    sample_bin = agent.discretize_state(INITIAL_STATE)
    sample_q_values = agent.q_table[sample_bin]

    print(
        "trained episodes =", EPISODES,
        "; Q", sample_bin, "=",
        np.array2string(sample_q_values, precision=3),
        "; epsilon =", f"{agent.epsilon:.4f}",
        "; final", window, "episode reward =", f"{np.mean(final_training_rewards):.4f}",
    )


if __name__ == "__main__":
    main()
