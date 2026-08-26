from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, TypedDict, cast

class TrainingHistory(TypedDict):
    episode_rewards: List[float]
    episode_lengths: List[int]
    epsilons: List[float]
    diverged: List[bool]
    unique_state_bins_visited: List[int]
    q_table_coverages: List[float]
    mean_abs_td_errors: List[float]
    rolling_mean_rewards: List[float]

import numpy as np

ArrayLike = Union[Sequence[Any], np.ndarray]
StateVector = Union[Sequence[float], np.ndarray]

RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
DT = 0.01
LYAPUNOV_EXP = 0.9056
EPS = 2.0
U_REF = 60.0
LAMBDA = 0.007
EPISODES = 1000

DEFAULT_STATE_BOUNDS: Tuple[Tuple[float, float], ...] = (
    (-30.0, 30.0), (-30.0, 30.0), (0.0, 60.0))


def phi(x: Union[float, np.ndarray], eps: float = EPS) -> Union[float, np.ndarray]:
    return 0.5 * (1.0 + np.tanh(x / eps))


def default_state_cost_fn(x: float) -> float:
    return float(phi(float(x)))


@dataclass(frozen=True)
class ExperimentDefaults:
    episodes: int = EPISODES
    eval_episodes: int = 5
    evaluation_interval: int = 50
    max_steps: Optional[int] = None
    eval_max_steps: Optional[int] = None
    print_every: int = 50

    ic: Tuple[float, float, float] = (0.0, 1.0, 1.05)
    training_ic_seed: Optional[int] = 0
    training_ic_perturbation: float = 0.01
    exploration_seed: Optional[int] = None
    evaluation_seed: Optional[int] = 0
    evaluation_ic_perturbation: float = 0.01

    training_lyapunov_times: float = 50.0
    evaluation_lyapunov_times: float = 100.0
    control_cost: float = LAMBDA
    regularized: bool = True
    action_low: float = -U_REF
    action_high: float = U_REF
    action_bins: int = 9
    u_ref: float = U_REF
    divergence_threshold: float = np.inf
    state_cost_fn: Callable[[float], float] = default_state_cost_fn

    state_bins: Tuple[int, int, int] = (15, 15, 15)
    learning_rate: float = 0.01
    discount_factor: float = 1.0
    epsilon: float = 0.99
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05

    def make_training_initial_state_sampler(self) -> Callable[[], np.ndarray]:
        return make_random_initial_state_sampler(
            self.ic, self.training_ic_perturbation, self.training_ic_seed
        )

    def make_evaluation_initial_states(self) -> np.ndarray:
        return make_evaluation_initial_states(
            self.ic,
            self.eval_episodes,
            self.evaluation_ic_perturbation,
            self.evaluation_seed,
        )

EXPERIMENT_DEFAULTS = ExperimentDefaults()
RUN_OUTPUT_PATH = Path(__file__).with_name("q_learning_run.pkl")


def _positive_int(value: Any, name: str, *, optional: bool = False) -> Optional[int]:
    """Validate integer configuration without silently accepting bools or floats."""
    if optional and value is None:
        return None
    suffix = " when supplied" if optional else ""
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer{suffix}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1{suffix}")
    return int(value)


def _finite_array(values: ArrayLike, shape: Tuple[int, ...], message: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(message)
    return result


def lyapunov_times_to_steps(lyapunov_times: float) -> int:
    return round(lyapunov_times / (LYAPUNOV_EXP * DT))


def make_evaluation_initial_states(
    reference_state: Sequence[float],
    num_episodes: int,
    perturbation: float = 0.01,
    random_seed: Optional[int] = 0,
) -> np.ndarray:
    reference = _finite_array(
        reference_state, (3,), "reference_state must contain three finite values"
    )
    num_episodes = cast(int, _positive_int(num_episodes, "num_episodes"))
    if not np.isfinite(perturbation) or perturbation < 0.0:
        raise ValueError("perturbation must be finite and nonnegative")

    starts = np.repeat(reference[None, :], int(num_episodes), axis=0)
    if perturbation > 0.0:
        rng = np.random.default_rng(random_seed)
        starts += rng.uniform(-perturbation, perturbation, size=(num_episodes, 3))
    return starts

def make_random_initial_state_sampler(
    reference_state: Sequence[float],
    perturbation: float = 0.01,
    random_seed: Optional[int] = None,
) -> Callable[[], np.ndarray]:
    reference = _finite_array(
        reference_state, (3,), "reference_state must contain three finite values"
    )
    if not np.isfinite(perturbation) or perturbation < 0.0:
        raise ValueError("perturbation must be finite and nonnegative")
    rng = np.random.default_rng(random_seed)
    return lambda: reference + rng.uniform(-perturbation, perturbation, size=3)


class LorenzEnvEuler:
    def __init__(
        self,
        settings: ExperimentDefaults,
        *,
        training: bool = False,
        controlled: bool = True,
    ):
        if not isinstance(settings, ExperimentDefaults):
            raise TypeError("settings must be an ExperimentDefaults instance")
        alpha = settings.control_cost
        lyapunov_times = (settings.training_lyapunov_times if training
                          else settings.evaluation_lyapunov_times)
        divergence_threshold = settings.divergence_threshold
        u_ref = settings.u_ref
        numeric_rules = (
            (np.isfinite(alpha) and alpha >= 0.0, "alpha must be finite and nonnegative"),
            (np.isfinite(lyapunov_times) and lyapunov_times > 0.0,
             "lyapunov_times must be finite and positive"),
            (np.isfinite(u_ref) and u_ref > 0.0, "u_ref must be finite and positive"),
            (not np.isnan(divergence_threshold) and divergence_threshold > 0.0,
             "divergence_threshold must be positive"),
        )
        for valid, message in numeric_rules:
            if not valid:
                raise ValueError(message)
        self.alpha = alpha
        self.lyapunov_times = lyapunov_times
        self.horizon = lyapunov_times_to_steps(lyapunov_times)
        if self.horizon < 1:
            raise ValueError(
                "lyapunov_times is too short to contain one integration step"
            )
        self.divergence_threshold = divergence_threshold
        self.regularized = settings.regularized if controlled else False
        self.u_ref = u_ref

        self.action_type = "discrete" if controlled else "continuous"
        self.action_low, self.action_high = settings.action_low, settings.action_high
        if controlled:
            self.n_action_bins = cast(
                int, _positive_int(settings.action_bins, "action_bins")
            )
            self.actions = np.linspace(self.action_low, self.action_high,
                                       self.n_action_bins, dtype=np.float64)
        if not np.isfinite([self.action_low, self.action_high]).all():
            raise ValueError("action bounds must be finite")
        if self.action_low >= self.action_high:
            raise ValueError("the lower action bound must be below the upper bound")

        reference_state = _finite_array(
            settings.ic, (3,), "ic must contain three finite values"
        ).copy()
        self.ic_dist = (settings.make_training_initial_state_sampler()
                        if training else lambda: reference_state.copy())
        if not callable(settings.state_cost_fn):
            raise TypeError("state_cost_fn must be callable")
        self.state_cost_fn = settings.state_cost_fn

        self.state: Optional[np.ndarray] = None
        self.step_count = 0
        self._initial_state_cost = 0.0
        self._terminated = False

    def reset(self, x0: Optional[np.ndarray] = None) -> np.ndarray:
        initial_state = x0 if x0 is not None else self.ic_dist()
        self.state = _finite_array(
            initial_state, (3,), "The initial Lorenz state must be finite and contain x, y, z"
        ).copy()
        self.step_count = 0
        self._initial_state_cost = float(self.state_cost_fn(self.state[0]))
        if not np.isfinite(self._initial_state_cost):
            raise ValueError("state_cost_fn must return a finite value")
        self._terminated = False
        return self.state.copy()

    def step(self, action: Union[int, float]) -> Tuple[np.ndarray, float, bool, Dict[str, bool]]:
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
            # Scale before taking the norm to prevent overflow for large finite states.
            state_scale = float(np.max(np.abs(self.state)))
            if state_scale == 0.0:
                diverged = False
            else:
                scaled_norm = float(np.linalg.norm(self.state / state_scale))
                diverged = bool(state_scale > self.divergence_threshold / scaled_norm)
        raw_state_cost = 1.0 if diverged else float(self.state_cost_fn(self.state[0]))
        if not np.isfinite(raw_state_cost):
            raise ValueError("state_cost_fn must return a finite value")
        remaining_state_terms = self.horizon - self.step_count + 1 if diverged else 1
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
        if bounds_array.shape != (3, 2) or not np.isfinite(bounds_array).all():
            raise ValueError("bounds must contain finite (low, high) pairs for x, y, and z")
        if np.any(bounds_array[:, 0] >= bounds_array[:, 1]):
            raise ValueError("each lower state bound must be below its upper bound")

        raw_bins = np.asarray([bins] * 3 if np.isscalar(bins) else bins)
        if raw_bins.shape != (3,):
            raise ValueError("bins must be an integer or three bin counts")
        if any(not isinstance(value, (int, np.integer)) or
               isinstance(value, (bool, np.bool_)) for value in raw_bins.tolist()):
            raise ValueError("bin counts must be positive integers")
        bins_array = raw_bins.astype(np.int64)
        if np.any(bins_array < 1):
            raise ValueError("bin counts must be positive integers")

        self.bounds = bounds_array
        self.bins = cast(Tuple[int, int, int], tuple(int(value) for value in bins_array))
        self._lower = bounds_array[:, 0]
        self._upper = bounds_array[:, 1]
        self._widths = (self._upper - self._lower) / bins_array

    def discretize(self, state: StateVector) -> Tuple[int, int, int]:
        state_array = _finite_array(
            state, (3,), "cannot discretize a non-finite or incorrectly shaped state"
        )
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

        n_actions = cast(int, _positive_int(n_actions, "n_actions"))

        self.learning_rate, self.discount_factor = float(learning_rate), float(discount_factor)
        numeric_rules = (
            (np.isfinite(self.learning_rate) and 0.0 < self.learning_rate <= 1.0,
             "learning_rate must be in (0, 1]"),
            (np.isfinite(self.discount_factor) and 0.0 <= self.discount_factor <= 1.0,
             "discount_factor must be in [0, 1]"),
            (np.isfinite([epsilon_min, epsilon, epsilon_decay]).all()
             and 0.0 <= epsilon_min <= epsilon <= 1.0,
             "require 0 <= epsilon_min <= epsilon <= 1"),
            (0.0 < epsilon_decay <= 1.0, "epsilon_decay must be in (0, 1]"),
        )
        for valid, message in numeric_rules:
            if not valid:
                raise ValueError(message)
        self.n_actions = int(n_actions)
        self.epsilon = float(epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.discretizer = StateDiscretizer(state_bins, state_bounds)
        self.rng = np.random.default_rng(random_seed)
        self.q_table = np.zeros(self.discretizer.bins + (self.n_actions,),
                                dtype=np.float64)

    def discretize_state(self, state: StateVector) -> Tuple[int, int, int]:
        return self.discretizer.discretize(state)

    def select_action(self, state: StateVector, training: bool = True) -> int:
        state_index = self.discretize_state(state)
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return int(np.argmax(self.q_table[state_index]))

    def update(
        self,
        state: StateVector,
        action: int,
        reward: float,
        next_state: StateVector,
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
            target = reward_value + self.discount_factor * float(np.max(self.q_table[next_index]))
        td_error = target - current_value
        self.q_table[table_index] += self.learning_rate * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon


EvaluationResults = Dict[str, Union[List[float], List[int], List[bool], List[np.ndarray]]]


@dataclass
class _EpisodeResult:
    reward: float
    length: int
    diverged: bool
    actions: np.ndarray
    controls: np.ndarray
    trajectory: np.ndarray
    visited_bins: set[Tuple[int, int, int]]
    updated_pairs: set[Tuple[int, int, int, int]]
    abs_td_errors: List[float]


def _validate_discrete_pair(env: LorenzEnvEuler, agent: QLearningAgent) -> None:
    if env.action_type != "discrete":
        raise ValueError("Tabular Q-learning requires env.action_type='discrete'")
    if len(env.actions) != agent.n_actions:
        raise ValueError("agent.n_actions must match the environment's action bins")


def _rolling_mean(values: List[float], window: int = 100) -> List[float]:
    return [float(np.mean(values[max(0, end - window):end]))
            for end in range(1, len(values) + 1)]


def _run_episode(
    env: LorenzEnvEuler,
    agent: QLearningAgent,
    max_steps: Optional[int],
    *,
    training: bool,
    x0: Optional[np.ndarray] = None,
) -> _EpisodeResult:
    current_state = env.reset(x0=x0)
    states = [] if training else [current_state.copy()]
    actions: List[int] = []
    controls: List[float] = []
    visited_bins = {agent.discretize_state(current_state)} if training else set()
    updated_pairs: set[Tuple[int, int, int, int]] = set()
    abs_td_errors: List[float] = []
    total_reward, steps, diverged = 0.0, 0, False
    done = False

    while not done and (max_steps is None or steps < max_steps):
        action = agent.select_action(current_state, training=training)
        next_state, reward, done, info = env.step(action)
        if training:
            state_index = agent.discretize_state(current_state)
            td_error = agent.update(
                current_state, action, reward, next_state, done
            )
            abs_td_errors.append(abs(td_error))
            updated_pairs.add(state_index + (action,))
            if np.isfinite(next_state).all():
                visited_bins.add(agent.discretize_state(next_state))
        else:
            actions.append(action)
            controls.append(float(env.actions[action]))
            states.append(next_state.copy())
        current_state = next_state
        total_reward += reward
        steps += 1
        diverged = diverged or bool(info.get("diverged", False))

    return _EpisodeResult(float(total_reward), steps, diverged,
                          np.asarray(actions, dtype=np.int64),
                          np.asarray(controls, dtype=np.float64),
                          np.asarray(states, dtype=np.float64),
                          visited_bins, updated_pairs, abs_td_errors)


def train_q_learning(
    env: LorenzEnvEuler,
    agent: QLearningAgent,
    num_episodes: int,
    max_steps: Optional[int] = None,
    print_every: int = EXPERIMENT_DEFAULTS.print_every,
    total_episodes: Optional[int] = None,
    start_episode: int = 0,
) -> TrainingHistory:
    _validate_discrete_pair(env, agent)
    num_episodes = cast(int, _positive_int(num_episodes, "num_episodes"))
    max_steps = _positive_int(max_steps, "max_steps", optional=True)
    print_every = cast(int, _positive_int(print_every, "print_every"))
    if total_episodes is None:
        total_episodes = num_episodes

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    epsilons: List[float] = []
    diverged: List[bool] = []
    unique_state_bins_visited: List[int] = []
    q_table_coverages: List[float] = []
    mean_abs_td_errors: List[float] = []
    updated_state_action_pairs = set()

    for episode in range(1, num_episodes + 1):
        result = _run_episode(env, agent, max_steps, training=True)
        updated_state_action_pairs.update(result.updated_pairs)
        agent.decay_epsilon()
        episode_rewards.append(result.reward)
        episode_lengths.append(result.length)
        epsilons.append(agent.epsilon)
        diverged.append(result.diverged)
        unique_state_bins_visited.append(len(result.visited_bins))
        q_table_coverages.append(100.0 * len(updated_state_action_pairs) /
                                 agent.q_table.size)
        mean_abs_td_errors.append(float(np.mean(result.abs_td_errors))
                                  if result.abs_td_errors else 0.0)

        if result.diverged:
            print(f"ALERT: Lorenz system diverged during episode {episode}.")

        if episode % print_every == 0:
            recent_rewards = episode_rewards[-print_every:]
            rolling_mean_reward = float(np.mean(episode_rewards[-100:]))
            global_episode = start_episode + episode
            print(
                f"Episode {global_episode}/{total_episodes} | "
                f"average reward: {np.mean(recent_rewards):.4f} | "
                f"Rolling mean reward: {rolling_mean_reward:.4f} | "
                f"epsilon: {agent.epsilon:.4f}"
            )
        
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "epsilons": epsilons,
        "diverged": diverged,
        "unique_state_bins_visited": unique_state_bins_visited,
        "q_table_coverages": q_table_coverages,
        "mean_abs_td_errors": mean_abs_td_errors,
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
    num_episodes = cast(int, _positive_int(num_episodes, "num_episodes"))
    if x0 is not None and initial_states is not None:
        raise ValueError("provide either x0 or initial_states, not both")
    episode_initial_states: Optional[np.ndarray] = None
    if initial_states is not None:
        episode_initial_states = np.asarray(initial_states, dtype=np.float64)
        if episode_initial_states.shape != (num_episodes, 3):
            raise ValueError("initial_states must have shape (num_episodes, 3)")
        if not np.isfinite(episode_initial_states).all():
            raise ValueError("initial_states must contain only finite values")
    max_steps = _positive_int(max_steps, "max_steps", optional=True)

    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    all_actions: List[np.ndarray] = []
    all_controls: List[np.ndarray] = []
    trajectories: List[np.ndarray] = []
    diverged: List[bool] = []

    for episode_index in range(num_episodes):
        episode_x0 = (episode_initial_states[episode_index]
                      if episode_initial_states is not None else x0)
        result = _run_episode(env, agent, max_steps, training=False, x0=episode_x0)
        episode_rewards.append(result.reward)
        episode_lengths.append(result.length)
        all_actions.append(result.actions)
        all_controls.append(result.controls)
        trajectories.append(result.trajectory)
        diverged.append(result.diverged)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "actions": all_actions,
        "control_values": all_controls,
        "trajectories": trajectories,
        "diverged": diverged,
    }


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
) -> Tuple[TrainingHistory, Dict[str, List[Union[int, float]]], EvaluationResults]:
    evaluation_interval = cast(int, _positive_int(evaluation_interval,
                                                  "evaluation_interval"))

    checkpoint_episodes = list(range(evaluation_interval, num_episodes, evaluation_interval))
    checkpoint_episodes.append(num_episodes)
    history_parts: List[TrainingHistory] = []
    evaluations: List[EvaluationResults] = []
    previous_episode = 0

    for episode in checkpoint_episodes:
        history_parts.append(train_q_learning(
            training_env, agent, episode - previous_episode, max_steps=max_steps,
            total_episodes=num_episodes, start_episode=previous_episode
        ))
        evaluations.append(
            evaluate_q_learning(
                evaluation_env,
                agent,
                evaluation_episodes,
                max_steps=evaluation_max_steps,
                initial_states=evaluation_initial_states,
            )
        )
        previous_episode = episode

    history = cast(TrainingHistory, {
        key: sum((part[key] for part in history_parts), [])
        for key in history_parts[0] if key != "rolling_mean_rewards"
    })
    history["rolling_mean_rewards"] = _rolling_mean(history["episode_rewards"])

    rewards = [np.asarray(result["episode_rewards"], dtype=float) for result in evaluations]
    checkpoints = {
        "episodes": checkpoint_episodes,
        "mean_rewards": [float(np.mean(values)) for values in rewards],
        "reward_standard_deviations": [float(np.std(values)) for values in rewards],
        "divergence_rates": [float(np.mean(result["diverged"])) for result in evaluations],
    }
    return history, checkpoints, evaluations[-1]



def run_q_learning(settings: ExperimentDefaults = EXPERIMENT_DEFAULTS) -> Dict[str, Any]:
    reference_state = np.asarray(settings.ic, dtype=np.float64)
    training_env = LorenzEnvEuler(settings, training=True, controlled=True)
    agent = QLearningAgent(
        n_actions=len(training_env.actions),
        learning_rate=settings.learning_rate,
        discount_factor=settings.discount_factor,
        epsilon=settings.epsilon,
        epsilon_decay=settings.epsilon_decay,
        epsilon_min=settings.epsilon_min,
        state_bins=settings.state_bins,
        random_seed=settings.exploration_seed,
    )
    evaluation_env = LorenzEnvEuler(settings, training=False, controlled=True)
    evaluation_starts = settings.make_evaluation_initial_states()
    history, checkpoints, evaluation = train_q_learning_with_evaluation(
        training_env,
        evaluation_env,
        agent,
        settings.episodes,
        settings.evaluation_interval,
        settings.eval_episodes,
        max_steps=settings.max_steps,
        evaluation_max_steps=settings.eval_max_steps,
        evaluation_initial_states=evaluation_starts.tolist(),
    )
    return {
        "history": history,
        "checkpoints": checkpoints,
        "evaluation": evaluation,
        "q_table": agent.q_table.copy(),
        "actions": training_env.actions.copy(),
        "state_bounds": agent.discretizer.bounds.copy(),
        "state_bins": agent.discretizer.bins,
        "reference_state": reference_state,
        "final_epsilon": agent.epsilon,
    }


def save_q_learning_run(run: Dict[str, Any], path: Union[str, Path] = RUN_OUTPUT_PATH) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output_file:
        pickle.dump(run, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def load_q_learning_run(path: Union[str, Path] = RUN_OUTPUT_PATH) -> Dict[str, Any]:
    with Path(path).open("rb") as input_file:
        run = pickle.load(input_file)
    if not isinstance(run, dict):
        raise ValueError("Q-learning run file must contain a result dictionary")
    return run


if __name__ == "__main__":
    run = run_q_learning()
    save_q_learning_run(run)
