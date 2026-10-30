from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
import gzip
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

# Use bundled fonts to avoid blocking macOS system font discovery on startup.
if sys.platform == "darwin":
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "DeepQLearning.py requires PyTorch; install it in the project environment "
        "before running the Double DQN controller."
    ) from exc

import QLearning as qlearning
import PlotQLearning as shared_plots


DEEP_RUN_FILENAME = "deep_q_learning_run.pkl.gz"
DEEP_RUNS_DIRECTORY = Path(__file__).with_name("q_learning_plots") / "Saved_Plots" / "DQN"


@dataclass(frozen=True)
class DeepExperimentDefaults:
    """Complete Double DQN configuration, including shared experiment values."""

    episodes: int = 800
    eval_episodes: int = 40
    evaluation_interval: int = 50
    max_steps: Optional[int] = None
    eval_max_steps: Optional[int] = None
    print_every: int = 50
    training_plot_samples: int = 20

    ic: Tuple[float, float, float] = (0.0, 1.0, 1.05)
    training_ic_seed: int = 0
    training_ic_perturbation: float = 0.01
    exploration_seed: int = 0
    evaluation_seed: int = 0
    evaluation_ic_perturbation: float = 0.01
    network_seed: int = 0
    use_attractor_initial_state_ensemble: bool = True
    attractor_initial_states_per_lobe: int = 20
    attractor_burn_in_steps: int = 2_000
    attractor_sample_spacing: int = 25

    training_lyapunov_times: float = 50.0
    evaluation_lyapunov_times: float = 50.0
    action_low: float = -2.5
    action_high: float = 2.5
    u_ref: float = 2.5
    action_bins: int = 9
    discount_factor: float = 0.991
    control_cost: float = 0.01
    regularized: bool = True
    potential_shaping_weight: float = 0.005
    divergence_threshold: float = np.inf
    state_bounds: Tuple[Tuple[float, float], ...] = qlearning.DEFAULT_STATE_BOUNDS

    hidden_layers: Tuple[int, int] = (128, 128)
    activation: str = "tanh"
    network_learning_rate: float = 0.0003
    batch_size: int = 128
    replay_buffer_size: int = 200_000
    minimum_replay_size: int = 10_000
    target_update_interval: int = 1_000
    gradient_clip_norm: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.02
    epsilon_decay_steps: int = 500_000
    device_mode: str = "auto"

    def __post_init__(self) -> None:
        positive_integers = (
            "episodes", "eval_episodes", "evaluation_interval", "print_every",
            "training_plot_samples", "action_bins", "batch_size",
            "replay_buffer_size", "minimum_replay_size", "target_update_interval",
            "epsilon_decay_steps", "attractor_initial_states_per_lobe",
            "attractor_burn_in_steps", "attractor_sample_spacing",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        for name in ("max_steps", "eval_max_steps"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value < 1
            ):
                raise ValueError(f"{name} must be None or a positive integer")
        if self.minimum_replay_size > self.replay_buffer_size:
            raise ValueError("minimum_replay_size cannot exceed replay_buffer_size")
        if self.batch_size > self.replay_buffer_size:
            raise ValueError("batch_size cannot exceed replay_buffer_size")
        if len(self.hidden_layers) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in self.hidden_layers
        ):
            raise ValueError("hidden_layers must contain two positive integers")
        if self.activation not in {"relu", "tanh"}:
            raise ValueError("activation must be relu or tanh")
        if self.action_bins < 2:
            raise ValueError("action_bins must be at least 2")
        finite_values = (
            self.action_low, self.action_high, self.u_ref, self.discount_factor,
            self.control_cost, self.potential_shaping_weight,
            self.network_learning_rate, self.gradient_clip_norm,
            self.epsilon_start, self.epsilon_end,
            self.training_lyapunov_times, self.evaluation_lyapunov_times,
        )
        if not np.isfinite(finite_values).all():
            raise ValueError("numeric settings must be finite")
        if self.action_low >= self.action_high:
            raise ValueError("action_low must be below action_high")
        if self.u_ref <= 0.0:
            raise ValueError("u_ref must be positive")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be in [0, 1]")
        if self.control_cost < 0.0 or self.potential_shaping_weight < 0.0:
            raise ValueError("cost and shaping weights must be nonnegative")
        if self.network_learning_rate <= 0.0 or self.gradient_clip_norm <= 0.0:
            raise ValueError("learning rate and gradient clip norm must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("require 0 <= epsilon_end <= epsilon_start <= 1")
        bounds = np.asarray(self.state_bounds, dtype=np.float64)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all():
            raise ValueError("state_bounds must contain three finite (low, high) pairs")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("each state lower bound must be below its upper bound")
        if self.device_mode not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device_mode must be one of auto, cpu, cuda, or mps")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def environment_settings(self) -> qlearning.ExperimentDefaults:
        """Create the shared settings object consumed by the original environment."""
        shared_names = {field.name for field in fields(qlearning.ExperimentDefaults)}
        values = {
            name: getattr(self, name)
            for name in shared_names
            if hasattr(self, name) and name != "state_cost_fn"
        }
        values["state_cost_fn"] = qlearning.default_state_cost_fn
        return replace(qlearning.EXPERIMENT_DEFAULTS, **values)


DEEP_EXPERIMENT_DEFAULTS = DeepExperimentDefaults()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(mode: str = "auto") -> torch.device:
    if mode == "cpu":
        return torch.device("cpu")
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if mode == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class StateNormalizer:
    """Map finite Lorenz coordinates from the existing bounds into [-1, 1]."""

    def __init__(self, bounds: Sequence[Tuple[float, float]]):
        values = np.asarray(bounds, dtype=np.float64)
        if values.shape != (3, 2) or not np.isfinite(values).all():
            raise ValueError("state bounds must contain three finite pairs")
        if np.any(values[:, 0] >= values[:, 1]):
            raise ValueError("state bound lows must be below highs")
        self.bounds = values.copy()
        self.lower = values[:, 0]
        self.upper = values[:, 1]
        self.center = (self.lower + self.upper) / 2.0
        self.half_width = (self.upper - self.lower) / 2.0

    def normalize(self, state: Union[Sequence[float], np.ndarray]) -> np.ndarray:
        array = np.asarray(state, dtype=np.float64)
        if array.shape[-1:] != (3,) or not np.isfinite(array).all():
            raise ValueError("state must end in three finite Lorenz coordinates")
        clipped = np.clip(array, self.lower, self.upper)
        return ((clipped - self.center) / self.half_width).astype(np.float32)


class QNetwork(nn.Module):
    def __init__(
        self, n_actions: int, hidden_layers: Tuple[int, int] = (128, 128),
        activation: str = "tanh",
    ):
        super().__init__()
        activation_class = {"relu": nn.ReLU, "tanh": nn.Tanh}[activation]
        self.layers = nn.Sequential(
            nn.Linear(3, hidden_layers[0]),
            activation_class(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            activation_class(),
            nn.Linear(hidden_layers[1], n_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


class ReplayBuffer:
    """Fixed-size, preallocated replay memory with deterministic sampling."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.states = np.empty((capacity, 3), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, 3), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        if batch_size > self.size:
            raise ValueError("cannot sample more transitions than the buffer contains")
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return (
            self.states[indices], self.actions[indices], self.rewards[indices],
            self.next_states[indices], self.dones[indices],
        )


class DoubleDQNAgent:
    """Online/target Double DQN pair over the continuous Lorenz state."""

    def __init__(self, settings: DeepExperimentDefaults, device: torch.device):
        self.settings = settings
        self.device = device
        self.n_actions = settings.action_bins
        actions = np.linspace(settings.action_low, settings.action_high, self.n_actions)
        self.preferred_action = int(np.argmin(np.abs(actions)))
        self.action_preference = sorted(range(self.n_actions), key=lambda i: (abs(actions[i]), i))
        self.normalizer = StateNormalizer(settings.state_bounds)
        self.online_network = QNetwork(self.n_actions, settings.hidden_layers, settings.activation).to(device)
        self.target_network = QNetwork(self.n_actions, settings.hidden_layers, settings.activation).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=settings.network_learning_rate
        )
        self.replay = ReplayBuffer(settings.replay_buffer_size, settings.exploration_seed)
        self.rng = np.random.default_rng(settings.exploration_seed)
        self.global_step = 0
        self.gradient_steps = 0

    @property
    def epsilon(self) -> float:
        progress = min(1.0, self.global_step / self.settings.epsilon_decay_steps)
        return float(
            self.settings.epsilon_start
            + progress * (self.settings.epsilon_end - self.settings.epsilon_start)
        )

    def _preferred_argmax(self, values: torch.Tensor) -> torch.Tensor:
        """Argmax with deterministic ties biased toward the zero-forcing action."""
        if values.ndim == 1:
            values = values.unsqueeze(0)
            squeeze = True
        elif values.ndim == 2:
            squeeze = False
        else:
            raise ValueError("Q-values must have one or two dimensions")
        maxima = values.max(dim=1, keepdim=True).values
        tied = values == maxima
        result = torch.full(
            (values.shape[0],), self.preferred_action, dtype=torch.long, device=values.device
        )
        unassigned = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
        for index in self.action_preference:
            choose = unassigned & tied[:, index]
            result[choose] = index
            unassigned &= ~choose
        return result[0] if squeeze else result

    def select_action(self, state: np.ndarray, *, training: bool) -> int:
        normalized = self.normalizer.normalize(state)
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            state_tensor = torch.as_tensor(normalized, device=self.device).unsqueeze(0)
            values = self.online_network(state_tensor)
            return int(self._preferred_argmax(values).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")
        # Divergence may produce a nonfinite terminal observation. Its value is
        # never bootstrapped, but replay still needs a finite network input.
        normalized_next = (
            np.zeros(3, dtype=np.float32)
            if done and not np.isfinite(next_state).all()
            else self.normalizer.normalize(next_state)
        )
        self.replay.add(
            self.normalizer.normalize(state), action, float(reward),
            normalized_next, done,
        )
        self.global_step += 1

    def train_step(self) -> Optional[Dict[str, float]]:
        if len(self.replay) < max(self.settings.minimum_replay_size, self.settings.batch_size):
            return None
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.settings.batch_size
        )
        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)

        predicted_all = self.online_network(states_t)
        predicted = predicted_all.gather(1, actions_t).squeeze(1)
        with torch.no_grad():
            # Double DQN: online selects, target evaluates.
            next_online = self.online_network(next_states_t)
            next_actions = self._preferred_argmax(next_online).unsqueeze(1)
            next_target = self.target_network(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + self.settings.discount_factor * (1.0 - dones_t) * next_target

        loss = F.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.settings.gradient_clip_norm
        )
        self.optimizer.step()
        self.gradient_steps += 1
        if self.global_step % self.settings.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return {
            "loss": float(loss.detach().cpu()),
            "mean_q": float(predicted_all.detach().mean().cpu()),
            "max_q": float(predicted_all.detach().max(dim=1).values.mean().cpu()),
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
        }

    def checkpoint(self, episode: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "episode": int(episode),
            "online_network_state": _cpu_state_dict(self.online_network.state_dict()),
            "target_network_state": _cpu_state_dict(self.target_network.state_dict()),
            "optimizer_state": _optimizer_to_cpu(self.optimizer.state_dict()),
            "global_step": self.global_step,
            "gradient_steps": self.gradient_steps,
            "epsilon": self.epsilon,
            "agent_rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "replay_rng_state": copy.deepcopy(self.replay.rng.bit_generator.state),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state().cpu(),
        }
        if torch.cuda.is_available():
            result["cuda_rng_states"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["mps_rng_state"] = torch.mps.get_rng_state().cpu()
        return result

    def load_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self.online_network.load_state_dict(checkpoint["online_network_state"])
        self.target_network.load_state_dict(checkpoint["target_network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        _move_optimizer(self.optimizer, self.device)
        self.global_step = int(checkpoint["global_step"])
        self.gradient_steps = int(checkpoint["gradient_steps"])
        self.rng.bit_generator.state = copy.deepcopy(checkpoint["agent_rng_state"])
        self.replay.rng.bit_generator.state = copy.deepcopy(checkpoint["replay_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if "cuda_rng_states" in checkpoint and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        if "mps_rng_state" in checkpoint and torch.backends.mps.is_available():
            torch.mps.set_rng_state(checkpoint["mps_rng_state"].cpu())


def _cpu_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state_dict.items()}


def _optimizer_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _optimizer_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_optimizer_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_optimizer_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _empty_history() -> Dict[str, List[Any]]:
    return {
        "episode_rewards": [], "episode_lengths": [], "epsilons": [],
        "diverged": [], "first_target_steps": [], "target_steps": [],
        "average_rewards": [], "rolling_mean_rewards": [],
        "sampled_episodes": [], "sampled_trajectories": [],
        "sampled_control_values": [], "losses": [], "mean_q_values": [],
        "max_q_values": [], "gradient_norms": [], "episode_mean_losses": [],
        "episode_mean_q_values": [], "episode_max_q_values": [],
        "replay_buffer_sizes": [], "global_steps": [],
    }


def _evaluation_metrics(evaluation: Mapping[str, Any], u_ref: float, runtime: float) -> Dict[str, Any]:
    controls_list = [np.asarray(values, dtype=np.float64) for values in evaluation["control_values"]]
    controls = np.concatenate(controls_list) if controls_list else np.empty(0)
    changes = sum(int(np.count_nonzero(values[1:] != values[:-1])) for values in controls_list)
    opportunities = sum(max(0, len(values) - 1) for values in controls_list)
    per_trial = [float(value) for value in evaluation["episode_negative_lobe_percentages"]]
    successful = int(np.count_nonzero(np.asarray(per_trial) >= 51.0))
    divergence_count = int(np.count_nonzero(evaluation["diverged"]))
    return {
        "episode_negative_lobe_percentages": per_trial,
        "successful_trial_count": successful,
        "successful_trial_percentage": 100.0 * successful / len(per_trial) if per_trial else 0.0,
        "aggregate_negative_lobe_occupancy": float(evaluation["negative_lobe_percentage"]),
        "mean_absolute_forcing": float(np.mean(np.abs(controls))) if controls.size else 0.0,
        "rms_forcing": float(np.sqrt(np.mean(controls ** 2))) if controls.size else 0.0,
        "normalized_control_effort": float(np.mean((controls / u_ref) ** 2)) if controls.size else 0.0,
        "zero_forcing_percentage": 100.0 * float(np.mean(np.isclose(controls, 0.0))) if controls.size else 0.0,
        "action_change_count": changes,
        "action_change_rate": changes / opportunities if opportunities else 0.0,
        "divergence_count": divergence_count,
        "divergence_rate": divergence_count / len(per_trial) if per_trial else 0.0,
        "mean_evaluation_reward": float(np.mean(evaluation["episode_rewards"])),
        "runtime_seconds": float(runtime),
    }


def evaluate_deep_q_learning(
    env: qlearning.LorenzEnvEuler,
    agent: DoubleDQNAgent,
    initial_states: Sequence[Sequence[float]],
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    starts = np.asarray(initial_states, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1] != 3 or starts.shape[0] < 1:
        raise ValueError("initial_states must have shape (n, 3)")
    if not np.isfinite(starts).all():
        raise ValueError("initial_states must contain finite values")
    started = time.perf_counter()
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    actions: List[np.ndarray] = []
    controls: List[np.ndarray] = []
    trajectories: List[np.ndarray] = []
    diverged: List[bool] = []
    first_target_steps: List[Optional[int]] = []
    target_steps: List[int] = []
    for start in starts:
        state = env.reset(x0=start)
        states = [state.copy()]
        trial_actions: List[int] = []
        trial_controls: List[float] = []
        total_reward = 0.0
        trial_diverged = False
        first_target: Optional[int] = None
        target_count = 0
        steps = 0
        done = False
        while not done and (max_steps is None or steps < max_steps):
            action = agent.select_action(state, training=False)
            next_state, environment_reward, done, info = env.step(action)
            reward = qlearning.potential_shaped_reward(
                environment_reward, state, next_state, agent.settings.discount_factor,
                env.potential_shaping_weight, done=done,
            )
            steps += 1
            total_reward += reward
            trial_actions.append(action)
            trial_controls.append(float(env.actions[action]))
            states.append(next_state.copy())
            if qlearning.fixed_point_check(next_state):
                first_target = steps if first_target is None else first_target
                target_count += 1
            trial_diverged = trial_diverged or bool(info.get("diverged", False))
            state = next_state
        episode_rewards.append(float(total_reward))
        episode_lengths.append(steps)
        actions.append(np.asarray(trial_actions, dtype=np.int64))
        controls.append(np.asarray(trial_controls, dtype=np.float64))
        trajectories.append(np.asarray(states, dtype=np.float64))
        diverged.append(trial_diverged)
        first_target_steps.append(first_target)
        target_steps.append(target_count)
    negative, positive = qlearning.lobe_time_percentages(trajectories)
    per_trial = [qlearning.lobe_time_percentages([trajectory])[0] for trajectory in trajectories]
    result: Dict[str, Any] = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "actions": actions,
        "control_values": controls,
        "trajectories": trajectories,
        "diverged": diverged,
        "first_target_steps": first_target_steps,
        "target_steps": target_steps,
        "negative_lobe_percentage": negative,
        "positive_lobe_percentage": positive,
        "episode_negative_lobe_percentages": per_trial,
        "negative_lobe_success_rate": 100.0 * float(np.mean(np.asarray(per_trial) >= 51.0)),
    }
    result["metrics"] = _evaluation_metrics(result, env.u_ref, time.perf_counter() - started)
    return result


def checkpoint_is_eligible(metrics: Mapping[str, Any], required_successes: int = 21) -> bool:
    return bool(
        metrics["divergence_count"] == 0
        and metrics["successful_trial_count"] >= required_successes
    )


def checkpoint_rank(metrics: Mapping[str, Any]) -> Tuple[float, float, int, float]:
    """Smaller tuples are better, following the requested lexicographic rule."""
    return (
        float(metrics["mean_absolute_forcing"]),
        float(metrics["rms_forcing"]),
        -int(metrics["successful_trial_count"]),
        -float(metrics["zero_forcing_percentage"]),
    )


def _checkpoint_summary(evaluation: Mapping[str, Any], episode: int) -> Dict[str, Any]:
    metrics = dict(evaluation["metrics"])
    return {"episode": episode, **metrics, "eligible": checkpoint_is_eligible(metrics)}


def _write_torch_checkpoint(checkpoint: Mapping[str, Any], path: Path, settings: DeepExperimentDefaults) -> None:
    payload = {"settings": settings.as_dict(), "checkpoint": dict(checkpoint)}
    torch.save(payload, path)


def run_deep_q_learning(
    settings: DeepExperimentDefaults = DEEP_EXPERIMENT_DEFAULTS,
    *,
    output_directory: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Train, evaluate, checkpoint, and return a complete Double DQN run."""
    seed_everything(settings.network_seed)
    device = select_device(settings.device_mode)
    shared = settings.environment_settings()
    training_env = qlearning.LorenzEnvEuler(shared, training=True, controlled=True)
    evaluation_env = qlearning.LorenzEnvEuler(shared, training=False, controlled=True)
    evaluation_starts = shared.make_evaluation_initial_states()
    agent = DoubleDQNAgent(settings, device)
    history = _empty_history()
    checkpoint_episodes: List[int] = []
    evaluations: List[Dict[str, Any]] = []
    checkpoint_records: List[Dict[str, Any]] = []
    checkpoint_summaries: List[Dict[str, Any]] = []
    output_path = Path(output_directory) if output_directory is not None else None
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=False)
        with (output_path / "parameters.json").open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(settings.as_dict()), stream, indent=2, allow_nan=False)

    sample_count = min(settings.training_plot_samples, settings.episodes)
    sampled_episode_set = set(
        np.rint(np.linspace(1, settings.episodes, sample_count)).astype(int).tolist()
    )
    sampled_episode_set.update(
        range(settings.evaluation_interval, settings.episodes + 1, settings.evaluation_interval)
    )
    sampled_episode_set.add(settings.episodes)
    training_started = time.perf_counter()

    for episode in range(1, settings.episodes + 1):
        state = training_env.reset()
        keep_rollout = episode in sampled_episode_set
        states = [state.copy()] if keep_rollout else []
        controls: List[float] = []
        total_reward = 0.0
        steps = 0
        done = False
        episode_diverged = False
        first_target: Optional[int] = None
        target_count = 0
        episode_losses: List[float] = []
        episode_means: List[float] = []
        episode_maxima: List[float] = []
        while not done and (settings.max_steps is None or steps < settings.max_steps):
            action = agent.select_action(state, training=True)
            next_state, environment_reward, done, info = training_env.step(action)
            reward = qlearning.potential_shaped_reward(
                environment_reward, state, next_state, settings.discount_factor,
                training_env.potential_shaping_weight, done=done,
            )
            agent.observe(state, action, reward, next_state, done)
            training_stats = agent.train_step()
            if training_stats is not None:
                history["losses"].append(training_stats["loss"])
                history["mean_q_values"].append(training_stats["mean_q"])
                history["max_q_values"].append(training_stats["max_q"])
                history["gradient_norms"].append(training_stats["gradient_norm"])
                episode_losses.append(training_stats["loss"])
                episode_means.append(training_stats["mean_q"])
                episode_maxima.append(training_stats["max_q"])
            steps += 1
            total_reward += reward
            if keep_rollout:
                states.append(next_state.copy())
                controls.append(float(training_env.actions[action]))
            if qlearning.fixed_point_check(next_state):
                first_target = steps if first_target is None else first_target
                target_count += 1
            episode_diverged = episode_diverged or bool(info.get("diverged", False))
            state = next_state

        history["episode_rewards"].append(float(total_reward))
        history["episode_lengths"].append(steps)
        history["epsilons"].append(agent.epsilon)
        history["diverged"].append(episode_diverged)
        history["first_target_steps"].append(first_target)
        history["target_steps"].append(target_count)
        history["episode_mean_losses"].append(float(np.mean(episode_losses)) if episode_losses else math.nan)
        history["episode_mean_q_values"].append(float(np.mean(episode_means)) if episode_means else math.nan)
        history["episode_max_q_values"].append(float(np.mean(episode_maxima)) if episode_maxima else math.nan)
        history["replay_buffer_sizes"].append(len(agent.replay))
        history["global_steps"].append(agent.global_step)
        if keep_rollout:
            history["sampled_episodes"].append(episode)
            history["sampled_trajectories"].append(np.asarray(states, dtype=np.float64))
            history["sampled_control_values"].append(np.asarray(controls, dtype=np.float64))

        if episode % settings.print_every == 0:
            recent = history["episode_rewards"][-settings.print_every:]
            print(
                f"Episode {episode}/{settings.episodes} | average reward: "
                f"{np.mean(recent):.4f} | epsilon: {agent.epsilon:.4f} | "
                f"replay: {len(agent.replay)}"
            )

        if episode % settings.evaluation_interval == 0 or episode == settings.episodes:
            evaluation = evaluate_deep_q_learning(
                evaluation_env, agent, evaluation_starts, settings.eval_max_steps
            )
            checkpoint = agent.checkpoint(episode)
            checkpoint_episodes.append(episode)
            evaluations.append(evaluation)
            checkpoint_records.append(checkpoint)
            checkpoint_summaries.append(_checkpoint_summary(evaluation, episode))
            if output_path is not None:
                _write_torch_checkpoint(
                    checkpoint, output_path / f"checkpoint_episode_{episode:04d}.pt", settings
                )

    history["average_rewards"] = qlearning.rolling_average(
        cast(List[float], history["episode_rewards"])
    )
    history["rolling_mean_rewards"] = list(history["average_rewards"])
    eligible_indices = [
        index for index, summary in enumerate(checkpoint_summaries) if summary["eligible"]
    ]
    best_index = (
        min(eligible_indices, key=lambda index: checkpoint_rank(checkpoint_summaries[index]))
        if eligible_indices else None
    )
    final_checkpoint = checkpoint_records[-1]
    best_checkpoint = checkpoint_records[best_index] if best_index is not None else None
    final_three = checkpoint_summaries[-3:]
    stable_final_three = len(final_three) == 3 and all(item["eligible"] for item in final_three)
    total_runtime = time.perf_counter() - training_started
    checkpoints = {
        "episodes": checkpoint_episodes,
        "evaluations": evaluations,
        "summaries": checkpoint_summaries,
        "network_checkpoints": checkpoint_records,
        "mean_rewards": [item["mean_evaluation_reward"] for item in checkpoint_summaries],
        "divergence_rates": [item["divergence_rate"] for item in checkpoint_summaries],
        "mean_control_efforts": [item["normalized_control_effort"] for item in checkpoint_summaries],
        "epsilons": [record["epsilon"] for record in checkpoint_records],
        "best_checkpoint_index": best_index,
        "best_checkpoint_episode": checkpoint_episodes[best_index] if best_index is not None else None,
        "criterion_satisfied_final_three": stable_final_three,
    }
    run: Dict[str, Any] = {
        "controller": "Double DQN",
        "history": history,
        "checkpoints": checkpoints,
        "evaluation": evaluations[-1],
        "actions": training_env.actions.copy(),
        "state_bounds": np.asarray(settings.state_bounds, dtype=np.float64),
        "reference_state": np.asarray(settings.ic, dtype=np.float64),
        "evaluation_initial_states": evaluation_starts.copy(),
        "final_epsilon": agent.epsilon,
        "settings": settings.as_dict(),
        "seeds": {
            "training_ic_seed": settings.training_ic_seed,
            "exploration_seed": settings.exploration_seed,
            "evaluation_seed": settings.evaluation_seed,
            "network_seed": settings.network_seed,
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "runtime_seconds": total_runtime,
        "final_checkpoint": final_checkpoint,
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_episode": checkpoints["best_checkpoint_episode"],
        "criterion_satisfied_final_three": stable_final_three,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if output_path is not None:
        _write_torch_checkpoint(final_checkpoint, output_path / "final_model.pt", settings)
        if best_checkpoint is not None:
            _write_torch_checkpoint(best_checkpoint, output_path / "best_model.pt", settings)
    return run


def restore_agent(
    checkpoint: Mapping[str, Any],
    settings: DeepExperimentDefaults,
    *,
    device_mode: str = "cpu",
) -> DoubleDQNAgent:
    """Restore model, optimizer, and RNGs; replay contents are not checkpointed."""
    device = select_device(device_mode)
    agent = DoubleDQNAgent(settings, device)
    agent.load_checkpoint(checkpoint)
    return agent


def save_deep_q_learning_run(run: Mapping[str, Any], path: Union[str, Path]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if output.suffix == ".gz" else Path.open
    with opener(output, "wb") as stream:
        pickle.dump(dict(run), stream, protocol=pickle.HIGHEST_PROTOCOL)
    return output


def load_deep_q_learning_run(path: Union[str, Path]) -> Dict[str, Any]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else Path.open
    with opener(source, "rb") as stream:
        run = pickle.load(stream)
    if not isinstance(run, dict) or run.get("controller") != "Double DQN":
        raise ValueError("run file does not contain a Double DQN result dictionary")
    # Archives made before activation was recorded used ReLU.
    run["settings"].setdefault("activation", "relu")
    return run


def compare_with_tabular_episode(
    deep_run: Mapping[str, Any],
    tabular_path: Union[str, Path],
    checkpoint_episode: int = 650,
) -> Dict[str, Any]:
    """Evaluate the saved tabular checkpoint on the DQN states and horizon."""
    tabular = qlearning.load_q_learning_run(tabular_path)
    episodes = list(tabular["checkpoints"]["episodes"])
    if checkpoint_episode not in episodes:
        raise ValueError(f"tabular run has no episode-{checkpoint_episode} checkpoint")
    index = episodes.index(checkpoint_episode)
    actions = np.asarray(deep_run["actions"], dtype=np.float64)
    tabular_actions = np.asarray(tabular["actions"], dtype=np.float64)
    if actions.shape != tabular_actions.shape or not np.allclose(actions, tabular_actions):
        raise ValueError("DQN and tabular saved runs do not use identical actions")
    settings = DeepExperimentDefaults(**dict(deep_run["settings"]))
    shared = settings.environment_settings()
    env = qlearning.LorenzEnvEuler(shared, training=False, controlled=True)
    table = np.asarray(tabular["checkpoints"]["q_tables"][index], dtype=np.float64)
    agent = qlearning.QLearningAgent(
        n_actions=len(actions), discount_factor=settings.discount_factor,
        epsilon=0.0, epsilon_min=0.0,
        state_bins=tuple(int(value) for value in tabular["state_bins"]),
        state_bounds=np.asarray(tabular["state_bounds"]),
        preferred_action=int(np.argmin(np.abs(actions))), random_seed=0,
    )
    if agent.q_table.shape != table.shape:
        raise ValueError("episode-650 Q-table shape is inconsistent with saved metadata")
    agent.q_table[...] = table
    started = time.perf_counter()
    evaluation = qlearning.evaluate_q_learning(
        env, agent, len(deep_run["evaluation_initial_states"]),
        max_steps=settings.eval_max_steps,
        initial_states=deep_run["evaluation_initial_states"],
    )
    tabular_metrics = _evaluation_metrics(
        evaluation, settings.u_ref, time.perf_counter() - started
    )
    deep_metrics = dict(deep_run["evaluation"]["metrics"])
    keys = (
        "successful_trial_count", "aggregate_negative_lobe_occupancy",
        "mean_absolute_forcing", "rms_forcing", "normalized_control_effort",
        "zero_forcing_percentage", "action_change_rate", "divergence_rate",
        "runtime_seconds",
    )
    return {
        "evaluation_initial_states_identical": True,
        "evaluation_horizon_steps": settings.eval_max_steps or env.horizon,
        "dqn": {key: deep_metrics[key] for key in keys},
        "tabular_episode_650": {key: tabular_metrics[key] for key in keys},
    }


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_deep_q_learning(run: Mapping[str, Any], output_directory: Union[str, Path]) -> List[Path]:
    """Render the tabular baseline plots plus DQN-specific diagnostics."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    history = run["history"]
    summaries = run["checkpoints"]["summaries"]
    evaluation = run["evaluation"]
    evaluation_label = f"Final model (episode {run['final_checkpoint']['episode']})"
    saved: List[Path] = []

    def finish(fig: plt.Figure, name: str) -> None:
        path = output / name
        _save_figure(fig, path)
        saved.append(path)

    # Keep all controller-independent plots identical to the tabular baseline.
    shared_plots.plot_training_diagnostics(
        history, output / "training_diagnostics.png", controller_name="Double DQN"
    )
    shared_plots.plot_training_rollouts(history, output)
    shared_plots.plot_target_acquisition(history, output / "target_acquisition.png")
    checkpoint_evaluations = run["checkpoints"]["evaluations"]
    checkpoint_history = {
        "episodes": run["checkpoints"]["episodes"],
        "mean_rewards": [np.mean(item["episode_rewards"]) for item in checkpoint_evaluations],
        "reward_standard_deviations": [np.std(item["episode_rewards"]) for item in checkpoint_evaluations],
        "divergence_rates": [np.mean(item["diverged"]) for item in checkpoint_evaluations],
        "mean_control_efforts": [item["metrics"]["normalized_control_effort"] for item in checkpoint_evaluations],
    }
    shared_plots.plot_checkpoint_evaluations(
        checkpoint_history, output / "checkpoint_evaluations.png"
    )
    shared_plots.plot_evaluation_diagnostics(
        evaluation, output, controller_name="Double DQN", evaluation_label=evaluation_label
    )
    shared_plots.plot_evaluation_lobe_occupancy(
        evaluation, output / "evaluation_lobe_occupancy.png"
    )
    shared_plots.plot_evaluation_trial_lobe_occupancy(
        evaluation, output / "evaluation_trial_lobe_occupancy.png",
        controller_name="Double DQN", evaluation_label=evaluation_label,
    )
    shared_plots.plot_uncontrolled_evaluation_reference(evaluation, output)
    saved.extend(output / name for name in (
        "training_diagnostics.png", "training_corrected_trajectory.png",
        "training_trajectory.png", "target_acquisition.png", "checkpoint_evaluations.png",
        "evaluation_corrected_trajectory.png", "evaluation_control_correction.png",
        "evaluation_trajectory.png", "evaluation_state_coordinates.png",
        "evaluation_lobe_occupancy.png", "evaluation_trial_lobe_occupancy.png",
        "evaluation_uncontrolled_lorenz.png",
        "evaluation_uncontrolled_state_path.png",
    ))

    episodes = np.arange(1, len(history["episode_rewards"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(episodes, history["episode_mean_losses"], linewidth=1)
    losses = np.asarray(history["episode_mean_losses"], dtype=float)
    finite_losses = losses[np.isfinite(losses)]
    if finite_losses.size and np.all(finite_losses > 0.0):
        ax.set_yscale("log")
    ax.set(xlabel="Episode", ylabel="Mean Huber loss", title="Double DQN training loss")
    finish(fig, "training_loss.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(episodes, history["episode_mean_q_values"], label="Mean Q")
    ax.plot(episodes, history["episode_max_q_values"], label="Mean batch max Q")
    ax.set(xlabel="Episode", ylabel="Predicted Q-value", title="Predicted Q-values")
    ax.legend()
    finish(fig, "q_values.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(episodes, history["replay_buffer_sizes"])
    ax.set(xlabel="Episode", ylabel="Transitions", title="Replay-buffer growth")
    finish(fig, "replay_buffer_growth.png")

    checkpoint_episodes = [item["episode"] for item in summaries]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(checkpoint_episodes, [item["zero_forcing_percentage"] for item in summaries], marker="o")
    ax.set(xlabel="Episode", ylabel="Zero-forcing actions (%)", title="Evaluation zero-action percentage")
    finish(fig, "zero_action_percentage.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(checkpoint_episodes, [100.0 * item["action_change_rate"] for item in summaries], marker="o")
    ax.set(xlabel="Episode", ylabel="Action changes (%)", title="Evaluation action-switching rate")
    finish(fig, "action_switching_rate.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(checkpoint_episodes, [item["successful_trial_count"] for item in summaries], marker="o")
    axes[0, 0].axhline(21, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set(ylabel="Successful trials", title="Checkpoint success")
    axes[0, 1].plot(checkpoint_episodes, [item["aggregate_negative_lobe_occupancy"] for item in summaries], marker="o")
    axes[0, 1].set(ylabel="Negative-lobe occupancy (%)", title="Aggregate occupancy")
    axes[1, 0].plot(checkpoint_episodes, [item["mean_absolute_forcing"] for item in summaries], marker="o")
    axes[1, 0].set(xlabel="Episode", ylabel="Mean |forcing|", title="Control magnitude")
    axes[1, 1].plot(checkpoint_episodes, [100.0 * item["divergence_rate"] for item in summaries], marker="o")
    axes[1, 1].set(xlabel="Episode", ylabel="Divergence rate (%)", title="Divergence")
    finish(fig, "evaluation_policy_metrics.png")

    # These legacy standalone plots are now panels in training_diagnostics.png.
    for obsolete in ("training_rewards.png", "epsilon_schedule.png"):
        (output / obsolete).unlink(missing_ok=True)
    return saved


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def make_run_directory(settings: DeepExperimentDefaults, root: Path = DEEP_RUNS_DIRECTORY) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    base = root / (
        f"{stamp}_ep{settings.episodes}_lr{settings.network_learning_rate:g}_"
        f"gamma{settings.discount_factor:g}_alpha{settings.control_cost:g}_"
        f"forcing{settings.u_ref:g}_actions{settings.action_bins}_"
        f"{settings.activation}_ddqn"
    )
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = Path(f"{base}_{suffix}")
        suffix += 1
    return candidate


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a tiny end-to-end experiment")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--output-directory", type=Path, default=DEEP_RUNS_DIRECTORY,
        help="archive root; each invocation creates a new parameter-labeled subfolder",
    )
    parser.add_argument("--plot-run", type=Path, help="regenerate plots from a saved DQN run without training")
    parser.add_argument("--tabular-run", type=Path, help="optionally compare with a saved tabular run")
    args = parser.parse_args(argv)
    if args.plot_run is not None:
        run = load_deep_q_learning_run(args.plot_run)
        settings = DeepExperimentDefaults(**run["settings"])
        output = make_run_directory(settings, args.output_directory)
        plots = plot_deep_q_learning(run, output)
        with (output / "parameters.json").open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(settings.as_dict()), stream, indent=2, allow_nan=False)
        print(f"Saved {len(plots)} Double DQN plots to {output}")
        return 0
    settings = replace(DEEP_EXPERIMENT_DEFAULTS, device_mode=args.device)
    if args.smoke:
        settings = replace(
            settings, episodes=3, eval_episodes=4, evaluation_interval=2,
            max_steps=12, eval_max_steps=12, minimum_replay_size=4,
            batch_size=4, replay_buffer_size=64, target_update_interval=3,
            hidden_layers=(16, 16), print_every=1, training_plot_samples=3,
            use_attractor_initial_state_ensemble=False,
        )
    output = make_run_directory(settings, args.output_directory)
    print(f"Saving Double DQN run to {output}", flush=True)
    run = run_deep_q_learning(settings, output_directory=output)
    save_deep_q_learning_run(run, output / DEEP_RUN_FILENAME)
    plot_deep_q_learning(run, output)
    with (output / "parameters.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(settings.as_dict()), stream, indent=2, allow_nan=False)
    print(f"Saved Double DQN run and plots to {output}")
    if args.tabular_run is not None:
        comparison = compare_with_tabular_episode(run, args.tabular_run)
        run["tabular_episode_650_comparison"] = comparison
        save_deep_q_learning_run(run, output / DEEP_RUN_FILENAME)
        with (output / "comparison.json").open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(comparison), stream, indent=2, allow_nan=False)
        print(json.dumps(_json_safe(comparison), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
