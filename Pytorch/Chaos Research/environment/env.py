import numpy as np
from typing import Callable, Optional, Tuple, Union

# Mirror existing constants from sim.py
RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
DT = 0.01
LYAPUNOV_EXP = 0.9056


class LorenzEnv:
    """
    Gym-style environment for the Lorenz system with control.

    Control is injected additively on dx (first state component), matching
    the existing sim.py convention: dx = PRANDTL * (y - x) + u

    Integration uses RK4. State is [x, y, z].
    """

    def __init__(
        self,
        alpha: float = 1.0,
        horizon: int = 1000,
        action_type: str = "continuous",
        action_bounds: Optional[Tuple[float, float]] = None,
        n_action_bins: Optional[int] = None,
        ic_dist: Optional[Callable] = None,
        state_cost_fn: Optional[Callable] = None,
        divergence_threshold: float = 100.0,
    ):
        """
        Args:
            alpha: Control cost weight in reward (sweepable parameter).
            horizon: Episode length in timesteps (sweepable parameter).
            action_type: "continuous" or "discrete".
            action_bounds: (low, high) for continuous; for discrete, defines
                           the range for uniform binning.
            n_action_bins: Number of action bins for discrete actions.
                           If action_type="discrete", must be provided.
            ic_dist: Callable returning [x0, y0, z0] initial condition.
                     If None, uses default: (0, 1, 1.05) + N(0, 0.1^2).
            state_cost_fn: Callable mapping state x -> cost. Default: 0.5*(1+tanh(x/2)).
            divergence_threshold: Episode terminates if norm(state) exceeds this.
        """
        self.alpha = alpha
        self.horizon = horizon
        self.divergence_threshold = divergence_threshold

        # Action space setup
        self.action_type = action_type
        if action_type == "continuous":
            if action_bounds is None:
                action_bounds = (-1.0, 1.0)
            self.action_low, self.action_high = action_bounds
        elif action_type == "discrete":
            if n_action_bins is None:
                raise ValueError("n_action_bins required for discrete action_type")
            if action_bounds is None:
                action_bounds = (-1.0, 1.0)
            self.action_low, self.action_high = action_bounds
            self.n_action_bins = n_action_bins
            # Uniform grid of actions
            self.actions = np.linspace(self.action_low, self.action_high, n_action_bins)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        # Initial condition distribution
        if ic_dist is None:
            # Default: small noise around (0, 1, 1.05)
            self.ic_dist = lambda: np.array([0.0, 1.0, 1.05]) + np.random.randn(3) * 0.1
        else:
            self.ic_dist = ic_dist

        # State cost function
        if state_cost_fn is None:
            # Default: 0.5 * (1 + tanh(x / 2))
            self.state_cost_fn = lambda x: 0.5 * (1.0 + np.tanh(x / 2.0))
        else:
            self.state_cost_fn = state_cost_fn

        # State tracking
        self.state = None
        self.step_count = 0

    def reset(self, x0: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Reset to initial condition.

        Args:
            x0: Fixed IC [x, y, z]. If None, sample from ic_dist.

        Returns:
            Initial state [x, y, z].
        """
        if x0 is None:
            self.state = self.ic_dist().astype(np.float64)
        else:
            self.state = np.array(x0, dtype=np.float64)

        self.step_count = 0
        return self.state.copy()

    def step(self, action: Union[int, float]) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one RK4 step with control action.

        Args:
            action: Control input (int index for discrete, float for continuous).

        Returns:
            (next_state, reward, done, info_dict)
        """
        # Map action to control value
        if self.action_type == "discrete":
            u = self.actions[int(action)]
        else:
            u = float(action)
            # Optionally clip to bounds (for safety)
            u = np.clip(u, self.action_low, self.action_high)

        # RK4 integration
        self.state = self._rk4_step(self.state, u)
        self.step_count += 1

        # Reward: -(state_cost + control_cost)
        x = self.state[0]
        state_cost = self.state_cost_fn(x)
        control_cost = self.alpha * (u ** 2)
        reward = -(state_cost + control_cost)

        # Termination checks
        done = False
        info = {}

        if np.isnan(self.state).any() or np.linalg.norm(self.state) > self.divergence_threshold:
            done = True
            info["diverged"] = True

        if self.step_count >= self.horizon:
            done = True
            info["horizon_reached"] = True

        return self.state.copy(), reward, done, info

    @staticmethod
    def _rk4_step(state: np.ndarray, u: float) -> np.ndarray:
        """
        RK4 step for Lorenz with control.

        Control u is added to dx (first component).
        """

        def derivatives(s, control):
            x, y, z = s
            dx = PRANDTL * (y - x) + control
            dy = x * (RAYLEIGH - z) - y
            dz = x * y - B * z
            return np.array([dx, dy, dz])

        k1 = derivatives(state, u)
        k2 = derivatives(state + 0.5 * DT * k1, u)
        k3 = derivatives(state + 0.5 * DT * k2, u)
        k4 = derivatives(state + DT * k3, u)

        return state + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def render(self):
        """Placeholder for render (gym compatibility)."""
        pass

    def close(self):
        """Placeholder for close (gym compatibility)."""
        pass


# ============================================================================
# Policy interface protocol (for type hints in training code)
# ============================================================================

class Policy:
    """
    Abstract policy interface. Implement this for DQN, SAC, PPO, etc.
    The environment makes no assumptions about the policy implementation.
    """

    def select_action(self, state: np.ndarray) -> Union[int, float]:
        """
        Select an action given a state.

        Args:
            state: [x, y, z]

        Returns:
            int (for discrete) or float (for continuous)
        """
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        Update policy (e.g., from a batch of experience).
        Signature is algorithm-specific.
        """
        raise NotImplementedError
