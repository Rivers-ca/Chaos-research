import numpy as np
from typing import Callable, Optional, Tuple, Union

"""Patricks Parameters for the Lorenz system"""
RAYLEIGH = 28
PRANDTL = 10
B = 8 / 3
DT = 0.01
LYAPUNOV_EXP = 0.9056


def lyapunov_times_to_steps(lyapunov_times: float) -> int:
    """Convert a duration in Lyapunov times to a step count, Patrick's convention."""
    return round(lyapunov_times / (LYAPUNOV_EXP * DT))


def steps_to_lyapunov_times(steps: int) -> float:
    """Inverse of the above, for reporting results on a Lyapunov-time axis."""
    return steps * LYAPUNOV_EXP * DT


"Environment Class for the Lorenz system with Euler integration and Q-learning support."
class LorenzEnvEuler:

    def __init__(
        self,
        alpha: float = 1.0,
        lyapunov_times: float = 5.0, #Used Patricks method to convert lyapunov times to steps: steps = round(lyapunov_times / (LYAPUNOV_EXP * DT)) or T = lyapunov_times / λ
        action_type: str = "continuous",
        action_bounds: Optional[Tuple[float, float]] = None,
        n_action_bins: Optional[int] = None,
        ic_dist: Optional[Callable] = None,
        state_cost_fn: Optional[Callable] = None,
        divergence_threshold: float = 100.0,
    ):
        """
        Args:
            alpha: Control cost weight in reward.
            lyapunov_times: Episode length, in Lyapunov times (converted to
                             steps internally via LYAPUNOV_EXP and DT).
            action_type: "continuous" or "discrete".
            action_bounds: (low, high) for the control u.
            n_action_bins: Required if action_type="discrete".
            ic_dist: Callable -> [x0, y0, z0]. Default: (0,1,1.05) + N(0, 0.1^2).
            state_cost_fn: Callable x -> cost. Default: 0.5*(1+tanh(x/2)),
                            matching sim.py's phi(x) with eps=2.0.
            divergence_threshold: Episode ends if norm(state) exceeds this.
        """
        self.alpha = alpha
        self.lyapunov_times = lyapunov_times
        self.horizon = lyapunov_times_to_steps(lyapunov_times)
        self.divergence_threshold = divergence_threshold

        self.action_type = action_type
        if action_type == "continuous":
            action_bounds = action_bounds or (-1.0, 1.0)
            self.action_low, self.action_high = action_bounds
        elif action_type == "discrete":
            if n_action_bins is None:
                raise ValueError("n_action_bins required for discrete action_type")
            action_bounds = action_bounds or (-1.0, 1.0)
            self.action_low, self.action_high = action_bounds
            self.n_action_bins = n_action_bins
            self.actions = np.linspace(self.action_low, self.action_high, n_action_bins)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        self.ic_dist = ic_dist or (
            lambda: np.array([0.0, 1.0, 1.05]) + np.random.randn(3) * 0.1
        )
        #Loss function from patricks code matches sim.py's phi(x, eps=2.0).
        self.state_cost_fn = state_cost_fn or (lambda x: 0.5 * (1.0 + np.tanh(x / 2.0)))

        self.state = None
        self.step_count = 0

    def reset(self, x0: Optional[np.ndarray] = None) -> np.ndarray:
        self.state = np.array(x0, dtype=np.float64) if x0 is not None else self.ic_dist().astype(np.float64)
        self.step_count = 0
        return self.state.copy()

    def step(self, action: Union[int, float]) -> Tuple[np.ndarray, float, bool, dict]:
        assert self.state is not None, "Must call reset() before step()"

        if self.action_type == "discrete":
            u = self.actions[int(action)]
        else:
            u = float(np.clip(action, self.action_low, self.action_high))

        self.state = self._euler_step(self.state, u)
        self.step_count += 1

        x = self.state[0]
        state_cost = self.state_cost_fn(x)
        control_cost = self.alpha * (u ** 2)
        reward = -(state_cost + control_cost)

        done, info = False, {}
        if np.isnan(self.state).any() or np.linalg.norm(self.state) > self.divergence_threshold:
            done, info["diverged"] = True, True
        if self.step_count >= self.horizon:
            done, info["horizon_reached"] = True, True

        return self.state.copy(), reward, done, info

    @staticmethod
    def _euler_step(state: np.ndarray, u: float) -> np.ndarray:
        #DX DY DZ copied from Patricks sim.py's array_data() with dt=DT, alpha=1.0, and u as the control input.
        x, y, z = state
        #Note: change to rk4 later
        dx = PRANDTL * (y - x) + u
        dy = x * (RAYLEIGH - z) - y
        dz = x * y - B * z
        return state + np.array([dx, dy, dz]) * DT

    def render(self):
        pass

    def close(self):
        pass



if __name__ == "__main__":
    # Sanity check against sim.py's array_data() with u=0.
    env = LorenzEnvEuler(alpha=0.0, lyapunov_times=1.0, action_type="continuous")
    print(f"horizon = {env.horizon} steps ({env.lyapunov_times} Lyapunov times)")

    state = env.reset(x0=np.array([-0.1, 1.0, 1.05]))
    for _ in range(5):
        state, reward, done, info = env.step(0.0)
        print(state, reward, done, info)
