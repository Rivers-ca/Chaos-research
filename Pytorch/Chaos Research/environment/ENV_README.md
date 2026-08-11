# Lorenz RL Environment

A Gym-style environment for learning control of the chaotic Lorenz system.

## Design Principles

1. **Fixed physics**: The Lorenz differential equations and RK4 integration are non-configurable.
2. **Sweepable parameters**: `alpha`, `horizon`, action space, initial conditions, and state-cost function are all configurable.
3. **Algorithm-agnostic**: The interface is stable and will not change regardless of which RL algorithm (DQN, SAC, PPO, etc.) is plugged in.
4. **Reproducible sweeps**: Same physics with varying parameters enables rigorous comparison of results.

## Quick Start

```python
from env import LorenzEnv

# Continuous control
env = LorenzEnv(
    alpha=0.1,           # Control cost weight
    horizon=200,          # Episode length
    action_type="continuous",
    action_bounds=(-1.0, 1.0),
)

state = env.reset()  # Returns [x, y, z]

for step in range(200):
    action = policy(state)  # Your policy
    state, reward, done, info = env.step(action)
    if done:
        break
```

## Constructor Parameters

### Required/Important

- **`alpha`** (float): Control cost weight. Higher α penalizes control effort more.
  - Common values: 0.01, 0.1, 1.0, 10.0
  - This is typically the primary sweep parameter.
  
- **`horizon`** (int): Number of timesteps per episode.
  - Measured in timesteps (not Lyapunov times).
  - To convert from Lyapunov times τ to timesteps: `h = round(τ / (LYAPUNOV_EXP * DT))`
  - Common values: 50, 100, 200, 500, 1000

### Action Space

- **`action_type`** (str): `"continuous"` or `"discrete"`.

**For continuous:**
```python
env = LorenzEnv(
    action_type="continuous",
    action_bounds=(-1.0, 1.0),  # [low, high]
)
# step(action) expects a float in [-1, 1]
```

**For discrete (e.g., DQN):**
```python
env = LorenzEnv(
    action_type="discrete",
    action_bounds=(-1.0, 1.0),
    n_action_bins=5,  # Gives 5 actions uniformly spaced
)
# step(action) expects an int in [0, 4]
# Internally maps to [-1.0, -0.5, 0.0, 0.5, 1.0]
```

### Initial Conditions

- **`ic_dist`** (Callable): Function returning `[x0, y0, z0]` array.
  - If `None`: uses default: `(0, 1, 1.05) + N(0, 0.1²)`
  - Can pass any distribution for train/test split experiments.

**Example:**
```python
def ic_dist_uniform():
    return np.random.uniform(-10, 10, size=3)

env = LorenzEnv(ic_dist=ic_dist_uniform)

state = env.reset()  # Samples from ic_dist
state = env.reset(x0=np.array([1.0, 2.0, 3.0]))  # Fixed IC
```

### State Cost Function

- **`state_cost_fn`** (Callable): Maps `x` (first state component) → cost.
  - If `None`: uses default: `φ(x) = 0.5 * (1 + tanh(x / 2))`
  - Can replace with custom cost functions (e.g., |x|, x², etc.)

**Example:**
```python
env = LorenzEnv(
    state_cost_fn=lambda x: np.abs(x),  # L1 cost
)
```

### Termination

- **`divergence_threshold`** (float): Episode ends if `norm(state) > this`.
  - Default: 100.0 (Lorenz attractor is bounded by ~60)
  - Rarely triggered in practice; mainly safety check.

## Reward Structure

At each step:
```
reward = -(state_cost + control_cost)
       = -(φ(x) + α * u²)
```

Where:
- `φ(x)` is the state cost function (penalizes departures of x from 0)
- `α * u²` is the control cost (penalizes control effort)
- Higher reward is better; agents learn to minimize φ(x) while conserving control.

## Interface (`reset` and `step`)

### `reset(x0=None) -> np.ndarray`

Reset to initial condition.

- **`x0`** (np.ndarray or None): Fixed IC [x, y, z].
  - If `None`: samples from `ic_dist`.
- **Returns**: Initial state [x, y, z].

**Example:**
```python
state = env.reset()  # Sample from ic_dist
state = env.reset(x0=np.array([0, 1, 1.05]))  # Fixed IC
```

### `step(action) -> (state, reward, done, info)`

Integrate one RK4 step with control `action`.

- **`action`** (int or float): Control input.
  - Discrete: int in [0, n_action_bins)
  - Continuous: float, automatically clipped to bounds
- **`state`** (np.ndarray): New state [x, y, z].
- **`reward`** (float): Reward for this step.
- **`done`** (bool): Episode terminated?
- **`info`** (dict): Keys like `"diverged"` or `"horizon_reached"`.

**Example:**
```python
state, reward, done, info = env.step(0.5)
if done:
    if info.get("diverged"):
        print("State diverged")
    elif info.get("horizon_reached"):
        print("Horizon reached")
```

## Constants

Defined in `env.py` and mirrored from `sim.py`:

```python
RAYLEIGH = 28    # ρ parameter
PRANDTL = 10     # σ parameter
B = 8/3          # β parameter
DT = 0.01        # Timestep size
LYAPUNOV_EXP = 0.9056  # λ (used to convert Lyapunov times ↔ timesteps)
```

These are **not configurable**. Control is injected additively on the first component:
```
dx = PRANDTL * (y - x) + u  (control injection point)
dy = x * (RAYLEIGH - z) - y
dz = x * y - B * z
```

Integration is RK4 with step size `DT`.

## Example: Sweep Across Configurations

See `sweep_example.py` for complete examples. Quick version:

```python
import numpy as np
from env import LorenzEnv

alphas = [0.01, 0.1, 1.0]
horizons = [100, 200, 500]

for alpha in alphas:
    for horizon in horizons:
        env = LorenzEnv(alpha=alpha, horizon=horizon)
        
        state = env.reset()
        total_reward = 0.0
        
        for _ in range(horizon):
            action = policy(state)  # Your policy
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        
        print(f"α={alpha}, h={horizon}: reward={total_reward:.4f}")
```

## Implementing a Policy

The environment expects policies to implement:

```python
def select_action(state: np.ndarray, env: LorenzEnv) -> Union[int, float]:
    """Select action given state."""
    if env.action_type == "discrete":
        return 0  # int in [0, n_action_bins)
    else:
        return 0.5  # float in action_bounds
```

You can also define a `Policy` class with an `update()` method:

```python
from env import Policy

class MyPolicy(Policy):
    def select_action(self, state):
        # Your algorithm here
        return action
    
    def update(self, experience_batch):
        # Your learning update here
        pass
```

The environment makes no assumptions about policy structure. You're free to implement:
- **DQN** (discrete actions)
- **SAC/TD3** (continuous actions)
- **PPO** (either)
- **DDPG**, **A3C**, etc.

## Validation

Run `test_env.py` to verify environment correctness:
```bash
python3 test_env.py
```

Run `sweep_example.py` to validate that behavior is consistent across configurations:
```bash
python3 sweep_example.py
```

Both should complete without errors and confirm:
- ✓ Physics is stable
- ✓ Reward scales predictably
- ✓ Results generalize across ICs
- ✓ Both action spaces work correctly

## Files

- **`env.py`**: Environment class and Policy protocol.
- **`test_env.py`**: Unit tests (instantiation, both action spaces, rewards, physics).
- **`sweep_example.py`**: Demonstration of sweeps (core validation before RL).
- **`sim.py`** (existing): Reference implementation of uncontrolled Lorenz dynamics.
- **`plots.py`** (existing): Plotting utilities (unchanged).

## Notes

- The environment uses **double precision** (`float64`) throughout for numerical stability.
- RK4 integration is more accurate than the Euler method in `sim.py`, but results should be qualitatively similar for uncontrolled trajectories.
- Control is always applied as `u_actual = clip(u, action_low, action_high)` for continuous actions.
- The environment is deterministic given a fixed seed and fixed IC; use `ic_dist` or `np.random.seed()` to vary runs.

## Contact & Extension

To add a new RL algorithm:
1. Implement a `Policy` class in a new file (e.g., `dqn.py`, `sac.py`).
2. Import `LorenzEnv` from `env.py`.
3. Create sweeps over `alpha`, `horizon`, and initial conditions as needed.
4. No changes to `env.py` should be required.

If you find yourself needing to modify the environment interface (e.g., new parameters, different reward structure), check with the project lead first — the interface should be stable.
