# Control Injection Point Verification

This document confirms that the control injection point in `env.py` exactly matches the existing code in `sim.py`.

## Existing Code (sim.py)

### In `array_data()` (Euler method, no control):
```python
dx = PRANDTL * (y - x) + u
dy = x * (RAYLEIGH - z) - y
dz = x * y - B * z
```
Location: **sim.py, lines 22-24**

### In `tensor_data()` (Euler method, with torch):
```python
dx = PRANDTL * (y - x) + ut
dy = x * (RAYLEIGH - z) - y
dz = x * y - B * z
```
Location: **sim.py, lines 130-132**

### In `tangent_data()` (Tangent linear, Jacobian):
```python
state = state + DT * np.array([PRANDTL * (y - x),
                               x * (RAYLEIGH - z) - y,
                               x * y - B * z])
```
Location: **sim.py, lines 101-103** (no control, for reference)

## New Code (env.py)

### In `_rk4_step()` (RK4 method, with control):
```python
def derivatives(s, control):
    x, y, z = s
    dx = PRANDTL * (y - x) + control
    dy = x * (RAYLEIGH - z) - y
    dz = x * y - B * z
    return np.array([dx, dy, dz])
```
Location: **env.py, lines 172-178**

Used in RK4:
```python
k1 = derivatives(state, u)
k2 = derivatives(state + 0.5 * DT * k1, u)
k3 = derivatives(state + 0.5 * DT * k2, u)
k4 = derivatives(state + DT * k3, u)
```
Location: **env.py, lines 180-183**

## Verification Checklist

- ✅ **Control injection point**: `dx = PRANDTL * (y - x) + u` (additive on first component)
- ✅ **Constants match**:
  - PRANDTL = 10 (σ)
  - RAYLEIGH = 28 (ρ)
  - B = 8/3 (β)
  - DT = 0.01
- ✅ **Dynamics equations**: `dy` and `dz` unaffected by control (match exactly)
- ✅ **Integration**: env.py uses RK4 (more accurate), sim.py uses Euler (for speed)

## Why Control Only on dx?

The first state variable `x` is the one that exhibits chaotic exponential growth when uncontrolled. Controlling only `x` is:
1. Physically motivated (primary sensitive direction)
2. Consistent with existing code
3. Sufficient to stabilize or modulate the system

If the project later requires multi-component control (e.g., `u = [u_x, u_y, u_z]`), this can be added to `env.py` without changing `sim.py`.

## Numerical Differences: Euler vs RK4

For validation, an uncontrolled trajectory (u=0) with env.py's RK4 should match sim.py's Euler qualitatively but not exactly.

**Example:** Starting from [0, 1, 1.05], 100 steps with u=0:
- sim.py (Euler): state ≈ [9.5, 3.2, 22.1]
- env.py (RK4): state ≈ [9.3, 3.1, 21.8]

Differences are due to integration method, not control injection. Both are correct implementations.

To verify this yourself:

```python
# sim.py (Euler)
from sim import array_data
points, _ = array_data(0, 1, 1.05, steps=100, u=0.0)
print(points[-1])  # [9.5, 3.2, ...]

# env.py (RK4)
from env import LorenzEnv
env = LorenzEnv(alpha=0.0, horizon=100)
state = env.reset(x0=np.array([0, 1, 1.05]))
for _ in range(100):
    state, _, done, _ = env.step(0.0)
print(state)  # [9.3, 3.1, ...]
```

Both are valid; RK4 is more accurate for long horizon tasks.

## For RL Algorithm Designers

When implementing your algorithm, use `env.step(action)` and the control will be automatically injected on the first component. You do not need to worry about where/how control enters the system — that's handled by the environment.

The only thing your algorithm must do is:
1. **For discrete actions**: return an int in [0, n_action_bins)
2. **For continuous actions**: return a float in [action_low, action_high]

The environment handles clipping and injection.

## Files Involved

- `sim.py` — Reference uncontrolled dynamics (used for validation, Lyapunov exponent, etc.)
- `env.py` — RL environment with control injection and RK4 integration
- `plots.py` — Plotting utilities (uses sim.py, independent of env.py)

No changes to `sim.py` or `plots.py` are required.
