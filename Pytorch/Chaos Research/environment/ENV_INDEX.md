# Lorenz RL Environment: File Index

Quick reference for all environment-related files. Start here if you're new to the setup.

## Core Files (Read First)

| File | Purpose | Read Time |
|------|---------|-----------|
| **ENVIRONMENT_SETUP.md** | Overview & quick start | 10 min |
| **ENV_README.md** | Complete API reference | 15 min |
| **env.py** | Implementation (LorenzEnv class) | 10 min |

## Validation Files (Run These)

| File | Purpose | Command |
|------|---------|---------|
| **test_env.py** | Unit tests for all functionality | `python3 test_env.py` |
| **sweep_example.py** | Demonstrates parameter sweeps | `python3 sweep_example.py` |

All tests should pass (✓) without errors.

## Implementation Template

| File | Purpose |
|------|---------|
| **template_rl_algorithm.py** | Skeleton for DQN/SAC/PPO algorithms; copy to create new algorithms |

## Reference & Verification

| File | Purpose |
|------|---------|
| **CONTROL_INJECTION.md** | Verifies control point matches sim.py; physics validation |
| **ENV_INDEX.md** | This file; quick reference |

## Existing Project Files (Do Not Modify)

| File | Purpose |
|------|---------|
| **sim.py** | Reference Lorenz dynamics (Euler, uncontrolled) |
| **plots.py** | Plotting utilities for analysis |

## Quick Start Paths

### I want to understand the environment (15 min)
1. Read: ENVIRONMENT_SETUP.md
2. Skim: ENV_README.md (constructor & interface sections)
3. Run: `python3 test_env.py` (verify installation)

### I want to see it working (10 min)
1. Run: `python3 test_env.py`
2. Run: `python3 sweep_example.py`

### I want to implement an RL algorithm (30 min)
1. Read: ENVIRONMENT_SETUP.md (overview)
2. Copy: `template_rl_algorithm.py` → `my_algorithm.py`
3. Modify: `MyRLPolicy` class with your algorithm
4. Reference: ENV_README.md (API) and CONTROL_INJECTION.md (physics)

### I want to verify the physics (20 min)
1. Read: CONTROL_INJECTION.md (verification checklist)
2. Compare: Lines in sim.py vs env.py (control equations)
3. Run: `sweep_example.py` → Validate numbers look reasonable

## File Dependencies

```
┌─────────────────────────────────────────────────────────┐
│  User's Algorithm (dqn.py, sac.py, ppo.py, ...)        │
│  └─ imports: env.LorenzEnv, env.Policy                 │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  env.py (LorenzEnv class)                               │
│  └─ imports: numpy                                       │
└─────────────────────────────────────────────────────────┘
                         ↓
    Existing project files (unchanged):
    sim.py, plots.py (independent)
```

## Parameters You'll Configure

These go in your algorithm's `__main__` or config dict:

```python
config = {
    "alpha": 0.1,              # Control cost weight (SWEEP THIS)
    "horizon": 200,            # Episode length (SWEEP THIS)
    "action_type": "continuous",  # or "discrete"
    "action_bounds": (-1.0, 1.0),  # For continuous
    "n_action_bins": 5,        # For discrete
    "ic_dist": None,           # None = default, or Callable
    "n_episodes": 100,         # Training episodes per config
    "n_seeds": 3,              # Seeds to average over
}
```

## Key Interfaces (What Your Algorithm Needs)

```python
# 1. Create environment
env = LorenzEnv(alpha=..., horizon=..., action_type=...)

# 2. Implement policy
class MyPolicy(Policy):
    def select_action(self, state: np.ndarray) -> Union[int, float]:
        # Return action
    
    def update(self, batch):
        # Update from experience

# 3. Training loop
state = env.reset()
for _ in range(env.horizon):
    action = policy.select_action(state)
    state, reward, done, info = env.step(action)
    policy.update(...)  # Your algorithm's learning
    if done:
        break
```

## Common Questions Answered

**Q: Where's the actual control code?**  
A: Inside `env._rk4_step()` — control is injected additively on `dx`.

**Q: Can I modify the Lorenz equations?**  
A: No, they're hardcoded. If you need different physics, contact project lead.

**Q: Can I use different action bounds?**  
A: Yes, pass `action_bounds=(-2.0, 2.0)` to constructor.

**Q: Can I change the reward function?**  
A: Partially. State cost via `state_cost_fn`, control cost is always `α*u²`.

**Q: How do I do a sweep?**  
A: See `sweep_example.py` and `template_rl_algorithm.py` for structure.

**Q: What if I need a different integrator (Euler, RK2)?**  
A: Not recommended without project lead approval. RK4 is validated; others require re-testing.

**Q: Can I run in a notebook?**  
A: Yes, import and use like any Python module. Copy code from examples.

**Q: How do I get reproducible results?**  
A: Use `np.random.seed()` before training. Environment is deterministic given a seed.

## Checklist Before Implementing Your Algorithm

- [ ] Read ENVIRONMENT_SETUP.md
- [ ] Run `python3 test_env.py` (all tests pass)
- [ ] Run `python3 sweep_example.py` (all outputs reasonable)
- [ ] Understand env.reset() and env.step() interfaces
- [ ] Know your action_type (discrete or continuous)
- [ ] Have a config dict ready (alpha, horizon, etc.)
- [ ] Copied template_rl_algorithm.py to your algorithm file

## Next Steps

1. **Immediate**: Run test_env.py and sweep_example.py to verify setup
2. **Short-term**: Copy template and implement your first algorithm (DQN?)
3. **Medium-term**: Run sweeps across (alpha, horizon) configurations
4. **Long-term**: Compare results across multiple algorithms

## Support

If you get stuck:
1. Check ENV_README.md API section
2. Look at test_env.py for usage examples
3. Check CONTROL_INJECTION.md if physics seems wrong
4. Read the inline comments in env.py

Good luck! 🚀
