# Lorenz RL Environment: Complete Implementation

## Summary

A production-ready Gym-style environment for learning control of the chaotic Lorenz system. Designed for sweepable parameter studies (alpha, horizon, initial conditions) with a stable, reusable interface for DQN, SAC, PPO, and other RL algorithms.

**Status**: ✅ Complete and validated

## What You Get

### Core Implementation
- **LorenzEnv** class: Gym-compatible Lorenz system with control
- **RK4 integration**: Accurate numerical method with dt=0.01
- **Configurable rewards**: Control-cost weight (`alpha`) and state-cost function
- **Flexible action spaces**: Discrete (DQN) or continuous (SAC/TD3/PPO)
- **Robust termination**: Divergence detection and horizon limits

### Validation
- ✅ All unit tests pass (test_env.py)
- ✅ Sweeps work correctly (sweep_example.py)
- ✅ Physics matches existing sim.py code
- ✅ Reproducible results with fixed seed
- ✅ Generalization across initial conditions

### Documentation
- **ENV_README.md**: Complete API reference
- **CONTROL_INJECTION.md**: Physics verification & equation matching
- **ENVIRONMENT_SETUP.md**: Quick start & next steps
- **ENV_INDEX.md**: File reference and quick lookup
- **template_rl_algorithm.py**: Skeleton for implementing algorithms

## Quick Start

### 1. Verify Installation (2 minutes)
```bash
cd /Users/riverscalareso/Desktop/Python/Pytorch/Chaos\ Research
python3 test_env.py          # Unit tests
python3 sweep_example.py     # Parameter sweep demo
```

Both should complete with all tests passing ✓.

### 2. Understand the Interface (10 minutes)
Read the first section of **ENV_README.md** for the three-line API:
```python
env = LorenzEnv(alpha=0.1, horizon=200)  # Configure
state = env.reset()                      # Initialize
state, reward, done, info = env.step(u)  # Integrate
```

### 3. Implement Your Algorithm (30-60 minutes)
1. Copy `template_rl_algorithm.py` to `dqn.py` (or your algorithm)
2. Implement `select_action()` and `update()` in your policy class
3. Run training using the sweep structure shown in the template
4. No changes to `env.py` required

## File Structure

```
Chaos Research/
├── Core Files
│   ├── env.py                      ← LorenzEnv class (205 lines)
│   ├── sim.py                      ← Reference (existing, unchanged)
│   └── plots.py                    ← Plotting utils (existing, unchanged)
│
├── Validation
│   ├── test_env.py                 ← Unit tests (317 lines, all tests pass)
│   └── sweep_example.py            ← Sweep demo (268 lines, validated)
│
├── Templates
│   └── template_rl_algorithm.py     ← Copy to create DQN/SAC/PPO (264 lines)
│
└── Documentation
    ├── README_ENVIRONMENT.md       ← This file
    ├── ENV_README.md               ← API reference (comprehensive)
    ├── CONTROL_INJECTION.md        ← Physics verification
    ├── ENVIRONMENT_SETUP.md        ← Setup overview & next steps
    └── ENV_INDEX.md                ← Quick file reference
```

## Key Parameters

All configurable via constructor (no hardcoded magic):

| Parameter | Default | Type | Use For |
|-----------|---------|------|---------|
| `alpha` | 1.0 | float | Control cost weight (primary sweep parameter) |
| `horizon` | 1000 | int | Episode length in timesteps |
| `action_type` | "continuous" | str | "continuous" or "discrete" |
| `action_bounds` | (-1, 1) | tuple | Range for action space |
| `n_action_bins` | None | int | Number of discrete actions (if discrete) |
| `ic_dist` | None | Callable | Initial condition distribution |
| `state_cost_fn` | None | Callable | State cost φ(x) |
| `divergence_threshold` | 100.0 | float | Bounds check for termination |

## Physics (Hardcoded, Matches sim.py)

```
Lorenz equations with control:
  σ = 10 (PRANDTL)
  ρ = 28 (RAYLEIGH)
  β = 8/3 (B)
  dt = 0.01 (DT)

  dx/dt = σ(y - x) + u     ← Control injected here
  dy/dt = x(ρ - z) - y
  dz/dt = xy - βz

Integration: RK4 (Runge-Kutta 4th order)
Reward: -(φ(x) + α*u²) where φ(x) = 0.5*(1 + tanh(x/2))
```

## Typical Sweep Structure

This is the pattern you'll follow for reproducible results:

```python
# For each RL algorithm (DQN, SAC, PPO, ...)
config = {
    "alphas": [0.001, 0.01, 0.1, 1.0],        # Control cost sweep
    "horizons": [100, 200, 500, 1000],        # Horizon sweep (Lyapunov times)
    "n_episodes": 100,                         # Training episodes
    "n_seeds": 5,                              # Runs to average over
    "action_type": "discrete",                 # or "continuous"
    "ic_dist": None,                           # None = default, or custom
}

# Grid sweep across configurations
results = {}
for alpha in config["alphas"]:
    for horizon in config["horizons"]:
        for seed in range(config["n_seeds"]):
            env = LorenzEnv(alpha=alpha, horizon=horizon)
            policy = YourPolicy(env)
            
            for episode in range(config["n_episodes"]):
                # Train one episode...
                pass
            
            results[alpha][horizon][seed] = final_performance

# Results show: [alpha x horizon x seed] matrix
# Ready for paper: "Results hold across α ∈ [0.001, 1.0], T ∈ [100, 1000], ..."
```

## Before Starting Your Algorithm

Make sure you can answer these:

1. **Action space**: Discrete (DQN) or continuous (SAC/PPO)?
2. **Reward sign**: Maximize reward (same sign convention)?
3. **Horizon**: Will you use a fixed horizon or let episodes terminate naturally?
4. **Sweep parameters**: Which alpha/horizon values do you want to test?
5. **Number of runs**: How many random seeds for averaging?

Default answers (for first implementation):
- Action space: continuous (easier to debug)
- Reward: maximize (standard)
- Horizon: 200 timesteps (≈2 Lyapunov times)
- Sweep: alpha ∈ [0.01, 0.1, 1.0], horizon ∈ [100, 200]
- Seeds: 3-5 for early testing, 10+ for final results

## Verification Checklist

Before claiming results are reproducible:

- [ ] Same random seed → same trajectory
- [ ] Sweep across alpha shows expected trend
- [ ] Sweep across horizon shows expected trend
- [ ] Results consistent across different ICs
- [ ] Control actually affects the system (compare to u=0 baseline)
- [ ] No NaNs or divergences in trajectories
- [ ] Reward values reasonable (not orders of magnitude off)

All of these are validated in `test_env.py` and `sweep_example.py`.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError: No module named 'env'" | Make sure you're in the Chaos Research directory |
| "Trajectory diverges" | Increase `divergence_threshold` or reduce `alpha` |
| "Reward is NaN" | Check state_cost_fn for division by zero |
| "Same seed different results" | Verify you set `np.random.seed()` before training |
| "Test fails on divergence check" | Run multiple times; occasional divergence possible |

## Next: Implementing an Algorithm

**Step 1**: Copy the template
```bash
cp template_rl_algorithm.py dqn.py
```

**Step 2**: Edit `dqn.py`
- Replace `MyRLPolicy` with `DQNPolicy`
- Implement neural network, replay buffer, loss function
- Implement `select_action()` with epsilon-greedy
- Implement `update()` with Bellman loss

**Step 3**: Run training with the sweep structure
```python
config = {
    "alphas": [0.01, 0.1, 1.0],
    "horizons": [100, 200],
    "n_episodes": 100,
    "action_type": "discrete",  # For DQN
}
results = train_sweep(config, n_seeds=3)
```

**Step 4**: Plot and analyze results

See `template_rl_algorithm.py` for the complete structure.

## Citation/Reference

If you use this environment in a paper, cite it as:
> "We use a custom Gym-style Lorenz control environment with RK4 integration
> (dt=0.01), Lorenz parameters σ=10, ρ=28, β=8/3. Control is injected
> additively on the first state component. Reward is -(φ(x) + α*u²) where
> φ(x) = 0.5*(1 + tanh(x/2))."

## Summary

✅ **What's done:**
- Environment is complete and tested
- Physics is verified against sim.py
- All sweepable parameters work
- Documentation is comprehensive
- Template is ready for algorithm implementation

✅ **What's next:**
- Implement DQN (discrete actions)
- Run alpha × horizon sweep
- Compare results across algorithms
- Publish findings

**Estimated time to first trained DQN**: 1-2 hours  
**Estimated time to paper-ready results**: 1-2 weeks (including sweeps & analysis)

Good luck! 🚀
