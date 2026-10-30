# DQN implementation and plotting audit

The Double DQN update is correct: the online network selects the next action,
the target network evaluates it, and terminal transitions do not bootstrap.
Replay samples contain normalized continuous coordinates; the update uses Huber
loss, Adam, gradient clipping, and periodic target-network copies.

## Verification

- All 61 tests across DQN, the shared Lorenz environment, and tabular plotting passed.
- Checked all 640 saved evaluation trajectories across 16 checkpoints for finite
  coordinates, state/action alignment, and forcing values matching action indices.
- Recomputed checkpoint metrics and verified the best-checkpoint ranking.
- Checked saved loss, gradient, and network values for finiteness.
- Fully re-evaluated both the best and final checkpoints on all 40 initial states.
  Their action sequences and metrics match the saved evaluations exactly.
- Regenerated 19 PNG plots using the tabular plotting functions for all shared
  diagnostics. Rollouts and control traces use Lyapunov times (t / τ); learning
  and checkpoint histories retain the tabular episode axes. DQN-specific loss,
  predicted Q-values, replay, and action-switching diagnostics remain separate.
- Evaluation trajectory figures show one trial per 3D grid panel with common axis limits.
- State-coordinate figures also separate trials into grid panels, each with stacked
  x/y/z traces, shared coordinate limits, and Lyapunov-time axes.
- Restored per-trial lobe occupancy alongside the aggregate summary, with a tick
  and initial-condition tuple for every bar, percentage labels, and the 51% threshold.
- Tests verify plotted occupancy, rewards, checkpoint summaries, state coordinates,
  and control data, including the final control interval and Lyapunov conversion.

| Model | Successful trials | Negative-lobe occupancy | Mean absolute forcing |
| --- | ---: | ---: | ---: |
| Best, episode 250 | 25/40 | 53.2123% | 0.8449 |
| Final, episode 800 | 19/40 | 49.6554% | 2.0510 |

A successful trial spends at least 51% of evaluated steps in the negative lobe.
An eligible checkpoint requires at least 21 successful trials and no divergence.
The last three checkpoints do not all satisfy this criterion. The plots show
unstable performance; low TD loss does not establish a successful controller.

## Fixes

- Restore Python, NumPy, PyTorch, and available accelerator RNG states from checkpoints.
- Wait until replay holds a complete batch; reject batches larger than capacity.
- Store a finite placeholder for nonfinite terminal observations so divergence
  cannot crash replay insertion. Nonterminal observations must remain finite.
- Break greedy ties using actual forcing magnitude, including asymmetric action ranges.
- Avoid blocking macOS font discovery when generating plots.
- Save run data and plots independently of the now-optional tabular comparison.
- Add `--plot-run` to regenerate figures without training.
- Display the final control interval, label evaluation figures with the final
  checkpoint episode, include training returns in the shared diagnostics, and use a logarithmic loss axis when
  all finite losses are positive so the initial spike does not hide later values.
- Repair a stale tabular plot filename test using explicit input settings.

## Interpretation limits

All 40 default evaluation starts also occur in the training ensemble. These are
repeatable evaluations on familiar initial states, not held-out generalization
tests. The current 21/40 checkpoint requirement is a fixed experiment rule.

Checkpoints restore model, optimizer, and random state, but do not contain replay
transitions or the training-environment sampler position. They cannot reproduce
an uninterrupted training continuation.

The discount factor 0.991 has an effective horizon of about 111 integration steps,
while evaluation lasts 5,521 steps. This is an experiment-design choice that may
limit long-term control. No hyperparameters were tuned and no new 800-episode
training run was performed during this audit.

## Outputs and reproduction

See [plot overview](plot_overview.png), [individual plots](plots/), and
[machine-readable checks](audit.json). The original saved experiment is preserved.

The project `.venv` contains cloud-only dependency files that blocked imports, so
verification used `/private/tmp/dqn-audit-venv` with dependencies recorded in
`audit.json`. From the workspace root:

```sh
MPLCONFIGDIR=/private/tmp/dqn-audit-mpl /private/tmp/dqn-audit-venv/bin/python -m unittest discover -s Q-Learning -p 'Test*.py'

MPLCONFIGDIR=/private/tmp/dqn-audit-mpl /private/tmp/dqn-audit-venv/bin/python Q-Learning/DeepQLearning.py --plot-run Q-Learning/deep_q_learning_runs/20261006-205250_ep800_lr0.0003_gamma0.991_actions9_ddqn/deep_q_learning_run.pkl.gz --output-directory Q-Learning/deep_q_learning_audit/plots
```
