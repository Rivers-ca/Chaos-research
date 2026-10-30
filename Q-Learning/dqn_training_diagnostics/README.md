# DQN training diagnostics

800 episodes · 4,416,800 environment steps · 4,406,801 gradient updates. Final checkpoint: 17/40 successful trials; zero forcing: 12/40. Mean occupancy gain: +1.43 percentage points.

Saved loss, gradient, and Q-value logs are all finite. Training divergence count: 0. The final checkpoint misses the experiment's 21-trial success requirement. The last three checkpoints do not all meet the selection rule. Selected eligible checkpoint: 350. A small training loss should be interpreted together with these policy metrics.

Evaluation starts are familiar training-ensemble states, not held-out generalization tests. Only one training seed is assessed. Saved data do not include TD-error distributions, Bellman targets, replay contents, or per-update target-network differences; these cannot be reconstructed faithfully. Numerical stability and policy success are assessed separately.

Open `index.html` to browse the full report. Plots 01–07 are the main diagnostic dashboards; `standard_plots/` contains the existing trajectory and evaluation plots.

Source archive: `/Users/riverscalareso/Desktop/Python/Q-Learning/q_learning_plots/Saved_Plots/DQN/20261022-215212_ep800_lr0.0003_gamma0.991_alpha0.01_forcing2.5_actions9_tanh_ddqn/deep_q_learning_run.pkl.gz`

Reproduce from the workspace root:

```sh
python Q-Learning/DiagnoseDQN.py --run "Q-Learning/q_learning_plots/Saved_Plots/DQN/20261022-215212_ep800_lr0.0003_gamma0.991_alpha0.01_forcing2.5_actions9_tanh_ddqn/deep_q_learning_run.pkl.gz" --output "Q-Learning/dqn_training_diagnostics"
```
