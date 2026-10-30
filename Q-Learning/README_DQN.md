# Lorenz Double DQN

`DeepQLearning.py` trains a Double DQN controller on the Lorenz environment in
`QLearning.py`. It uses normalized continuous states, discrete forcing actions,
experience replay, a target network, Huber loss, and gradient clipping.
`PlotQLearning.py` supplies the shared trajectory and evaluation plots.

## Run the code

From the repository root, use a Python environment with NumPy, PyTorch,
Matplotlib, and Pillow installed:

```sh
python Q-Learning/DeepQLearning.py --smoke --device cpu
python Q-Learning/DeepQLearning.py --device auto
```

The first command performs a small end-to-end check. The second uses the full
experiment defaults, including 800 episodes and two hidden layers of 128 neurons.
Experiment settings are defined by `DeepExperimentDefaults` in the source.
Each invocation creates a new directory under
`Q-Learning/q_learning_plots/Saved_Plots/DQN/` containing run data, plots,
parameters, and model checkpoints.

## Plot an existing run

```sh
python Q-Learning/DeepQLearning.py --plot-run /path/to/deep_q_learning_run.pkl.gz
python Q-Learning/DiagnoseDQN.py --run /path/to/deep_q_learning_run.pkl.gz --output Q-Learning/dqn_training_diagnostics
```

The diagnostic command creates seven overview figures, nineteen standard plots,
an HTML report, and machine-readable metrics. It does not train the agent.
It simulates zero forcing from each saved evaluation start over the same horizon.
Only load trusted run archives; the archive format uses Python pickle.

The [October 22 diagnostic report](dqn_training_diagnostics/README.md) summarizes
the existing 800-episode tanh run. Download the report directory and open
`index.html` locally to browse all plots. Evaluation starts belong to the training
ensemble, so these results do not establish held-out generalization.

## Tests

```sh
python -m unittest discover -s Q-Learning -p 'Test*.py'
```

The macOS startup regression requires a Matplotlib release that supports
`MPL_IGNORE_SYSTEM_FONTS`; older releases can fail that check even when plotting
works with an existing font cache.

## Saved artifacts

Plots, settings, small smoke-run data, metrics, and audit notes are included in
the repository. The two full 800-episode pickle archives are approximately
150 MiB each and remain local because they exceed GitHub's normal per-file
limit. Model checkpoints are also local and excluded by the existing ignore
rules. To regenerate the full diagnostic report, supply the original archive or
train a new run. Checkpoints do not contain replay-buffer transitions and cannot
reproduce an uninterrupted training continuation.
