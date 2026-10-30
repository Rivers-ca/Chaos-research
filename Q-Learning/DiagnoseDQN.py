"""Create a browsable diagnostic report from an existing Double DQN archive.

No training is performed. Zero-forcing references are integrated from the saved
evaluation starts over the same per-trial horizons. Run with --help for usage.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import DeepQLearning as deep
import QLearning as qlearning
import numpy as np
from matplotlib import pyplot as plt


def rolling(values, window=25):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    total = np.r_[0.0, np.cumsum(np.where(valid, values, 0))]
    count = np.r_[0, np.cumsum(valid)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(0, end - window)
    return np.divide(total[end] - total[start], count[end] - count[start],
                     out=np.full(len(values), np.nan), where=count[end] > count[start])


def binned(values, bins=600):
    """Preserve update spread and spikes without drawing millions of vertices."""
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.linspace(0, len(values), min(bins, len(values)) + 1).astype(int))
    x, stats = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        finite = values[left:right]
        finite = finite[np.isfinite(finite)]
        if finite.size:
            x.append((left + right + 1) / 2)
            stats.append(np.percentile(finite, [10, 50, 90, 100]))
    return np.asarray(x), np.asarray(stats).reshape(-1, 4)


def create_report(run_path: Path, output: Path, include_standard=True):
    run = deep.load_deep_q_learning_run(run_path)
    output.mkdir(parents=True, exist_ok=True)
    h, settings = run['history'], run['settings']
    evaluations = run['checkpoints']['evaluations']
    summaries = run['checkpoints']['summaries']
    final = run['evaluation']
    episodes = np.arange(1, len(h['episode_rewards']) + 1)
    checkpoints = np.asarray(run['checkpoints']['episodes'])
    sections = []
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': .18, 'font.size': 10})

    def save(fig, name, caption):
        fig.tight_layout(rect=(0, 0, 1, .95))
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        sections.append((name, caption))

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Learning progress | exploration and greedy evaluation')
    rewards = np.asarray(h['episode_rewards'])
    ax[0, 0].plot(episodes, rewards, alpha=.25, label='Training episode')
    ax[0, 0].plot(episodes, rolling(rewards), label='Trailing 25-episode mean')
    means = [np.mean(e['episode_rewards']) for e in evaluations]
    stds = [np.std(e['episode_rewards']) for e in evaluations]
    ax[0, 0].errorbar(checkpoints, means, yerr=stds, fmt='o-', label='Greedy evaluation ± trial SD')
    ax[0, 0].set(ylabel='Undiscounted shaped return', title='Does the policy improve?')
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(episodes, h['epsilons'])
    ax[0, 1].set(ylabel='Epsilon', title='Exploration probability', ylim=(0, 1.05))
    lengths = np.asarray(h['episode_lengths'])
    ax[1, 0].plot(episodes, lengths, label='Episode length')
    ax[1, 0].scatter(episodes[np.asarray(h['diverged'], bool)], lengths[np.asarray(h['diverged'], bool)], color='red', label='Diverged')
    ax[1, 0].set(ylabel='Environment steps', title='Early termination / divergence')
    ax[1, 0].legend()
    occupancy = 100 * np.asarray(h['target_steps']) / np.maximum(lengths, 1)
    ax[1, 1].plot(episodes, occupancy, alpha=.25)
    ax[1, 1].plot(episodes, rolling(occupancy))
    ax[1, 1].set(ylabel='Steps in fixed-point target (%)', title='Training target residence')
    for a in ax.flat: a.set_xlabel('Training episode')
    save(fig, '01_learning_progress.png', 'Training includes exploratory actions; greedy evaluation does not. Error bars show spread across familiar initial states, not independent training seeds. Fixed-point residence and negative-lobe occupancy are different objectives.')

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Optimizer stability | actual saved gradient updates')
    for a, key, title in [(ax[0, 0], 'losses', 'Huber loss'), (ax[0, 1], 'gradient_norms', 'Gradient norm before clipping')]:
        x, stats = binned(h[key])
        if len(x):
            a.fill_between(x, stats[:, 0], stats[:, 2], alpha=.22, label='10–90% within update bin')
            a.plot(x, stats[:, 1], label='Median')
            a.plot(x, stats[:, 3], alpha=.35, lw=.7, label='Maximum')
            if np.all(stats > 0): a.set_yscale('log')
        a.set(xlabel='Gradient update', ylabel=title, title=title)
        a.legend(fontsize=8)
    clip = settings['gradient_clip_norm']
    ax[0, 1].axhline(clip, color='red', ls='--', label='Clip threshold')
    ax[0, 1].legend(fontsize=8)
    gradients = np.asarray(h['gradient_norms'], float)
    # Mean, rather than percentile, is needed for a clipping frequency.
    edges = np.unique(np.linspace(0, len(gradients), min(600, len(gradients)) + 1).astype(int))
    fractions = [100 * np.mean(gradients[l:r] > clip) for l, r in zip(edges[:-1], edges[1:])]
    ax[1, 0].plot((edges[:-1] + edges[1:] + 1) / 2, fractions)
    ax[1, 0].set(xlabel='Gradient update', ylabel='Updates clipped (%)', ylim=(0, 100), title='How often clipping intervenes')
    for key, label in [('mean_q_values', 'Batch mean Q'), ('max_q_values', 'Mean over batch of max-action Q')]:
        x, stats = binned(h[key])
        if len(x): ax[1, 1].plot(x, stats[:, 1], label=label)
    ax[1, 1].set(xlabel='Gradient update', ylabel='Predicted discounted value', title='Q-value drift')
    ax[1, 1].legend(fontsize=8)
    save(fig, '02_optimizer_stability.png', 'Binned quantiles retain update variability and maxima. A finite loss or falling loss alone does not establish policy quality. Q-values are discounted estimates and cannot be compared directly with full-episode undiscounted returns.')

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Data collection and update accounting')
    steps = np.asarray(h['global_steps'])
    ax[0].plot(episodes, h['replay_buffer_sizes'])
    ax[0].axhline(settings['minimum_replay_size'], ls='--', color='orange', label='Learning warm-up')
    ax[0].axhline(settings['replay_buffer_size'], ls=':', color='black', label='Capacity')
    ax[0].set(xlabel='Episode', ylabel='Stored transitions', title='Replay fill'); ax[0].legend(fontsize=8)
    ax[1].plot(episodes, steps)
    ax[1].set(xlabel='Episode', ylabel='Cumulative environment steps', title='Data collected')
    ax[2].plot(steps, h['epsilons'])
    ax[2].axvline(settings['epsilon_decay_steps'], ls='--', color='orange', label='Scheduled decay end')
    ax[2].set(xlabel='Environment step', ylabel='Epsilon', title='Exploration vs data'); ax[2].legend(fontsize=8)
    save(fig, '03_replay_and_exploration.png', 'Replay capacity and warm-up are shown explicitly. Update counts and finite-value checks are recorded in metrics.json.')

    trial_occupancy = np.asarray([e['episode_negative_lobe_percentages'] for e in evaluations], float)
    successes = np.sum(trial_occupancy >= 51, axis=1)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Checkpoint policy quality and consistency')
    ax[0, 0].plot(checkpoints, successes, 'o-')
    selected = run.get('best_checkpoint_episode')
    if selected is not None:
        selected_index = list(checkpoints).index(selected)
        ax[0, 0].scatter([selected], [successes[selected_index]], marker='*', s=200,
                         color='orange', zorder=5, label=f'Selected model: episode {selected}')
    ax[0, 0].axhline(21, color='red', ls='--', label='Saved experiment rule: 21 trials')
    ax[0, 0].set(ylabel='Trials with ≥51% negative-lobe occupancy', ylim=(0, trial_occupancy.shape[1] + 1)); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(checkpoints, np.mean(trial_occupancy, axis=1), 'o-', label='Mean across trials')
    ax[0, 1].fill_between(checkpoints, np.percentile(trial_occupancy, 10, axis=1), np.percentile(trial_occupancy, 90, axis=1), alpha=.2, label='10–90% across trials')
    ax[0, 1].axhline(51, color='red', ls='--'); ax[0, 1].set(ylabel='Negative-lobe occupancy (%)', ylim=(0, 100)); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(checkpoints, [s['normalized_control_effort'] for s in summaries], 'o-')
    ax[1, 0].set(ylabel='Mean (forcing / u_ref)²', title='Control effort')
    ax[1, 1].plot(checkpoints, [100 * s['divergence_rate'] for s in summaries], 'o-')
    ax[1, 1].set(ylabel='Diverged trials (%)', title='Evaluation divergence', ylim=(-1, 101))
    for a in ax.flat: a.set_xlabel('Training episode')
    save(fig, '04_checkpoint_quality.png', 'The threshold is the existing experiment selection rule. Among eligible checkpoints, selection prioritizes lower forcing before success count. Later checkpoints can regress; compare the final model with the selected eligible checkpoint.')

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle('Every evaluation start at every checkpoint')
    im = ax.imshow(trial_occupancy.T, origin='lower', aspect='auto', vmin=0, vmax=100, cmap='viridis')
    ax.set_xticks(np.arange(len(checkpoints)), labels=checkpoints, rotation=45)
    ax.set(xlabel='Training episode', ylabel='Evaluation trial (same start across columns)')
    fig.colorbar(im, ax=ax, label='Negative-lobe occupancy (%)')
    save(fig, '05_trial_consistency.png', 'Columns share the same evaluation starts. This reveals gains or failures hidden by averages; these starts also belong to the training ensemble in the default experiment.')

    actions = np.asarray(run['actions'])
    usage = np.asarray([np.bincount(np.concatenate(e['actions']), minlength=len(actions)) for e in evaluations], float)
    usage = 100 * usage / usage.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Policy behavior | control actions')
    im = ax[0, 0].imshow(usage.T, origin='lower', aspect='auto', vmin=0, vmax=100, cmap='magma')
    ax[0, 0].set_xticks(np.arange(len(checkpoints)), labels=checkpoints, rotation=60)
    ax[0, 0].set_yticks(np.arange(len(actions)), labels=[f'{v:g}' for v in actions])
    ax[0, 0].set(xlabel='Episode', ylabel='Forcing'); fig.colorbar(im, ax=ax[0, 0], label='Actions (%)')
    ax[0, 1].bar(actions, usage[-1], width=(actions[1] - actions[0]) * .8)
    ax[0, 1].set(xlabel='Forcing', ylabel='Actions (%)', title='Final policy action distribution')
    ax[1, 0].plot(checkpoints, [100 * s['action_change_rate'] for s in summaries], 'o-')
    ax[1, 0].set(xlabel='Episode', ylabel='Consecutive actions changed (%)', title='Chattering / switching')
    entropy = -np.sum(np.where(usage > 0, (usage / 100) * np.log(np.maximum(usage / 100, 1e-15)), 0), axis=1) / np.log(len(actions))
    ax[1, 1].plot(checkpoints, entropy, 'o-')
    ax[1, 1].set(xlabel='Episode', ylabel='Normalized action entropy', ylim=(0, 1.05), title='Action diversity across evaluation states')
    save(fig, '06_action_behavior.png', 'Low action diversity may be a valid deterministic control strategy or policy collapse; interpret it alongside occupancy and control effort. This is action-frequency entropy across states, not stochastic policy entropy.')

    # Reuse the exact saved starts and horizon, without training or selecting new models.
    baseline = [qlearning.integrate_uncontrolled_lorenz(np.asarray(t)[0], len(t) - 1) for t in final['trajectories']]
    zero_occupancy = np.asarray([qlearning.lobe_time_percentages([t])[0] for t in baseline])
    final_occupancy = np.asarray(final['episode_negative_lobe_percentages'])
    delta = final_occupancy - zero_occupancy
    order = np.argsort(zero_occupancy)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Final model versus zero forcing | paired starts and horizons')
    ax[0, 0].plot(zero_occupancy[order], 'o-', label='Zero forcing')
    ax[0, 0].plot(final_occupancy[order], 'o-', label='Final DQN')
    ax[0, 0].axhline(51, color='gray', ls='--'); ax[0, 0].legend()
    ax[0, 0].set(xlabel='Trials sorted by zero-forcing occupancy', ylabel='Negative-lobe occupancy (%)', ylim=(0, 100))
    ax[0, 1].bar(np.arange(1, len(delta) + 1), delta, color=np.where(delta >= 0, '#20856c', '#c75c4c'))
    ax[0, 1].axhline(0, color='black', lw=.7)
    ax[0, 1].set(xlabel='Saved evaluation trial', ylabel='Occupancy gain (percentage points)')
    for trajectory, label in [(baseline[0], 'Zero forcing'), (np.asarray(final['trajectories'][0]), 'Final DQN')]:
        time = np.arange(len(trajectory)) * qlearning.DT
        ax[1, 0].plot(time, trajectory[:, 0], label=label, alpha=.8, lw=.8)
        cumulative = 100 * np.cumsum(trajectory[1:, 0] < 0) / np.arange(1, len(trajectory))
        ax[1, 1].plot(time[1:], cumulative, label=label)
    ax[1, 0].set(xlabel='Simulation time', ylabel='x coordinate', title='Trial 1 state evolution'); ax[1, 0].legend(fontsize=8)
    ax[1, 1].set(xlabel='Simulation time', ylabel='Cumulative negative-lobe occupancy (%)', title='Trial 1 residence over time'); ax[1, 1].legend(fontsize=8)
    save(fig, '07_zero_forcing_comparison.png', 'Zero forcing is newly simulated with the existing Euler integrator at each saved start for the matching horizon. Chaotic trajectories separate rapidly; compare residence statistics rather than pointwise tracking errors.')

    checks = {}
    for key in ['losses', 'gradient_norms', 'mean_q_values', 'max_q_values']:
        values = np.asarray(h[key], float)
        checks[key] = {'count': len(values), 'nonfinite_count': int(np.sum(~np.isfinite(values)))}
    metrics = {
        'source_run': str(run_path.resolve()), 'episodes': len(episodes),
        'environment_steps': int(steps[-1]), 'gradient_updates': len(h['losses']),
        'final_epsilon': float(h['epsilons'][-1]), 'finite_checks': checks,
        'gradient_clip_fraction': float(np.mean(gradients > clip)) if len(gradients) else None,
        'first_checkpoint_successes': int(successes[0]), 'final_checkpoint_successes': int(successes[-1]),
        'peak_successes': int(np.max(successes)), 'best_selected_episode': run.get('best_checkpoint_episode'),
        'final_three_meet_selection_rule': bool(run['criterion_satisfied_final_three']),
        'training_divergences': int(np.sum(h['diverged'])),
        'final_evaluation_divergences': int(np.sum(final['diverged'])),
        'final_mean_negative_lobe_occupancy': float(np.mean(final_occupancy)),
        'zero_forcing_mean_negative_lobe_occupancy': float(np.mean(zero_occupancy)),
        'mean_occupancy_gain_percentage_points': float(np.mean(delta)),
        'trials_improved_over_zero_forcing': int(np.sum(delta > 0)),
        'zero_forcing_successes': int(np.sum(zero_occupancy >= 51)),
    }
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    if include_standard:
        standard = deep.plot_deep_q_learning(run, output / 'standard_plots')
        sections.extend((f'standard_plots/{p.name}', p.stem.replace('_', ' ').capitalize()) for p in standard if p.exists())
    overview = (f"{len(episodes)} episodes · {int(steps[-1]):,} environment steps · "
                f"{len(h['losses']):,} gradient updates. Final checkpoint: "
                f"{successes[-1]}/{trial_occupancy.shape[1]} successful trials; "
                f"zero forcing: {metrics['zero_forcing_successes']}/{trial_occupancy.shape[1]}. "
                f"Mean occupancy gain: {np.mean(delta):+.2f} percentage points.")
    limitations = ('Evaluation starts are familiar training-ensemble states, not held-out generalization tests. '
                   'Only one training seed is assessed. Saved data do not include TD-error distributions, '
                   'Bellman targets, replay contents, or per-update target-network differences; these cannot be reconstructed faithfully. '
                   'Numerical stability and policy success are assessed separately.')
    finite = all(item['nonfinite_count'] == 0 for item in checks.values())
    assessment = (f"Saved loss, gradient, and Q-value logs are {'all finite' if finite else 'not all finite'}. "
                  f"Training divergence count: {metrics['training_divergences']}. "
                  f"The final checkpoint {'meets' if successes[-1] >= 21 and metrics['final_evaluation_divergences'] == 0 else 'misses'} "
                  f"the experiment's 21-trial success requirement. "
                  f"The last three checkpoints {'all meet' if metrics['final_three_meet_selection_rule'] else 'do not all meet'} the selection rule. "
                  f"Selected eligible checkpoint: {selected if selected is not None else 'none'}. "
                  'A small training loss should be interpreted together with these policy metrics.')
    body = ''.join(f'<section><h2>{html.escape(caption.split(".")[0])}</h2><a href="{name}"><img loading="lazy" src="{name}"></a><p>{html.escape(caption)}</p></section>' for name, caption in sections)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>DQN diagnostics</title>'
        '<style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#243344;background:#f5f7fa}section{background:white;padding:24px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}h2{font-size:20px}p{line-height:1.6}</style>'
        f'<h1>DQN training diagnostics</h1><p>{html.escape(overview)}</p><p>{html.escape(assessment)}</p><p>{html.escape(limitations)}</p>'
        f'<p>Source: {html.escape(str(run_path))}</p><a href="metrics.json">Machine-readable metrics</a>' + body)
    (output / 'README.md').write_text(f'# DQN training diagnostics\n\n{overview}\n\n{assessment}\n\n{limitations}\n\n'
        'Open `index.html` to browse the full report. Plots 01–07 are the main diagnostic dashboards; '
        '`standard_plots/` contains the existing trajectory and evaluation plots.\n\n'
        f'Source archive: `{run_path.resolve()}`\n\nReproduce from the workspace root:\n\n'
        f'```sh\npython Q-Learning/DiagnoseDQN.py --run "{run_path}" --output "{output}"\n```\n')
    print(json.dumps(metrics, indent=2))
    print(f'Report: {(output / "index.html").resolve()}')
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Existing deep_q_learning_run.pkl.gz')
    parser.add_argument('--output', type=Path, help='Default: diagnostics beside the source archive')
    parser.add_argument('--skip-standard', action='store_true', help='Only generate the seven diagnostic dashboards')
    args = parser.parse_args()
    create_report(args.run, args.output or args.run.parent / 'diagnostics', not args.skip_standard)
