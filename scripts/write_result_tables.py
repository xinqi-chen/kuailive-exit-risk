"""Regenerate docs/results.md from the verified aggregate outputs of a finished run."""
from pathlib import Path
import json
import shutil
import pandas as pd
from kuailive_exit.config import load_config
from kuailive_exit.provenance import sha256

ROOT = Path(__file__).resolve().parents[1]
KEY_MODELS = ['M0_fixed', 'M1_fixed', 'M_all_fixed', 'M_all_tuned']


def table(frame, columns):
    rows = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for values in frame[columns].itertuples(index=False, name=None):
        rows.append('| ' + ' | '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values) + ' |')
    return '\n'.join(rows)


def manifest(results, run, name):
    """Prefer the processed run directory; fall back to the copy kept under results/."""
    if (run / name).exists():
        shutil.copyfile(run / name, results / name)
    return json.loads((results / name).read_text())


def main():
    cfg = load_config(); run = cfg['data']['run_dir']; results = cfg['data']['result_dir']
    final = results / 'exp_final_ablation'
    run_manifest = json.loads((final / 'run_manifest.json').read_text())
    for cohort, filename, field in [('B', 'metrics_long.csv', 'B_metrics_sha256'), ('C', 'C_metrics_long.csv', 'C_metrics_sha256')]:
        if sha256(final / filename) != run_manifest[field]:
            raise ValueError(f'{cohort} metrics do not match run manifest')
    features = manifest(results, run, 'feature_manifest.json')
    training = manifest(results, run, 'training_manifest.json')
    if (run / 'r0_r4_manifest.json').exists():
        shutil.copyfile(run / 'r0_r4_manifest.json', results / 'r0_r4_manifest.json')
    edges = features['edges_seconds']
    span = {i: f'{edges[i]}–{edges[i + 1]} s' for i in range(len(edges) - 1)}

    text = ['# Results', '',
        f"Run `{cfg['data']['run_id']}`. All tables are generated from the aggregate metric files under `results/{cfg['data']['run_id']}/` by `scripts/write_result_tables.py`.", '',
        'The models are discrete-time survival models: each test row is one visit in one of the non-uniform intervals '
        f"`{edges}` (seconds), and the model predicts the probability that the user leaves within that interval. "
        'The interval metrics below evaluate exactly this output.', '',
        'Cohorts: A trains (first 18 days) and early-stops (next 5 days); B (300 users) and C (1,000 users) are '
        'test cohorts disjoint from A on the last 5 days. Model fitting and early stopping use only supervised '
        'rows and labels from A; the tuned parameters come from an earlier search on A validation. B and C '
        'contribute no supervised training rows or labels. Room-context features aggregate already observed '
        'logs from all released sampled users, including users in B and C, without using their future labels. '
        'C was also evaluated as a test cohort in earlier model experiments. The code, inputs and model files '
        'are hash-frozen before the single C evaluation within this run.', '']

    primaries = {}
    for cohort, filename in [('C', 'C_metrics_long.csv'), ('B', 'metrics_long.csv')]:
        metrics = pd.read_csv(final / filename)
        overall = metrics[metrics.task.eq('interval')]
        text += [f'## {cohort}: interval predictions', '',
                 'All interval rows of the test period, weighted by row. Lower log loss and Brier are better.', '',
                 table(overall, ['model', 'n', 'positive_rate', 'log_loss', 'brier', 'auc', 'ap']), '']
        bins = metrics[metrics.task.eq('interval_bin') & metrics.model.isin(KEY_MODELS)].copy()
        bins['interval_index'] = bins.interval_index.astype(int)
        base = bins[bins.model.eq('M1_fixed')].set_index('interval_index')
        wide = pd.DataFrame({'interval': [span[i] for i in base.index], 'n': base.n.values,
                             'exit_rate': base.positive_rate.values}, index=base.index)
        for m in KEY_MODELS:
            sub = bins[bins.model.eq(m)].set_index('interval_index')
            wide[f'{m}_log_loss'] = sub.log_loss
            wide[f'{m}_auc'] = sub.auc
        text += [f'### {cohort}: by interval', '', 'AUC within an interval compares visits that reached the same elapsed time.', '',
                 table(wide.reset_index(drop=True), list(wide.columns)), '']
        primaries[cohort] = metrics[metrics.task.eq('H30')].set_index('model')

    text += ['## Derived 30-second horizon (H30)', '',
        'For comparison with fixed-horizon formulations, interval hazards are also converted to the probability of '
        'leaving in the next 30 s at checkpoints t = 30, 60, …, 570 s, assuming a constant hazard inside each interval '
        '(a 30 s window inside a 255 s interval receives 30/255 of that interval\'s cumulative hazard). Elapsed time, '
        'broadcast age and clock fields move with the window; other features stay at their value at t. The labels are '
        'observed exits in (t, t + 30 s]: an exit exactly at t + 30 s is included; a visit ending at t is not at risk. This is a secondary, approximate summary; it does not cover the first 30 s of a visit.', '']
    for cohort in ('C', 'B'):
        p = primaries[cohort].reset_index()
        bs = pd.DataFrame([{**{k: r[k] for k in ['model', 'baseline', 'users', 'rows']}, **r['log_loss']}
                           for r in json.loads((final / f'{cohort}_user_bootstrap.json').read_text())])
        text += [f'### {cohort}: H30', '', table(p, ['model', 'n', 'positive_rate', 'log_loss', 'brier', 'auc', 'ap']), '',
                 f'### {cohort}: H30 paired user bootstrap', '',
                 'Difference = candidate minus baseline (negative = better), 2,000 user resamples, seed 20261001. '
                 'Intervals condition on the fitted models and exclude training randomness and dependence between users sharing rooms; '
                 'no multiplicity adjustment.', '',
                 table(bs, ['model', 'baseline', 'users', 'rows', 'difference', 'ci95_lower', 'ci95_upper']), '']

    text += ['## Training and learning curve', '',
             'M_all_tuned uses parameters from an earlier search on the validation days of cohort A; they were not re-searched in this run. '
             'Best iteration per model (cap 3,000):', '',
             table(pd.DataFrame([{'model': k, 'best_iteration': v['best_iteration']} for k, v in training['models'].items()]), ['model', 'best_iteration']), '']
    lc = pd.read_csv(results / 'exp_learning_curve/metrics_long.csv')
    text += ['Learning curve on B (models trained on nested subsets of A):', '',
             table(lc[lc.task.eq('interval')], ['model', 'training_users', 'n', 'log_loss', 'auc', 'best_iteration']), '',
             f'![Learning curve](../results/{cfg["data"]["run_id"]}/exp_learning_curve/learning_curve.png)', '']
    audit = json.loads((results / 'exp_protocol_audit/summary.json').read_text())
    text += ['## Label audit', '',
             f"A/train visits: {audit['visits']:,}; end classification: `{audit['cause_counts']}`. The delay histogram and the nine broadcast-end window settings are in `results/{cfg['data']['run_id']}/exp_protocol_audit/`.", '',
             'Test-period duration summaries (quality-passing visits classified as user exits):', '',
             table(pd.DataFrame([{'cohort': k, **v} for k, v in features['test_exit_duration'].items()]),
                   ['cohort', 'valid_user_exit_visits', 'exit_within_5s_fraction', 'exit_within_30s_fraction', 'exit_within_35s_fraction']), '',
             '## Scope', '',
             '- Results describe these LightGBM models and features; small increments do not show that a kind of information carries no signal.',
             '- One trained seed, shared-room dependence and iteration caps limit inference. Repeated seeds, room/time resampling and data from another period would strengthen the conclusions.',
             '- Raw inputs, processed rows, model files and predictions are not distributed. Download the pinned raw inputs and run the commands in `reproduce.md`.', '']
    (ROOT / 'docs/results.md').write_text('\n'.join(text))
    print('Updated docs/results.md from verified aggregate results')


if __name__ == '__main__':
    main()
