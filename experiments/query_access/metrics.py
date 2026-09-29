"""Stream/fact statistics, stream bootstrap, and leave-stream-out calibration."""
import csv
import json
from pathlib import Path

import numpy as np


PREDICTORS = ('cum_state', 'cum_fixed', 'cum_delta_norm')
TARGETS = ('access_margin', 'access_gap', 'access_specific_failure')


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def correlation(x, y, spearman=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if spearman:
        x, y = ranks(x), ranks(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap(rows, statistic, repetitions, seed):
    # Resample whole streams. All facts (and their dependencies) stay together.
    groups = {}
    for row in rows:
        groups.setdefault(row['seed'], []).append(row)
    if len(groups) < 3:
        return None
    rng = np.random.default_rng(seed)
    streams = list(groups.values())
    values = []
    for _ in range(repetitions):
        sample = [r for idx in rng.integers(len(streams), size=len(streams)) for r in streams[idx]]
        value = statistic(sample)
        if value is not None and np.isfinite(value):
            values.append(value)
    if len(values) < max(20, repetitions // 2):
        return None
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def association(rows, x, y, repetitions, seed):
    result = {'n_rows': len(rows),
              'n_stream_facts': len({(r['seed'], r.get('fact_id', index)) for index, r in enumerate(rows)})}
    for name, ranked in [('pearson', False), ('spearman', True)]:
        def statistic(sample):
            return correlation([r[x] for r in sample], [r[y] for r in sample], ranked)
        result[name] = statistic(rows)
        result[name + '_ci95'] = bootstrap(rows, statistic, repetitions, seed)
    return result


def leave_stream_out(rows):
    seeds = sorted({r['seed'] for r in rows})
    if len(seeds) < 3:
        return {'status': 'insufficient_streams', 'required_streams': 3}
    predictions = []
    for held in seeds:
        train = [r for r in rows if r['seed'] != held]
        test = [r for r in rows if r['seed'] == held]
        for predictor in PREDICTORS:
            x = np.array([r[predictor] for r in train], dtype=float)
            mean, scale = float(x.mean()), float(x.std()) or 1.0
            design = np.column_stack([np.ones(len(train)), (x - mean) / scale])
            test_x = np.array([r[predictor] for r in test])
            test_design = np.column_stack([np.ones(len(test)), (test_x - mean) / scale])
            for target in TARGETS:
                y = np.array([r[target] for r in train], dtype=float)
                coef = np.linalg.lstsq(design, y, rcond=None)[0]
                threshold = None
                if target == 'access_specific_failure':
                    fitted = design @ coef
                    unique = np.unique(fitted)
                    candidates = np.r_[np.nextafter(unique[0], -np.inf),
                                       (unique[:-1] + unique[1:]) / 2,
                                       np.nextafter(unique[-1], np.inf)]
                    threshold = float(max(candidates, key=lambda t: np.mean((fitted >= t) == y)))
                for row, pred in zip(test, test_design @ coef):
                    predictions.append({'seed': held, 'fact_id': row['fact_id'],
                                        'training_seeds': [s for s in seeds if s != held],
                                        'predictor': predictor, 'target': target,
                                        'prediction': float(pred), 'observed': float(row[target]),
                                        'train_x_mean': mean, 'train_x_scale': scale,
                                        'train_coefficients': coef.tolist(), 'train_threshold': threshold,
                                        'predicted_failure': None if threshold is None else bool(pred >= threshold)})
    scores = {}
    for predictor in PREDICTORS:
        scores[predictor] = {}
        for target in TARGETS:
            subset = [r for r in predictions if r['predictor'] == predictor and r['target'] == target]
            p, y = [r['prediction'] for r in subset], [r['observed'] for r in subset]
            score = {'pearson': correlation(p, y), 'spearman': correlation(p, y, True),
                     'mse': float(np.mean((np.array(p) - y) ** 2))}
            if target == 'access_specific_failure':
                score['accuracy'] = float(np.mean([r['predicted_failure'] == r['observed'] for r in subset]))
            scores[predictor][target] = score
    return {'status': 'ok', 'scores': scores, 'predictions': predictions}


def summarize(trajectory, interference, output, repetitions=1000, bootstrap_seed=0):
    seeds = sorted({r['seed'] for r in trajectory})
    end = {s: max(r['write_idx'] for r in trajectory if r['seed'] == s) for s in seeds}
    final = [r for r in trajectory if r['write_idx'] == end[r['seed']]]
    immediate = [r for r in trajectory if r['write_idx'] == r['eval_fact_idx']]
    # The last fact has no later writes; exclude it from forgetting prediction and rates.
    exposed = [r for r in final if r['age'] > 0]
    step_groups = {}
    for r in interference:
        step_groups.setdefault((r['seed'], r['fact_id']), []).append(r)
    step_facts = [{'seed': s, 'fact_id': f, **{key: float(np.mean([r[key] for r in records]))
                   for key in ('I_state', 'I_fixed', 'actual_delta_D')}}
                  for (s, f), records in step_groups.items()]
    associations = {f'{x}_vs_{y}': association(exposed, x, y, repetitions, bootstrap_seed)
                    for x in PREDICTORS for y in TARGETS}
    one_step = {x: association(step_facts, x, 'actual_delta_D', repetitions, bootstrap_seed)
                for x in ('I_state', 'I_fixed')}
    transitions = {x: association(interference, x, 'actual_delta_D', repetitions, bootstrap_seed)
                   for x in ('I_state', 'I_fixed')}
    def difference(rows):
        a = correlation([r['cum_state'] for r in rows], [r['access_margin'] for r in rows], True)
        b = correlation([r['cum_fixed'] for r in rows], [r['access_margin'] for r in rows], True)
        return None if a is None or b is None else a - b
    def average(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None
    result = {
        'num_streams': len(seeds), 'num_final_stream_facts': len(exposed),
        'immediate_edit_success': average(immediate, 'write_em'),
        'immediate_access_em': average(immediate, 'access_em'),
        'final_retention': average(exposed, 'access_em'),
        'final_positive_access_margin_rate': (float(np.mean([r['access_margin'] > 0 for r in exposed])) if exposed else None),
        'access_specific_failure_rate': average(exposed, 'access_specific_failure'),
        'latest_write_intrusion_rate': average(exposed, 'latest_write_intrusion'),
        'rho_state_one_step': one_step['I_state']['spearman'],
        'rho_fixed_one_step': one_step['I_fixed']['spearman'],
        'rho_cumulative_state_final_access': associations['cum_state_vs_access_margin']['spearman'],
        'rho_cumulative_fixed_final_access': associations['cum_fixed_vs_access_margin']['spearman'],
        'rho_cumulative_norm_final_access': associations['cum_delta_norm_vs_access_margin']['spearman'],
        'delta_rho_state_minus_fixed': difference(exposed),
        'bootstrap_ci_delta_rho': bootstrap(exposed, difference, repetitions, bootstrap_seed),
        'associations': associations, 'one_step': one_step,
        'one_step_transition_diagnostics': transitions,
        'leave_stream_out': leave_stream_out(exposed),
        'definitions': {
            'one_step_unit': 'One mean over later writes per stream/fact; raw paired steps are in interference JSONL.',
            'transition_diagnostics': 'Descriptive paired-transition correlation with whole-stream bootstrap, not independent timepoint inference.',
            'bootstrap': 'Percentile 95% CI; whole streams resampled; unavailable with fewer than 3 streams.',
            'prediction_cohort': 'Final rows with at least one later write (age > 0).',
            'final_retention': 'Mean final access EM over exposed facts; not conditioned on initial success.',
            'delta_rho': 'Signed rho(state, final access) minus rho(fixed, final access); negative may be stronger forgetting association.',
            'thresholds': 'Storage/context thresholds prespecified by CLI; failure classifier cutoff and linear calibration fit on training streams only.',
        },
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'query_access_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    flat = {}
    def flatten(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                flatten(prefix + '.' + k if prefix else k, v)
        elif not isinstance(value, (list, tuple)):
            flat[prefix] = value
        elif prefix.endswith('ci95') or prefix.startswith('bootstrap_ci'):
            flat[prefix] = json.dumps(value)
    flatten('', {k: v for k, v in result.items() if k not in ('definitions', 'leave_stream_out')})
    if result['leave_stream_out']['status'] == 'ok':
        flatten('leave_stream_out', result['leave_stream_out']['scores'])
    with (output / 'query_access_summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(flat))
        writer.writeheader()
        writer.writerow(flat)
    return result
