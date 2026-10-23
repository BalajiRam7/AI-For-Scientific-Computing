"""Collect saved experiment results and measure validation query costs."""
from pathlib import Path
import argparse
import json
import numpy as np
import burgers_reference as br
import metrics
from common import configure, median_seconds, write_json


def read(path):
    return json.loads(Path(path).read_text())


def fmt(x):
    return "undefined" if x is None else f"{x:.7g}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def classical_query(initial, nu, T, solver_n, method="implicit"):
    """Solve on solver_n nodes and return values on the input grid."""
    initial = np.asarray(initial, dtype=np.float64)
    n = len(initial)
    if n % solver_n:
        raise ValueError("the internal grid must be a subset of the input grid")
    grid_input = initial[::n//solver_n].copy()
    solver = {"explicit": br.burgers_fd_explicit, "implicit": br.burgers_fd_implicit}[method]
    _, field = solver(solver_n, nu, T, grid_input)
    return br.periodic_linear_resample(field, n)


def calibrate_cost(root, fno, data_path):
    """Measure validation errors and query times of the Euler candidates and select the
    fastest one whose mean validation error is no larger than the FNO's (Question 13(d))."""
    with np.load(data_path, allow_pickle=False) as data:
        n = int(data['train_resolution']); nu, T = float(data['nu']), float(data['T'])
        initial, target, seeds = [data[k].copy() for k in
                                 (f'val_u0_{n}', f'val_uT_{n}', 'val_seeds')]
    prediction = np.load(root/'fno/validation_prediction.npz', allow_pickle=False)['pred']
    learned_error = float(np.mean([metrics.rel_l2(p, y) for p, y in zip(prediction, target)]))
    rows = []
    for method in ('explicit', 'implicit'):
        for solver_n in (64, 128, 256):
            outputs = [classical_query(u0, nu, T, solver_n, method) for u0 in initial]
            errors = [metrics.rel_l2(p, y) for p, y in zip(outputs, target)]
            timings = [median_seconds(lambda u0=u0: classical_query(u0, nu, T, solver_n, method),
                                      warmup=1, repeats=5) for u0 in initial[:5]]
            seconds = float(np.median([r['median_seconds'] for r in timings]))
            rows.append({'method': method, 'solver_n': solver_n, 'output_n': n,
                         'validation_mean_rel_l2': float(np.mean(errors)),
                         'validation_std_rel_l2': float(np.std(errors)), 'validation_count': len(initial),
                         'query_seconds': seconds, 'per_ic_timings': timings,
                         'meets_fno_mean_error': float(np.mean(errors)) <= learned_error})
    feasible = [r for r in rows if r['meets_fno_mean_error']]
    best = min(feasible, key=lambda r: r['query_seconds']) if feasible else None
    data_meta = read(Path(data_path).with_suffix('.json'))
    label_seconds = sum(data_meta['split_generation_seconds'][s] for s in ('train','val'))
    query = fno['family_query_timing_256']['median_seconds']
    timing_seeds = fno['family_query_timing_256']['validation_seed_ids']
    if timing_seeds != seeds[:5].tolist():
        raise ValueError('FNO/classical timing examples do not match')
    setup_seconds = label_seconds + fno['training_seconds']
    result = {'validation_seed_ids': seeds.tolist(), 'validation_count': len(initial),
              'timing_validation_seed_ids': timing_seeds, 'timing_warmup': 1, 'timing_repeats': 5,
              'fno_mean_error_same_validation_subset': learned_error,
              'fno_query_seconds': query, 'classical_candidates': rows, 'selected': best,
              'training_plus_checkpoint_selection_seconds': fno['training_seconds'],
              'fitting_label_generation_seconds': label_seconds,
              'test_label_generation_seconds_excluded': data_meta['split_generation_seconds']['test'],
              'one_time_reference_checks_seconds_excluded': data_meta['check_seconds'],
              'setup_seconds': setup_seconds,
              'qualification': 'Each query begins with the same float64 array of 256 input samples and '
              'returns 256 output values. Both methods use the first five validation inputs, one '
              'untimed call and five timed calls per input; the reported cost is the median of the '
              'five per-input medians. FNO timing includes float32 conversion and prediction; Euler '
              'timing includes restriction to its internal grid, initial-value copying, any sparse '
              'factorization, all timesteps, and periodic linear output interpolation. Model loading, '
              'imports and reference-error calculations are excluded. The selected classical candidate '
              'is the fastest among the two Euler methods at internal n=64,128,256 whose mean error '
              'on all 40 validation examples is no larger than the FNO mean. Setup includes training '
              'and validation label generation plus training/checkpoint selection; it excludes test '
              'labels, numerical refinement checks, benchmark calibration, and post-training diagnostics.'}
    write_json(root/'cost_calibration.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results'))
    parser.add_argument('--data', type=Path)
    parser.add_argument('--reuse-cost', action='store_true')
    args = parser.parse_args(); root = args.root
    configure(0,1,'cpu')
    data_path = args.data or root/'burgers_data.npz'
    classical = read(root/'classical/metrics.json')
    pinn = [read(root/f'pinn_seed{i}/metrics.json') for i in range(3)]
    focused = read(root/'pinn_focused/metrics.json')
    fno = read(root/'fno/metrics.json')
    cost = read(root/'cost_calibration.json') if args.reuse_cost else calibrate_cost(root,fno,data_path)
    metric_key = 'metrics' if 'metrics' in pinn[0] else 'final_metrics'
    prows = [r[metric_key] for r in pinn]
    intervention = focused[metric_key]
    lines = ['# Experiment results', '',
             'Generated from the saved predictions and metrics of your runs.', '',
             '## Environment', '', '```json', json.dumps(pinn[0]['environment'],indent=2), '```', '',
             '## Classical convergence', '',
             table(['method','n','relative L2','p_n = log2(e_n/e_2n)','mass drift'],
                   [[r['method'],r['n'],fmt(r['relative_l2']),fmt(r['observed_order']),fmt(r['mass_drift'])]
                    for r in classical['convergence']]), '',
             'Reference checks: `'+json.dumps(classical['reference_checks'])+'`.', '',
             '## PINN runs', '',
             table(['run','relative L2','absolute L2','local rel. inside','local rel. outside','training seconds'],
                   [[f'seed {i}',fmt(r['rel_l2']),fmt(r['abs_l2']),fmt(r['err_in_layer']),
                     fmt(r['err_outside_layer']),fmt(pinn[i]['training_seconds'])] for i,r in enumerate(prows)] +
                   [['focused seed 0',fmt(intervention['rel_l2']),fmt(intervention['abs_l2']),
                     fmt(intervention['err_in_layer']),fmt(intervention['err_outside_layer']),fmt(focused['training_seconds'])]]), '',
             table(['run','E_I (squared error inside)','E_O (squared error outside)','E_I/(E_I+E_O)'],
                   [[name,fmt(r['squared_error_in_layer']),fmt(r['squared_error_outside_layer']),
                     fmt(r['error_fraction_in_layer'])]
                    for name,r in [(f'seed {i}',r) for i,r in enumerate(prows)] + [('focused seed 0',intervention)]]), '',
             f"Mean over seeds: {np.mean([r['rel_l2'] for r in prows]):.8g}; sample standard deviation: "
             f"{np.std([r['rel_l2'] for r in prows],ddof=1):.8g}.", '',
             '## FNO', '',
             f"Selected epoch {fno['best_epoch']}; training with validation: {fno['training_seconds']:.6f} s.", '',
             table(['split at n=256','mean relative L2','population standard deviation'],
                   [[name,fmt(fno[key]['rel_l2']['mean']),fmt(fno[key]['rel_l2']['std'])]
                    for name,key in [('train','train_metrics'),('validation','validation_metrics')]] +
                   [['test',fmt(fno['resolutions']['256']['aggregate']['rel_l2']['mean']),
                     fmt(fno['resolutions']['256']['aggregate']['rel_l2']['std'])]]), '',
             table(['n','test mean relative L2','population SD','mean difference from 1024-node prediction'],
                   [[n,fmt(r['aggregate']['rel_l2']['mean']),fmt(r['aggregate']['rel_l2']['std']),
                     fmt(r['coarse_fine_prediction_difference_over_true_norm']['mean'])]
                    for n,r in fno['resolutions'].items()]), '',
             '## Stress test', '',
             table(['case','absolute error','relative error','reference norm','grid sensitivity','time sensitivity'],
                   [[name]+[fmt(r[k]) for k in ('abs_l2','rel_l2','true_l2',
                     'reference_refinement_relative_to_truth','reference_fixed_grid_time_halving_relative_to_truth')]
                    for name,r in fno['stress_tests'].items()]), '',
             '## Query costs', '',cost['qualification'], '',
             table(['Euler method','internal n','mean validation error','seconds per query','meets FNO mean'],
                   [[r['method'],r['solver_n'],fmt(r['validation_mean_rel_l2']),fmt(r['query_seconds']),r['meets_fno_mean_error']]
                    for r in cost['classical_candidates']]), '',
             f"FNO query: {cost['fno_query_seconds']:.9f} s; train+validation labels: "
             f"{cost['fitting_label_generation_seconds']:.6f} s; training+selection: "
             f"{cost['training_plus_checkpoint_selection_seconds']:.6f} s.", '',
             f"Setup S = {cost['fitting_label_generation_seconds'] + cost['training_plus_checkpoint_selection_seconds']:.6f} s.", '',
             f"Excluded from S: test-label generation {cost['test_label_generation_seconds_excluded']:.6f} s; "
             f"numerical reference checks {cost['one_time_reference_checks_seconds_excluded']:.6f} s.", '']
    if cost['selected']:
        r=cost['selected']
        lines += [f"Fastest qualifying candidate: {r['method']} at n={r['solver_n']}.", '',
                  f"Measured query times: c_N = {cost['fno_query_seconds']:.9f} s; "
                  f"c_C = {r['query_seconds']:.9f} s.", '']
    else:
        lines += ['No tested Euler candidate meets the FNO validation mean error, '
                  'so no classical competitor is selected.', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))
    print('wrote',root/'RESULTS.md')


if __name__ == '__main__':
    main()
