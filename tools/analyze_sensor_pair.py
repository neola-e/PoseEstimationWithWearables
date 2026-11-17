"""Produce paired CSV tables, sequence-cluster bootstrap summaries and publication figures."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'imuposer-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from sensor_metrics import (COMBOS, JOINT_NAMES, REGIONS, METRICS, cluster_bootstrap,
                            direction_counts, pair_windows, validate_coverage, weighted_mean)

ALIASES = {'position_cm': 'mpjpe_cm', 'local_deg': 'local_deg', 'global_deg': 'global_deg'}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison(a, b):
    result = {}
    for metric in METRICS:
        name = ALIASES[metric]
        av, bv = float(a[metric]), float(b[metric])
        result.update({f'global_{name}': av, f'lw_rp_h_{name}': bv,
                       f'delta_{name}': bv - av,
                       f'percent_change_{name}': 100 * (bv - av) / av if av else None})
    return result


def aggregate(rows):
    frames = [r['frames'] for r in rows]
    return {k: float(weighted_mean([r[k] for r in rows], frames)) for k in METRICS}


def joint_aggregate(rows):
    return {k: weighted_mean([r['joint_errors'][k] for r in rows], [r['frames'] for r in rows]) for k in METRICS}


def joint_and_region_tables(a, b, metadata):
    joints, regions = [], []
    for j, name in enumerate(JOINT_NAMES):
        av = {k: a[k][j] for k in METRICS}
        bv = {k: b[k][j] for k in METRICS}
        joints.append({**metadata, 'joint_id': j, 'joint': name, **comparison(av, bv)})
    for name, indices in REGIONS.items():
        av = {k: np.asarray(a[k])[indices].mean() for k in METRICS}
        bv = {k: np.asarray(b[k])[indices].mean() for k in METRICS}
        regions.append({**metadata, 'region': name, 'joint_ids': ' '.join(map(str, indices)),
                        **comparison(av, bv)})
    return joints, regions


def analyze(args):
    run = Path(args.run_dir)
    # Analysis writes its own fresh subdirectory and does not overwrite raw evaluations.
    out = run / args.analysis_name
    out.mkdir(exist_ok=False)
    provenance = json.loads((run / 'provenance.json').read_text())
    reports = {c: json.loads((run / f'{c}_metrics.json').read_text()) for c in COMBOS}
    pairs = pair_windows(reports['global']['per_window'], reports['lw_rp_h']['per_window'])
    coverage = validate_coverage([a for a, b in pairs], provenance['data']['sequence_lengths'])
    if any(a['target_sha256'] != b['target_sha256'] for a, b in pairs):
        raise ValueError('Paired GT hash mismatch')
    window_table, window_joint, window_region = [], [], []
    for a, b in pairs:
        meta = {k: a[k] for k in ['sequence_id', 'start_frame', 'frames']}
        window_table.append({**meta, **comparison(a, b)})
        j, r = joint_and_region_tables(a['joint_errors'], b['joint_errors'], meta)
        window_joint.extend(j)
        window_region.extend(r)
    sequence_table, seq_joint, seq_region = [], [], []
    seq_matrices = {c: [] for c in COMBOS}
    for seq in range(coverage['sequences']):
        rows = {c: [r for r in reports[c]['per_window'] if r['sequence_id'] == seq] for c in COMBOS}
        frame_count = sum(r['frames'] for r in rows['global'])
        meta = {'sequence_id': seq, 'frames': frame_count, 'windows': len(rows['global'])}
        sequence_table.append({**meta, **comparison(aggregate(rows['global']), aggregate(rows['lw_rp_h']))})
        ja, jb = joint_aggregate(rows['global']), joint_aggregate(rows['lw_rp_h'])
        for combo, matrix in [('global', ja), ('lw_rp_h', jb)]:
            seq_matrices[combo].append(matrix)
        j, r = joint_and_region_tables(ja, jb, meta)
        seq_joint.extend(j)
        seq_region.extend(r)
    overall = []
    joint_table, region_table = [], []
    for weighting in ['frame_weighted', 'sequence_equal']:
        joint = {}
        values = {}
        for combo in COMBOS:
            weights = [r['frames'] for r in sequence_table] if weighting == 'frame_weighted' else [1] * len(sequence_table)
            joint[combo] = {k: weighted_mean([m[k] for m in seq_matrices[combo]], weights) for k in METRICS}
            values[combo] = {k: joint[combo][k].mean() for k in METRICS}
            if weighting == 'frame_weighted':
                for k in METRICS:
                    if not np.isclose(values[combo][k], reports[combo]['metrics'][k], atol=2e-5, rtol=0):
                        raise ValueError('Scalar/per-joint aggregation disagreement')
        overall.append({'weighting': weighting, **comparison(values['global'], values['lw_rp_h'])})
        j, r = joint_and_region_tables(joint['global'], joint['lw_rp_h'], {'weighting': weighting})
        joint_table.extend(j)
        region_table.extend(r)
    for filename, rows in [('overall.csv', overall), ('per_sequence.csv', sequence_table),
                           ('per_window.csv', window_table), ('per_joint.csv', joint_table),
                           ('per_region.csv', region_table), ('per_sequence_joint.csv', seq_joint),
                           ('per_sequence_region.csv', seq_region), ('per_window_joint.csv', window_joint),
                           ('per_window_region.csv', window_region)]:
        save_csv(out / filename, rows)
    uncertainty = {metric: cluster_bootstrap([r[f'delta_{ALIASES[metric]}'] for r in sequence_table],
                   [r['frames'] for r in sequence_table], args.bootstrap_draws, args.seed) for metric in METRICS}
    delta = np.array([r['delta_mpjpe_cm'] for r in window_table])
    seqdelta = np.array([r['delta_mpjpe_cm'] for r in sequence_table])
    global_errors = np.array([r['global_mpjpe_cm'] for r in window_table])
    threshold = float(np.quantile(global_errors, .9))
    high = [r for r in window_table if r['global_mpjpe_cm'] >= threshold]
    seq18 = next(r for r in window_table if r['sequence_id'] == 18 and r['start_frame'] == 750)
    seq8 = max((r for r in window_table if r['sequence_id'] == 8), key=lambda r: r['global_mpjpe_cm'])
    focus = []
    for seq in [8, 18]:
        candidates = [r for r in window_table if r['sequence_id'] == seq]
        focus.append({'selection': f'sequence_{seq}_global_best', **min(candidates, key=lambda r: r['global_mpjpe_cm'])})
        focus.append({'selection': f'sequence_{seq}_global_worst', **max(candidates, key=lambda r: r['global_mpjpe_cm'])})
    focus += [{'selection': 'largest_improvement', **min(window_table, key=lambda r: r['delta_mpjpe_cm'])},
              {'selection': 'largest_worsening', **max(window_table, key=lambda r: r['delta_mpjpe_cm'])}]
    save_csv(out / 'failure_cases.csv', focus)
    motion = json.loads((run / 'motion_descriptors.json').read_text())
    paired_motion = pair_windows(window_table, motion)
    exploratory = {}
    for descriptor in ['gt_arm_speed_m_s', 'gt_leg_speed_m_s']:
        x = np.array([b[descriptor] for a, b in paired_motion])
        rho = float(spearmanr(x, delta).statistic)
        cuts = np.quantile(x, [.25, .5, .75])
        bins = np.digitize(x, cuts)
        exploratory[descriptor] = {'spearman_rho_with_mpjpe_delta': rho,
            'quartile_cut_points': cuts.tolist(),
            'quartiles': [{'quartile': q + 1, 'windows': int((bins == q).sum()),
                'mean_delta_cm': float(delta[bins == q].mean()),
                'improved_proportion': float((delta[bins == q] < -1e-5).mean())} for q in range(4)],
            'caution': 'Exploratory correlated windows, post hoc bins; root-relative GT speed, not activity/complexity labels or causal evidence.'}
    # These are input consistency proxies, not proof of corruption or causes of error.
    inputs = json.loads((run / 'input_consistency.json').read_text())
    sensor_table = []
    for s, name in enumerate(['Left wrist proxy', 'Right wrist proxy', 'Left pocket proxy', 'Right pocket proxy', 'Head']):
        sensor_table.append({'sensor_index': s, 'sensor': name, 'retained_in_lw_rp_h': s in COMBOS['lw_rp_h'],
            **{k: float(weighted_mean([r[k][s] for r in inputs], [r['frames'] for r in inputs])) for k in
               ['acceleration_norm_mean_m_s2', 'orientation_orthogonality_frobenius_mean', 'approx_orientation_vs_gt_segment_deg']}})
    save_csv(out / 'input_consistency_by_sensor.csv', sensor_table)
    region_direction = {name: direction_counts([r['delta_mpjpe_cm'] for r in seq_region if r['region'] == name]) for name in REGIONS}
    leave_one_out = []
    for omit in range(len(sequence_table)):
        included = [r for r in sequence_table if r['sequence_id'] != omit]
        leave_one_out.append({'omitted_sequence': omit,
            'frame_weighted_delta_cm': float(weighted_mean([r['delta_mpjpe_cm'] for r in included], [r['frames'] for r in included])),
            'sequence_equal_delta_cm': float(np.mean([r['delta_mpjpe_cm'] for r in included]))})
    save_csv(out / 'leave_one_sequence_out.csv', leave_one_out)
    summary = {'coverage': coverage, 'overall': overall, 'sequence_direction': direction_counts(seqdelta),
        'window_direction': direction_counts(delta),
        'improved_window_frame_proportion': float(sum(r['frames'] for r in window_table if r['delta_mpjpe_cm'] < -1e-5) / coverage['frames']),
        'window_delta_quantiles': dict(zip(['min', 'q25', 'median', 'q75', 'max'], np.quantile(delta, [0, .25, .5, .75, 1]).tolist())),
        'uncertainty': uncertainty, 'region_sequence_direction': region_direction,
        'high_global_error_windows': {'selection': 'Post hoc top decile of global MPJPE; descriptive, susceptible to regression to mean.',
            'threshold_cm': threshold, 'windows': len(high), 'direction': direction_counts([r['delta_mpjpe_cm'] for r in high]),
            'frame_weighted_delta_cm': float(weighted_mean([r['delta_mpjpe_cm'] for r in high], [r['frames'] for r in high]))},
        'sequence18_window750': seq18, 'sequence8_global_worst': seq8,
        'seq18_improvement_rank_fraction': float((delta <= seq18['delta_mpjpe_cm']).mean()),
        'exploratory_motion': exploratory, 'bootstrap_caution': uncertainty['position_cm']['caution'],
        'region_joint_ids': REGIONS, 'validation': 'Exact window keys/GT hashes/coverage checked; scalar, joint and sequence aggregation agree.'}
    dump(out / 'summary.json', summary)
    figures(out, sequence_table, window_table, region_table, joint_table)
    print(json.dumps(summary, indent=2))


def figures(out, sequences, windows, regions, joints):
    out = out / 'figures'
    out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'savefig.dpi': 180, 'axes.spines.top': False, 'axes.spines.right': False})
    blue, orange = '#1768ac', '#d97706'
    fig, ax = plt.subplots(figsize=(8, 7))
    for row in sequences:
        seq = row['sequence_id']
        ax.plot([row['global_mpjpe_cm'], row['lw_rp_h_mpjpe_cm']], [seq, seq], color='.65', lw=1)
    ax.scatter([r['global_mpjpe_cm'] for r in sequences], range(len(sequences)), color=blue, label='global (5 locations)', zorder=3)
    ax.scatter([r['lw_rp_h_mpjpe_cm'] for r in sequences], range(len(sequences)), color=orange, marker='s', label='lw_rp_h (3 locations)', zorder=3)
    ax.set(yticks=range(len(sequences)), ylabel='Processed sequence ID', xlabel='Root-relative MPJPE (cm)',
           title='Paired full-sequence MPJPE | Epoch 59 | DIP test')
    ax.invert_yaxis()
    fig.legend(loc='lower center', bbox_to_anchor=(.5, .005), ncol=2)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(out / 'per_sequence_mpjpe.png')
    fig.savefig(out / 'per_sequence_mpjpe.pdf')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist([r['delta_mpjpe_cm'] for r in windows], bins=25, color=blue, edgecolor='white')
    ax.axvline(0, color='black', lw=1)
    ax.set(xlabel='Delta MPJPE = three locations − five locations (cm)', ylabel='Number of windows',
           title='203 paired windows | negative favors three locations')
    fig.text(.5, .01, 'Windows within a sequence are correlated; histogram is descriptive.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out / 'window_delta_distribution.png')
    fig.savefig(out / 'window_delta_distribution.pdf')
    plt.close(fig)
    selected = [r for r in regions if r['weighting'] == 'frame_weighted']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(selected))
    labels = [r['region'].replace('_', '\n') for r in selected]
    for ax, metric, title in zip(axes, ['mpjpe_cm', 'local_deg', 'global_deg'], ['Position (cm)', 'Local rotation (degrees)', 'Global rotation (degrees)']):
        ax.bar(x - .18, [r[f'global_{metric}'] for r in selected], width=.36, color=blue, label='global')
        ax.bar(x + .18, [r[f'lw_rp_h_{metric}'] for r in selected], width=.36, color=orange, label='lw_rp_h')
        ax.set(xticks=x, xticklabels=labels, title=title)
        ax.tick_params(axis='x', labelsize=8, rotation=30)
    axes[0].legend()
    fig.suptitle('Body regions | valid-frame weighted | additional groups overlap')
    fig.tight_layout()
    fig.savefig(out / 'body_region_errors.png')
    fig.savefig(out / 'body_region_errors.pdf')
    plt.close(fig)
    selected = [r for r in joints if r['weighting'] == 'frame_weighted']
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    x = np.arange(24)
    for ax, metric, label in zip(axes, ['mpjpe_cm', 'local_deg', 'global_deg'], ['Position (cm)', 'Local rotation (degrees)', 'Global rotation (degrees)']):
        ax.plot(x, [r[f'global_{metric}'] for r in selected], 'o-', color=blue, label='global')
        ax.plot(x, [r[f'lw_rp_h_{metric}'] for r in selected], 's-', color=orange, label='lw_rp_h')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set(xticks=x, xticklabels=JOINT_NAMES)
    axes[-1].tick_params(axis='x', rotation=65)
    fig.suptitle('Per-joint errors | valid-frame weighted | Epoch 59')
    fig.tight_layout()
    fig.savefig(out / 'per_joint_errors.png')
    fig.savefig(out / 'per_joint_errors.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--analysis-name', default='analysis')
    parser.add_argument('--bootstrap-draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261027)
    analyze(parser.parse_args())
