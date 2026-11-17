"""Analyze, visualize and independently validate the reserved placement study."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from eval_wrist_pocket_dev import COMBOS, dump, fingerprint, read_protocol, validate_records
from runtime import load_body_model, pose_rotations, sha256
from sensor_metrics import JOINT_NAMES, METRICS, REGIONS, pair_windows, weighted_mean

OUTCOMES = {
    'left_arm_global_deg': ('global_deg', REGIONS['left_arm']),
    'right_arm_global_deg': ('global_deg', REGIONS['right_arm']),
    'left_wrist_position_cm': ('position_cm', [20]),
    'right_wrist_position_cm': ('position_cm', [21]),
    'whole_body_position_cm': ('position_cm', list(range(24))),
    'whole_body_local_deg': ('local_deg', list(range(24))),
    'whole_body_global_deg': ('global_deg', list(range(24))),
    'left_arm_position_cm': ('position_cm', REGIONS['left_arm']),
    'right_arm_position_cm': ('position_cm', REGIONS['right_arm']),
    'left_arm_local_deg': ('local_deg', REGIONS['left_arm']),
    'right_arm_local_deg': ('local_deg', REGIONS['right_arm']),
    'legs_position_cm': ('position_cm', REGIONS['legs']),
    'torso_position_cm': ('position_cm', REGIONS['torso']),
    'head_position_cm': ('position_cm', REGIONS['head']),
    'root_local_deg': ('local_deg', [0]),
}
LABELS = {'lw_rp_h': 'Left wrist\nRight pocket', 'rw_rp_h': 'Right wrist\nRight pocket',
          'lw_lp_h': 'Left wrist\nLeft pocket', 'rw_lp_h': 'Right wrist\nLeft pocket'}
COLORS = ['#226aab', '#c85826', '#35855b', '#9258a5']


def contrasts(values):
    """All contrasts use right minus left; input values can be scalars or arrays."""
    a, b, c, d = (values[k] for k in COMBOS)
    return {'wrist_right_minus_left': .5 * ((b - a) + (d - c)),
            'pocket_right_minus_left': .5 * ((a - c) + (b - d)),
            'interaction': (b - a) - (d - c),
            'wrist_right_minus_left_right_pocket': b - a,
            'wrist_right_minus_left_left_pocket': d - c,
            'pocket_right_minus_left_left_wrist': a - c,
            'pocket_right_minus_left_right_wrist': b - d}


def save_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def joint_mean(rows):
    return {k: weighted_mean([r['joint_errors'][k] for r in rows], [r['frames'] for r in rows]) for k in METRICS}


def endpoints(joint):
    return {name: float(np.asarray(joint[metric])[indices].mean())
            for name, (metric, indices) in OUTCOMES.items()}


def save_figure(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def analyze(run):
    protocol = read_protocol(run)
    manifest = json.loads((run / 'split_manifest.json').read_text())
    ev = run / 'evaluation'
    reports = {c: json.loads((ev / f'{c}_metrics.json').read_text())['per_window'] for c in COMBOS}
    base = reports[next(iter(COMBOS))]
    for c in COMBOS:
        for a, b in pair_windows(base, reports[c]):
            if a['target_sha256'] != b['target_sha256'] or a['subject'] != b['subject']:
                raise ValueError('Reference or subject pairing mismatch')
        validate_records(reports[c], manifest['selected'])
    out = run / 'analysis'
    out.mkdir(exist_ok=False)
    subjects = protocol['development_subjects']
    sequences = [r['sequence_id'] for r in manifest['selected']]
    window_rows, seq_rows, sub_rows, overall, joint_rows = [], [], [], [], []
    seq_joint, sub_joint = {}, {}
    for c, rows in reports.items():
        for row in rows:
            meta = {k: row[k] for k in ['sequence_id', 'subject', 'source_file', 'start_frame', 'frames']}
            window_rows.append({'configuration': c, **meta, **endpoints(row['joint_errors'])})
        seq_joint[c], sub_joint[c] = {}, {}
        for seq in sequences:
            chosen = [r for r in rows if r['sequence_id'] == seq]
            values = joint_mean(chosen)
            seq_joint[c][seq] = values
            seq_rows.append({'configuration': c, 'sequence_id': seq, 'subject': chosen[0]['subject'],
                             'source_file': chosen[0]['source_file'], 'frames': sum(r['frames'] for r in chosen),
                             'windows': len(chosen), **endpoints(values)})
        for subject in subjects:
            chosen = [r for r in rows if r['subject'] == subject]
            values = joint_mean(chosen)
            sub_joint[c][subject] = values
            sub_rows.append({'configuration': c, 'subject': subject,
                             'frames': sum(r['frames'] for r in chosen), **endpoints(values)})
        for weighting, values in [('frame_weighted', joint_mean(rows)),
             ('sequence_equal', {k: np.mean([seq_joint[c][s][k] for s in sequences], axis=0) for k in METRICS}),
             ('subject_equal', {k: np.mean([sub_joint[c][s][k] for s in subjects], axis=0) for k in METRICS})]:
            overall.append({'configuration': c, 'weighting': weighting, **endpoints(values)})
            for j, name in enumerate(JOINT_NAMES):
                joint_rows.append({'configuration': c, 'weighting': weighting, 'joint_id': j, 'joint': name,
                                   **{k: float(values[k][j]) for k in METRICS}})
    contrast_rows = []
    for unit, entries, group_key in [('overall', overall, 'weighting'), ('subject', sub_rows, 'subject'),
                                      ('sequence', seq_rows, 'sequence_id')]:
        for identity in dict.fromkeys(r[group_key] for r in entries):
            selected = {r['configuration']: r for r in entries if r[group_key] == identity}
            for outcome in OUTCOMES:
                for name, value in contrasts({c: selected[c][outcome] for c in COMBOS}).items():
                    contrast_rows.append({'unit': unit, 'identity': identity, 'outcome': outcome,
                                          'contrast': name, 'value': float(value)})
    for name, rows in [('per_window', window_rows), ('per_sequence', seq_rows), ('per_subject', sub_rows),
                       ('overall', overall), ('per_joint', joint_rows), ('contrasts', contrast_rows)]:
        save_csv(out / f'{name}.csv', rows)
    primary = [r for r in contrast_rows if r['unit'] == 'overall' and r['identity'] == 'subject_equal'
               and r['outcome'] in protocol['primary_outcomes']]
    summary = {'coverage': validate_records(base, manifest['selected']),
               'subject_coverage': {s: {'sequences': sum(r['subject'] == s for r in manifest['selected']),
                                      'frames': sum(r['frames'] for r in manifest['selected'] if r['subject'] == s)} for s in subjects},
               'overall': overall, 'per_subject': sub_rows, 'primary_contrasts': primary,
               'uncertainty': 'Two reserved subjects. Subject effects are descriptive; no population confidence intervals or frame-level p values.',
               'best_observed_whole_body_subject_equal': min((r for r in overall if r['weighting'] == 'subject_equal'), key=lambda r: r['whole_body_position_cm'])['configuration']}
    dump(out / 'summary.json', summary)
    figures(run, overall, sub_rows, contrast_rows)
    print(json.dumps(summary, indent=2), flush=True)


def figures(run, overall, sub_rows, contrast_rows):
    out = run / 'analysis' / 'figures'
    out.mkdir(exist_ok=False)
    selected = {r['configuration']: r for r in overall if r['weighting'] == 'subject_equal'}
    outcomes = [('whole_body_position_cm', 'Whole body position (cm)'),
                ('left_arm_global_deg', 'Left arm global rotation (degrees)'),
                ('right_arm_global_deg', 'Right arm global rotation (degrees)'),
                ('left_wrist_position_cm', 'Left wrist position (cm)'),
                ('right_wrist_position_cm', 'Right wrist position (cm)')]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.3), layout='constrained')
    for ax, (name, title) in zip(axes, outcomes):
        matrix = np.array([[selected['lw_lp_h'][name], selected['lw_rp_h'][name]],
                           [selected['rw_lp_h'][name], selected['rw_rp_h'][name]]])
        image = ax.imshow(matrix, cmap='YlOrRd')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=14,
                        color='white' if matrix[i, j] > (matrix.min() + matrix.max()) / 2 else 'black')
        ax.set(xticks=[0, 1], xticklabels=['Left', 'Right'], yticks=[0, 1], yticklabels=['Left', 'Right'],
               xlabel='Pocket input side', ylabel='Wrist input side', title=title)
    fig.suptitle('Equal weight per subject | two reserved subjects | lower error is better\nEach panel has its own color scale; head input is always present', fontsize=13)
    save_figure(fig, out, 'placement_grid')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, (name, title) in zip(axes.flat, outcomes[1:]):
        for subject, color in [('s_07', '#226aab'), ('s_08', '#c85826')]:
            for pocket, style in [('lp', '--'), ('rp', '-')]:
                a = next(r for r in sub_rows if r['subject'] == subject and r['configuration'] == f'lw_{pocket}_h')
                b = next(r for r in sub_rows if r['subject'] == subject and r['configuration'] == f'rw_{pocket}_h')
                ax.plot([0, 1], [a[name], b[name]], marker='o', linestyle=style, color=color,
                        label=f'{subject}, {"left" if pocket == "lp" else "right"} pocket')
        ax.set(xticks=[0, 1], xticklabels=['Left wrist input', 'Right wrist input'], ylabel='Error', title=title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Paired wrist switches on the same motions\nA rising left-arm line and falling right-arm line favor direct wrist coverage', fontsize=13)
    save_figure(fig, out, 'wrist_switch_by_subject')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for ax, subject in zip(axes, ['s_07', 's_08']):
        values = [next(r['whole_body_position_cm'] for r in sub_rows if r['subject'] == subject and r['configuration'] == c) for c in COMBOS]
        ax.bar(range(4), values, color=COLORS)
        ax.set(xticks=range(4), xticklabels=[LABELS[c] for c in COMBOS], ylabel='Mean joint position error (cm)', title=subject)
        for i, v in enumerate(values):
            ax.text(i, v + .12, f'{v:.2f}', ha='center')
        ax.set_ylim(0, max(values) * 1.2)
    fig.suptitle('Whole body error for each reserved subject | head input always present')
    save_figure(fig, out, 'whole_body_by_subject')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, names, unit in [(axes[0], ['left_arm_global_deg', 'right_arm_global_deg'], 'degrees'),
                            (axes[1], ['left_wrist_position_cm', 'right_wrist_position_cm'], 'cm')]:
        for identity, offset, color in [('s_07', -.12, '#226aab'), ('s_08', .12, '#c85826'), ('subject_equal', 0, 'black')]:
            values = [next(r['value'] for r in contrast_rows if r['identity'] == identity and r['outcome'] == name
                           and r['contrast'] == 'wrist_right_minus_left') for name in names]
            ax.scatter(values, np.arange(2) + offset, color=color, label=identity, s=50)
        ax.axvline(0, color='gray', linewidth=1)
        ax.set(yticks=[0, 1], yticklabels=['Left side', 'Right side'], xlabel=f'Right wrist input minus left wrist input ({unit})',
               title='Arm global rotation' if unit == 'degrees' else 'Wrist position')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Wrist effects averaged over pocket sides | points show subjects and their mean\nPositive means switching to right wrist input raises this error; no confidence intervals are implied')
    save_figure(fig, out, 'primary_wrist_effects')


def render(run):
    ev = run / 'evaluation'
    provenance = json.loads((ev / 'provenance.json').read_text())
    manifest = json.loads((run / 'split_manifest.json').read_text())
    out = run / 'visuals'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    body = load_body_model(provenance['smpl']['path'])
    edges = [(p, c) for c, p in enumerate(body.parent) if p is not None]
    cases = []
    with np.load(ev / 'predictions.npz', allow_pickle=False) as cache, torch.inference_mode():
        for subject in ['s_07', 's_08']:
            recording = min((r for r in manifest['selected'] if r['subject'] == subject), key=lambda r: r['source_file'])
            i = next(i for i, (s, f) in enumerate(zip(cache['sequence_ids'], cache['start_frames']))
                     if s == recording['sequence_id'] and f == 0)
            sl = slice(cache['offsets'][i], cache['offsets'][i + 1])
            keys = [f'pose_{c}' for c in COMBOS] + ['target_pose']
            positions = [body.forward_kinematics(pose_rotations(torch.from_numpy(cache[k][sl])), calc_mesh=False)[1].numpy() for k in keys]
            xyz = [p[..., [0, 2, 1]].copy() for p in positions]
            for p in xyz:
                p[..., 1] *= -1
            fig = plt.figure(figsize=(16, 7), dpi=130)
            for row, frame in enumerate([41, 83]):
                for col in range(5):
                    ax = fig.add_subplot(2, 5, row * 5 + col + 1, projection='3d')
                    title = LABELS[list(COMBOS)[col]] if col < 4 else 'Reference pose'
                    ax.set(title=f'{title}\nframe {frame}', xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), zlim=(-1.15, 1.15))
                    ax.set_box_aspect((1, 1, 1))
                    ax.view_init(elev=15, azim=-65)
                    points = xyz[col][frame]
                    for parent, child in edges:
                        pair = points[[parent, child]]
                        ax.plot(pair[:, 0], pair[:, 1], pair[:, 2], color=COLORS[col] if col < 4 else '#333333', linewidth=2)
                    ax.set_xticks([-.8, 0, .8])
                    ax.set_yticks([-.8, 0, .8])
                    ax.set_zticks([-.8, 0, .8])
            fig.suptitle(f'{subject}/{recording["source_file"]}, first window | fixed example selected before inference\nSame camera, scale and times; zero translation, original root rotation; head input present in every prediction')
            fig.subplots_adjust(left=.01, right=.99, bottom=.03, top=.84, hspace=.24, wspace=.02)
            fig.savefig(out / f'{subject}_fixed_example.png')
            plt.close(fig)
            cases.append({'subject': subject, 'sequence_id': recording['sequence_id'], 'source_file': recording['source_file'],
                          'start_frame': 0, 'snapshot_frames': [41, 83]})
    dump(out / 'manifest.json', cases)


def validate(run):
    ev = run / 'evaluation'
    protocol = read_protocol(run)
    provenance = json.loads((ev / 'provenance.json').read_text())
    manifest = json.loads((run / 'split_manifest.json').read_text())
    if sha256(run / 'PROTOCOL.json') != provenance['protocol_sha256'] or sha256(run / 'split_manifest.json') != provenance['split_manifest_sha256']:
        raise ValueError('Protocol or reservation changed')
    for asset in ['checkpoint', 'data', 'smpl']:
        if sha256(provenance[asset]['path']) != provenance[asset]['sha256']:
            raise ValueError(f'Changed {asset}')
    for path, digest in provenance['source_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Changed inference source {path}')
    for row in manifest['selected']:
        if sha256(row['raw_path']) != row['raw_sha256']:
            raise ValueError('Raw recording changed')
    reports = {c: json.loads((ev / f'{c}_metrics.json').read_text())['per_window'] for c in COMBOS}
    torch.set_num_threads(2)
    body = load_body_model(provenance['smpl']['path'])
    max_diff = {k: 0. for k in METRICS}
    arrays = {c: {k: [] for k in METRICS} for c in COMBOS}
    count = 0
    with np.load(ev / 'predictions.npz', allow_pickle=False) as cache, torch.inference_mode():
        for c, rows in reports.items():
            coverage = validate_records(rows, manifest['selected'])
            if len(cache['offsets']) != len(rows) + 1 or cache['offsets'][0] != 0:
                raise ValueError('Wrong cache offsets')
            if cache['offsets'][-1] != coverage['frames'] or len(cache['target_pose']) != coverage['frames']:
                raise ValueError('Wrong cache length')
            for i, row in enumerate(rows):
                sl = slice(cache['offsets'][i], cache['offsets'][i + 1])
                if (cache['sequence_ids'][i] != row['sequence_id'] or cache['start_frames'][i] != row['start_frame']
                        or cache['subjects'][i] != row['subject'] or sl.stop - sl.start != row['frames']):
                    raise ValueError('Cached window identity mismatch')
                target = cache['target_pose'][sl]
                if fingerprint(target) != row['target_sha256']:
                    raise ValueError('Reference changed')
                ref = pose_rotations(torch.from_numpy(target))
                pred = pose_rotations(torch.from_numpy(cache[f'pose_{c}'][sl]))
                rg, rp = body.forward_kinematics(ref, calc_mesh=False)
                pg, pp = body.forward_kinematics(pred, calc_mesh=False)
                def angle(a, b):
                    cosine = ((a.transpose(-1, -2) @ b).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
                    return torch.acos(torch.clamp(cosine, -1, 1)) * (180 / torch.pi)
                computed = {'position_cm': torch.linalg.vector_norm(pp - rp, dim=-1).mean(0).numpy() * 100,
                            'local_deg': angle(pred, ref).mean(0).numpy(),
                            'global_deg': angle(pg, rg).mean(0).numpy()}
                for k, value in computed.items():
                    difference = float(np.max(np.abs(value - np.array(row['joint_errors'][k]))))
                    max_diff[k] = max(max_diff[k], difference)
                    if difference > 1e-4:
                        raise ValueError(f'Metric mismatch {k}: {difference}')
                    arrays[c][k].append(value)
                count += 1
    # Recompute all aggregate endpoints directly from cached window joint arrays.
    with (run / 'analysis' / 'overall.csv').open() as stream:
        overall = list(csv.DictReader(stream))
    aggregate_max_diff = 0.
    for row in overall:
        c, mode = row['configuration'], row['weighting']
        windows = reports[c]
        if mode == 'frame_weighted':
            joint = {k: weighted_mean(arrays[c][k], [w['frames'] for w in windows]) for k in METRICS}
        else:
            field = 'subject' if mode == 'subject_equal' else 'sequence_id'
            ids = list(dict.fromkeys(w[field] for w in windows))
            joint = {}
            for k in METRICS:
                groups = []
                for identity in ids:
                    indices = [i for i, w in enumerate(windows) if w[field] == identity]
                    groups.append(weighted_mean([arrays[c][k][i] for i in indices], [windows[i]['frames'] for i in indices]))
                joint[k] = np.mean(groups, axis=0)
        for outcome, value in endpoints(joint).items():
            difference = abs(value - float(row[outcome]))
            aggregate_max_diff = max(aggregate_max_diff, difference)
            if difference > 1e-4:
                raise ValueError('CSV aggregate mismatch')
    with (run / 'analysis' / 'contrasts.csv').open() as stream:
        contrast_rows = list(csv.DictReader(stream))
    contrast_max_diff = 0.
    for unit, filename, group_key in [('overall', 'overall', 'weighting'), ('subject', 'per_subject', 'subject'),
                                      ('sequence', 'per_sequence', 'sequence_id')]:
        with (run / 'analysis' / f'{filename}.csv').open() as stream:
            source = list(csv.DictReader(stream))
        for row in (r for r in contrast_rows if r['unit'] == unit):
            values = {r['configuration']: float(r[row['outcome']]) for r in source if r[group_key] == row['identity']}
            expected = contrasts(values)[row['contrast']]
            difference = abs(expected - float(row['value']))
            contrast_max_diff = max(contrast_max_diff, difference)
            if difference > 1e-10:
                raise ValueError('Contrast table mismatch')
    result = {'status': 'passed', 'coverage_per_configuration': coverage,
              'configuration_window_predictions_recomputed': count,
              'max_joint_metric_difference': max_diff, 'max_aggregate_difference': aggregate_max_diff,
              'max_contrast_difference': contrast_max_diff, 'metric_tolerance': 1e-4,
              'protocol_and_reservation_unchanged': True, 'asset_and_raw_recording_hashes_unchanged': True,
              'inference_sources_unchanged': True, 'test_data_inference': False,
              'population_significance_claim': False}
    dump(run / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['analyze', 'render', 'validate'])
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    {'analyze': analyze, 'render': render, 'validate': validate}[args.stage](Path(args.run_dir))
