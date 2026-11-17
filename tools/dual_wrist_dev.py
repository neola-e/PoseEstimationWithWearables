"""Evaluate one new wrist mask and compare it with frozen E03 prediction caches.

Stages refuse to overwrite completed outputs. Only evaluate performs inference.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import imuposer
from imuposer.config import amass_combos

from eval_wrist_pocket_dev import (
    COMBOS, ReservedDipSubset, dump, fingerprint, source_hashes, validate_records,
)
from analyze_wrist_pocket_dev import endpoints, joint_mean, save_csv, save_figure
from runtime import PosePredictor, load_body_model, load_torch, peak_ram_mib, pose_rotations, sha256
from sensor_metrics import (
    JOINT_NAMES, METRICS, direction_counts, joint_errors, mask_input, pair_windows, sensor_mask,
)

NEW = 'lw_rw_h'
SENSORS = [0, 1, 4]
ALL = [*COMBOS, NEW]
LABELS = ['L wrist\nR pocket', 'R wrist\nR pocket', 'L wrist\nL pocket',
          'R wrist\nL pocket', 'Both wrists\nNo pocket']
COLORS = ['#226aab', '#c85826', '#35855b', '#9258a5', '#202020']


def read(path):
    return json.loads(Path(path).read_text())


def context(run):
    p = read(run / 'PROTOCOL.json')
    if (p['new_configuration'] != {NEW: SENSORS} or p['comparators'] != COMBOS
            or p['development_subjects'] != ['s_07', 's_08'] or p['checkpoint_epoch'] != 59):
        raise ValueError('Unexpected frozen protocol')
    base = Path(p['baseline_run'])
    if (sha256(base / 'PROTOCOL.json') != p['baseline_protocol_sha256']
            or sha256(base / 'split_manifest.json') != p['split_manifest_sha256']):
        raise ValueError('Baseline protocol or reservation changed')
    return p, base, read(base / 'split_manifest.json'), read(base / 'evaluation/provenance.json')


def check_baseline(run):
    record = read(run / 'BASELINE_RECORD.json')
    base = Path(record['baseline_run'])
    files = {str(p.relative_to(base)): sha256(p) for p in sorted(base.rglob('*')) if p.is_file()}
    if files != record['files']:
        raise ValueError('Original study files changed')
    return len(files)


def check_pairs(left, right):
    for a, b in pair_windows(left, right):
        for key in ('target_sha256', 'subject', 'source_file'):
            if a[key] != b[key]:
                raise ValueError(f'Paired {key} mismatch')


def outcomes(joint):
    result = endpoints(joint)
    left, right = result['left_wrist_position_cm'], result['right_wrist_position_cm']
    result['bilateral_wrist_mean_position_cm'] = (left + right) / 2
    result['worse_wrist_position_cm'] = max(left, right)
    return result


def aggregate(rows, mode):
    if mode == 'frame_weighted':
        return joint_mean(rows)
    field = {'subject_equal': 'subject', 'sequence_equal': 'sequence_id'}[mode]
    groups = [joint_mean([r for r in rows if r[field] == key])
              for key in dict.fromkeys(r[field] for r in rows)]
    return {metric: np.mean([g[metric] for g in groups], axis=0) for metric in METRICS}


def reports_for(run, base):
    return {c: read((run if c == NEW else base) / 'evaluation' / f'{c}_metrics.json')['per_window']
            for c in ALL}


def evaluate(run):
    protocol, base, manifest, old = context(run)
    old_count = check_baseline(run)
    if read(base / 'validation.json')['status'] != 'passed':
        raise ValueError('Prior numerical validation not passed')
    for path, digest in old['source_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Prior evaluated source changed: {path}')
    for name in ('checkpoint', 'data', 'smpl'):
        if sha256(old[name]['path']) != old[name]['sha256']:
            raise ValueError(f'Changed {name} asset')
    if (old['checkpoint']['sha256'] != protocol['checkpoint_sha256']
            or Path(old['data']['path']).name != 'dip_train.pt' or amass_combos[NEW] != SENSORS):
        raise ValueError('Wrong checkpoint, split, or training mask')
    for row in manifest['selected']:
        if sha256(row['raw_path']) != row['raw_sha256']:
            raise ValueError('Reserved raw recording changed')
    software = {'python': platform.python_version(), **{n: importlib.metadata.version(n)
                for n in ['torch', 'numpy', 'pytorch-lightning', 'matplotlib', 'Pillow', 'scipy']}}
    if software != old['software']:
        raise ValueError('Environment differs from baseline')
    out = run / 'evaluation'
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    predictor = PosePredictor(old['checkpoint']['path'], device='cpu', threads=2)
    if predictor.epoch != 59:
        raise ValueError('Wrong epoch')
    data = load_torch(old['data']['path'])
    for row in manifest['selected']:
        for key, digest in row['processed_sha256'].items():
            if fingerprint(data[key][row['sequence_id']]) != digest:
                raise ValueError('Selected processed sequence changed')
    dataset = ReservedDipSubset(data, manifest['selected'], predictor.training_config)
    coverage = validate_records(dataset.records, manifest['selected'])
    if coverage != old['coverage']:
        raise ValueError('Coverage differs from baseline')
    body = load_body_model(old['smpl']['path'])
    with np.load(base / 'evaluation/predictions.npz', allow_pickle=False) as cache:
        reference = {k: cache[k] for k in ['target_pose', 'offsets', 'sequence_ids', 'start_frames', 'subjects']}
    old_rows = read(base / 'evaluation/lw_rp_h_metrics.json')['per_window']
    root = Path(__file__).resolve().parents[1]
    sources = source_hashes(root)
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    test = root / 'tests/test_dual_wrist_dev.py'
    sources[str(test)] = sha256(test)
    package = Path(imuposer.__file__).resolve().parent
    for path in package.rglob('*.py'):
        sources[str(path)] = sha256(path)
    snapshot = run / 'source_snapshot'
    snapshot.mkdir(exist_ok=False)
    for path in sources:
        src = Path(path)
        dest = snapshot / ('team' if src.is_relative_to(root) else 'original_package') / (
            src.relative_to(root) if src.is_relative_to(root) else src.relative_to(package))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    dump(snapshot / 'SHA256.json', {str(p.relative_to(snapshot)): sha256(p)
                                   for p in snapshot.rglob('*.py')})
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
                  'protocol_sha256': sha256(run / 'PROTOCOL.json'),
                  'baseline_record_sha256': sha256(run / 'BASELINE_RECORD.json'),
                  'baseline_files_checked': old_count, 'coverage': coverage,
                  **{key: old[key] for key in ['checkpoint', 'data', 'smpl', 'imuposer_import']},
                  'source_sha256': sources, 'software': software,
                  'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  'git_status_at_start': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
                  'new_inference_configurations': [NEW], 'reused_configurations': list(COMBOS),
                  'test_container_opened': False, 'training_performed': False}
    dump(out / 'provenance.json', provenance)
    rows, predictions, targets, offsets = [], [], [], [0]
    keep = sensor_mask(SENSORS)
    with torch.inference_mode():
        for i, meta in enumerate(dataset.records):
            imu, target = dataset[i]
            original = imu.clone()
            features = mask_input(imu.numpy(), SENSORS)
            if (not np.array_equal(features[:, keep], imu.numpy()[:, keep])
                    or np.count_nonzero(features[:, ~keep])):
                raise ValueError('Masking failed')
            offsets.append(offsets[-1] + len(target))
            sl = slice(offsets[-2], offsets[-1])
            old_row = old_rows[i]
            if (not np.array_equal(target.numpy(), reference['target_pose'][sl])
                    or offsets[-2] != reference['offsets'][i]
                    or offsets[-1] != reference['offsets'][i + 1]
                    or meta['sequence_id'] != reference['sequence_ids'][i]
                    or meta['start_frame'] != reference['start_frames'][i]
                    or meta['subject'] != reference['subjects'][i]
                    or fingerprint(target) != old_row['target_sha256']):
                raise ValueError('New inference does not match frozen baseline windows')
            prediction = predictor.predict(features)
            errors = joint_errors(pose_rotations(prediction), pose_rotations(target), body)
            rows.append({**meta, 'target_sha256': fingerprint(target),
                         'joint_errors': {k: v.mean(0).tolist() for k, v in errors.items()}})
            predictions.append(prediction.numpy())
            targets.append(target.numpy())
            if not torch.equal(imu, original):
                raise ValueError('Masking mutated source')
            if (i + 1) % 50 == 0 or i + 1 == len(dataset):
                print(f'New-mask inference: {i + 1}/{len(dataset)} windows', flush=True)
    check_pairs(old_rows, rows)
    dump(out / f'{NEW}_metrics.json', {'configuration': NEW, 'per_window': rows})
    np.savez_compressed(out / 'predictions.npz', **{f'pose_{NEW}': np.concatenate(predictions)},
                        target_pose=np.concatenate(targets), offsets=np.array(offsets),
                        **{k: reference[k] for k in ['sequence_ids', 'start_frames', 'subjects']})
    dump(out / 'resources.json', {'seconds_including_cache_write': time.perf_counter() - start,
                                  'peak_process_rss_mib': peak_ram_mib(), 'gpu_used': False})
    print(json.dumps({'coverage': coverage, 'new_inference_configurations': [NEW]}), flush=True)


def analyze(run):
    protocol, base, manifest, _ = context(run)
    reports = reports_for(run, base)
    for rows in reports.values():
        check_pairs(reports[NEW], rows)
        validate_records(rows, manifest['selected'])
    out = run / 'analysis'
    out.mkdir(exist_ok=False)
    tables = {name: [] for name in ['overall', 'per_subject', 'per_sequence', 'per_window', 'per_joint']}
    for c, rows in reports.items():
        for row in rows:
            meta = {key: row[key] for key in ['subject', 'source_file', 'sequence_id', 'start_frame', 'frames']}
            tables['per_window'].append({'configuration': c, **meta, **outcomes(row['joint_errors'])})
        for field, name in [('subject', 'per_subject'), ('sequence_id', 'per_sequence')]:
            for identity in dict.fromkeys(r[field] for r in rows):
                selected = [r for r in rows if r[field] == identity]
                meta = {field: identity, 'frames': sum(r['frames'] for r in selected), 'windows': len(selected)}
                if field == 'sequence_id':
                    meta.update(subject=selected[0]['subject'], source_file=selected[0]['source_file'])
                tables[name].append({'configuration': c, **meta, **outcomes(joint_mean(selected))})
        for mode in ['subject_equal', 'frame_weighted', 'sequence_equal']:
            joint = aggregate(rows, mode)
            tables['overall'].append({'configuration': c, 'weighting': mode, **outcomes(joint)})
            for i, name in enumerate(JOINT_NAMES):
                tables['per_joint'].append({'configuration': c, 'weighting': mode, 'joint_id': i,
                                             'joint': name, **{k: float(joint[k][i]) for k in METRICS}})
    comparisons = []
    for table, keys in [('overall', ['weighting']), ('per_subject', ['subject']),
                        ('per_sequence', ['sequence_id']), ('per_window', ['sequence_id', 'start_frame'])]:
        entries = tables[table]
        for new in (r for r in entries if r['configuration'] == NEW):
            for c in COMBOS:
                old = next(r for r in entries if r['configuration'] == c and all(r[k] == new[k] for k in keys))
                for metric in outcomes(reports[NEW][0]['joint_errors']):
                    comparisons.append({'unit': table,
                                        **{k: new[k] if k in keys else '' for k in ['weighting', 'subject', 'sequence_id', 'start_frame']},
                                        'comparator': c,
                                        'outcome': metric, 'new': new[metric], 'baseline': old[metric],
                                        'delta_new_minus_baseline': new[metric] - old[metric]})
    for name, rows in tables.items():
        save_csv(out / f'{name}.csv', rows)
    save_csv(out / 'comparisons.csv', comparisons)
    consistency = []
    for c in COMBOS:
        for metric in protocol['primary_outcomes'] + ['left_wrist_position_cm', 'right_wrist_position_cm', 'legs_position_cm']:
            values = [r['delta_new_minus_baseline'] for r in comparisons
                      if r['unit'] == 'per_sequence' and r['comparator'] == c and r['outcome'] == metric]
            consistency.append({'comparator': c, 'outcome': metric, **direction_counts(values)})
    summary = {'primary_aggregation': 'subject_equal', 'overall': tables['overall'],
               'per_subject': tables['per_subject'], 'sequence_direction_counts': consistency,
               'decision_rule': protocol['decision_rule'], 'population_significance_claim': False}
    dump(out / 'summary.json', summary)
    plot_results(run, tables)
    print(json.dumps([r for r in tables['overall'] if r['weighting'] == 'subject_equal'], indent=2), flush=True)


def plot_results(run, tables):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = run / 'analysis/figures'
    out.mkdir(exist_ok=False)
    primary = [next(r for r in tables['overall'] if r['configuration'] == c and r['weighting'] == 'subject_equal') for c in ALL]
    panels = [('bilateral_wrist_mean_position_cm', 'Mean of both wrists (cm)'),
              ('worse_wrist_position_cm', 'Worse wrist mean (cm)'),
              ('legs_position_cm', 'Leg position (cm)'), ('root_local_deg', 'Root orientation (degrees)'),
              ('whole_body_position_cm', 'Whole-body position (cm)')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (metric, title) in zip(axes.flat, panels):
        values = [r[metric] for r in primary]
        bars = ax.bar(np.arange(5), values, color=COLORS)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.set(xticks=np.arange(5), xticklabels=LABELS, title=title, ylim=(0, max(values) * 1.2))
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.2)
    ax = axes.flat[-1]
    for i, row in enumerate(primary):
        ax.scatter(row['legs_position_cm'], row['bilateral_wrist_mean_position_cm'], color=COLORS[i], s=65)
        ax.annotate(ALL[i], (row['legs_position_cm'], row['bilateral_wrist_mean_position_cm']),
                    xytext=(4, 5), textcoords='offset points', fontsize=8)
    ax.set(xlabel='Leg position error (cm)', ylabel='Mean wrist position error (cm)', title='Arm versus leg tradeoff; lower is better')
    ax.margins(.25)
    ax.grid(alpha=.2)
    fig.suptitle('Head plus both wrists versus four wrist/pocket configurations\nEqual subject weighting; two development subjects; no application acceptance threshold')
    fig.tight_layout(rect=(0, 0, 1, .94))
    save_figure(fig, out, 'sensor_allocation_tradeoff')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, subject in zip(axes, ['s_07', 's_08']):
        for i, c in enumerate(ALL):
            row = next(r for r in tables['per_subject'] if r['configuration'] == c and r['subject'] == subject)
            ax.scatter(row['legs_position_cm'], row['bilateral_wrist_mean_position_cm'], color=COLORS[i], s=65)
            ax.annotate(c, (row['legs_position_cm'], row['bilateral_wrist_mean_position_cm']),
                        xytext=(4, 5), textcoords='offset points', fontsize=8)
        ax.set(title=subject, xlabel='Leg position error (cm)', ylabel='Mean wrist position error (cm)')
        ax.margins(.3)
        ax.grid(alpha=.2)
    fig.suptitle('The allocation tradeoff for each subject; lower is better')
    fig.tight_layout()
    save_figure(fig, out, 'tradeoff_by_subject')


def load_cache(path):
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def validate(run):
    protocol, base, manifest, old = context(run)
    provenance = read(run / 'evaluation/provenance.json')
    if (sha256(run / 'PROTOCOL.json') != provenance['protocol_sha256']
            or sha256(run / 'BASELINE_RECORD.json') != provenance['baseline_record_sha256']):
        raise ValueError('Frozen records changed')
    baseline_count = check_baseline(run)
    for key in ('checkpoint', 'data', 'smpl'):
        if sha256(provenance[key]['path']) != provenance[key]['sha256']:
            raise ValueError(f'Asset changed: {key}')
    for path, digest in provenance['source_sha256'].items():
        if sha256(path) != digest:
            raise ValueError(f'Evaluated source changed: {path}')
    for row in manifest['selected']:
        if sha256(row['raw_path']) != row['raw_sha256']:
            raise ValueError('Reserved raw recording changed')
    snapshot = run / 'source_snapshot'
    for path, digest in read(snapshot / 'SHA256.json').items():
        if sha256(snapshot / path) != digest:
            raise ValueError('Source snapshot changed')
    old_cache = load_cache(base / 'evaluation/predictions.npz')
    new_cache = load_cache(run / 'evaluation/predictions.npz')
    for key in ['target_pose', 'offsets', 'sequence_ids', 'start_frames', 'subjects']:
        if not np.array_equal(old_cache[key], new_cache[key]):
            raise ValueError(f'Cache pairing mismatch: {key}')
    reports = reports_for(run, base)
    torch.set_num_threads(2)
    body = load_body_model(provenance['smpl']['path'])
    difference = {k: 0. for k in METRICS}
    recomputed = {}
    count = 0
    with torch.inference_mode():
        for c, rows in reports.items():
            coverage = validate_records(rows, manifest['selected'])
            check_pairs(reports[NEW], rows)
            cache = new_cache if c == NEW else old_cache
            if (len(cache['offsets']) != len(rows) + 1 or cache['offsets'][0] != 0
                    or cache['offsets'][-1] != coverage['frames']
                    or cache[f'pose_{c}'].shape != (coverage['frames'], 144)
                    or not np.isfinite(cache[f'pose_{c}']).all()):
                raise ValueError('Invalid cached prediction shape or offsets')
            recomputed[c] = []
            for i, row in enumerate(rows):
                a, b = map(int, cache['offsets'][i:i + 2])
                if (b - a != row['frames'] or cache['sequence_ids'][i] != row['sequence_id']
                        or cache['start_frames'][i] != row['start_frame'] or cache['subjects'][i] != row['subject']
                        or fingerprint(cache['target_pose'][a:b]) != row['target_sha256']):
                    raise ValueError('Cache identity mismatch')
                reference = pose_rotations(torch.from_numpy(cache['target_pose'][a:b]))
                prediction = pose_rotations(torch.from_numpy(cache[f'pose_{c}'][a:b]))
                rg, rp = body.forward_kinematics(reference, calc_mesh=False)
                pg, pp = body.forward_kinematics(prediction, calc_mesh=False)
                def angle(x, y):
                    cosine = ((x.transpose(-1, -2) @ y).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
                    return torch.acos(torch.clamp(cosine, -1, 1)) * (180 / torch.pi)
                values = {'position_cm': torch.linalg.vector_norm(pp - rp, dim=-1).mean(0).numpy() * 100,
                          'local_deg': angle(prediction, reference).mean(0).numpy(),
                          'global_deg': angle(pg, rg).mean(0).numpy()}
                for key, value in values.items():
                    diff = float(np.max(np.abs(value - row['joint_errors'][key])))
                    difference[key] = max(difference[key], diff)
                    if not np.isfinite(value).all() or diff > 1e-4:
                        raise ValueError(f'Joint metric mismatch: {c}, {i}, {key}, {diff}')
                recomputed[c].append({**row, 'joint_errors': values})
                count += 1
            print(f'Independent metric validation: {c}, {len(rows)} windows', flush=True)
    # Independent flat-weight construction checks all aggregate and unit endpoints.
    max_endpoint = 0.
    checked = 0
    tables = {}
    for name in ['overall', 'per_subject', 'per_sequence', 'per_window', 'per_joint']:
        with (run / f'analysis/{name}.csv').open() as stream:
            tables[name] = list(csv.DictReader(stream))
        for row in tables[name]:
            windows = recomputed[row['configuration']]
            for field in ['subject', 'sequence_id', 'start_frame']:
                if field in row:
                    windows = [w for w in windows if str(w[field]) == row[field]]
            weights = np.array([w['frames'] for w in windows], dtype=np.float64)
            mode = row.get('weighting', 'frame_weighted')
            if mode != 'frame_weighted':
                field = 'subject' if mode == 'subject_equal' else 'sequence_id'
                groups = list(dict.fromkeys(w[field] for w in windows))
                weights = np.array([w['frames'] / sum(t['frames'] for t in windows if t[field] == w[field]) / len(groups)
                                    for w in windows])
            weights /= weights.sum()
            joint = {k: np.sum(np.array([w['joint_errors'][k] for w in windows]) * weights[:, None], axis=0)
                     for k in METRICS}
            expected = {k: joint[k][int(row['joint_id'])] for k in METRICS} if name == 'per_joint' else outcomes(joint)
            for key, value in expected.items():
                diff = abs(float(row[key]) - value)
                max_endpoint = max(max_endpoint, diff)
                if diff > 1e-4:
                    raise ValueError(f'Endpoint mismatch: {name}, {key}, {diff}')
                checked += 1
    with (run / 'analysis/comparisons.csv').open() as stream:
        comparisons = list(csv.DictReader(stream))
    max_delta = 0.
    for row in comparisons:
        entries = tables[row['unit']]
        fields = [k for k in ['weighting', 'subject', 'sequence_id', 'start_frame'] if row.get(k)]
        values = {}
        for c in [NEW, row['comparator']]:
            source = next(r for r in entries if r['configuration'] == c and all(r[k] == row[k] for k in fields))
            values[c] = float(source[row['outcome']])
        error = max(abs(float(row['new']) - values[NEW]), abs(float(row['baseline']) - values[row['comparator']]),
                    abs(float(row['delta_new_minus_baseline']) - (values[NEW] - values[row['comparator']])))
        max_delta = max(max_delta, error)
        if error > 1e-10:
            raise ValueError('Paired delta mismatch')
    summary = read(run / 'analysis/summary.json')
    for row in summary['sequence_direction_counts']:
        values = [float(r['delta_new_minus_baseline']) for r in comparisons
                  if r['unit'] == 'per_sequence' and r['comparator'] == row['comparator'] and r['outcome'] == row['outcome']]
        expected = direction_counts(values)
        if any(row[k] != v for k, v in expected.items()):
            raise ValueError('Direction count mismatch')
    result = {'status': 'passed', 'coverage_per_configuration': coverage,
              'new_inference_windows': len(reports[NEW]), 'reused_prediction_windows': sum(len(reports[c]) for c in COMBOS),
              'configuration_window_predictions_recomputed': count, 'max_joint_metric_difference': difference,
              'independent_csv_endpoint_checks': checked, 'max_csv_endpoint_difference': max_endpoint,
              'paired_delta_rows_checked': len(comparisons), 'max_paired_delta_difference': max_delta,
              'original_study_files_unchanged': baseline_count, 'source_snapshot_checked': True,
              'source_and_asset_hashes_unchanged': True, 'protocol_unchanged': True,
              'identical_reference_and_window_arrays': True, 'test_data_inference': False,
              'training_performed': False, 'population_significance_claim': False}
    dump(run / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


def render(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, base, manifest, old = context(run)
    out = run / 'visuals'
    out.mkdir(exist_ok=False)
    cache = load_cache(base / 'evaluation/predictions.npz')
    cache.update(load_cache(run / 'evaluation/predictions.npz'))
    torch.set_num_threads(2)
    body = load_body_model(old['smpl']['path'])
    edges = [(p, c) for c, p in enumerate(body.parent) if p is not None]
    cases = []
    with torch.inference_mode():
        for subject in ['s_07', 's_08']:
            record = min((r for r in manifest['selected'] if r['subject'] == subject), key=lambda r: r['source_file'])
            i = next(i for i, (seq, start) in enumerate(zip(cache['sequence_ids'], cache['start_frames']))
                     if seq == record['sequence_id'] and start == 0)
            a, b = map(int, cache['offsets'][i:i + 2])
            poses = [cache[f'pose_{c}'][a:b] for c in ALL] + [cache['target_pose'][a:b]]
            xyz = []
            for pose in poses:
                positions = body.forward_kinematics(pose_rotations(torch.from_numpy(pose)), calc_mesh=False)[1].numpy()
                points = positions[..., [0, 2, 1]].copy()
                points[..., 1] *= -1
                xyz.append(points)
            fig = plt.figure(figsize=(19, 7), dpi=130)
            for row, frame in enumerate([41, 83]):
                for col in range(6):
                    ax = fig.add_subplot(2, 6, row * 6 + col + 1, projection='3d')
                    title = LABELS[col] if col < 5 else 'Reference'
                    ax.set(title=f'{title}\nframe {frame}', xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), zlim=(-1.15, 1.15))
                    ax.set_box_aspect((1, 1, 1))
                    ax.view_init(elev=15, azim=-65)
                    for parent, child in edges:
                        pair = xyz[col][frame][[parent, child]]
                        ax.plot(*pair.T, color=COLORS[col] if col < 5 else '#777777', linewidth=2)
                    ax.set_xticks([-.8, 0, .8])
                    ax.set_yticks([-.8, 0, .8])
                    ax.set_zticks([-.8, 0, .8])
            fig.suptitle(f'{subject}/{record["source_file"]}, first window | fixed examples inherited from E03\nSame camera, scale, and times; zero translation and original root orientation; head input in every mask')
            fig.subplots_adjust(left=.01, right=.99, bottom=.03, top=.84, hspace=.24, wspace=.02)
            fig.savefig(out / f'{subject}_fixed_example.png')
            plt.close(fig)
            cases.append({'subject': subject, 'sequence_id': record['sequence_id'], 'source_file': record['source_file'],
                          'start_frame': 0, 'snapshot_frames': [41, 83]})
    dump(out / 'manifest.json', cases)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['evaluate', 'analyze', 'validate', 'render'])
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    {'evaluate': evaluate, 'analyze': analyze, 'validate': validate, 'render': render}[args.stage](args.run_dir)
