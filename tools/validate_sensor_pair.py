"""Independently recompute all cached-window metrics and audit the study's exported artifacts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from runtime import load_body_model, pose_rotations, sha256
from sensor_metrics import COMBOS, METRICS, pair_windows, validate_coverage


def validate(args):
    run = Path(args.run_dir)
    destination = run / 'final_validation.json'
    if destination.exists():
        raise FileExistsError(destination)
    provenance = json.loads((run / 'provenance.json').read_text())
    for asset in ['checkpoint', 'data', 'smpl']:
        if sha256(provenance[asset]['path']) != provenance[asset]['sha256']:
            raise ValueError(f'Asset changed: {asset}')
    root = Path(__file__).resolve().parents[1]
    for name, digest in provenance['source_sha256'].items():
        if sha256(root / name) != digest:
            raise ValueError(f'Evaluation source changed since inference: {name}')
    reports = {c: json.loads((run / f'{c}_metrics.json').read_text()) for c in COMBOS}
    pairs = pair_windows(reports['global']['per_window'], reports['lw_rp_h']['per_window'])
    coverage = validate_coverage([a for a, b in pairs], provenance['data']['sequence_lengths'])
    torch.set_num_threads(2)
    body = load_body_model(provenance['smpl']['path'])
    maximum = {k: 0. for k in METRICS}
    with np.load(run / 'predictions.npz', allow_pickle=False) as cache, torch.inference_mode():
        expected_shape = (coverage['frames'], 144)
        if any(cache[k].shape != expected_shape or not np.isfinite(cache[k]).all()
               for k in ['target_pose', 'pose_global', 'pose_lw_rp_h']):
            raise ValueError('Prediction cache shape/finiteness mismatch')
        if not np.array_equal(np.diff(cache['offsets']), cache['frames']):
            raise ValueError('Cache offsets mismatch')
        for i, (a, b) in enumerate(pairs):
            if (int(cache['sequence_ids'][i]), int(cache['start_frames'][i]), int(cache['frames'][i])) != (a['sequence_id'], a['start_frame'], a['frames']):
                raise ValueError('Cache/JSON alignment mismatch')
            sl = slice(cache['offsets'][i], cache['offsets'][i + 1])
            target = cache['target_pose'][sl]
            if hashlib.sha256(target.tobytes()).hexdigest() != a['target_sha256'] or a['target_sha256'] != b['target_sha256']:
                raise ValueError('Cache/JSON ground truth hash mismatch')
            gt = pose_rotations(torch.from_numpy(target))
            gt_global, gt_pos = body.forward_kinematics(gt, calc_mesh=False)
            for combo, row in [('global', a), ('lw_rp_h', b)]:
                rotation = pose_rotations(torch.from_numpy(cache[f'pose_{combo}'][sl]))
                global_rot, pos = body.forward_kinematics(rotation, calc_mesh=False)
                position = (pos - gt_pos).norm(dim=-1) * 100
                local_cos = ((rotation.transpose(-1, -2) @ gt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
                global_cos = ((global_rot.transpose(-1, -2) @ gt_global).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
                values = {'position_cm': position, 'local_deg': torch.rad2deg(local_cos.clamp(-1, 1).acos()),
                          'global_deg': torch.rad2deg(global_cos.clamp(-1, 1).acos())}
                if position[:, 0].abs().max() != 0:
                    raise ValueError('Nonzero root position error under zero translation')
                for metric, tensor in values.items():
                    difference = max(abs(tensor.mean().item() - row[metric]),
                        float(np.abs(tensor.mean(0).numpy() - row['joint_errors'][metric]).max()))
                    maximum[metric] = max(maximum[metric], difference)
                    if difference > 1e-4:
                        raise ValueError(f'Cached prediction metric mismatch: {combo}/{i}/{metric}')
    # Independently audit frame aggregation and all exported scalar sequence/window deltas.
    sequence_csv = list(csv.DictReader((run / 'analysis/per_sequence.csv').open()))
    window_csv = list(csv.DictReader((run / 'analysis/per_window.csv').open()))
    if len(sequence_csv) != 19 or len(window_csv) != len(pairs):
        raise ValueError('CSV row count mismatch')
    aliases = {'position_cm': 'mpjpe_cm', 'local_deg': 'local_deg', 'global_deg': 'global_deg'}
    for combo, report in reports.items():
        n = sum(r['frames'] for r in report['per_window'])
        for metric, alias in aliases.items():
            value = sum(r[metric] * r['frames'] for r in report['per_window']) / n
            if abs(value - report['metrics'][metric]) > 1e-5:
                raise ValueError('Overall frame weighting mismatch')
            for row in sequence_csv:
                subset = [r for r in report['per_window'] if r['sequence_id'] == int(row['sequence_id'])]
                value = sum(r[metric] * r['frames'] for r in subset) / sum(r['frames'] for r in subset)
                if abs(value - float(row[f'{combo}_{alias}'])) > 1e-5:
                    raise ValueError('Per-sequence aggregation mismatch')
    for row, (a, b) in zip(window_csv, pairs):
        if tuple(int(row[k]) for k in ['sequence_id', 'start_frame', 'frames']) != (a['sequence_id'], a['start_frame'], a['frames']):
            raise ValueError('CSV window alignment mismatch')
        for metric, alias in aliases.items():
            if abs(float(row[f'delta_{alias}']) - (b[metric] - a[metric])) > 1e-5:
                raise ValueError('Paired delta mismatch')
    historical = root / 'outputs/local_dip/seq18_3sensor_metrics.json'
    checked_historical = 0
    if historical.exists():
        old = json.loads(historical.read_text())
        for row in old['per_window']:
            new = next(r for r in reports['lw_rp_h']['per_window'] if
                       (r['sequence_id'], r['start_frame'], r['frames']) == (row['sequence_id'], row['start_frame'], row['frames']))
            for metric, old_key in [('position_cm', 'root_relative_mpjpe_cm'), ('local_deg', 'mean_local_joint_angle_deg'), ('global_deg', 'mean_global_joint_angle_deg')]:
                if abs(new[metric] - row[old_key]) > 1e-4:
                    raise ValueError('Historical three-location metrics changed')
            checked_historical += 1
    # Capture code only; never copy licensed data/model assets or private run metadata.
    snapshot = run / 'source_snapshot'
    snapshot.mkdir(exist_ok=False)
    files = [root / p for p in provenance['source_sha256']]
    files += [root / 'tools' / name for name in ['analyze_sensor_pair.py', 'diagnose_sensor_pair.py',
              'render_sensor_cases.py', 'validate_sensor_pair.py']]
    files += [root / 'tests/test_sensor_metrics.py', root / 'docs/SENSOR_EVALUATION.md', root / 'LICENSE']
    imported = json.loads((run / 'diagnostics/training_source_audit.json').read_text())
    for p, digest in imported['actual_imported_source_sha256'].items():
        if sha256(p) != digest:
            raise ValueError(f'Imported source changed: {p}')
    files += [Path(p) for p in imported['actual_imported_source_sha256']]
    files += list(Path(imported['actual_imported_package']).rglob('*.py'))
    source_manifest = {}
    original_root = Path(imported['actual_imported_package']).parents[1]
    for path in files:
        if path.is_relative_to(root):
            relative = Path('team') / path.relative_to(root)
        else:
            relative = Path('original') / path.relative_to(original_root)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        source_manifest[str(relative)] = sha256(target)
    (snapshot / 'SHA256.json').write_text(json.dumps(source_manifest, indent=2) + '\n')
    result = {'status': 'passed', 'coverage': coverage, 'prediction_windows_recomputed': len(pairs) * 2,
        'all_joint_metric_max_absolute_difference': maximum, 'asset_hashes_rechecked': True,
        'evaluation_sources_unchanged_since_inference': True, 'csv_deltas_and_sequence_aggregation_checked': True,
        'historical_three_sensor_windows_reproduced': checked_historical,
        'source_snapshot_files': len(source_manifest), 'tolerance': 1e-4}
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    validate(parser.parse_args())
