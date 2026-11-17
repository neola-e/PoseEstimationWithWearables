"""Full DIP paired sensor evaluation. Preserve old outputs; emit auditable joint metrics/cache."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import imuposer

from dip_subset import DipTestSubset
from runtime import PosePredictor, load_body_model, load_torch, peak_ram_mib, pose_rotations, sha256
from sensor_metrics import (COMBOS, JOINT_NAMES, REGIONS, METRICS, joint_errors, mask_input,
                            pair_windows, sensor_mask, validate_coverage, weighted_mean)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def source_hashes(root):
    names = ['tools/eval_sensor_pair.py', 'tools/sensor_metrics.py', 'tools/dip_subset.py',
             'tools/runtime.py', 'tools/eval_dip.py', 'src/imuposer/config.py',
             'src/imuposer/datasets/globalModelDataset.py', 'src/imuposer/datasets/utils.py',
             'scripts/2. Train/1. Train Global Model.py',
             'scripts/1. Preprocessing/1. preprocess_all.py',
             'scripts/1. Preprocessing/2. preprocess_all_to_imuposer_at_25fps.py']
    return {name: sha256(root / name) for name in names if (root / name).is_file()}


def run(args):
    root = Path(__file__).resolve().parents[1]
    for path in (args.checkpoint, args.data, args.smpl_model):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    out = Path(args.output_dir)
    # Refuse reuse even for empty directories, so a run cannot overwrite another run.
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    predictor = PosePredictor(args.checkpoint, device='cpu', threads=args.threads)
    if predictor.epoch != 59:
        raise ValueError('This protocol requires the preselected Epoch 59 checkpoint')
    dataset = DipTestSubset(args.data, predictor.training_config, None, None, 'global')
    raw = load_torch(args.data)
    sequence_lengths = [len(x) for x in raw['acc']]
    if any(len(raw[k]) != len(sequence_lengths) for k in ('ori', 'pose')):
        raise ValueError('Raw sequence counts differ')
    del raw
    coverage = validate_coverage(dataset.records, sequence_lengths)
    body = load_body_model(args.smpl_model)
    checkpoint_hash = sha256(args.checkpoint)
    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'checkpoint': {'path': str(Path(args.checkpoint).resolve()),
            'sha256': checkpoint_hash, 'epoch': predictor.epoch, 'global_step': predictor.global_step,
            'training_validation_metric': predictor.validation_score,
            'config': {k: str(v) if isinstance(v, (Path, torch.device)) else v
                       for k, v in vars(predictor.training_config).items()}},
        'data': {'path': str(Path(args.data).resolve()), 'sha256': sha256(args.data),
                 'sequence_lengths': sequence_lengths},
        'smpl': {'path': str(Path(args.smpl_model).resolve()), 'sha256': sha256(args.smpl_model)},
        'coverage': coverage, 'combos': COMBOS,
        'inference': {'device': 'cpu', 'threads': args.threads, 'batch_size': 1, 'fps': 25,
            'window_size': 125, 'independent_windows': True, 'eval_mode': True,
            'inference_mode': True, 'ik_refinement': False, 'mean_male_shape': True,
            'zero_root_translation': True, 'rigid_alignment': False},
        'metric_conventions': {'position_cm': 'Mean Euclidean joint position distance in cm; all 24 joints including zero-error root; no root-rotation or rigid alignment.',
            'local_deg': 'SO(3) geodesic of local SMPL rotations, including global root orientation.',
            'global_deg': 'SO(3) geodesic after SMPL forward kinematics.',
            'aggregation': 'Each window has per-joint frame means; aggregate by valid frame counts, then equal joints. No padding.'},
        'joint_names': JOINT_NAMES, 'regions': REGIONS,
        'software': {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: importlib.metadata.version(name) for name in
                         ['torch', 'numpy', 'pytorch-lightning', 'scipy', 'matplotlib', 'Pillow', 'chumpy']},
            'imuposer_import': str(Path(imuposer.__file__).resolve())},
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'git_status_at_start': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
        'source_sha256': source_hashes(root),
    }
    write_json(out / 'provenance.json', provenance)
    reports = {combo: [] for combo in COMBOS}
    poses = {combo: [] for combo in COMBOS}
    targets = []
    diagnostics = []
    input_checks = []
    with torch.inference_mode():
        for idx, record in enumerate(dataset.records):
            imu, target = dataset[idx]
            if len(imu) != record['frames'] or target.shape != (record['frames'], 144):
                raise ValueError('Window tensor/metadata mismatch')
            gt_rotation = pose_rotations(target)
            gt_global, gt_position = body.forward_kinematics(gt_rotation, calc_mesh=False)
            targets.append(target.numpy())
            orientation = imu[:, 15:].reshape(-1, 5, 3, 3)
            gram_error = (orientation.transpose(-1, -2) @ orientation - torch.eye(3)).square().sum((-1, -2)).sqrt()
            # Trace/acos on interpolated IMU matrices is only an approximate consistency diagnostic.
            segment = gt_global[:, [18, 19, 1, 2, 15]]
            trace = (orientation.transpose(-1, -2) @ segment).diagonal(dim1=-2, dim2=-1).sum(-1)
            mismatch = torch.rad2deg(((trace - 1) / 2).clamp(-1, 1).acos())
            input_checks.append({'sequence_id': record['sequence_id'], 'start_frame': record['start_frame'],
                'frames': record['frames'], 'acceleration_norm_mean_m_s2': (imu[:, :15].reshape(-1, 5, 3).norm(dim=-1) * 30).mean(0).tolist(),
                'orientation_orthogonality_frobenius_mean': gram_error.mean(0).tolist(),
                'approx_orientation_vs_gt_segment_deg': mismatch.mean(0).tolist()})
            # Motion descriptors come only from GT, not activity labels. No cross-window differences.
            speed = (gt_position[1:] - gt_position[:-1]).norm(dim=-1) * 25
            arm_speed = speed[:, REGIONS['arms']].mean().item() if len(speed) else 0.
            leg_speed = speed[:, REGIONS['legs']].mean().item() if len(speed) else 0.
            diagnostics.append({**record, 'gt_arm_speed_m_s': arm_speed, 'gt_leg_speed_m_s': leg_speed})
            for combo, sensors in COMBOS.items():
                features = torch.from_numpy(mask_input(imu.numpy(), sensors))
                mask = sensor_mask(sensors)
                if not torch.equal(features[:, mask], imu[:, mask]) or torch.count_nonzero(features[:, ~mask]):
                    raise ValueError('Masking altered retained values or failed to zero missing values')
                prediction = predictor.predict(features)
                poses[combo].append(prediction.numpy())
                errors = joint_errors(pose_rotations(prediction), gt_rotation, body, gt_global, gt_position)
                joint = {k: v.mean(0).tolist() for k, v in errors.items()}
                reports[combo].append({**record, 'joint_errors': joint,
                    **{k: v.mean().item() for k, v in errors.items()},
                    'target_sha256': hashlib.sha256(target.numpy().tobytes()).hexdigest()})
            if (idx + 1) % 25 == 0 or idx + 1 == len(dataset):
                print(f'Evaluated paired windows {idx + 1}/{len(dataset)}', flush=True)
    pairs = pair_windows(reports['global'], reports['lw_rp_h'])
    if any(a['target_sha256'] != b['target_sha256'] for a, b in pairs):
        raise ValueError('Ground truth differs between configurations')
    reference_validation = None
    if args.reference_global:
        reference = json.loads(Path(args.reference_global).read_text())
        if reference['checkpoint']['sha256'] != checkpoint_hash or reference['sensor_combo'] != 'global':
            raise ValueError('Reference checkpoint/config mismatch')
        keys = {'position_cm': 'root_relative_mpjpe_cm', 'local_deg': 'mean_local_joint_angle_deg',
                'global_deg': 'mean_global_joint_angle_deg'}
        comparisons = pair_windows(reference['per_window'], reports['global'])
        max_diff = {k: max(abs(a[old] - b[k]) for a, b in comparisons) for k, old in keys.items()}
        if any(v > 1e-4 for v in max_diff.values()):
            raise ValueError(f'Existing global baseline changed: {max_diff}')
        reference_validation = {'path': str(Path(args.reference_global).resolve()),
            'sha256': sha256(args.reference_global), 'max_absolute_window_metric_differences': max_diff,
            'tolerance': 1e-4, 'windows_checked': len(comparisons)}
    frames = [r['frames'] for r in dataset.records]
    for combo, rows in reports.items():
        report = {'sensor_combo': combo, 'sensor_indices': COMBOS[combo], 'coverage': coverage,
            'metrics': {k: float(weighted_mean([r[k] for r in rows], frames)) for k in METRICS},
            'joint_errors': {k: weighted_mean([r['joint_errors'][k] for r in rows], frames).tolist() for k in METRICS},
            'per_window': rows}
        write_json(out / f'{combo}_metrics.json', report)
    np.savez_compressed(out / 'predictions.npz', target_pose=np.concatenate(targets),
        **{f'pose_{combo}': np.concatenate(poses[combo]) for combo in COMBOS},
        offsets=np.concatenate([[0], np.cumsum(frames)]),
        sequence_ids=[r['sequence_id'] for r in dataset.records],
        start_frames=[r['start_frame'] for r in dataset.records], frames=frames, fps=25)
    write_json(out / 'input_consistency.json', input_checks)
    write_json(out / 'motion_descriptors.json', diagnostics)
    write_json(out / 'validation.json', {'coverage': coverage, 'exact_pairs_checked': len(pairs),
        'identical_ground_truth_hashes': True, 'mask_channels_checked_every_window': True,
        'reference_global': reference_validation, 'elapsed_seconds': time.perf_counter() - started,
        'peak_process_ram_mib': peak_ram_mib(), 'status': 'passed'})
    print(json.dumps({'output_dir': str(out), 'coverage': coverage, 'status': 'passed'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--smpl-model', required=True)
    parser.add_argument('--reference-global')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--threads', type=int, default=2)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
