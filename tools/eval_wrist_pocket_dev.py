"""Reserve DIP development subjects by content and evaluate four fixed placements.

This entry point never opens dip_test.pt or changes a model or dataset asset.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import imuposer
from imuposer import math
from imuposer.config import amass_combos
from imuposer.datasets.globalModelDataset import GlobalModelDataset

from runtime import PosePredictor, load_body_model, load_torch, peak_ram_mib, pose_rotations, sha256
from sensor_metrics import JOINT_NAMES, METRICS, REGIONS, joint_errors, mask_input, sensor_mask

COMBOS = {'lw_rp_h': [0, 3, 4], 'rw_rp_h': [1, 3, 4],
          'lw_lp_h': [0, 2, 4], 'rw_lp_h': [1, 2, 4]}


def dump(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def fingerprint(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def resample(value):
    """Reproduce the inherited 60 to 25 FPS linear interpolation exactly."""
    indices = torch.arange(0, value.shape[0], 60 / 25)
    first, last = indices.floor().long(), indices.ceil().long()
    last[last >= len(value)] = len(value) - 1
    fraction = indices - first
    for _ in range(value.ndim - 1):
        fraction = fraction.unsqueeze(1)
    weights = torch.ones_like(value[first]) * fraction
    return torch.lerp(value[first], value[last], weights)


def smooth(value):
    edge = torch.zeros((2, value.shape[1], value.shape[2])) * torch.nan
    padded = torch.cat((edge, value, edge))
    return torch.stack([padded[i:len(padded) - (5 - i - 1)] for i in range(5)]).nanmean(0)


def raw_inputs(path):
    # Existing locally obtained research assets, with the same trust assumption
    # as the original preprocessing script. No untrusted download is loaded.
    with path.open('rb') as stream:
        source = pickle.load(stream, encoding='latin1')
    acc = torch.from_numpy(source['imu_acc'][:, [7, 8, 9, 10, 0, 2]]).float()
    ori = torch.from_numpy(source['imu_ori'][:, [7, 8, 9, 10, 0, 2]]).float()
    pose = torch.from_numpy(source['gt']).float()
    del source
    for _ in range(4):
        for value in (acc, ori):
            value[1:].masked_scatter_(torch.isnan(value[1:]), value[:-1][torch.isnan(value[1:])])
            value[:-1].masked_scatter_(torch.isnan(value[:-1]), value[1:][torch.isnan(value[:-1])])
    acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
    if not all(torch.isfinite(x).all() for x in (acc, ori, pose)):
        return None
    return {'acc': smooth(resample(acc)), 'ori': resample(ori),
            'pose': math.axis_angle_to_rotation_matrix(resample(pose).contiguous()).view(-1, 24, 3, 3)}


def validate_records(records, selected, window=125):
    expected = {r['sequence_id']: r for r in selected}
    if len(expected) != len(selected) or {r['sequence_id'] for r in records} != set(expected):
        raise ValueError('Reserved sequence coverage mismatch')
    for seq, source in expected.items():
        cursor = 0
        for record in sorted((r for r in records if r['sequence_id'] == seq), key=lambda r: r['start_frame']):
            if (record['start_frame'] != cursor
                    or record['frames'] != min(window, source['frames'] - cursor)
                    or record['frames'] < 1 or record['subject'] != source['subject']):
                raise ValueError('Gap, overlap, wrong length or wrong subject')
            cursor += record['frames']
        if cursor != source['frames']:
            raise ValueError('Truncated reserved sequence')
    return {'subjects': len({s['subject'] for s in selected}), 'sequences': len(selected),
            'windows': len(records), 'frames': sum(s['frames'] for s in selected),
            'partial_windows': sum(r['frames'] < window for r in records)}


class ReservedDipSubset(GlobalModelDataset):
    """Use the installed dataset's original target conversion, without test loading."""
    def __init__(self, data, selected, config):
        self.source_data, self.selected = data, selected
        super().__init__(split='development', config=config)

    def load_data(self):
        self.imu, self.pose, self.imu_masks, self.records = [], [], [], []
        window = self.config.max_sample_len * 25 // 60
        for row in self.selected:
            seq = row['sequence_id']
            acc, ori, pose = (self.source_data[k][seq] for k in ('acc', 'ori', 'pose'))
            n = row['frames']
            if acc.shape != (n, 6, 3) or ori.shape != (n, 6, 3, 3) or pose.shape != (n, 24, 3, 3):
                raise ValueError('Unexpected development tensor schema')
            x = torch.cat(((acc[:, :5] / self.config.acc_scale).flatten(1),
                           ori[:, :5].float().flatten(1)), dim=1)
            if not torch.isfinite(x).all() or not torch.isfinite(pose).all():
                raise ValueError('Non-finite source')
            for start in range(0, n, window):
                stop = min(start + window, n)
                self.imu.append(x[start:stop])
                self.pose.append(pose[start:stop].reshape(stop - start, -1))
                self.imu_masks.append(torch.ones(60, dtype=torch.bool))
                self.records.append({**row, 'start_frame': start, 'frames': stop - start, 'original_frames': n})


def read_protocol(run):
    p = json.loads((run / 'PROTOCOL.json').read_text())
    if (p['development_subjects'] != ['s_07', 's_08'] or p['configurations'] != COMBOS
            or set(p['development_subjects']) & set(p['excluded_test_subjects'])):
        raise ValueError('Unexpected or overlapping protocol')
    return p


def reserve(args):
    out = Path(args.run_dir)
    protocol = read_protocol(out)
    if Path(args.data).name != 'dip_train.pt':
        raise ValueError('Only dip_train.pt is allowed')
    if (out / 'split_manifest.json').exists():
        raise FileExistsError('Reservation already exists')
    torch.set_num_threads(2)
    data = load_torch(args.data)
    n = len(data['acc'])
    if any(len(data[k]) != n for k in ('ori', 'pose')):
        raise ValueError('Sequence counts differ')
    index = {}
    for seq in range(n):
        key = tuple(fingerprint(data[k][seq]) for k in ('acc', 'ori', 'pose'))
        if key in index:
            raise ValueError('Ambiguous duplicate processed sequence')
        index[key] = seq
    selected, excluded = [], []
    for subject in protocol['development_subjects']:
        files = sorted((Path(args.raw_dip) / subject).glob('*.pkl'))
        if not files:
            raise FileNotFoundError(f'Raw recordings missing for {subject}')
        for path in files:
            values = raw_inputs(path)
            meta = {'subject': subject, 'source_file': path.name, 'raw_path': str(path.resolve()),
                    'raw_sha256': sha256(path)}
            if values is None:
                excluded.append({**meta, 'reason': 'Fails original finite-value preprocessing criterion'})
                continue
            key = tuple(fingerprint(values[k]) for k in ('acc', 'ori', 'pose'))
            if key not in index:
                raise ValueError(f'No exact acc/ori/pose match in dip_train.pt for {subject}/{path.name}')
            seq = index[key]
            for k in values:
                if not torch.equal(values[k], data[k][seq]):
                    raise ValueError('Digest match did not reproduce values')
            selected.append({**meta, 'sequence_id': seq, 'frames': len(values['acc']),
                             'processed_sha256': dict(zip(('acc', 'ori', 'pose'), key))})
            print(f'Exact match: {subject}/{path.name} -> train sequence {seq}, {len(values["acc"])} frames', flush=True)
    selected_ids = [r['sequence_id'] for r in selected]
    if len(selected_ids) != len(set(selected_ids)) or {r['subject'] for r in selected} != set(protocol['development_subjects']):
        raise ValueError('Duplicate or missing reserved subject')
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
                'protocol_sha256': sha256(out / 'PROTOCOL.json'),
                'data': {'path': str(Path(args.data).resolve()), 'sha256': sha256(args.data),
                         'sequence_count': n},
                'identity_check': 'Bitwise equality of acceleration, orientation and reference rotations after raw preprocessing',
                'selected': selected, 'excluded_raw_recordings': excluded,
                'remaining_sequence_ids': sorted(set(range(n)) - set(selected_ids)),
                'test_container_opened': False,
                'reservation_note': 'Reserved now for development. The inspected global training loader excludes DIP. A complete historical training corpus audit is unavailable.'}
    dump(out / 'split_manifest.json', manifest)
    print(f'Reserved {len(selected)} sequences / {sum(r["frames"] for r in selected)} frames', flush=True)


def source_hashes(root):
    paths = [root / 'tools' / n for n in ['eval_wrist_pocket_dev.py', 'analyze_wrist_pocket_dev.py',
             'sensor_metrics.py', 'runtime.py']]
    package = Path(imuposer.__file__).resolve().parent
    paths += [package / n for n in ['config.py', 'datasets/globalModelDataset.py',
              'models/LSTMs/IMUPoser_Model.py', 'models/LSTMs/RNN.py', 'math/angular.py',
              'smpl/parametricModel.py']]
    return {str(p): sha256(p) for p in paths if p.is_file()}


def evaluate(args):
    out = Path(args.run_dir)
    protocol = read_protocol(out)
    manifest = json.loads((out / 'split_manifest.json').read_text())
    if sha256(out / 'PROTOCOL.json') != manifest['protocol_sha256']:
        raise ValueError('Protocol changed after reservation')
    if Path(args.data).name != 'dip_train.pt' or sha256(args.data) != manifest['data']['sha256']:
        raise ValueError('Reserved data changed or wrong split supplied')
    if sha256(args.checkpoint) != protocol['checkpoint_sha256']:
        raise ValueError('Wrong checkpoint')
    for c, sensors in COMBOS.items():
        if amass_combos[c] != sensors:
            raise ValueError('Installed training configuration differs')
    result_dir = out / 'evaluation'
    result_dir.mkdir(exist_ok=False)
    started = time.perf_counter()
    predictor = PosePredictor(args.checkpoint, device='cpu', threads=2)
    if predictor.epoch != 59:
        raise ValueError('Wrong checkpoint epoch')
    data = load_torch(args.data)
    for row in manifest['selected']:
        for k, digest in row['processed_sha256'].items():
            if fingerprint(data[k][row['sequence_id']]) != digest:
                raise ValueError('Reserved sequence changed')
    dataset = ReservedDipSubset(data, manifest['selected'], predictor.training_config)
    coverage = validate_records(dataset.records, manifest['selected'])
    body = load_body_model(args.smpl_model)
    root = Path(__file__).resolve().parents[1]
    provenance = {'created_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
                  'protocol_sha256': sha256(out / 'PROTOCOL.json'),
                  'split_manifest_sha256': sha256(out / 'split_manifest.json'), 'coverage': coverage,
                  'checkpoint': {'path': str(Path(args.checkpoint).resolve()), 'sha256': sha256(args.checkpoint),
                                 'epoch': predictor.epoch, 'global_step': predictor.global_step,
                                 'training_validation_metric': predictor.validation_score},
                  'data': manifest['data'],
                  'smpl': {'path': str(Path(args.smpl_model).resolve()), 'sha256': sha256(args.smpl_model)},
                  'source_sha256': source_hashes(root), 'imuposer_import': str(Path(imuposer.__file__).resolve()),
                  'software': {'python': platform.python_version(), **{n: importlib.metadata.version(n) for n in
                               ['torch', 'numpy', 'pytorch-lightning', 'matplotlib', 'Pillow', 'scipy']}},
                  'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                  'git_status_at_start': subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
                  'joint_names': JOINT_NAMES, 'regions': REGIONS, 'test_container_opened': False}
    dump(result_dir / 'provenance.json', provenance)
    poses, reports = {c: [] for c in COMBOS}, {c: [] for c in COMBOS}
    targets, offsets = [], [0]
    with torch.inference_mode():
        for i, row in enumerate(dataset.records):
            imu, target = dataset[i]
            original = imu.clone()
            gt_rot = pose_rotations(target)
            gt_g, gt_p = body.forward_kinematics(gt_rot, calc_mesh=False)
            target_hash = fingerprint(target)
            targets.append(target.numpy())
            offsets.append(offsets[-1] + len(target))
            for combo, sensors in COMBOS.items():
                features = mask_input(imu.numpy(), sensors)
                mask = sensor_mask(sensors)
                if not np.array_equal(features[:, mask], imu.numpy()[:, mask]) or np.count_nonzero(features[:, ~mask]):
                    raise ValueError('Invalid sensor masking')
                pred = predictor.predict(features)
                errors = joint_errors(pose_rotations(pred), gt_rot, body, gt_g, gt_p)
                poses[combo].append(pred.numpy())
                reports[combo].append({**row, 'target_sha256': target_hash,
                                       'joint_errors': {k: v.mean(0).tolist() for k, v in errors.items()}})
            if not torch.equal(imu, original):
                raise ValueError('Masking modified shared input')
            if (i + 1) % 25 == 0 or i + 1 == len(dataset):
                print(f'Evaluated {i + 1}/{len(dataset)} windows for all four placements', flush=True)
    for combo in COMBOS:
        dump(result_dir / f'{combo}_metrics.json', {'configuration': combo, 'per_window': reports[combo]})
    np.savez_compressed(result_dir / 'predictions.npz',
                        **{f'pose_{c}': np.concatenate(v) for c, v in poses.items()},
                        target_pose=np.concatenate(targets), offsets=np.array(offsets),
                        sequence_ids=np.array([r['sequence_id'] for r in dataset.records]),
                        start_frames=np.array([r['start_frame'] for r in dataset.records]),
                        subjects=np.array([r['subject'] for r in dataset.records]))
    dump(result_dir / 'resources.json', {'seconds_including_cache_write': time.perf_counter() - started,
                                       'peak_process_rss_mib': peak_ram_mib(), 'gpu_used': False})
    print(f'Completed: {coverage}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['reserve', 'evaluate'])
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--raw-dip')
    parser.add_argument('--checkpoint')
    parser.add_argument('--smpl-model')
    args = parser.parse_args()
    if args.stage == 'reserve' and not args.raw_dip:
        parser.error('--raw-dip is required for reserve')
    if args.stage == 'evaluate' and (not args.checkpoint or not args.smpl_model):
        parser.error('--checkpoint and --smpl-model are required for evaluate')
    (reserve if args.stage == 'reserve' else evaluate)(args)
