"""Freeze an explicit DIP fine-tuning recipe and content-matched subject split.

No training or test-set loading occurs here. Existing protocol directories are
never overwritten. Fine-tuning budget/selection choices are ours, not recovered
author hyperparameters.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import torch
import imuposer

from dip_finetuning import EXPECTED_INITIAL_SHA256, MASKS, json_write, validate_masks
from eval_wrist_pocket_dev import fingerprint, raw_inputs
from runtime import load_torch, sha256


def prepare(args):
    validate_masks(MASKS)
    if Path(args.data).name != 'dip_train.pt':
        raise ValueError('Only dip_train.pt is allowed; test data is never loaded')
    if sha256(args.checkpoint) != EXPECTED_INITIAL_SHA256:
        raise ValueError('Expected the audited original Epoch 59 checkpoint')
    reservation = json.loads(Path(args.reservation).read_text())
    if sha256(args.data) != reservation['data']['sha256']:
        raise ValueError('DIP container differs from the original reservation')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    data = load_torch(args.data)
    lookup = {}
    for seq in range(len(data['acc'])):
        key = tuple(fingerprint(data[k][seq]) for k in ['acc', 'ori', 'pose'])
        if key in lookup:
            raise ValueError('Duplicate processed motion content')
        lookup[key] = seq
    records, excluded = [], []
    for i in range(1, 9):
        subject = f's_{i:02d}'
        paths = sorted((Path(args.raw_dip)/subject).glob('*.pkl'))
        if not paths:
            raise FileNotFoundError(f'Missing raw recordings for {subject}')
        for path in paths:
            values = raw_inputs(path)
            raw = {'subject': subject, 'source_file': path.name, 'raw_path': str(path.resolve()),
                   'raw_sha256': sha256(path)}
            if values is None:
                excluded.append({**raw, 'reason': 'Fails inherited finite-value preprocessing after NaN filling/cropping'})
                continue
            key = tuple(fingerprint(values[k]) for k in ['acc', 'ori', 'pose'])
            if key not in lookup:
                raise ValueError(f'Finite raw recording has no exact processed match: {subject}/{path.name}')
            seq = lookup[key]
            record = {**raw, 'sequence_id': seq, 'frames': len(values['acc']),
                      'processed_sha256': dict(zip(['acc', 'ori', 'pose'], key)),
                      'role': 'train' if i <= 6 else 'validation'}
            records.append(record)
            print(f'Exact subject match: {subject}/{path.name} -> sequence {seq} ({record["frames"]} frames)', flush=True)
    if sorted(r['sequence_id'] for r in records) != list(range(len(data['acc']))):
        raise ValueError('Subject mapping must cover every processed sequence exactly once')
    old = {r['sequence_id']: r for r in reservation['selected']}
    dev = {r['sequence_id']: r for r in records if r['role'] == 'validation'}
    if old.keys() != dev.keys():
        raise ValueError('Development reservation changed')
    for seq in old:
        if any(old[seq][k] != dev[seq][k] for k in ['subject', 'raw_sha256', 'processed_sha256', 'frames']):
            raise ValueError('Development reservation identity changed')
    stats = {}
    for role in ['train', 'validation', 'final_first_eight']:
        rows = [r for r in records if role == 'final_first_eight' or r['role'] == role]
        windows = sum((r['frames']+124)//125 for r in rows)
        stats[role] = {'subjects': sorted({r['subject'] for r in rows}), 'sequences': len(rows),
                       'frames': sum(r['frames'] for r in rows), 'motion_windows': windows,
                       'partial_windows': sum(r['frames'] % 125 != 0 for r in rows),
                       'masked_samples': windows*24, 'batches_per_epoch_at_256': (windows*24+255)//256}
    manifest = {'data_path': str(Path(args.data).resolve()), 'data_sha256': sha256(args.data),
                'identity_method': 'Bitwise acc/ori/pose match to locally licensed raw preprocessing',
                'reservation_path': str(Path(args.reservation).resolve()),
                'reservation_sha256': sha256(args.reservation),
                'records': sorted(records, key=lambda r:r['sequence_id']),
                'excluded_raw_recordings': excluded, 'statistics': stats,
                'test_subjects_opened': [], 'test_data_opened': False}
    json_write(out/'data_manifest.json', manifest)
    source_files = [Path(__file__), Path(__file__).with_name('dip_finetuning.py'),
                    Path(__file__).with_name('smoke_dip_finetuning.py'),
                    Path(__file__).with_name('runtime.py'), Path(__file__).with_name('eval_wrist_pocket_dev.py'),
                    Path(__file__).with_name('window_inference.py'), Path(__file__).with_name('paper_metrics.py'),
                    Path(__file__).parents[1]/'tests/test_dip_finetuning.py']
    package = Path(imuposer.__file__).parent
    source_files += sorted(package.rglob('*.py'))
    sources = {}
    for source in source_files:
        relative = Path('installed_imuposer')/source.relative_to(package) if source.is_relative_to(package) else Path('team')/source.name
        dest = out/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        sources[str(source.resolve())] = sha256(source)
    json_write(out/'SOURCE_HASHES.json', sources)
    protocol = {
        'protocol_id': 'DIP_FINETUNING_V1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'argv': sys.argv, 'scope': 'Freeze future recipe and run engineering smoke only; no full training in this task',
        'paper': 'https://arxiv.org/html/2304.12518v1',
        'paper_vs_choice': 'Paper establishes 8/2 subject split and AMASS initialization. Our 30-epoch cap, seed, internal development split, and selection rule are explicit reproduction choices, not recovered author fine-tuning details.',
        'checkpoint': {'path': str(Path(args.checkpoint).resolve()), 'sha256': EXPECTED_INITIAL_SHA256,
                       'epoch': 59, 'global_step': 165060,
                       'known_difference': 'Initialization trained with 25 masks including five-sensor global; 24-mask fine-tuning does not undo that exposure'},
        'data_manifest_sha256': sha256(out/'data_manifest.json'),
        'source_hashes_sha256': sha256(out/'SOURCE_HASHES.json'),
        'smpl_model': {'path': str(Path(args.smpl_model).resolve()), 'sha256': sha256(args.smpl_model)},
        'data': {'fps': 25, 'feature_dim': 60, 'sensor_order': ['lw','rw','lp','rp','h'], 'acc_scale': 30,
                 'smoothing': 'Inherited centered five-frame acceleration mean only; do not re-smooth',
                 'orientation': 'Inherited elementwise resampling; no new orientation filter',
                 'window': 125, 'training_stride': 125, 'retain_partial_tail': True,
                 'batch_padding': 'zero pad inputs and 6D targets to the batch maximum; pass valid lengths to packed LSTM'},
        'masks': MASKS, 'mask_order': list(MASKS), 'mask_expansion': 'All 24 masks per motion window, after subject split; exclude global',
        'seed': 42, 'deterministic_algorithms': True, 'precision': 'float32; no AMP or TF32',
        'optimizer': {'name': 'Adam', 'lr': 0.0003, 'betas': [0.9,0.999], 'eps': 1e-8,
                      'weight_decay': 0.0, 'amsgrad': False, 'foreach': False,
                      'initial_state': 'Fresh Adam; never inherit AMASS optimizer moments'},
        'scheduler': None, 'gradient_clipping': None,
        'loss': {'pose': 'MSE over 144 column-6D values', 'joint': 'MSE over SMPL FK positions in meters, all 24 joints',
                 'weights': [1.0,1.0], 'padding': 'Include padding exactly as inherited public training loss; differs from padding-excluded reporting metrics',
                 'body': 'Mean male shape, zero translation, pose blend shapes disabled'},
        'development': {'train_subjects': [f's_{i:02d}' for i in range(1,7)],
                        'validation_subjects': ['s_07','s_08'], 'split_before_masks': True,
                        'maximum_epochs': 30, 'batch_size': 256, 'shuffle': True, 'workers': 0, 'drop_last': False,
                        'early_stopping': False, 'validation_every_epochs': 1,
                        'checkpoint_selection': 'Smallest full-development combined validation loss; exact ties choose earlier completed epoch. Never use training loss or test metrics to select.',
                        'validation_loss_reduction': 'Sum each batch mean times its padded B*T, divide by total padded B*T; do not average unequal batches equally',
                        'checkpoints': 'Keep last and best with optimizer, RNG, epoch/step, sampler generator/cursor, protocol hash; save after every completed epoch'},
        'final_first_eight': {'train_subjects': [f's_{i:02d}' for i in range(1,9)],
                              'test_subjects': ['s_09','s_10'],
                              'initialization': 'Reset to the same Epoch59 and fresh optimizer/seed, not the development fine-tuned weights',
                              'epochs': 'K = number of completed development epochs at selected checkpoint, 1..30; fixed before final training',
                              'checkpoint_selection': 'The checkpoint after exactly K epochs; no test-driven early stopping or selection',
                              'reservation_role_change': 's07/s08 enter training only in this separately labeled final stage and no longer form independent validation for that model'},
        'reporting': {'before_after': 'Same frozen checkpoint-independent pipeline and same sequences/masks for both models',
                      'inference': 'rolling last-frame, window125, stride1, prefix startup; no hidden-state carry',
                      'filter': 'Retain inherited centered acceleration filter and disclose up to 80ms lookahead plus upstream interpolation/filling; trailing is separate sensitivity analysis',
                      'metrics': ['mpjre_deg','mpjpe_cm','mpjve_cm','jitter_100_m_s3'],
                      'shape_alignment': 'Same mean male SMPL, all24 joints/all6890 vertices, pelvis translation only for spatial errors; zero trajectory translation',
                      'aggregation': 'Within each mask frame-weight spatial errors and T-3-weight jitter over recordings; arithmetic mean across 24 masks. Also retain per-recording and per-mask results.',
                      'refinement': 'Raw network before/after; IMU-only IK is a separate future pipeline component',
                      'success': 'Engineering checks and all four metrics with known differences; low training loss is not paper reproduction success'},
        'smoke': {'steps': 3, 'batch_size': 8, 'samples': 24, 'coverage': 'Exactly one sample for each mask; varied training windows, including a short tail',
                  'validation_use': 'Loader sanity only; no checkpoint selection or tuning in smoke',
                  'additional_resume_probe_steps': 1, 'resume_probe': 'Same prescribed next batch and RNG; compare uninterrupted/reloaded parameter and optimizer states',
                  'eligible_for_final_model': False},
        'environment': {'torch': torch.__version__, 'imported_imuposer': imuposer.__file__,
                        'cuda_available_at_freeze': torch.cuda.is_available()},
        'changes': 'Never silently alter this protocol in place. New recipe changes require a new protocol ID and run directory.'}
    json_write(out/'PROTOCOL.json', protocol)
    json_write(out/'FREEZE_MANIFEST.json', {str(p.relative_to(out)): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({'protocol': str(out/'PROTOCOL.json'), 'sha256': sha256(out/'PROTOCOL.json'),
                      'statistics': stats, 'raw_exclusions': len(excluded)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--reservation', required=True)
    p.add_argument('--raw-dip', required=True)
    p.add_argument('--smpl-model', required=True)
    p.add_argument('--output', required=True)
    prepare(p.parse_args())


if __name__ == '__main__':
    main()
