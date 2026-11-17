"""Read-only inventory of the Epoch 59 run, data, preprocessing, and sources.

Run in the existing imuposer environment. Never constructs the training dataset,
trains a model, performs pose inference, or rewrites an input asset. Outputs are
metadata, hashes, and source snapshots; no weights or dataset arrays are copied.
"""
import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import zipfile
import zlib

import numpy as np
import torch
import imuposer
from imuposer import math as pose_math
from imuposer.config import Config, amass_combos, amass_datasets
from eval_wrist_pocket_dev import resample, smooth, fingerprint
from runtime import sha256, peak_ram_mib


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def file_info(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'bytes': stat.st_size, 'sha256': sha256(path),
            'mtime_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()}


def checkpoint_info(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    config = saved['hyper_parameters']['config']
    states = saved['state_dict']
    result = {**file_info(path), 'epoch': saved['epoch'], 'global_step': saved['global_step'],
              'lightning_version': saved.get('pytorch-lightning_version'),
              'serialized_config_instance': clean(vars(config)),
              'effective_current_class_defaults_not_serialized': {
                  k: {'value': clean(getattr(config, k)), 'serialized_as_instance_attribute': k in vars(config)}
                  for k in ['batch_size', 'max_sample_len', 'acc_scale', 'train_pct', 'torch_seed']},
              'state_dict_shapes': {k: list(v.shape) for k, v in states.items()},
              'network_parameters': sum(v.numel() for v in states.values()),
              'network_tensors_finite': all(bool(torch.isfinite(v).all()) for v in states.values()),
              'callbacks': clean(saved.get('callbacks', {})), 'loops': clean(saved.get('loops', {})),
              'optimizer_parameter_groups': [clean(o['param_groups']) for o in saved.get('optimizer_states', [])],
              'optimizer_state_steps': [sorted(set(float(s['step']) for s in o['state'].values() if 'step' in s))
                                        for o in saved.get('optimizer_states', [])],
              'lr_schedulers': clean(saved.get('lr_schedulers', [])),
              'rng_state_or_split_keys': [k for k in saved if any(t in k.lower() for t in ['rng', 'random', 'split', 'sampler'])]}
    del saved, states
    gc.collect()
    return result


def history_payloads(path):
    """Read W&B log records without the removed SDK DataStore module.

    Format and CRC convention: wandb/wandb v0.19.11,
    wandb/sdk/internal/datastore.py. Open read-only; verify every fragment CRC.
    """
    fragments = None
    with path.open('rb') as stream:
        if stream.read(7) != struct.pack('<4sHB', b':W&B', 0xBEE1, 0):
            raise ValueError('Unexpected W&B log header')
        while True:
            remaining = 32768 - stream.tell() % 32768
            if remaining < 7:
                if stream.read(remaining) != bytes(remaining):
                    raise ValueError('Nonzero log padding')
            header = stream.read(7)
            if not header:
                if fragments is not None:
                    raise ValueError('Truncated fragmented record')
                return
            if len(header) != 7:
                raise ValueError('Truncated log header')
            checksum, length, kind = struct.unpack('<IHB', header)
            value = stream.read(length)
            if kind not in [1, 2, 3, 4] or len(value) != length:
                raise ValueError('Invalid log fragment')
            if zlib.crc32(value, zlib.crc32(bytes([kind]))) & 0xFFFFFFFF != checksum:
                raise ValueError('Log fragment CRC mismatch')
            if kind == 1:
                if fragments is not None:
                    raise ValueError('Unexpected complete record')
                yield value
            elif kind == 2:
                if fragments is not None:
                    raise ValueError('Unexpected first fragment')
                fragments = [value]
            else:
                if fragments is None:
                    raise ValueError('Missing first fragment')
                fragments.append(value)
                if kind == 4:
                    yield b''.join(fragments)
                    fragments = None


def run_history(path):
    from wandb.proto import wandb_internal_pb2
    result, count = [], 0
    for payload in history_payloads(path):
        count += 1
        record = wandb_internal_pb2.Record()
        record.ParseFromString(payload)
        if record.HasField('history'):
            row = {('.'.join(i.nested_key) if i.nested_key else i.key): json.loads(i.value_json)
                   for i in record.history.item}
            if 'validation_step_loss' in row:
                result.append({k: row.get(k) for k in ['epoch', 'trainer/global_step', 'validation_step_loss', 'val_loss']})
    return result, count


def inspect_raw(raw_root):
    rows, summaries = [], []
    for subset in amass_datasets:
        files = list((raw_root / subset).glob('*/*_poses.npz'))
        rates = {}
        for ordinal, path in enumerate(files):
            with np.load(path, allow_pickle=False) as source:
                fps = float(source['mocap_framerate'])
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo('poses.npy')
                with archive.open('poses.npy') as stream:
                    version = np.lib.format.read_magic(stream)
                    header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)(stream)
                shape = header[0]
            stat = path.stat()
            accepted = int(fps) in [59, 60, 120]
            step = 2 if int(fps) == 120 else 1
            frames_60 = math.ceil(shape[0] / step) if accepted else 0
            rows.append({'subset': subset, 'current_glob_ordinal': ordinal,
                         'relative_path': str(path.relative_to(raw_root)), 'bytes': stat.st_size,
                         'mtime_utc': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                         'mocap_framerate': fps, 'pose_shape': str(shape),
                         'poses_zip_crc32_not_crypto_hash': f'{info.CRC:08x}',
                         'fps_accepted_by_current_preprocessor': accepted,
                         'length_accepted': accepted and frames_60 > 12,
                         'expected_60fps_frames': frames_60})
            rates[str(fps)] = rates.get(str(fps), 0) + 1
        selected = [r for r in rows if r['subset'] == subset]
        summaries.append({'subset': subset, 'raw_recordings_matching_glob': len(files), 'framerates': rates,
                          'accepted_fps_recordings': sum(r['fps_accepted_by_current_preprocessor'] for r in selected),
                          'accepted_fps_and_length_recordings': sum(r['length_accepted'] for r in selected)})
        print(f'Raw inventory: {subset}, {len(files)} recordings, FPS={rates}', flush=True)
    return rows, summaries


def inspect_processed(root, processed_dir):
    # Preserve iterdir order because the installed loader does not sort files.
    files = list(processed_dir.iterdir())
    selected = [p.name for p in files if 'dip' not in p.name]
    rows, sequence_rows, reconstruction = [], [], []
    for ordinal, path in enumerate(files):
        if not path.is_file() or path.suffix != '.pt':
            raise ValueError(f'Unexpected loader input: {path}')
        data = torch.load(path, map_location='cpu', weights_only=True)
        lengths = [len(x) for x in data['acc']]
        if not all(len(data[k]) == len(lengths) for k in ['ori', 'pose']):
            raise ValueError('Unpaired processed arrays')
        for i, n in enumerate(lengths):
            if (data['acc'][i].shape != (n, 6, 3) or data['ori'][i].shape != (n, 6, 3, 3)
                    or data['pose'][i].shape != (n, 24, 3, 3)):
                raise ValueError('Unexpected processed tensor shape')
            if not all(bool(torch.isfinite(data[k][i]).all()) for k in ['acc', 'ori', 'pose']):
                raise ValueError('Nonfinite processed tensor')
            sequence_rows.append({'file': path.name, 'sequence_index': i, 'frames_25fps': n,
                                  'windows_125_with_tail': math.ceil(n / 125),
                                  'tail_frames': n % 125,
                                  **{f'{k}_content_sha256': fingerprint(data[k][i]) for k in ['acc', 'ori', 'pose']}})
        row = {'current_directory_ordinal': ordinal, 'file': path.name,
               'selected_by_current_global_train_loader': path.name in selected,
               **file_info(path), 'sequences': len(lengths), 'frames_25fps': sum(lengths),
               'hours_25fps': sum(lengths) / 25 / 3600,
               'windows_125_with_tail': sum(math.ceil(n / 125) for n in lengths),
               'partial_windows': sum(n % 125 != 0 for n in lengths),
               'minimum_sequence_frames': min(lengths), 'maximum_sequence_frames': max(lengths),
               'pose_dtype': str(data['pose'][0].dtype), 'acc_dtype': str(data['acc'][0].dtype),
               'ori_dtype': str(data['ori'][0].dtype)}
        rows.append(row)
        stem = path.stem
        intermediate = root / 'data/processed_imuposer' / (
            f'DIP_IMU/{stem[4:]}' if stem.startswith('dip_') else f'AMASS/{stem}')
        for key, filename in [('acc', 'accs.pt' if stem.startswith('dip_') else 'vacc.pt'),
                              ('ori', 'oris.pt' if stem.startswith('dip_') else 'vrot.pt'), ('pose', 'pose.pt')]:
            src = intermediate / filename
            values = torch.load(src, map_location='cpu', weights_only=True)
            if len(values) != len(lengths):
                raise ValueError('Intermediate sequence count mismatch')
            exact, max_diff = 0, 0.
            for i, value in enumerate(values):
                current = resample(value)
                if key == 'acc':
                    current = smooth(current)
                elif key == 'pose':
                    current = pose_math.axis_angle_to_rotation_matrix(current.contiguous()).view(-1, 24, 3, 3)
                target = data[key][i]
                if current.shape != target.shape:
                    raise ValueError('Reconstructed shape mismatch')
                equal = torch.equal(current, target)
                exact += equal
                if not equal:
                    max_diff = max(max_diff, float((current - target).abs().max()))
                del current
            reconstruction.append({'processed_file': path.name, 'field': key, 'intermediate_path': str(src),
                                   'intermediate_sha256': sha256(src), 'sequences_checked': len(values),
                                   'bitwise_equal_sequences': exact, 'maximum_absolute_difference': max_diff,
                                   'within_1e_6': max_diff <= 1e-6})
            del values
        print(f'Processed audit: {path.name}, {len(lengths)} sequences, {sum(lengths)} frames', flush=True)
        del data
        gc.collect()
    return rows, sequence_rows, reconstruction, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.original_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    run = root / 'checkpoints/IMUPoserGlobalModel_global-10062026-021332'
    wandb_dir = run / 'wandb/run-20261006_021333-wk8tfuyl'
    evidence_paths = [run / 'best_model.txt', *[wandb_dir / 'files' / n for n in [
        'config.yaml', 'wandb-metadata.json', 'wandb-summary.json', 'output.log', 'requirements.txt']],
        wandb_dir / 'run-wk8tfuyl.wandb', root / 'preprocess_full.log', root / 'preprocess_full_retry.log']
    evidence = {str(p): file_info(p) for p in evidence_paths}
    best = checkpoint_info(run / 'epoch=epoch=59-val_loss=validation_step_loss=0.01017.ckpt')
    last = checkpoint_info(run / 'last.ckpt')
    write_json(out / 'checkpoint_metadata.json', {'best': best, 'last': last})
    metadata = json.loads((wandb_dir / 'files/wandb-metadata.json').read_text())
    summary = json.loads((wandb_dir / 'files/wandb-summary.json').read_text())
    history, binary_records = run_history(wandb_dir / 'run-wk8tfuyl.wandb')
    write_csv(out / 'validation_history.csv', history)
    write_json(out / 'run_metadata.json', {
        'run_id': 'wk8tfuyl',
        'metadata_selected_fields': {k: metadata.get(k) for k in ['startedAt', 'program', 'codePath', 'args', 'python', 'os', 'git', 'gpu', 'gpu_count']},
        'final_summary_selected_fields': {k: summary.get(k) for k in ['epoch', 'trainer/global_step', 'validation_step_loss', 'train_loss', 'val_loss', '_runtime']},
        'validation_history_records': len(history), 'wandb_binary_records_read': binary_records,
        'minimum_logged_validation': min(history, key=lambda r: r['validation_step_loss']),
        'config_yaml_model_config_value': 'Config object repr rather than serialized hyperparameter fields',
        'privacy_note': 'W&B email, host, device UUID and unrelated telemetry are omitted.'})
    processed = root / 'data/processed_imuposer_25fps'
    data_rows, seq_rows, checks, order = inspect_processed(root, processed)
    write_csv(out / 'processed_data_manifest.csv', data_rows)
    write_csv(out / 'processed_sequence_manifest.csv', seq_rows)
    write_csv(out / 'preprocessing_reconstruction.csv', checks)
    raw_rows, raw_summary = inspect_raw(root / 'data/raw/AMASS')
    write_csv(out / 'raw_amass_inventory.csv', raw_rows)
    write_json(out / 'raw_amass_summary.json', raw_summary)
    local_sources = [root / n for n in [
        'src/imuposer/config.py', 'src/imuposer/datasets/globalModelDataset.py',
        'src/imuposer/datasets/globalModelDatasetFineTuneDIP.py', 'src/imuposer/datasets/utils.py',
        'src/imuposer/models/LSTMs/IMUPoser_Model.py', 'src/imuposer/models/LSTMs/IMUPoser_Model_FineTune.py',
        'src/imuposer/models/LSTMs/RNN.py', 'src/imuposer/math/angular.py',
        'src/imuposer/smpl/parametricModel.py', 'src/imuposer/utils.py',
        'scripts/1. Preprocessing/1. preprocess_all.py',
        'scripts/1. Preprocessing/2. preprocess_all_to_imuposer_at_25fps.py',
        'scripts/2. Train/1. Train Global Model.py', 'scripts/2. Train/run_combos.sh']]
    source_rows = []
    for path in local_sources:
        relative = str(path.relative_to(root))
        upstream = subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=root)
        original_hash = hashlib.sha256(upstream).hexdigest()
        info = file_info(path)
        source_rows.append({'relative_path': relative, 'current_sha256': info['sha256'],
                            'git_head_sha256': original_hash, 'matches_git_head': original_hash == info['sha256']})
        dest = out / 'source_snapshot/original' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    this = Path(__file__).resolve()
    shutil.copyfile(this, out / 'source_snapshot/audit_reproduction.py')
    write_csv(out / 'source_comparison.csv', source_rows)
    selected = [r for r in data_rows if r['selected_by_current_global_train_loader']]
    windows = sum(r['windows_125_with_tail'] for r in selected)
    sample_count = windows * len(amass_combos)
    train_count = int(sample_count * Config.train_pct)
    val_count = sample_count - train_count
    steps = math.ceil(train_count / Config.batch_size)
    train_frames = sum(r['frames_25fps'] for r in selected)
    summary = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
        'original_root': str(root), 'imported_package': str(Path(imuposer.__file__).resolve()),
        'original_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'current_software': {'python': platform.python_version(), **{k: importlib.metadata.version(k) for k in ['torch', 'numpy', 'pytorch-lightning', 'wandb', 'scipy']}},
        'evidence_source_files': evidence,
        'configured_amass_subsets': list(amass_datasets), 'processed_global_train_files_current_iteration_order': order,
        'missing_configured_processed_subsets': [s for s in amass_datasets if s + '.pt' not in order],
        'global_train_subset_count': len(selected), 'global_train_sequences': sum(r['sequences'] for r in selected),
        'global_train_frames_25fps': train_frames, 'global_train_hours': train_frames / 25 / 3600,
        'unmasked_motion_windows': windows, 'mask_count_current_source': len(amass_combos),
        'mask_definitions_current_source': amass_combos, 'one_two_three_five_location_mask_counts': {
            str(n): sum(len(v) == n for v in amass_combos.values()) for n in [1, 2, 3, 5]},
        'enumerated_samples_25_masks_before_split': sample_count,
        'current_rule_training_samples': train_count, 'current_rule_validation_samples': val_count,
        'current_rule_train_batches_per_epoch': steps, 'current_rule_val_batches': math.ceil(val_count / Config.batch_size),
        'epoch59_global_step': best['global_step'], 'last_global_step': last['global_step'],
        'epoch59_steps_match_current_inventory': steps * (best['epoch'] + 1) == best['global_step'],
        'last_steps_match_current_inventory': steps * (last['epoch'] + 1) == last['global_step'],
        'all_stage2_reconstructions_bitwise_equal': all(r['bitwise_equal_sequences'] == r['sequences_checked'] for r in checks),
        'all_stage2_reconstructions_within_1e_6': all(r['within_1e_6'] for r in checks),
        'reconstruction_sequence_field_checks': sum(r['sequences_checked'] for r in checks),
        'full_raw_archives_sha256_computed': False,
        'historical_limits': ['Current processed hashes were recorded now, not at training time.',
                              'Exact historical data ordering, split indices, RNG state and per-mask batch exposure were not recovered.',
                              'Current-source reconstruction and step-count agreement support consistency but do not prove historical data identity.',
                              'W&B git commit does not capture uncommitted source changes.',
                              'AMASS release/archive versions and original per-recording preprocessing identity mapping are not archived.'],
        'training_performed': False, 'model_inference_performed': False,
        'peak_process_rss_mib': peak_ram_mib()}
    write_json(out / 'audit_summary.json', summary)
    # Ensure source/checkpoint/run evidence remained unchanged during inspection.
    all_inputs = [*evidence.values(), best, last, *data_rows]
    for record in all_inputs:
        if sha256(record['path']) != record['sha256']:
            raise ValueError('An audited input changed during collection')
    write_json(out / 'collection_validation.json', {
        'status': 'passed', 'input_hashes_rechecked': len(all_inputs),
        'original_sources_rechecked': all(sha256(root / r['relative_path']) == r['current_sha256'] for r in source_rows),
        'checkpoint_history_best_matches': abs(min(r['validation_step_loss'] for r in history) - next(
            v['best_model_score'] for v in best['callbacks'].values() if 'best_model_score' in v)) < 1e-12,
        'data_manifest_files': len(data_rows), 'raw_candidate_files': len(raw_rows),
        'model_or_dataset_asset_copied': False})
    print(json.dumps({k: v for k, v in summary.items() if k not in [
        'evidence_source_files', 'mask_definitions_current_source', 'historical_limits']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
