"""Run three real DIP optimizer steps and verify checkpoint continuation.

This command does not execute the 30-epoch development training protocol.
Its checkpoint is explicitly ineligible for scientific model selection.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from imuposer.smpl.parametricModel import ParametricModel

from dip_finetuning import (DipFineTuneDataset, assert_tree_equal, collate, combined_loss,
                           json_write, load_training_checkpoint, make_optimizer,
                           restore_rng, rng_state, save_training_checkpoint, seed_all, train_step)
from eval_wrist_pocket_dev import fingerprint
from runtime import PosePredictor, load_torch, peak_ram_mib, sha256


def smoke(args):
    root = Path(args.protocol_dir)
    protocol_path = root/'PROTOCOL.json'
    p = json.loads(protocol_path.read_text())
    if p['protocol_id'] != 'DIP_FINETUNING_V1' or p['smoke']['steps'] != 3:
        raise ValueError('Unexpected frozen protocol')
    for name, digest in json.loads((root/'FREEZE_MANIFEST.json').read_text()).items():
        if sha256(root/name) != digest:
            raise ValueError(f'Frozen artifact changed: {name}')
    source_hashes = json.loads((root/'SOURCE_HASHES.json').read_text())
    for name, digest in source_hashes.items():
        if sha256(name) != digest:
            raise ValueError(f'Frozen source changed: {name}; use a new protocol version/run')
    if sha256(root/'data_manifest.json') != p['data_manifest_sha256']:
        raise ValueError('Data manifest changed')
    manifest = json.loads((root/'data_manifest.json').read_text())
    data_path = Path(manifest['data_path'])
    if data_path.name != 'dip_train.pt' or sha256(data_path) != manifest['data_sha256']:
        raise ValueError('Training container changed or points at test data')
    if sha256(p['checkpoint']['path']) != p['checkpoint']['sha256']:
        raise ValueError('Initialization changed')
    if sha256(p['smpl_model']['path']) != p['smpl_model']['sha256']:
        raise ValueError('SMPL asset changed')
    out = root/'smoke'
    out.mkdir(exist_ok=False)
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA explicitly requested but unavailable')
    started = time.perf_counter()
    predictor = PosePredictor(p['checkpoint']['path'], device, args.threads)
    body = ParametricModel(p['smpl_model']['path'], device=predictor.device)
    seed_all(p['seed'])
    data = load_torch(data_path)
    for row in manifest['records']:
        for field, digest in row['processed_sha256'].items():
            if fingerprint(data[field][row['sequence_id']]) != digest:
                raise ValueError('Sequence content no longer matches subject manifest')
    train = DipFineTuneDataset(data, manifest['records'], 'train', p['masks'])
    validation = DipFineTuneDataset(data, manifest['records'], 'validation', p['masks'])
    del data
    train_subjects = {r['subject'] for r in train.windows}
    val_subjects = {r['subject'] for r in validation.windows}
    train_sequences = {r['sequence_id'] for r in train.windows}
    val_sequences = {r['sequence_id'] for r in validation.windows}
    if train_subjects & val_subjects or train_sequences & val_sequences:
        raise ValueError('Training/development leakage')
    if train_subjects != set(p['development']['train_subjects']) or val_subjects != set(p['development']['validation_subjects']):
        raise ValueError('Subject sets differ from frozen protocol')
    generator = torch.Generator().manual_seed(p['seed'])
    chosen_windows = torch.randperm(len(train.windows), generator=generator)[:24].tolist()
    # Exercise padded training loss as well as full windows. Keep windows unique.
    partial = next(i for i, row in enumerate(train.windows) if row['frames'] < 125)
    if partial not in chosen_windows:
        chosen_windows[0] = partial
    sample_indices = [w*24+mask for mask, w in enumerate(chosen_windows)]
    loader_generator = torch.Generator().manual_seed(p['seed'])
    loader = DataLoader(Subset(train, sample_indices), batch_size=p['smoke']['batch_size'],
                        shuffle=False, num_workers=0, collate_fn=collate, generator=loader_generator)
    batches = list(loader)
    val_loader = DataLoader(Subset(validation, [0, len(validation)-1]), batch_size=2,
                            shuffle=False, num_workers=0, collate_fn=collate,
                            generator=torch.Generator().manual_seed(p['seed']+1))
    val_batch = next(iter(val_loader))
    if len(batches) != 3 or len({m['mask'] for b in batches for m in b[3]}) != 24:
        raise ValueError('Smoke must cover all 24 masks in three batches')
    details = {'started_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
               'protocol_sha256': sha256(protocol_path), 'device': str(predictor.device),
               'cuda_available': torch.cuda.is_available(), 'python': platform.python_version(),
               'torch': torch.__version__, 'threads': args.threads, 'seed': p['seed'],
               'optimizer_initial_state_entries': 0,
               'formal_training_launched': False, 'selection_eligible': False,
               'overrides_from_development_training': {'batch_size': 8, 'steps': 3, 'epochs_completed': 0},
               'data': {'train_windows': len(train.windows), 'train_masked_samples': len(train),
                        'validation_windows': len(validation.windows), 'validation_masked_samples': len(validation),
                        'train_subjects': sorted(train_subjects), 'validation_subjects': sorted(val_subjects),
                        'train_sequence_ids': sorted(train_sequences), 'validation_sequence_ids': sorted(val_sequences),
                        'test_data_opened': False},
               'sample_plan': [train.metadata(i) for i in sample_indices],
               'validation_loader_examples': val_batch[3],
               'validation_loader_lengths': val_batch[2].tolist()}
    json_write(out/'RUN.json', details)
    optimizer = make_optimizer(predictor.model, p['optimizer'])
    if optimizer.state:
        raise ValueError('Expected fresh Adam without AMASS moments')
    probe = batches[0]
    before = predictor.predict(probe[0], probe[2])
    initial_parameters = {name: value.detach().clone() for name, value in predictor.model.named_parameters()}
    rows = []
    for step, batch in enumerate(batches, 1):
        tick = time.perf_counter()
        row = train_step(predictor.model, body, optimizer, batch)
        row.update(step=step, seconds=time.perf_counter()-tick, samples=batch[3])
        rows.append(row)
        print(f'Smoke optimizer step {step}/3: loss={row["combined_loss"]:.8f}, '
              f'valid={row["valid_frames"]}, padding={row["padded_frames"]}, seconds={row["seconds"]:.2f}', flush=True)
    predictor.model.eval()
    after = predictor.predict(probe[0], probe[2])
    parameter_changes = {name: (value.detach()-initial_parameters[name]).abs().max().item()
                         for name, value in predictor.model.named_parameters()}
    if not any(parameter_changes.values()) or torch.equal(before, after):
        raise ValueError('Training produced no change in model outputs/parameters')
    with torch.no_grad():
        _, probe_loss = combined_loss(predictor.model, body, probe)
    checkpoint = out/'smoke_step3.ckpt'
    sampler_state = {'sample_indices': sample_indices, 'next_batch': 3, 'batch_size': 8,
                     'loader_generator_state': loader_generator.get_state(),
                     'resume_probe_batch': 0, 'resume_probe_purpose': 'Controlled continuation check; not another selected training epoch'}
    save_training_checkpoint(checkpoint, predictor, optimizer, protocol_path, 3, sampler_state)
    saved = load_torch(checkpoint, trusted=True)
    if saved['global_step'] != 3 or saved['selection_eligible']:
        raise ValueError('Checkpoint step/kind metadata is wrong')
    reloaded, reloaded_optimizer, reloaded_state = load_training_checkpoint(
        checkpoint, p['optimizer'], protocol_path, device, args.threads)
    assert_tree_equal(predictor.model.state_dict(), reloaded.model.state_dict())
    assert_tree_equal(optimizer.state_dict(), reloaded_optimizer.state_dict())
    assert_tree_equal(sampler_state, reloaded_state['sampler_state'])
    prediction_reload = reloaded.predict(probe[0], probe[2])
    if not torch.equal(after, prediction_reload):
        raise ValueError('Prediction changed after strict checkpoint reload')
    # Replay the SAME explicitly recorded next input from the SAME RNG. This
    # catches missing Adam moments, optimizer counters, and dropout RNG state.
    restore_rng(saved['rng_state'])
    uninterrupted = train_step(predictor.model, body, optimizer, probe)
    uninterrupted_rng = rng_state()
    expected_weights = copy.deepcopy(predictor.model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    restore_rng(reloaded_state['rng_state'])
    resumed = train_step(reloaded.model, body, reloaded_optimizer, probe)
    assert_tree_equal(expected_weights, reloaded.model.state_dict())
    assert_tree_equal(expected_optimizer, reloaded_optimizer.state_dict())
    assert_tree_equal(uninterrupted_rng, rng_state())
    if uninterrupted['combined_loss'] != resumed['combined_loss']:
        raise ValueError('Continuation loss differs')
    json_write(out/'steps.json', rows)
    np.savez_compressed(out/'probe_predictions.npz', pose_before=before.numpy(), pose_after=after.numpy(),
                        pose_reloaded=prediction_reload.numpy(), target_pose=probe[1].numpy(), lengths=probe[2].numpy())
    results = {'passed': True, 'protocol_sha256': sha256(protocol_path), 'checkpoint_sha256': sha256(checkpoint),
               'checkpoint': str(checkpoint.resolve()), 'checkpoint_kind': saved['kind'],
               'epoch': -1, 'optimizer_steps_in_saved_checkpoint': 3, 'epochs_completed': 0,
               'mask_coverage': sorted({m['mask'] for b in batches for m in b[3]}),
               'validation_data_used_for_gradients': False, 'test_data_opened': False,
               'gradient_checks': 'All trainable tensors have finite gradients; at least one nonzero per step',
               'parameter_tensors': len(parameter_changes),
               'parameter_tensors_changed': sum(v > 0 for v in parameter_changes.values()),
               'per_tensor_max_abs_parameter_change': parameter_changes,
               'max_abs_probe_prediction_change': (after-before).abs().max().item(),
               'probe_loss_after_3_steps': probe_loss,
               'reload_model_and_optimizer_bitwise_equal': True,
               'reload_predictions_max_abs_difference': (after-prediction_reload).abs().max().item(),
               'continuation_model_optimizer_and_rng_bitwise_equal': True,
               'continuation_combined_loss': resumed['combined_loss'],
               'additional_probe_updates': 'One step in uninterrupted branch and one in reloaded branch; step3 checkpoint is retained unchanged',
               'formal_training_launched': False, 'selection_eligible': False,
               'elapsed_seconds': time.perf_counter()-started, 'peak_ram_mib': peak_ram_mib(),
               'interpretation': 'Engineering training chain passed; this is not validation/test accuracy or reproduction success'}
    for name, digest in source_hashes.items():
        if sha256(name) != digest:
            raise ValueError('Source changed during smoke')
    if sha256(data_path) != manifest['data_sha256'] or sha256(p['checkpoint']['path']) != p['checkpoint']['sha256']:
        raise ValueError('Input assets changed during smoke')
    json_write(out/'results.json', results)
    json_write(out/'ARTIFACT_MANIFEST.json', {str(f.relative_to(out)):sha256(f)
                                            for f in sorted(out.rglob('*')) if f.is_file()})
    print(json.dumps(results), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol-dir', required=True)
    p.add_argument('--device', default='auto', choices=['auto','cpu','cuda:0'])
    p.add_argument('--threads', type=int, default=2)
    smoke(p.parse_args())


if __name__ == '__main__':
    main()
