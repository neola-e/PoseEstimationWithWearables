"""DIP fine-tuning data, inherited loss, and reproducible training state helpers.

Architecture and licensed assets stay in the installed original IMUPoser tree.
Subject splitting happens BEFORE mask expansion. No helper opens dip_test.pt.
"""
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from imuposer import math
from imuposer.config import amass_combos

from runtime import PosePredictor, load_torch, sha256

MASKS = {k: list(v) for k, v in amass_combos.items() if k != 'global'}
EXPECTED_INITIAL_SHA256 = '02b9211b57ca36ae3fb13dcacc4b008b49a31fd360ebea516b73d103bd2a5058'


def json_write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_masks(masks):
    counts = {n: sum(len(v) == n for v in masks.values()) for n in [1, 2, 3]}
    if (len(masks) != 24 or counts != {1: 5, 2: 10, 3: 9} or 'global' in masks
            or len({tuple(sorted(v)) for v in masks.values()}) != 24
            or any(len(set(v)) != len(v) or any(s not in range(5) for s in v) for v in masks.values())):
        raise ValueError('Expected exactly 24 distinct sparse masks (5 single, 10 pair, 9 triple)')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        if not torch.cuda.is_available():
            raise ValueError('Cannot restore CUDA RNG on a CPU-only device')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def sensor_mask(sensors):
    result = torch.zeros(60, dtype=torch.bool)
    result[:15].view(5, 3)[sensors] = True
    result[15:].view(5, 9)[sensors] = True
    return result


class DipFineTuneDataset(Dataset):
    """Share each motion window across masks; do not duplicate entire data 24x."""
    def __init__(self, data, records, role, masks=None):
        self.masks = MASKS if masks is None else masks
        validate_masks(self.masks)
        if role not in ['train', 'validation']:
            raise ValueError('Only train or validation roles are supported')
        selected = [r for r in records if r['role'] == role]
        expected = {f's_{i:02d}' for i in (range(1, 7) if role == 'train' else range(7, 9))}
        if not selected or any(r['subject'] not in expected for r in selected):
            raise ValueError('Subject-role mismatch or empty dataset')
        self.windows, self.features, self.targets = [], {}, {}
        self.mask_names = list(self.masks)
        self.mask_tensors = [sensor_mask(self.masks[name]) for name in self.mask_names]
        for row in selected:
            seq, n = row['sequence_id'], row['frames']
            if seq in self.features:
                raise ValueError('Duplicate sequence in manifest')
            acc, ori, pose = [data[k][seq] for k in ['acc', 'ori', 'pose']]
            if acc.shape != (n, 6, 3) or ori.shape != (n, 6, 3, 3) or pose.shape != (n, 24, 3, 3):
                raise ValueError('Unexpected DIP schema')
            x = torch.cat([(acc[:, :5]/30).flatten(1), ori[:, :5].float().flatten(1)], 1).float()
            target = math.rotation_matrix_to_r6d(pose).reshape(n, 144).float()
            if not torch.isfinite(x).all() or not torch.isfinite(target).all():
                raise ValueError('Non-finite DIP input or target')
            self.features[seq], self.targets[seq] = x, target
            for start in range(0, n, 125):
                self.windows.append({'sequence_id': seq, 'subject': row['subject'],
                                     'start_frame': start, 'frames': min(125, n-start)})

    def __len__(self):
        return len(self.windows)*len(self.masks)

    def metadata(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        window, mask = divmod(index, len(self.masks))
        return {**self.windows[window], 'mask': self.mask_names[mask], 'sample_index': index}

    def __getitem__(self, index):
        meta = self.metadata(index)
        seq, start, length = [meta[k] for k in ['sequence_id', 'start_frame', 'frames']]
        x = self.features[seq][start:start+length].clone()
        x[:, ~self.mask_tensors[index % len(self.masks)]] = 0
        return x, self.targets[seq][start:start+length], meta


def collate(batch):
    x, target, meta = zip(*batch)
    lengths = torch.tensor([len(v) for v in x], dtype=torch.long)
    return pad_sequence(x, batch_first=True), pad_sequence(target, batch_first=True), lengths, list(meta)


def combined_loss(model, body, batch):
    """Preserve inherited padded 6D MSE + padded FK-position MSE, weights 1:1.

    This explicit source-compatible choice includes padding in training loss.
    Reported pose metrics always exclude padding. Do not silently change recipe.
    """
    device = next(model.parameters()).device
    x, target, lengths, _ = batch
    x, target = x.to(device), target.to(device)
    prediction = model(x, lengths)
    pose_loss = (prediction-target).square().mean()
    pred_rot = math.r6d_to_rotation_matrix(prediction).reshape(-1, 216)
    target_rot = math.r6d_to_rotation_matrix(target).reshape(-1, 216)
    pred_joint = body.forward_kinematics(pred_rot, calc_mesh=False)[1]
    with torch.no_grad():
        target_joint = body.forward_kinematics(target_rot, calc_mesh=False)[1]
    joint_loss = (pred_joint-target_joint).square().mean()
    loss = pose_loss + joint_loss
    if not torch.isfinite(loss):
        raise ValueError('Non-finite training loss')
    return loss, {'pose_mse': pose_loss.detach().item(), 'joint_mse_m2': joint_loss.detach().item(),
                  'combined_loss': loss.detach().item(), 'valid_frames': int(lengths.sum()),
                  'padded_frames': int(len(lengths)*x.shape[1]-lengths.sum())}


def make_optimizer(model, settings):
    if settings['name'] != 'Adam':
        raise ValueError('Frozen recipe requires Adam')
    return torch.optim.Adam(model.parameters(), lr=settings['lr'],
                            betas=tuple(settings['betas']), eps=settings['eps'],
                            weight_decay=settings['weight_decay'], amsgrad=False, foreach=False)


def train_step(model, body, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before = {k: v.detach().clone() for k, v in model.named_parameters()}
    loss, values = combined_loss(model, body, batch)
    loss.backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f'Missing or non-finite gradient: {name}')
        gradients[name] = parameter.grad.detach().double().norm().item()
    if not any(v > 0 for v in gradients.values()):
        raise ValueError('All gradients are zero')
    optimizer.step()
    updates = {}
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f'Non-finite parameter after update: {name}')
        updates[name] = (parameter.detach()-before[name]).abs().max().item()
    if not any(v > 0 for v in updates.values()):
        raise ValueError('Optimizer did not change parameters')
    return {**values, 'gradient_norms': gradients, 'parameter_max_abs_updates': updates}


def save_training_checkpoint(path, predictor, optimizer, protocol_path, step, sampler_state):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    config = copy.deepcopy(predictor.training_config)
    config.device = predictor.device
    config.mkdir = False
    # Keep original network key names for strict PosePredictor compatibility.
    saved = {'format': 'imuposer-dip-training-v1', 'state_dict': predictor.model.state_dict(),
             'hyper_parameters': {'config': config}, 'epoch': -1, 'global_step': step,
             'optimizer_states': [optimizer.state_dict()], 'lr_schedulers': [],
             'rng_state': rng_state(), 'sampler_state': sampler_state,
             'training_protocol_sha256': sha256(protocol_path),
             'pretrained_epoch': predictor.epoch, 'pretrained_global_step': predictor.global_step,
             'kind': 'engineering_smoke_only', 'selection_eligible': False,
             'training_validation_metric': None}
    temporary = path.with_suffix('.tmp')
    torch.save(saved, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(path, optimizer_settings, protocol_path, device='cpu', threads=2):
    saved = load_torch(path, trusted=True)
    if saved.get('format') != 'imuposer-dip-training-v1':
        raise ValueError('Expected a DIP training checkpoint')
    if saved['training_protocol_sha256'] != sha256(protocol_path):
        raise ValueError('Checkpoint belongs to a different protocol')
    predictor = PosePredictor(path, device, threads)
    optimizer = make_optimizer(predictor.model, optimizer_settings)
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]))
    return predictor, optimizer, saved


def assert_tree_equal(a, b):
    """Strict continuation check for model, Adam moments/counters, and RNG."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.dtype != b.dtype or not torch.equal(a.cpu(), b.cpu()):
            raise AssertionError('Tensor state differs')
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise AssertionError('Array state differs')
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError('State dictionary keys differ')
        for key in a:
            assert_tree_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError('State sequence lengths differ')
        for x, y in zip(a, b):
            assert_tree_equal(x, y)
    elif a != b:
        raise AssertionError('Scalar state differs')
