"""Explicit IMUPoser pose metrics; all positions/translations are in meters.

Spatial errors align pelvis translation only. Jitter is predicted joint jerk
before spatial alignment, following TransPose's FullMotionEvaluator convention.
It is not a prediction-minus-reference error. Sequences are evaluated separately.
"""
import math

import numpy as np
import torch

from runtime import pose_rotations

ERROR_NAMES = ['mpjre_deg', 'mpjpe_cm', 'mpjve_cm', 'local_rotation_deg']
JITTER_NAMES = ['jitter_m_s3', 'reference_jitter_m_s3']


def finite_tensor(value, name, dtype=torch.float64):
    result = torch.as_tensor(value, device='cpu').detach().to(dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f'{name} must be finite')
    return result


def validate_rotations(value):
    r = finite_tensor(value, 'rotations')
    if r.ndim < 3 or r.shape[-2:] != (3, 3) or r.numel() == 0:
        raise ValueError('Expected nonempty rotation matrices')
    if (not torch.allclose(r.transpose(-1, -2) @ r, torch.eye(3, dtype=r.dtype).expand_as(r), atol=1e-4, rtol=0)
            or not torch.allclose(torch.linalg.det(r), torch.ones(r.shape[:-2], dtype=r.dtype), atol=1e-4, rtol=0)):
        raise ValueError('Expected proper SO(3) rotations, without scale or reflection')
    return r


def rotation_angles_deg(predicted, target):
    """SO(3) geodesic using atan2; stable at identical, small and pi rotations."""
    p, t = validate_rotations(predicted), validate_rotations(target)
    if p.shape != t.shape:
        raise ValueError('Rotation shapes must match')
    rel = p.transpose(-1, -2) @ t
    skew = torch.stack([rel[..., 2, 1] - rel[..., 1, 2], rel[..., 0, 2] - rel[..., 2, 0],
                        rel[..., 1, 0] - rel[..., 0, 1]], dim=-1)
    sine = torch.linalg.vector_norm(skew, dim=-1) / 2
    cosine = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    return torch.rad2deg(torch.atan2(sine, cosine))


def aligned_position_errors(predicted_joints_m, target_joints_m, predicted_vertices_m, target_vertices_m):
    """Per-frame/per-point distances in cm after each pose's pelvis subtraction."""
    p, t, v, w = [finite_tensor(x, 'positions') for x in
                   [predicted_joints_m, target_joints_m, predicted_vertices_m, target_vertices_m]]
    if (p.ndim != 3 or p.shape != t.shape or p.shape[-1] != 3 or p.shape[0] < 1 or p.shape[1] < 1
            or v.ndim != 3 or v.shape != w.shape or v.shape[0] != p.shape[0] or v.shape[-1] != 3 or v.shape[1] < 1):
        raise ValueError('Expected paired [T,J,3] joints and [T,V,3] vertices')
    return (torch.linalg.vector_norm((p - p[:, :1]) - (t - t[:, :1]), dim=-1) * 100,
            torch.linalg.vector_norm((v - p[:, :1]) - (w - t[:, :1]), dim=-1) * 100)


def jerk_norms(positions_m, fps):
    """Forward third difference on four consecutive frames, in m/s^3."""
    p = finite_tensor(positions_m, 'joint trajectory')
    if p.ndim != 3 or p.shape[-1] != 3 or p.shape[1] < 1 or p.shape[0] < 1:
        raise ValueError('Expected nonempty [T,J,3] trajectory')
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')
    if len(p) < 4:
        return torch.empty((0, p.shape[1]), dtype=torch.float64)
    return torch.linalg.vector_norm((p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3]) * fps ** 3, dim=-1)


def local_rotations(pose):
    p = finite_tensor(pose, 'pose', torch.float32)
    if p.ndim == 4 and p.shape[1:] == (24, 3, 3):
        validate_rotations(p)
        return p
    if p.ndim == 3 and p.shape[1:] == (24, 6):
        p = p.reshape(len(p), 144)
    if p.ndim == 2 and p.shape[1] == 144:
        return pose_rotations(p)
    raise ValueError('Expected [T,144], [T,24,6], or [T,24,3,3] local pose')


class PoseMetricEvaluator:
    """Bounded mesh FK, full-sequence jerk, shared fixed shape for each pair."""
    def __init__(self, body, fps=25., fk_chunk=32):
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError('fps must be positive and finite')
        if isinstance(fk_chunk, bool) or not isinstance(fk_chunk, int) or fk_chunk < 1:
            raise ValueError('fk_chunk must be a positive integer')
        if len(body.parent) != 24 or len(body._v_template) != 6890:
            raise ValueError('Expected the 24-joint, 6890-vertex SMPL body model')
        self.body, self.fps, self.fk_chunk = body, float(fps), fk_chunk

    @torch.inference_mode()
    def evaluate_sequence(self, predicted, target, sequence_id, shape=None,
                          predicted_translation_m=None, target_translation_m=None):
        p, t = local_rotations(predicted), local_rotations(target)
        if p.shape != t.shape or len(p) == 0:
            raise ValueError('Expected paired nonempty sequences of the same length')
        n = len(p)
        beta = None if shape is None else finite_tensor(shape, 'shared shape', torch.float32)
        if beta is not None:
            if beta.shape not in [(10,), (1, 10)]:
                raise ValueError('Shared shape must contain 10 fixed betas, without per-frame changes')
            beta = beta.reshape(1, 10)
        translations = []
        for value in [predicted_translation_m, target_translation_m]:
            value = torch.zeros(n, 3) if value is None else finite_tensor(value, 'translation_m', torch.float32)
            if value.shape != (n, 3):
                raise ValueError('Expected one [T,3] translation in meters for each trajectory')
            translations.append(value)
        per_frame = {name: [] for name in ERROR_NAMES}
        per_joint = {name: torch.zeros(24, dtype=torch.float64) for name in ['mpjre_deg', 'mpjpe_cm', 'local_rotation_deg']}
        positions_p, positions_t = [], []
        for start in range(0, n, self.fk_chunk):
            stop = min(start + self.fk_chunk, n)
            gp, jp, vp = self.body.forward_kinematics(p[start:stop], beta, translations[0][start:stop], calc_mesh=True)
            gt, jt, vt = self.body.forward_kinematics(t[start:stop], beta, translations[1][start:stop], calc_mesh=True)
            je, ve = aligned_position_errors(jp, jt, vp, vt)
            angles = rotation_angles_deg(gp, gt)
            local = rotation_angles_deg(p[start:stop], t[start:stop])
            for key, values in [('mpjre_deg', angles), ('mpjpe_cm', je), ('local_rotation_deg', local)]:
                per_joint[key] += values.sum(0)
                per_frame[key].append(values.mean(1))
            per_frame['mpjve_cm'].append(ve.mean(1))
            positions_p.append(jp.double())
            positions_t.append(jt.double())
        jitter = jerk_norms(torch.cat(positions_p), self.fps)
        reference_jitter = jerk_norms(torch.cat(positions_t), self.fps)
        result = {'sequence_id': str(sequence_id), 'frames': n, 'jerk_samples': max(n - 3, 0),
                  'metrics': {k: torch.cat(v).mean().item() for k, v in per_frame.items()},
                  'per_joint': {k: (v / n).tolist() for k, v in per_joint.items()},
                  'shape_betas': None if beta is None else beta.flatten().tolist(),
                  'settings': {'fps': self.fps, 'joints': list(range(24)), 'vertices': 6890,
                               'root_alignment': 'pelvis translation only for spatial errors',
                               'jitter_alignment': 'unaligned joint positions, including supplied translation',
                               'pose_blendshape': bool(self.body.use_pose_blendshape)}}
        for key, values in [('jitter_m_s3', jitter), ('reference_jitter_m_s3', reference_jitter)]:
            result['metrics'][key] = values.mean().item() if len(values) else None
            result['metrics'][key.replace('_m_s3', '_100_m_s3')] = result['metrics'][key] / 100 if len(values) else None
            result['per_joint'][key] = values.mean(0).tolist() if len(values) else None
        return result


def aggregate_sequences(results):
    if not results or len({r['sequence_id'] for r in results}) != len(results):
        raise ValueError('Need nonempty results with unique sequence IDs')
    if any(r['settings'] != results[0]['settings'] for r in results):
        raise ValueError('Cannot aggregate different metric settings')
    output = {}
    for mode in ['frame_weighted', 'sequence_equal']:
        output[mode] = {}
        for name in ERROR_NAMES + JITTER_NAMES:
            count = 'jerk_samples' if name in JITTER_NAMES else 'frames'
            valid = [r for r in results if r[count] > 0]
            if not valid:
                value = None
            else:
                weights = [r[count] if mode == 'frame_weighted' else 1 for r in valid]
                value = float(np.average([r['metrics'][name] for r in valid], weights=weights))
            output[mode][name] = value
            if name in JITTER_NAMES:
                output[mode][name.replace('_m_s3', '_100_m_s3')] = None if value is None else value / 100
    return {'sequences': len(results), 'frames': sum(r['frames'] for r in results),
            'jerk_samples': sum(r['jerk_samples'] for r in results), 'metrics': output,
            'settings': results[0]['settings']}


def cache_sequences(cache, pose_key, target_key='target_pose'):
    """Reassemble consecutive cached windows; never differentiate across recordings.

    offsets alone denote independent sequences. With sequence_ids/start_frames,
    windows must cover each recording exactly once from frame zero, without gaps.
    """
    p, t = cache[pose_key], cache[target_key]
    if p.shape != t.shape or p.ndim not in [2, 3, 4] or len(p) < 1:
        raise ValueError('Prediction and reference arrays must have matching nonempty shapes')
    if 'offsets' not in cache:
        if 'sequence_ids' in cache or 'start_frames' in cache:
            raise ValueError('Window identity arrays require offsets')
        valid_frames = len(p)
        if 'lengths' in cache:
            lengths = np.asarray(cache['lengths'])
            if (lengths.ndim > 1 or lengths.size != 1 or lengths.dtype.kind not in 'iu'
                    or not 1 <= int(lengths.reshape(-1)[0]) <= len(p)):
                raise ValueError('Single-sequence lengths must contain one valid positive integer')
            valid_frames = int(lengths.reshape(-1)[0])
        chunks = [('0', [(0, valid_frames)])]
    else:
        if 'lengths' in cache:
            raise ValueError('Use offsets or lengths, not both')
        offsets = cache['offsets']
        if (offsets.ndim != 1 or offsets.dtype.kind not in 'iu' or len(offsets) < 2
                or offsets[0] != 0 or offsets[-1] != len(p) or np.any(offsets[1:] <= offsets[:-1])):
            raise ValueError('Invalid cache offsets')
        intervals = [(int(a), int(b)) for a, b in zip(offsets[:-1], offsets[1:])]
        present = ['sequence_ids' in cache, 'start_frames' in cache]
        if any(present) and not all(present):
            raise ValueError('Both sequence_ids and start_frames are required for reassembly')
        if not all(present):
            chunks = [(str(i), [sl]) for i, sl in enumerate(intervals)]
        else:
            ids, starts = cache['sequence_ids'], cache['start_frames']
            if (ids.shape != (len(intervals),) or starts.shape != ids.shape
                    or ids.dtype.kind not in 'iu' or starts.dtype.kind not in 'iu' or np.any(starts < 0)):
                raise ValueError('Invalid cache sequence/window identities')
            chunks = []
            for seq in sorted(set(ids.tolist())):
                chosen = sorted((int(starts[i]), intervals[i]) for i in range(len(intervals)) if ids[i] == seq)
                cursor = 0
                spans = []
                for start, sl in chosen:
                    if start != cursor:
                        raise ValueError('Gap, overlap or duplicate window in cached recording')
                    spans.append(sl)
                    cursor += sl[1] - sl[0]
                chunks.append((str(seq), spans))
    output = []
    for identity, spans in chunks:
        result = {'sequence_id': identity, 'predicted': np.concatenate([p[a:b] for a, b in spans]),
                  'target': np.concatenate([t[a:b] for a, b in spans]), 'source_windows': len(spans)}
        for key in ['predicted_translation_m', 'target_translation_m']:
            if key in cache:
                value = cache[key]
                if value.shape != (len(p), 3):
                    raise ValueError(f'Invalid {key} shape')
                result[key] = np.concatenate([value[a:b] for a, b in spans])
        output.append(result)
    return output
