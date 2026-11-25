"""Pairing, coverage, aggregation and sequence-cluster uncertainty for sensor studies."""
import numpy as np

JOINT_NAMES = [
    'Pelvis', 'L Hip', 'R Hip', 'Spine1', 'L Knee', 'R Knee', 'Spine2',
    'L Ankle', 'R Ankle', 'Spine3', 'L Foot', 'R Foot', 'Neck', 'L Collar',
    'R Collar', 'Head', 'L Shoulder', 'R Shoulder', 'L Elbow', 'R Elbow',
    'L Wrist', 'R Wrist', 'L Hand', 'R Hand',
]
# The first four groups partition all 24 joints. Additional groups overlap.
REGIONS = {
    'arms': [16, 17, 18, 19, 20, 21, 22, 23],
    'legs': [1, 2, 4, 5, 7, 8, 10, 11],
    'torso': [0, 3, 6, 9, 13, 14],
    'head': [12, 15],
    'upper_body': [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    'left_arm': [16, 18, 20, 22],
    'right_arm': [17, 19, 21, 23],
}
METRICS = ['position_cm', 'local_deg', 'global_deg']
COMBOS = {'global': [0, 1, 2, 3, 4], 'lw_rp_h': [0, 3, 4]}


def sensor_mask(sensors):
    sensors = list(sensors)
    if not sensors or len(set(sensors)) != len(sensors) or any(s not in range(5) for s in sensors):
        raise ValueError('Expected distinct sensor indices in [0,4]')
    mask = np.zeros(60, dtype=bool)
    mask[:15].reshape(5, 3)[sensors] = True
    mask[15:].reshape(5, 9)[sensors] = True
    return mask


def mask_input(imu, sensors):
    """Copy a [T,60] array; preserve present channels bitwise; zero both missing blocks."""
    x = np.asarray(imu)
    if x.ndim != 2 or x.shape[1] != 60 or not np.isfinite(x).all():
        raise ValueError('Expected finite [T,60] input')
    result = x.copy()
    result[:, ~sensor_mask(sensors)] = 0
    return result


def window_key(row):
    return (int(row['sequence_id']), int(row['start_frame']), int(row['frames']))


def pair_windows(left, right):
    """Require exact key coverage, including lengths; align independently of row order."""
    def index(rows):
        result = {}
        for row in rows:
            key = window_key(row)
            if key in result:
                raise ValueError(f'Duplicate window: {key}')
            if key[1] < 0 or key[2] < 1:
                raise ValueError(f'Invalid window: {key}')
            result[key] = row
        return result
    a, b = index(left), index(right)
    if not a or a.keys() != b.keys():
        raise ValueError(f'Window mismatch: only left={a.keys() - b.keys()}, only right={b.keys() - a.keys()}')
    return [(a[k], b[k]) for k in sorted(a)]


def validate_coverage(rows, sequence_lengths, window=125):
    """Require every frame exactly once, including partial tails and all sequences."""
    ids = set(range(len(sequence_lengths)))
    if {r['sequence_id'] for r in rows} != ids:
        raise ValueError('Sequence coverage mismatch')
    for seq, total in enumerate(sequence_lengths):
        selected = sorted((r for r in rows if r['sequence_id'] == seq), key=lambda r: r['start_frame'])
        cursor = 0
        for row in selected:
            if row['start_frame'] != cursor or row['frames'] != min(window, total - cursor):
                raise ValueError(f'Gap, overlap, duplicate or wrong window length in sequence {seq}')
            if row['frames'] < 1:
                raise ValueError('Empty window')
            cursor += row['frames']
        if cursor != total:
            raise ValueError(f'Truncated sequence {seq}: {cursor} != {total}')
    return {'sequences': len(sequence_lengths), 'windows': len(rows), 'frames': sum(sequence_lengths),
            'partial_windows': sum(r['frames'] < window for r in rows)}


def weighted_mean(values, frames):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(frames, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] != len(weights) or not len(weights):
        raise ValueError('Values/weights mismatch')
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Non-finite values or invalid frame counts')
    return np.average(values, axis=0, weights=weights)


def direction_counts(delta, tolerance=1e-5):
    x = np.asarray(delta)
    return {'improved': int((x < -tolerance).sum()), 'worsened': int((x > tolerance).sum()),
            'tied': int((np.abs(x) <= tolerance).sum()), 'total': len(x), 'tolerance': tolerance,
            'improved_proportion': float((x < -tolerance).mean()),
            'worsened_proportion': float((x > tolerance).mean())}


def cluster_bootstrap(sequence_delta, sequence_frames, draws=10000, seed=20261027):
    """Resample entire paired sequences; ratio of weighted sums for frame estimand."""
    d = np.asarray(sequence_delta, dtype=np.float64)
    w = np.asarray(sequence_frames, dtype=np.float64)
    if len(d) < 2 or d.shape != w.shape or np.any(w <= 0) or not np.isfinite(d).all():
        raise ValueError('Need at least two finite paired sequence effects')
    indices = np.random.default_rng(seed).integers(0, len(d), (draws, len(d)))
    frame = (d[indices] * w[indices]).sum(1) / w[indices].sum(1)
    equal = d[indices].mean(1)
    return {'draws': draws, 'seed': seed, 'unit': 'paired sequence',
            'frame_weighted_delta_ci95': np.quantile(frame, [.025, .975]).tolist(),
            'sequence_equal_delta_ci95': np.quantile(equal, [.025, .975]).tolist(),
            'caution': 'Descriptive percentile intervals; sequences may share subjects. Only 19 sequences, not independent frames or subjects.'}


def rotation_angle_deg(left, right):
    """Torch SO(3) geodesic angles, matching eval_dip's trace/acos convention."""
    import torch
    rel = left.transpose(-1, -2) @ right
    cosine = (rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(cosine.clamp(-1, 1).acos())


def joint_errors(rotation, gt_rotation, body, gt_global=None, gt_position=None):
    import torch
    if gt_global is None or gt_position is None:
        gt_global, gt_position = body.forward_kinematics(gt_rotation, calc_mesh=False)
    global_rotation, position = body.forward_kinematics(rotation, calc_mesh=False)
    result = {
        'position_cm': torch.linalg.vector_norm(position - gt_position, dim=-1) * 100,
        'local_deg': rotation_angle_deg(rotation, gt_rotation),
        'global_deg': rotation_angle_deg(global_rotation, gt_global),
    }
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise ValueError('Non-finite joint errors')
    return result
