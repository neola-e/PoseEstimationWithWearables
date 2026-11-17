"""Frame-independent orientation IK; inputs are predictions and available IMUs only.

Reconstruction of IMUPoser section 4.4, with explicitly chosen missing details.
No ground truth, body positions, training data, or network weights are accessed.
SMPL sensor segments follow the public synthesis ji_mask, not the stale
config.leaf_joints list. All selected constraints are optimized simultaneously.
"""
from dataclasses import dataclass
import math as scalar_math

import torch
from imuposer import math

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)
SENSOR_NAMES = ('lw', 'rw', 'lp', 'rp', 'h')
SENSOR_JOINTS = (18, 19, 1, 2, 15)
# Sensed segment plus its immediate parent: wrist sensor -> shoulder/elbow;
# pocket -> pelvis/hip; head -> neck/head. Head/hip selection is a reconstruction
# convention, because IMUPoser does not publish those exact index lists.
OPTIMIZED_BY_SENSOR = ((16, 18), (17, 19), (0, 1), (0, 2), (12, 15))


@dataclass(frozen=True)
class IKSettings:
    iterations: int = 10
    learning_rate: float = 1e-3
    frame_batch: int = 256
    device: str = 'cpu'

    def __post_init__(self):
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int) or self.iterations < 0:
            raise ValueError('iterations must be a nonnegative integer')
        if isinstance(self.frame_batch, bool) or not isinstance(self.frame_batch, int) or self.frame_batch < 1:
            raise ValueError('frame_batch must be a positive integer')
        if isinstance(self.learning_rate, bool) or not scalar_math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')
        device = torch.device(self.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Expected cpu or cuda[:index]')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')


def sensor_ids(sensors):
    selected = tuple(sensors)
    if (len(set(selected)) != len(selected)
            or any(isinstance(s, bool) or not isinstance(s, int) or s not in range(5) for s in selected)):
        raise ValueError('Expected distinct integer sensor IDs in [0,4]')
    return tuple(sorted(selected))


def exp_map(vector):
    """SO(3) exponential with finite derivatives at zero, using analytic sinc."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(vector, dim=-1)[..., None, None]
    return (torch.eye(3, device=vector.device, dtype=vector.dtype)
            + torch.sinc(theta / scalar_math.pi) * skew
            + .5 * torch.sinc(theta / (2 * scalar_math.pi)).square() * (skew @ skew))


def global_rotations(local):
    rotations = [local[:, 0]]
    for joint in range(1, 24):
        rotations.append(rotations[PARENTS[joint]] @ local[:, joint])
    return torch.stack(rotations, dim=1)


def project_orientations(value):
    """Nearest proper rotation for positive, full-rank interpolated matrices.

    This projection is applied to IK targets only. The processed data and the
    network input remain untouched. Reject missing/invalid active measurements.
    """
    if not torch.isfinite(value).all():
        raise ValueError('Available IMU orientations must be finite')
    u, singular, vh = torch.linalg.svd(value)
    if (singular[..., -1] < 1e-6).any() or (torch.linalg.det(value) <= 0).any():
        raise ValueError('Available orientation is singular or has nonpositive determinant')
    return u @ vh


def angular_distance_deg(left, right):
    relative = left.transpose(-1, -2) @ right
    skew = torch.stack((relative[..., 2, 1]-relative[..., 1, 2],
                        relative[..., 0, 2]-relative[..., 2, 0],
                        relative[..., 1, 0]-relative[..., 0, 1]), -1)
    sine = torch.linalg.vector_norm(skew, dim=-1)/2
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)
    return torch.rad2deg(torch.atan2(sine, cosine))


def _rotations(pose):
    rotation = math.r6d_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
    identity = torch.eye(3, device=pose.device, dtype=pose.dtype)
    if (not torch.isfinite(rotation).all()
            or not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity.expand_as(rotation), atol=1e-4, rtol=0)
            or not torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=1e-4, rtol=0)):
        raise ValueError('Invalid or degenerate predicted pose')
    return rotation


@torch.inference_mode(False)
def refine_pose(predicted_pose, imu_orientations, sensors, settings=IKSettings()):
    """Return refined float32 [T,144], frame diagnostics, and run settings.

    Optimized variables are local right-multiplicative rotation-vector increments
    around each frame's prediction. Adam moments reset per frame (elementwise
    batched computation). A sum of per-frame MSEs preserves independent-frame
    gradients including Adam's epsilon across batch sizes. Return the actual
    final update, not a pre-update pose or a GT-selected iterate.
    """
    selected = sensor_ids(sensors)
    device = torch.device(settings.device)
    pose = torch.as_tensor(predicted_pose, dtype=torch.float32, device=device).detach().clone()
    if pose.ndim != 2 or pose.shape[1] != 144 or len(pose) == 0 or not torch.isfinite(pose).all():
        raise ValueError('Expected nonempty finite [T,144] predicted pose')
    base = _rotations(pose.double()).detach()
    orientations = torch.as_tensor(imu_orientations, dtype=torch.float64, device=device).detach()
    if orientations.shape != (len(pose), 5, 3, 3):
        raise ValueError('Expected [T,5,3,3] orientations')
    joints = tuple(sorted({j for s in selected for j in OPTIMIZED_BY_SENSOR[s]}))
    sensed = tuple(SENSOR_JOINTS[s] for s in selected)
    output = pose.clone().reshape(-1, 24, 6)
    if not selected:
        return {'pose': pose.cpu(), 'loss_history': torch.zeros(len(pose), settings.iterations+1, dtype=torch.float64),
                'orientation_angles_deg': torch.empty(len(pose), 0, 2, dtype=torch.float64),
                'summary': {'frames': len(pose), 'sensors': [], 'optimized_joints': [], 'iterations': 0}}
    # Unavailable channels are not read by projection or loss, even if NaN.
    active = orientations[:, selected].clone()
    target = project_orientations(active)
    projection = torch.linalg.matrix_norm(target-active, ord='fro')
    history, angles = [], []
    for start in range(0, len(pose), settings.frame_batch):
        stop = min(start+settings.frame_batch, len(pose))
        initial, goal = base[start:stop], target[start:stop]
        delta = torch.zeros(stop-start, len(joints), 3, dtype=torch.float64, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=settings.learning_rate, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0, foreach=False, fused=False)

        def compose():
            local = initial.clone()
            local[:, joints] = initial[:, joints] @ exp_map(delta)
            return local

        trajectory = []
        with torch.enable_grad():
            for _ in range(settings.iterations):
                local = compose()
                error = (global_rotations(local)[:, sensed]-goal).square().mean((1, 2, 3))
                trajectory.append(error.detach())
                optimizer.zero_grad(set_to_none=True)
                error.sum().backward()
                if delta.grad is None or not torch.isfinite(delta.grad).all():
                    raise ValueError('Nonfinite orientation IK gradient')
                optimizer.step()
        # Recompute after the last optimizer step; no best-iterate selection.
        with torch.no_grad():
            local = compose()
            final_global = global_rotations(local)[:, sensed]
            trajectory.append((final_global-goal).square().mean((1, 2, 3)))
            history.append(torch.stack(trajectory, dim=1).cpu())
            angles.append(torch.stack((angular_distance_deg(global_rotations(initial)[:, sensed], goal),
                                       angular_distance_deg(final_global, goal)), dim=-1).cpu())
            if settings.iterations:
                output[start:stop, joints] = math.rotation_matrix_to_r6d(local[:, joints]).reshape(stop-start, len(joints), 6).float()
    losses, angles = torch.cat(history), torch.cat(angles)
    result = output.reshape(-1, 144).detach().cpu()
    _rotations(result.double())
    fixed = [j for j in range(24) if j not in joints]
    if not torch.equal(result.reshape(-1, 24, 6)[:, fixed], pose.cpu().reshape(-1, 24, 6)[:, fixed]):
        raise AssertionError('Unoptimized local pose entries changed')
    return {'pose': result, 'loss_history': losses, 'orientation_angles_deg': angles,
            'summary': {'frames': len(pose), 'sensors': list(selected), 'sensed_joints': list(sensed),
                        'optimized_joints': list(joints), 'iterations': settings.iterations,
                        'learning_rate': settings.learning_rate, 'device': str(device),
                        'optimization_dtype': 'float64', 'output_dtype': 'float32',
                        'orientation_mse_before': losses[:, 0].mean().item(),
                        'orientation_mse_after': losses[:, -1].mean().item(),
                        'orientation_deg_before': angles[..., 0].mean().item(),
                        'orientation_deg_after': angles[..., 1].mean().item(),
                        'imu_mse_increased_frames': int((losses[:, -1] > losses[:, 0]+1e-12).sum()),
                        'projection_frobenius_mean': projection.mean().item(),
                        'projection_frobenius_max': projection.max().item(),
                        'unoptimized_pose6d_max_abs_change': 0.,
                        'frame_information': 'prediction[t] and available imu_orientation[t] only',
                        'state_carry': False, 'ground_truth_input': False}}
