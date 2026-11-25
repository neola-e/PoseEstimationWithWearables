"""Checkpoint inference and file IO using the original IMUPoser architecture."""
import copy
import hashlib
import inspect
import os
import resource
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imuposer-matplotlib"))

from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
from imuposer.math import r6d_to_rotation_matrix


def load_torch(path, trusted=False):
    """Old torch compatibility. Config-containing Lightning files require trust."""
    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = not trusted
    return torch.load(path, **kwargs)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_ram_mib():
    """Linux process maximum resident set, including imported libraries."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)


class PosePredictor:
    def __init__(self, checkpoint, device="cpu", threads=2):
        if threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda[:index]")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        self.checkpoint = Path(checkpoint)
        # Lightning pickles Config; only load checkpoints from a trusted source.
        saved = load_torch(self.checkpoint, trusted=True)
        if saved.get("format") == "imuposer-inference-v1":
            self.training_config = SimpleNamespace(**saved["config"], batch_size=256)
        else:
            self.training_config = copy.deepcopy(saved["hyper_parameters"]["config"])
        cfg = copy.deepcopy(self.training_config)
        if (cfg.model != "GlobalModelIMUPoser" or cfg.joints_set != list(range(5))
                or cfg.pred_joints_set != list(range(24)) or not cfg.r6d):
            raise ValueError("Expected the five-sensor, 24-joint 6D global checkpoint")
        if cfg.acc_scale != 30 or cfg.max_sample_len * 25 // 60 != 125:
            raise ValueError("Config class defaults differ from this checkpoint's training preprocessing")
        # This flag only constructs the loss-time FK helper, not network layers.
        # Keep training_config intact for provenance. Never call Config() or mkdir.
        cfg.use_joint_loss = False
        cfg.device = self.device
        cfg.mkdir = False
        self.model = IMUPoserModel(cfg)
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.to(self.device).eval()
        self.epoch = int(saved["epoch"])
        self.global_step = int(saved.get("global_step", 0))
        self.validation_score = saved.get("training_validation_metric")
        for callback in saved.get("callbacks", {}).values():
            if callback.get("monitor") == "validation_step_loss":
                score = callback.get("best_model_score")
                if score is not None:
                    self.validation_score = float(score)
        # Drop optimizer tensors and checkpoint weights after copying into model.
        del saved

    @torch.inference_mode()
    def predict(self, imu, lengths=None):
        x = torch.as_tensor(imu, dtype=torch.float32, device="cpu")
        single = x.ndim == 2
        if x.ndim not in (2, 3) or x.shape[-1] != 60:
            raise ValueError("imu must have shape [T,60] or [B,T,60]")
        if single:
            x = x.unsqueeze(0)
        batch, frames, _ = x.shape
        if batch == 0 or frames == 0 or not torch.isfinite(x).all():
            raise ValueError("imu must be nonempty and finite, including padded frames")
        if lengths is None:
            lens = torch.full((batch,), frames, dtype=torch.long)
        else:
            raw = torch.as_tensor(lengths, device="cpu")
            if raw.ndim != 1 or raw.numel() != batch:
                raise ValueError("lengths must contain one integer per sequence")
            if raw.dtype == torch.bool or raw.is_complex() or not torch.isfinite(raw).all():
                raise ValueError("lengths must be finite integers")
            lens = raw.long()
            if not torch.equal(raw, lens.to(raw.dtype)):
                raise ValueError("lengths must be integers")
        if (lens < 1).any() or (lens > frames).any():
            raise ValueError("lengths must be between 1 and T")
        pred = self.model(x.to(self.device), lens).cpu()
        if pred.shape != (batch, int(lens.max()), 144) or not torch.isfinite(pred).all():
            raise RuntimeError("Invalid model output shape or non-finite prediction")
        # The original RNN unpacks only to max(lengths), and linear bias fills pads.
        # Preserve the caller's T and explicitly zero all invalid output frames.
        result = torch.zeros(batch, frames, 144)
        result[:, :pred.shape[1]] = pred
        mask = torch.arange(frames)[None, :] >= lens[:, None]
        result[mask] = 0
        return result[0] if single else result


def read_arrays(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    if path.suffix == ".pt":
        data = load_torch(path)
        if isinstance(data, torch.Tensor):
            return {"imu": data}
        if not isinstance(data, dict):
            raise ValueError(".pt file must contain a tensor or dictionary")
        return data
    raise ValueError("Use .npz or .pt")


def write_arrays(path, **arrays):
    path = Path(path)
    if path.suffix not in (".pt", ".npz"):
        raise ValueError("Output must end in .pt or .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".pt":
        torch.save({k: torch.as_tensor(v) for k, v in arrays.items()}, path)
    else:
        np.savez_compressed(path, **{k: v.detach().cpu().numpy()
                                   if isinstance(v, torch.Tensor) else np.asarray(v)
                                   for k, v in arrays.items()})


def pose_rotations(pose):
    """Validate the repository's Gram-Schmidt conversion, including degeneracy."""
    pose = torch.as_tensor(pose, dtype=torch.float32, device="cpu")
    if pose.ndim != 2 or pose.shape[1] != 144 or len(pose) == 0:
        raise ValueError("pose must be nonempty [T,144]")
    if not torch.isfinite(pose).all():
        raise ValueError("pose contains non-finite values")
    rotation = r6d_to_rotation_matrix(pose).reshape(-1, 24, 3, 3)
    gram = rotation.transpose(-1, -2) @ rotation
    if (not torch.isfinite(rotation).all()
            or not torch.allclose(gram, torch.eye(3).expand_as(gram), atol=1e-4)
            or not torch.allclose(torch.linalg.det(rotation),
                                  torch.ones(rotation.shape[:2]), atol=1e-4)):
        raise ValueError("Degenerate 6D pose: conversion did not produce SO(3) rotations")
    return rotation


def load_body_model(path):
    from imuposer.smpl.parametricModel import ParametricModel
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Install a licensed SMPL male model locally and pass --smpl-model")
    return ParametricModel(str(path), device=torch.device("cpu"))
