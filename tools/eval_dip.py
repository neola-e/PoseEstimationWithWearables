"""Evaluate a bounded DIP test subset, one training-sized window at a time."""
import argparse
import json
import time
from pathlib import Path

import torch
from imuposer.config import amass_combos

from dip_subset import DipTestSubset, masked_pose_mse
from runtime import (PosePredictor, load_body_model, peak_ram_mib, pose_rotations,
                     sha256, write_arrays)


def evaluate(predictor, dataset, body=None, example_dir=None):
    rows = []
    started = time.perf_counter()
    for idx in range(len(dataset)):
        imu, target = dataset[idx]  # inherited original sample conversion
        if not torch.isfinite(target).all():
            raise ValueError("Non-finite DIP ground truth")
        pred = predictor.predict(imu)
        rotation, gt_rotation = pose_rotations(pred), pose_rotations(target)
        row = dict(dataset.records[idx])
        row["pose_representation_mse"] = masked_pose_mse(
            pred[None], target[None], [len(imu)])
        # Geodesic angle between local SMPL rotations, including root and hands.
        relative = rotation.transpose(-1, -2) @ gt_rotation
        cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
        row["mean_local_joint_angle_deg"] = torch.rad2deg(cosine.acos()).mean().item()
        if body is not None:
            pred_global, pred_joint = body.forward_kinematics(rotation, calc_mesh=False)
            gt_global, gt_joint = body.forward_kinematics(gt_rotation, calc_mesh=False)
            if not torch.isfinite(pred_joint).all() or not torch.isfinite(gt_joint).all():
                raise ValueError("Non-finite FK output")
            # Same mean male shape and zero root translation for both motions.
            # Include root (zero error); no Procrustes or root rotation alignment.
            row["root_relative_mpjpe_cm"] = (pred_joint - gt_joint).norm(dim=-1).mean().item() * 100
            row["joint_position_mse_m2"] = (pred_joint - gt_joint).square().mean().item()
            rel_global = pred_global.transpose(-1, -2) @ gt_global
            cos_global = ((rel_global.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
            row["mean_global_joint_angle_deg"] = torch.rad2deg(cos_global.acos()).mean().item()
        rows.append(row)
        if idx == 0 and example_dir is not None:
            example_dir = Path(example_dir)
            write_arrays(example_dir / "dip_input.npz", imu=imu, lengths=[len(imu)], fps=25)
            write_arrays(example_dir / "dip_prediction.npz", pose=pred, target_pose=target,
                         lengths=[len(imu)], fps=25, sequence_id=row["sequence_id"],
                         start_frame=row["start_frame"])
    frames = sum(row["frames"] for row in rows)
    names = [key for key in rows[0] if key not in dataset.records[0]]
    metrics = {key: sum(row[key] * row["frames"] for row in rows) / frames for key in names}
    return {"split": "DIP-IMU test (preprocessing subjects s_09, s_10)",
            "data_file": str(dataset.path), "available_sequences": dataset.total_sequences,
            "selected_sequence_ids": dataset.sequence_ids, "sensor_combo": dataset.combo,
            "max_frames_per_sequence": dataset.max_frames, "frames_evaluated": frames,
            "windows_evaluated": len(rows), "fps": 25, "padding_in_metrics": False,
            "context": "Independent windows of at most 125 frames, as in training; tails truncated by max_frames",
            "metrics": metrics, "per_window": rows,
            "metric_conventions": {"pose_representation_mse": "Mean squared error over valid frames and 144 raw 6D values; not training validation loss",
                                   "angular": "SO(3) geodesic mean over all 24 joints; root included",
                                   "joint_position": "Repository SMPL FK, mean male shape, zero translation, all 24 joints including root; no rigid alignment; meters internally"},
            "checkpoint": {"path": str(predictor.checkpoint), "epoch": predictor.epoch,
                           "global_step": predictor.global_step,
                           "sha256": sha256(predictor.checkpoint),
                           "training_validation_metric": predictor.validation_score},
            "device": str(predictor.device), "elapsed_seconds": round(time.perf_counter() - started, 3),
            "peak_process_ram_mib": peak_ram_mib(),
            "scope": "Small deterministic subset; not full DIP benchmark or paper-comparable metrics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/processed_imuposer_25fps/dip_test.pt")
    parser.add_argument("--sequences", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-frames", type=int, default=125)
    parser.add_argument("--combo", choices=list(amass_combos), default="global")
    parser.add_argument("--output", default="outputs/local_dip/dip_test_metrics.json")
    parser.add_argument("--example-dir", help="Save first window's IMU/prediction/GT locally; licensed data")
    parser.add_argument("--smpl-model", help="Optional locally licensed model for joint-position metrics")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    predictor = PosePredictor(args.checkpoint, args.device, args.threads)
    dataset = DipTestSubset(args.data, predictor.training_config, args.sequences,
                            args.max_frames, args.combo)
    body = load_body_model(args.smpl_model) if args.smpl_model else None
    with torch.inference_mode():
        report = evaluate(predictor, dataset, body, args.example_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output), "metrics": report["metrics"],
                      "frames": report["frames_evaluated"], "peak_process_ram_mib": peak_ram_mib()}))


if __name__ == "__main__":
    main()
