"""Copyable inference example. Default input is synthetic, not an accuracy demo."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from runtime import PosePredictor, read_arrays, write_arrays, peak_ram_mib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(Path(os.environ.get("IMUPOSER_CHECKPOINT", ROOT / "checkpoints/imuposer_global_best.ckpt"))))
    parser.add_argument("--input", help="Optional preprocessed .npz/.pt containing 'imu'")
    parser.add_argument("--output", default=str(ROOT / "outputs/example_prediction.npz"))
    args = parser.parse_args()
    if args.input:
        data = read_arrays(args.input)
        if "fps" in data and float(data["fps"]) != 25:
            raise ValueError("Expected preprocessed 25 FPS input")
    else:
        # Identity orientations and small arbitrary acceleration values test IO.
        # They are not simulated measurements consistent with an SMPL motion.
        frames = 125
        imu = np.zeros((frames, 60), dtype=np.float32)
        imu[:, 15:] = np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (frames, 5))
        imu[:, 0] = 0.01 * np.sin(np.arange(frames) * 2 * np.pi / frames)
        data = {"imu": imu, "lengths": np.array([frames]), "fps": 25}
        write_arrays(Path(args.output).parent / "synthetic_imu.npz", **data)
    predictor = PosePredictor(args.checkpoint, device="cpu")
    pose = predictor.predict(data["imu"], data.get("lengths"))
    batch = 1 if pose.ndim == 2 else pose.shape[0]
    write_arrays(args.output, pose=pose,
                 lengths=data.get("lengths", [pose.shape[-2]] * batch), fps=25)
    print(f"Saved {tuple(pose.shape)} to {args.output}; peak RAM {peak_ram_mib():.1f} MiB")


if __name__ == "__main__":
    main()
