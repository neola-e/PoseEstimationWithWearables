"""Infer on already preprocessed 25 FPS IMU features."""
import argparse
import json

from runtime import PosePredictor, peak_ram_mib, read_arrays, write_arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    data = read_arrays(args.input)
    if "imu" not in data:
        parser.error("Input dictionary/archive must contain 'imu'")
    if "fps" in data and float(data["fps"]) != 25:
        parser.error("Input must already be sampled at 25 FPS")
    predictor = PosePredictor(args.checkpoint, args.device, args.threads)
    pose = predictor.predict(data["imu"], data.get("lengths"))
    batch = 1 if pose.ndim == 2 else pose.shape[0]
    lengths = data.get("lengths", [pose.shape[-2]] * batch)
    write_arrays(args.output, pose=pose, lengths=lengths, fps=25,
                 checkpoint_epoch=predictor.epoch)
    print(json.dumps({"output": args.output, "shape": list(pose.shape),
                      "checkpoint_epoch": predictor.epoch, "device": args.device,
                      "peak_process_ram_mib": peak_ram_mib()}))


if __name__ == "__main__":
    main()
