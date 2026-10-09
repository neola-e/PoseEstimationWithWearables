"""Stream an SMPL FK skeleton animation to GIF, without storing rendered frames."""
import argparse
import json
import os
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "imuposer-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, GifImagePlugin

from runtime import load_body_model, peak_ram_mib, pose_rotations, read_arrays


def select_pose(data, key="pose", sequence=0):
    if key not in data:
        raise ValueError(f"Missing '{key}' in prediction file")
    pose = torch.as_tensor(data[key], dtype=torch.float32)
    if pose.ndim == 3:
        if not 0 <= sequence < len(pose):
            raise ValueError("Sequence index out of range")
        pose = pose[sequence]
    elif pose.ndim != 2 or sequence != 0:
        raise ValueError("Expected [T,144] or [B,T,144]")
    lengths = torch.as_tensor(data.get("lengths", [len(pose)]))
    if lengths.ndim != 1 or sequence >= len(lengths):
        raise ValueError("Invalid pose lengths")
    length = int(lengths[sequence])
    if length < 1 or length > len(pose) or length != float(lengths[sequence]):
        raise ValueError("Invalid pose length")
    return pose[:length]


@torch.inference_mode()
def render(pose, body, output, target=None, frames=125, fps=25, label="Predicted"):
    if frames < 1 or fps != 25:
        raise ValueError("frames must be positive; this model uses 25 FPS")
    count = min(frames, len(pose))
    if target is not None and len(target) < count:
        raise ValueError("Ground truth shorter than prediction")
    predictions = [pose[:count]] + ([target[:count]] if target is not None else [])
    labels = [label] + (["Ground truth"] if target is not None else [])
    joints = [body.forward_kinematics(pose_rotations(p), calc_mesh=False)[1].numpy()
              for p in predictions]
    if not all(np.isfinite(j).all() for j in joints):
        raise ValueError("Non-finite skeleton positions")
    output = Path(output)
    if output.suffix.lower() != ".gif":
        raise ValueError("This lightweight renderer saves .gif files")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(4 * len(joints), 4), dpi=90)
    axes, lines, dots = [], [], []
    edges = [(parent, child) for child, parent in enumerate(body.parent) if parent is not None]
    for i, name in enumerate(labels):
        ax = fig.add_subplot(1, len(joints), i + 1, projection="3d")
        ax.set_title(textwrap.fill(name, width=32))
        # Display-only proper rotation: (X,Y,Z) -> (X,-Z,Y), Y remains up.
        # FK tensors themselves are never transformed or fitted to GT.
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_zlim(-1.15, 1.15)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("-Z (m)")
        ax.set_zlabel("Y (m), up")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-65)
        color = "#1768ac" if i == 0 else "#d97706"
        axes.append(ax)
        lines.append([ax.plot([], [], [], color=color, linewidth=2)[0] for _ in edges])
        dots.append(ax.plot([], [], [], "o", color=color, markersize=3)[0])
    title = fig.suptitle("")
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.08, top=0.82, wspace=0.18)
    try:
        with output.open("wb") as stream:
            for frame in range(count):
                for points, skeleton, dot in zip(joints, lines, dots):
                    xyz = points[frame][:, [0, 2, 1]].copy()
                    xyz[:, 1] *= -1
                    for (parent, child), line in zip(edges, skeleton):
                        pair = xyz[[parent, child]]
                        line.set_data(pair[:, 0], pair[:, 1])
                        line.set_3d_properties(pair[:, 2])
                    dot.set_data(xyz[:, 0], xyz[:, 1])
                    dot.set_3d_properties(xyz[:, 2])
                title.set_text(f"{frame / fps:.2f} s | root fixed | mean male SMPL | 25 FPS")
                fig.canvas.draw()
                rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
                image = Image.fromarray(rgb).quantize(colors=128)
                if frame == 0:
                    header, _ = GifImagePlugin.getheader(image, info={"loop": 0})
                    for block in header:
                        stream.write(block)
                # Each frame has its own palette; no PillowWriter frame list.
                for block in GifImagePlugin.getdata(image, duration=40, disposal=2,
                                                    include_color_table=True):
                    stream.write(block)
            stream.write(b";")
    finally:
        plt.close(fig)
    return {"output": str(output), "frames": count, "fps": fps,
            "side_by_side": target is not None, "peak_process_ram_mib": peak_ram_mib()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Archive with 'pose' and optional 'target_pose'")
    parser.add_argument("--ground-truth", help="Optional separate archive with 'pose'")
    parser.add_argument("--smpl-model", required=True, help="Locally licensed male SMPL .pkl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence", type=int, default=0)
    parser.add_argument("--frames", type=int, default=125)
    parser.add_argument("--label", default="Predicted")
    args = parser.parse_args()
    torch.set_num_threads(2)
    data = read_arrays(args.input)
    if "fps" in data and float(data["fps"]) != 25:
        parser.error("Expected a 25 FPS prediction")
    pose = select_pose(data, sequence=args.sequence)
    target = None
    if args.ground_truth:
        gt_data = read_arrays(args.ground_truth)
        if "fps" in gt_data and float(gt_data["fps"]) != 25:
            parser.error("Expected 25 FPS ground truth")
        target = select_pose(gt_data, sequence=args.sequence)
    elif "target_pose" in data:
        target = select_pose(data, key="target_pose", sequence=args.sequence)
    body = load_body_model(args.smpl_model)
    print(json.dumps(render(pose, body, args.output, target, args.frames, label=args.label)))


if __name__ == "__main__":
    main()
