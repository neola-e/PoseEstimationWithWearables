"""Render matched three-panel skeletons from cached predictions; identical axes and times."""
import argparse
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'imuposer-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, GifImagePlugin
import torch

from runtime import load_body_model, pose_rotations


def display_coordinates(joints):
    xyz = joints[..., [0, 2, 1]].copy()
    xyz[..., 1] *= -1
    return xyz


def setup_axis(ax, title):
    ax.set(title=title, xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), zlim=(-1.15, 1.15),
           xlabel='X (m)', ylabel='−Z (m)', zlabel='Y (m), up')
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=15, azim=-65)


def draw_skeleton(ax, points, edges, color):
    for parent, child in edges:
        pair = points[[parent, child]]
        ax.plot(pair[:, 0], pair[:, 1], pair[:, 2], color=color, linewidth=2)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=8)


def main(args):
    run = Path(args.run_dir)
    out = run / 'visuals'
    out.mkdir(exist_ok=False)
    provenance = json.loads((run / 'provenance.json').read_text())
    torch.set_num_threads(2)
    body = load_body_model(provenance['smpl']['path'])
    summary = json.loads((run / 'analysis/summary.json').read_text())
    # Predeclared focus cases plus the largest observed worsening, explicitly selected post hoc.
    cases = [(18, 0), (18, 750), (8, summary['sequence8_global_worst']['start_frame']), (8, 750)]
    edges = [(p, c) for c, p in enumerate(body.parent) if p is not None]
    manifest = []
    with np.load(run / 'predictions.npz', allow_pickle=False) as cache, torch.inference_mode():
        for seq, start in cases:
            idx = next(i for i, (s, f) in enumerate(zip(cache['sequence_ids'], cache['start_frames'])) if s == seq and f == start)
            sl = slice(cache['offsets'][idx], cache['offsets'][idx + 1])
            joints = [body.forward_kinematics(pose_rotations(torch.from_numpy(cache[k][sl])), calc_mesh=False)[1].numpy()
                      for k in ['pose_global', 'pose_lw_rp_h', 'target_pose']]
            xyz = [display_coordinates(j) for j in joints]
            errors = [np.linalg.norm(j - joints[2], axis=-1).mean(1) * 100 for j in joints[:2]]
            titles = [f'global (5) | {errors[0].mean():.2f} cm', f'lw_rp_h (3) | {errors[1].mean():.2f} cm', 'Ground truth']
            colors = ['#1768ac', '#14855f', '#d97706']
            # Two common times, selected by fixed index rather than maximum visual discrepancy.
            selected = [len(joints[0]) // 3, 2 * len(joints[0]) // 3]
            fig = plt.figure(figsize=(12, 8), dpi=120)
            for row, frame in enumerate(selected):
                for col in range(3):
                    ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection='3d')
                    setup_axis(ax, f'{titles[col]}\nframe {start + frame}, {(start + frame) / 25:.2f} s')
                    draw_skeleton(ax, xyz[col][frame], edges, colors[col])
            fig.suptitle(f'Sequence {seq} | frames {start}–{start + len(joints[0]) - 1} | mean male SMPL, zero translation\nSame camera, scale and time; titles show window MPJPE. No IK or rigid alignment.')
            fig.subplots_adjust(left=.03, right=.96, top=.84, bottom=.06, wspace=.15, hspace=.24)
            name = f'seq{seq}_frame{start}'
            fig.savefig(out / f'{name}.png')
            plt.close(fig)
            if (seq, start) in [(18, 750), (8, summary['sequence8_global_worst']['start_frame'])]:
                fig = plt.figure(figsize=(12, 4), dpi=90)
                lines, dots = [], []
                for col in range(3):
                    ax = fig.add_subplot(1, 3, col + 1, projection='3d')
                    setup_axis(ax, titles[col])
                    lines.append([ax.plot([], [], [], color=colors[col], linewidth=2)[0] for _ in edges])
                    dots.append(ax.plot([], [], [], 'o', color=colors[col], markersize=3)[0])
                title = fig.suptitle('')
                fig.subplots_adjust(left=.02, right=.96, top=.78, bottom=.1, wspace=.15)
                with (out / f'{name}.gif').open('wb') as stream:
                    for frame in range(len(joints[0])):
                        for col in range(3):
                            points = xyz[col][frame]
                            for line, (p, c) in zip(lines[col], edges):
                                pair = points[[p, c]]
                                line.set_data(pair[:, 0], pair[:, 1])
                                line.set_3d_properties(pair[:, 2])
                            dots[col].set_data(points[:, 0], points[:, 1])
                            dots[col].set_3d_properties(points[:, 2])
                        title.set_text(f'Sequence {seq} | frame {start + frame} | {(start + frame) / 25:.2f} s\nRoot translation fixed; original root rotation retained; 25 FPS')
                        fig.canvas.draw()
                        frame_image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(colors=128)
                        if frame == 0:
                            for block in GifImagePlugin.getheader(frame_image, info={'loop': 0})[0]:
                                stream.write(block)
                        for block in GifImagePlugin.getdata(frame_image, duration=40, disposal=2, include_color_table=True):
                            stream.write(block)
                    stream.write(b';')
                plt.close(fig)
                with Image.open(out / f'{name}.gif') as image:
                    if image.n_frames != len(joints[0]):
                        raise ValueError('GIF frame count mismatch')
                    for frame in range(image.n_frames):
                        image.seek(frame)
                        image.load()
                        if image.info.get('duration') != 40:
                            raise ValueError('GIF timing mismatch')
            manifest.append({'sequence_id': seq, 'start_frame': start, 'frames': len(joints[0]),
                'snapshot_frames': [start + f for f in selected], 'png': f'{name}.png',
                'window_mpjpe_cm': {'global': float(errors[0].mean()), 'lw_rp_h': float(errors[1].mean())}})
            print(f'Rendered and checked {name}', flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    main(parser.parse_args())
