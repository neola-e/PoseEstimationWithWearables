"""Prepare and render three frozen, four-way DIP cache explanations.

No network inference or IK optimization is performed. Frame indices refer to the
processed 25 Hz recording, with half-open clip intervals. Only display axes are
rotated; neither predictions nor GT are fitted, shifted in time, or smoothed.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'imuposer-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from PIL import Image, GifImagePlugin
import torch
import imuposer
from imuposer.math import rotation_matrix_to_r6d
from paper_metrics import PoseMetricEvaluator, jerk_norms, local_rotations, rotation_angles_deg
from runtime import load_body_model, load_torch, peak_ram_mib, sha256
from sensor_metrics import JOINT_NAMES
ROOT = Path(__file__).resolve().parents[1]
ORDER = ['network', 'framewise_ik', 'temporal_ik', 'gt']
LABELS = ['Network / no IK', 'Framewise IK', 'Temporal IK', 'GT reference']
COLORS = ['#1768ac', '#c24c3c', '#18845b', '#555555']
PRIMARY = ['mpjre_deg', 'mpjpe_cm', 'mpjve_cm', 'jitter_100_m_s3']
DISPLAY_ROTATION = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def display_coordinates(positions):
    return np.asarray(positions) @ DISPLAY_ROTATION.T

def fixed_cases(records, diagnosis, length=250):
    """Nominal clips are selected by identity/center, diagnostic by archived J01."""
    cases = []
    for name, filename, category in [('walking', '03_a.pkl', 'locomotion'), ('upper_body', '01_a.pkl', 'upper_body')]:
        record = next((r for r in records if r['subject'] == 's_09' and r['source_file'] == filename))
        start = (record['frames'] - length) // 2
        cases.append({**record, 'case': name, 'mask': 'rw_lp', 'category': category, 'start': start, 'stop': start + length, 'selection': 'Fixed s09 identity; central 250 frames, without inspecting new scores'})
    example = diagnosis['top_pairs'][0]
    record = next((r for r in records if r['sequence_id'] == example['sequence_id']))
    start = max(0, min(record['frames'] - length, example['peak_output_frame'] - length // 2))
    cases.append({**record, 'case': 'high_jitter', 'mask': example['mask'], 'category': example['category'], 'start': start, 'stop': start + length, 'peak_output_frame': example['peak_output_frame'], 'peak_joint': example['peak_joint'], 'selection': 'Archived J01 largest net-contribution pair; center on its recorded jerk peak; exploratory'})
    for case in cases:
        if not 0 <= case['start'] < case['stop'] <= case['frames'] or case['stop'] - case['start'] != length:
            raise ValueError('Invalid fixed clip boundary')
    return cases

def case_paths(protocol, case):
    name = f"{case['mask']}_seq{case['sequence_id']:02d}"
    raw, ik, temporal = [Path(protocol[k]) for k in ['baseline_raw_run', 'baseline_ik_run', 'temporal_run']]
    return {'raw_cache': raw / 'predictions' / (name + '.npz'), 'ik_input': ik / 'inputs' / (name + '.npz'), 'ik_cache': ik / 'refined' / (name + '.npz'), 'temporal_cache': temporal / 'refined' / (name + '.npz'), 'ik_row': ik / 'rows' / (name + '.json'), 'temporal_row': temporal / 'rows' / (name + '.json')}

def prepare(temporal_run, diagnostic_run, output):
    if output.exists():
        raise FileExistsError('Use a new output directory; existing artifacts are never replaced')
    temporal_run, diagnostic_run = (temporal_run.resolve(), diagnostic_run.resolve())
    p = json.loads((temporal_run / 'PROTOCOL.json').read_text())
    cases = fixed_cases(p['records'], json.loads((diagnostic_run / 'summary.json').read_text()))
    protocol = {'format': 'dip-final-four-way-visualization-v1', 'prepared_utc': datetime.now(timezone.utc).isoformat(), 'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'temporal_run': str(temporal_run), 'baseline_raw_run': p['baseline_raw_run'], 'baseline_ik_run': p['baseline_ik_run'], 'checkpoint': p['checkpoint'], 'data': p['data'], 'smpl': p['smpl'], 'cases': cases, 'fps': 25, 'branch_order': ORDER, 'clip_frames': 250, 'display_rotation': DISPLAY_ROTATION.tolist(), 'camera': {'elevation': 15, 'azimuth': -65}, 'shape': 'shared mean male, beta=0; inherited pose blendshape disabled', 'root': 'zero root translation; original predicted and GT root rotations retained', 'time': 'same source frame for all four panels; no lag fitting, interpolation, or dropped frames', 'jitter': 'Only complete four-frame stencils inside each clip (247); no clip-boundary joins', 'smoothing': 'No additional smoothing or angle correction; inherited input preprocessing unchanged', 'animation': '25 FPS, 40 ms GIF frames; high_jitter additionally 0.25x at 160 ms; no interpolation', 'interpretation': 'Three explanatory test clips, not new benchmark scores or selection of model/settings', 'protected_files': {}, 'execution_sources': {}}
    protected = protocol['protected_files']
    for key in ['checkpoint', 'data', 'smpl']:
        file = Path(p[key]['path'])
        if sha256(file) != p[key]['sha256']:
            raise ValueError('Frozen asset hash mismatch: ' + str(file))
        protected[str(file)] = p[key]['sha256']
    for run in [temporal_run, Path(p['baseline_raw_run']), Path(p['baseline_ik_run'])]:
        for name in ['PROTOCOL.json', 'ARTIFACT_MANIFEST.json', 'VALIDATION_MANIFEST.json']:
            file = run / name
            protected[str(file)] = sha256(file)
    for file in [diagnostic_run / 'summary.json', ROOT / 'docs/protocols/dip_finetuning_v1_20261114/PROTOCOL.json']:
        protected[str(file)] = sha256(file)
    for case in cases:
        for file in case_paths(protocol, case).values():
            protected[str(file)] = sha256(file)
    for file in [Path(__file__), ROOT / 'tools/paper_metrics.py', ROOT / 'tools/runtime.py', ROOT / 'tools/validate_final_comparison.py', ROOT / 'tests/test_final_comparison.py']:
        protocol['execution_sources'][str(file.resolve())] = sha256(file)
    package = Path(imuposer.__file__).parent
    for file in sorted(package.rglob('*.py')):
        protocol['execution_sources'][str(file.resolve())] = sha256(file)
    output.mkdir(parents=True)
    for file in protocol['execution_sources']:
        source = Path(file)
        relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('package') / source.relative_to(package)
        destination = output / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output / 'PROTOCOL.json', protocol)
    write_json(output / 'PREPARE_MANIFEST.json', {str(f.relative_to(output)): sha256(f) for f in sorted(output.rglob('*')) if f.is_file()})
    print(json.dumps({'prepared': str(output), 'cases': [{k: c[k] for k in ['case', 'mask', 'sequence_id', 'start', 'stop']} for c in cases]}), flush=True)

def verify_files(files):
    for file, expected in files.items():
        if sha256(file) != expected:
            raise ValueError('Protected file changed: ' + str(file))

def load_paired(protocol, case):
    paths = case_paths(protocol, case)
    a, b, c, inp = [dict(np.load(paths[k], allow_pickle=False)) for k in ['raw_cache', 'ik_cache', 'temporal_cache', 'ik_input']]
    for cache in [a, b, c, inp]:
        if int(cache['sequence_id']) != case['sequence_id'] or str(cache['mask']) != case['mask']:
            raise ValueError('Cache recording/mask identity mismatch')
    for key, cache, run in [('raw_cache', a, protocol['baseline_raw_run']), ('ik_cache', b, protocol['baseline_ik_run']), ('temporal_cache', c, protocol['temporal_run'])]:
        if str(cache['protocol_sha256']) != sha256(Path(run) / 'PROTOCOL.json'):
            raise ValueError('Cache protocol identity mismatch: ' + key)
    if int(a['fps']) != protocol['fps'] or str(inp['baseline_cache_sha256']) != sha256(paths['raw_cache']):
        raise ValueError('Raw cache lineage mismatch')
    if not np.array_equal(a['pose_dip_final_2ep'], inp['predicted_pose']):
        raise ValueError('IK did not start from this final network prediction')
    for cache in [b, c]:
        if str(cache['input_sha256']) != sha256(paths['ik_input']):
            raise ValueError('Refinement input lineage mismatch')
    ir, tr = [json.loads(paths[k].read_text()) for k in ['ik_row', 'temporal_row']]
    if ir['refined_sha256'] != sha256(paths['ik_cache']) or tr['output_sha256'] != sha256(paths['temporal_cache']) or ir['baseline_cache_sha256'] != sha256(paths['raw_cache']):
        raise ValueError('Cached output differs from historical metric row')
    poses = {'network': a['pose_dip_final_2ep'], 'framewise_ik': b['pose'], 'temporal_ik': c['pose'], 'gt': a['target_pose']}
    for value in poses.values():
        if value.shape != (case['frames'], 144) or not np.isfinite(value).all():
            raise ValueError('Invalid complete-recording pose cache')
    return {key: value[case['start']:case['stop']].copy() for key, value in poses.items()}

def kinematics(body, pose):
    local = local_rotations(pose)
    with torch.inference_mode():
        global_r, joints = body.forward_kinematics(local, calc_mesh=False)
    return (global_r.double().numpy(), joints.double().numpy())

def clip_statistics(joints, globals_):
    """Shared time indexing and unmodified FK positions for display diagnostics."""
    result = {}
    gt = joints['gt'] - joints['gt'][:, :1]
    for branch in ORDER:
        p = joints[branch]
        errors = np.linalg.norm(p - p[:, :1] - gt, axis=-1) * 100
        jerk = jerk_norms(p, 25).numpy() / 100
        root_error = rotation_angles_deg(globals_[branch][:, :1], globals_['gt'][:, :1]).numpy()[:, 0]
        step = rotation_angles_deg(globals_[branch][1:, :1], globals_[branch][:-1, :1]).numpy()[:, 0]
        result[branch] = {'errors': errors, 'jerk': jerk, 'mpjpe_cm': float(errors.mean()), 'jitter_100_m_s3': float(jerk.mean()), 'root_error_mean_deg': float(root_error.mean()), 'root_error_max_deg': float(root_error.max()), 'max_root_step_deg': float(step.max()), 'worst_position_joints': [{'joint': int(j), 'name': JOINT_NAMES[j], 'mpjpe_cm': float(errors[:, j].mean())} for j in np.argsort(errors.mean(0))[-5:][::-1]], 'worst_jitter_joints': [{'joint': int(j), 'name': JOINT_NAMES[j], 'jitter_100_m_s3': float(jerk[:, j].mean())} for j in np.argsort(jerk.mean(0))[-5:][::-1]]}
    return result

def frame_record(case, local_frame, fps=25):
    frame = case['start'] + local_frame
    return {'clip_frame': local_frame, 'recording_frame': frame, 'recording_seconds': frame / fps, 'clip_seconds': local_frame / fps, **{branch + '_source_frame': frame for branch in ORDER}}

def write_gif_frame(stream, rgb, first, duration):
    img = Image.fromarray(rgb).quantize(colors=128)
    if first:
        for block in GifImagePlugin.getheader(img, info={'loop': 0})[0]:
            stream.write(block)
    for block in GifImagePlugin.getdata(img, duration=duration, disposal=2, include_color_table=True):
        stream.write(block)

def render_clip(output, case, joints, stats, parent, limits):
    n = len(joints['gt'])
    xyz = {k: display_coordinates(v) for k, v in joints.items()}
    edges = [(p, j) for j, p in enumerate(parent) if p is not None]
    fig = plt.figure(figsize=(16, 8), dpi=100)
    grid = fig.add_gridspec(2, 4, height_ratios=[2.4, 1], left=0.035, right=0.97, bottom=0.075, top=0.82, hspace=0.28, wspace=0.18)
    artists = []
    for col, branch in enumerate(ORDER):
        ax = fig.add_subplot(grid[0, col], projection='3d')
        ax.set(xlim=limits, ylim=limits, zlim=limits, xlabel='X (m)', ylabel='-Z (m)', zlabel='Y (m), up')
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=15, azim=-65)
        ax.tick_params(labelsize=7, pad=0)
        lines = [ax.plot([], [], [], color=COLORS[col], linewidth=2.3)[0] for _ in edges]
        ghosts = [] if branch == 'gt' else [ax.plot([], [], [], color='#777777', alpha=0.35, linewidth=1, linestyle='--')[0] for _ in edges]
        dots = ax.scatter(np.zeros(24), np.zeros(24), np.zeros(24), c=np.zeros(24), cmap='inferno', norm=Normalize(0, 40), s=15, depthshade=False)
        labels = [ax.text(0, 0, 0, side, fontsize=7) for side in ['L', 'R']]
        ax.set_title(f"{LABELS[col]}\nClip MPJPE {stats[branch]['mpjpe_cm']:.2f} cm | jitter {stats[branch]['jitter_100_m_s3']:.3f}", fontsize=9)
        artists.append((lines, ghosts, dots, labels))
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 40), cmap='inferno'), ax=fig.axes[:4], orientation='horizontal', fraction=0.035, pad=0.12, shrink=0.4)
    colorbar.set_label('Same-frame joint position error (cm); colors saturate at 40 cm', fontsize=8)
    time = np.arange(case['start'], case['stop']) / 25
    err_ax = fig.add_subplot(grid[1, :2])
    jerk_ax = fig.add_subplot(grid[1, 2:])
    for i, branch in enumerate(ORDER):
        if branch != 'gt':
            err_ax.plot(time, stats[branch]['errors'].mean(1), color=COLORS[i], label=LABELS[i], linewidth=1.2)
        jerk_ax.plot(time[3:], stats[branch]['jerk'].mean(1), color=COLORS[i], label=LABELS[i], linewidth=1.1, alpha=0.85)
    err_ax.set(xlabel='Processed recording time (s)', ylabel='MPJPE (cm)', title='Accuracy: mean over 24 joints')
    jerk_ax.set(xlabel='Processed recording time (s)', ylabel='Jerk / 100 (m/s³)', title='Smoothness: complete four-frame stencils only')
    cursors = []
    for ax in [err_ax, jerk_ax]:
        ax.set_xlim(time[0], time[-1])
        ax.grid(alpha=0.15)
        ax.legend(fontsize=7, ncol=2)
        cursors.append(ax.axvline(time[0], color='black', alpha=0.6, linewidth=1))
        if 'peak_output_frame' in case:
            ax.axvline(case['peak_output_frame'] / 25, color='#9c3eaa', linestyle=':', linewidth=1)
    top = fig.suptitle('', fontsize=12, y=0.98)
    fig.text(0.5, 0.875, 'Same camera, scale and source frame | Root translation fixed; original root rotation retained\nDashed gray skeleton = same-frame GT | L/R mark wrists | No output smoothing | Jitter is not a GT error', ha='center', fontsize=9)
    fig.text(0.5, 0.012, 'Final DIP checkpoint: 2 epochs / 162 updates | Explanatory clip, not a new benchmark | Data: 25 Hz', ha='center', fontsize=9)
    stem = case['case']
    streams = [(open(output / (stem + '.gif'), 'xb'), 40)]
    if stem == 'high_jitter':
        streams.append((open(output / (stem + '_slow_025x.gif'), 'xb'), 160))
    selected = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    screenshots = []
    try:
        for t in range(n):
            for branch, (lines, ghosts, dots, labels) in zip(ORDER, artists):
                points = xyz[branch][t]
                for line, (p, child) in zip(lines, edges):
                    pair = points[[p, child]]
                    line.set_data(pair[:, 0], pair[:, 1])
                    line.set_3d_properties(pair[:, 2])
                for line, (p, child) in zip(ghosts, edges):
                    pair = xyz['gt'][t, [p, child]]
                    line.set_data(pair[:, 0], pair[:, 1])
                    line.set_3d_properties(pair[:, 2])
                dots._offsets3d = points.T
                dots.set_array(stats[branch]['errors'][t])
                for text, joint in zip(labels, [20, 21]):
                    text.set_position((points[joint, 0], points[joint, 1]))
                    text.set_3d_properties(points[joint, 2])
            for cursor in cursors:
                cursor.set_xdata([time[t], time[t]])
            top.set_text(f"{stem} | {case['subject']}/{case['source_file']} | {case['mask']} (right wrist + left pocket)\nProcessed frame {case['start'] + t} | recording time {time[t]:.2f} s | clip {t / 25:.2f} s")
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
            for stream, duration in streams:
                write_gif_frame(stream, rgb, t == 0, duration)
            if t in selected:
                Image.fromarray(rgb).save(output / f"{stem}_frame{case['start'] + t:04d}.png")
                screenshots.append(Image.fromarray(rgb).resize((1200, 600)))
            if (t + 1) % 50 == 0:
                print(json.dumps({'case': stem, 'rendered': t + 1, 'total': n}), flush=True)
        for stream, _ in streams:
            stream.write(b';')
    finally:
        for stream, _ in streams:
            stream.close()
        plt.close(fig)
    sheet = Image.new('RGB', (1200, 600 * len(screenshots)), 'white')
    for i, shot in enumerate(screenshots):
        sheet.paste(shot, (0, i * 600))
    sheet.save(output / (stem + '_contact_sheet.png'))

def render(output):
    p = json.loads((output / 'PROTOCOL.json').read_text())
    verify_files({str(output / k): v for k, v in json.loads((output / 'PREPARE_MANIFEST.json').read_text()).items()})
    verify_files(p['protected_files'])
    verify_files(p['execution_sources'])
    if (output / 'completion.json').exists() or any(output.glob('*.gif')):
        raise FileExistsError('Use a newly prepared directory to render; no artifacts are overwritten')
    torch.set_num_threads(2)
    write_json(output / 'RUNTIME.json', {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'device': 'cpu', 'optimizer_steps': 0, 'network_forward_calls': 0, 'additional_ik_optimization': False})
    body = load_body_model(p['smpl']['path'])
    evaluator = PoseMetricEvaluator(body, fps=25, fk_chunk=32)
    processed = load_torch(p['data']['path'])
    prepared = []
    rows = []
    for case in p['cases']:
        poses = load_paired(p, case)
        gt = rotation_matrix_to_r6d(processed['pose'][case['sequence_id']]).reshape(case['frames'], 144).numpy()
        if not np.array_equal(gt[case['start']:case['stop']], poses['gt']):
            raise ValueError('GT cache frame alignment differs from processed DIP')
        kinematics_ = {k: kinematics(body, v) for k, v in poses.items()}
        globals_ = {k: v[0] for k, v in kinematics_.items()}
        joints = {k: v[1] for k, v in kinematics_.items()}
        stats = clip_statistics(joints, globals_)
        rows_case = {}
        for branch in ORDER:
            metrics = evaluator.evaluate_sequence(poses[branch], poses['gt'], f"{case['case']}_{branch}")['metrics']
            if abs(metrics['mpjpe_cm'] - stats[branch]['mpjpe_cm']) > 1e-05 or abs(metrics['jitter_100_m_s3'] - stats[branch]['jitter_100_m_s3']) > 1e-05:
                raise ValueError('Plot statistics differ from canonical metric evaluator')
            row = {'case': case['case'], 'branch': branch, **{k: metrics[k] for k in PRIMARY}}
            rows.append(row)
            rows_case[branch] = {**row, **{k: v for k, v in stats[branch].items() if k not in ['errors', 'jerk']}}
        write_json(output / (case['case'] + '_summary.json'), {'case': case, 'branches': rows_case, 'jerk_samples': 247})
        np.savez_compressed(output / (case['case'] + '_replay.npz'), frame_ids=np.arange(case['start'], case['stop']), **{f'pose_{k}': v for k, v in poses.items()}, **{f'joints_{k}': v for k, v in joints.items()}, **{f'error_cm_{k}': v['errors'] for k, v in stats.items()}, **{f'jerk_100_m_s3_{k}': v['jerk'] for k, v in stats.items()})
        write_json(output / (case['case'] + '_timeline.json'), [frame_record(case, t) for t in range(250)])
        prepared.append((case, joints, stats))
    with (output / 'clip_metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'branch'] + PRIMARY)
        writer.writeheader()
        writer.writerows(rows)
    extent = max((float(np.abs(display_coordinates(j)).max()) for _, joints, _ in prepared for j in joints.values()))
    limit = float(np.ceil((extent + 0.05) * 10) / 10)
    write_json(output / 'DISPLAY.json', {'symmetric_axis_limits_m': [-limit, limit], 'max_absolute_coordinate_m': extent, 'error_color_limits_cm': [0, 40], 'camera': p['camera']})
    for case, joints, stats in prepared:
        render_clip(output, case, joints, stats, body.parent, (-limit, limit))
    verify_files(p['protected_files'])
    verify_files(p['execution_sources'])
    html = '<!doctype html><meta charset="utf-8"><title>Final DIP four-way replay</title>\n<style>body{font:16px system-ui;max-width:1600px;margin:30px auto;padding:0 18px;background:#f4f5f6;color:#172127}img{width:100%;background:white}section{margin:35px 0}a{color:#1768ac}</style>\n<h1>Final DIP model: four-way synchronized offline replay</h1>\n<p>Left to right: network / no IK, framewise IK, Temporal IK, GT. All three clips use right wrist + left pocket;25 Hz,250 frames(10 seconds) each. Root translation is fixed and each branch retains its original orientation. Dashed gray is same-frame GT;point colors show joint-position error. No additional filtering,training or IK optimization.</p>\n<p>Walking and upper-body clips use fixed central intervals;the jitter example comes from archived J01 diagnosis. These explain existing results,not overall performance. GIFs encode25 FPS;browser playback speed is not a real-time performance measurement.</p>'
    for case in p['cases']:
        name = case['case']
        label = {'walking': 'Walking category', 'upper_body': 'Upper-body category', 'high_jitter': 'Archived high-jitter example(upper body)'}[name]
        html += f'''<section><h2>{label}</h2><p>{case['subject']}/{case['source_file']}, processed frames [{case['start']}, {case['stop']}). <a href="{name}_summary.json">Clip metrics</a> · <a href="{name}_contact_sheet.png">Fixed-time previews</a></p><img loading="lazy" src="{name}.gif" alt="{label} four-way comparison animation"></section>'''
    html += '<h2>High-jitter example:0.25x slow replay</h2><p>The same250 frames,160ms each;no interpolation or smoothing. Curves and labels retain original25Hz data time.</p><img loading="lazy" src="high_jitter_slow_025x.gif"><p><a href="clip_metrics.csv">All clip metrics CSV</a> · <a href="PROTOCOL.json">Selection/display protocol</a></p>'
    (output / 'index.html').write_text(html)
    write_json(output / 'completion.json', {'status': 'rendered_pending_independent_validation', 'cases': 3, 'unique_source_frames': 750, 'gifs': 4, 'optimizer_steps': 0, 'network_forward_calls': 0, 'protected_files_unchanged': True, 'peak_ram_mib': peak_ram_mib()})
    write_json(output / 'ARTIFACT_MANIFEST.json', {str(f.relative_to(output)): sha256(f) for f in sorted(output.rglob('*')) if f.is_file()})
    print(json.dumps({'status': 'rendered', 'output': str(output), 'peak_ram_mib': peak_ram_mib()}), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--temporal-run', type=Path, required=True)
    prep.add_argument('--diagnostic-run', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    run = sub.add_parser('render')
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.temporal_run, args.diagnostic_run, args.output.resolve())
    else:
        render(args.output.resolve())
