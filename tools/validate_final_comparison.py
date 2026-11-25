"""Independently audit four-way replay pairing, FK, metrics and GIF timing."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from runtime import load_body_model, load_torch, pose_rotations, sha256


def audit(output):
    p = json.loads((output / 'PROTOCOL.json').read_text())
    manifest = json.loads((output / 'ARTIFACT_MANIFEST.json').read_text())
    for rel, digest in manifest.items():
        if sha256(output / rel) != digest:
            raise ValueError('Replay artifact changed: ' + rel)
    for file, digest in {**p['protected_files'], **p['execution_sources']}.items():
        if sha256(file) != digest:
            raise ValueError('Protected/source file changed: ' + file)
    body = load_body_model(p['smpl']['path'])
    data = load_torch(p['data']['path'])
    branch_order = ['network', 'framewise_ik', 'temporal_ik', 'gt']
    table = list(csv.DictReader((output / 'clip_metrics.csv').open()))
    checks = []; gifs = []; max_joint_diff = 0.; max_stat_diff = 0.
    rotation = np.array(p['display_rotation'])
    if not np.array_equal(rotation.T @ rotation, np.eye(3)) or np.linalg.det(rotation) != 1:
        raise ValueError('Display transform changes handedness or scale')
    display = json.loads((output / 'DISPLAY.json').read_text()); limit = display['symmetric_axis_limits_m'][1]
    for case in p['cases']:
        name = f"{case['mask']}_seq{case['sequence_id']:02d}.npz"
        cache_paths = [Path(p['baseline_raw_run']) / 'predictions' / name,
                       Path(p['baseline_ik_run']) / 'refined' / name,
                       Path(p['temporal_run']) / 'refined' / name]
        with np.load(cache_paths[0], allow_pickle=False) as a, np.load(cache_paths[1], allow_pickle=False) as b, np.load(cache_paths[2], allow_pickle=False) as c:
            source = {'network': a['pose_dip_final_2ep'], 'framewise_ik': b['pose'], 'temporal_ik': c['pose'], 'gt': a['target_pose']}
        timeline = json.loads((output / (case['case'] + '_timeline.json')).read_text())
        ids = np.arange(case['start'], case['stop'])
        if len(timeline) != 250 or not np.array_equal(ids, [r['recording_frame'] for r in timeline]):
            raise ValueError('Timeline lost or reordered frames')
        for t, row in enumerate(timeline):
            if row['clip_frame'] != t or row['recording_seconds'] != int(ids[t]) / 25 or row['clip_seconds'] != t / 25:
                raise ValueError('Incorrect displayed clock')
            if any(row[k + '_source_frame'] != int(ids[t]) for k in branch_order):
                raise ValueError('Panel time mismatch')
        with np.load(output / (case['case'] + '_replay.npz'), allow_pickle=False) as replay:
            if not np.array_equal(replay['frame_ids'], ids):
                raise ValueError('Replay pose frame mapping differs')
            joints = {}
            for branch in branch_order:
                pose = replay['pose_' + branch]
                if not np.array_equal(pose, source[branch][case['start']:case['stop']]):
                    raise ValueError('Cached pose was modified: ' + branch)
                rot = pose_rotations(torch.from_numpy(pose))
                if branch == 'gt':
                    actual = data['pose'][case['sequence_id']][case['start']:case['stop']]
                    if not torch.allclose(rot, actual, atol=2e-6, rtol=0):
                        raise ValueError('GT differs from exact processed recording frames')
                with torch.inference_mode():
                    joints[branch] = np.concatenate([body.forward_kinematics(rot[first:first + 11], calc_mesh=False)[1].double().numpy()
                                                     for first in range(0, 250, 11)])
                diff = float(np.abs(joints[branch] - replay['joints_' + branch]).max())
                max_joint_diff = max(max_joint_diff, diff)
                if diff > 2e-6:
                    raise ValueError('Independent FK differs')
                if np.abs(replay['joints_' + branch] @ rotation.T).max() >= limit:
                    raise ValueError('Skeleton falls outside common display limits')
            for branch in branch_order:
                j = joints[branch]; gt = joints['gt']
                error = np.linalg.norm((j - j[:, :1]) - (gt - gt[:, :1]), axis=-1) * 100
                jerk = np.linalg.norm(np.diff(j, n=3, axis=0), axis=-1) * (25 ** 3 / 100)
                if jerk.shape != (247, 24):
                    raise ValueError('Jerk support crosses clip boundary')
                error_diff = float(np.abs(error - replay['error_cm_' + branch]).max())
                jerk_diff = float(np.abs(jerk - replay['jerk_100_m_s3_' + branch]).max())
                # FP32 FK chunk differences are magnified by third differentiation.
                if error_diff > 5e-4 or jerk_diff > .005:
                    raise ValueError('Independent per-frame diagnostics differ')
                row = next(r for r in table if r['case'] == case['case'] and r['branch'] == branch)
                for key, value in [('mpjpe_cm', error.mean()), ('jitter_100_m_s3', jerk.mean())]:
                    diff = abs(value - float(row[key])); max_stat_diff = max(max_stat_diff, diff)
                    if diff > 1e-4:
                        raise ValueError('Independent clip summary differs')
                if branch == 'gt' and float(row['mpjpe_cm']) != 0:
                    raise ValueError('GT self-position error must be zero')
        normal = output / (case['case'] + '.gif')
        files = [(normal, 40)]
        if case['case'] == 'high_jitter':
            files.append((output / 'high_jitter_slow_025x.gif', 160))
        decoded_selected = {}
        for file, duration in files:
            with Image.open(file) as image:
                if image.n_frames != 250 or image.size != (1600, 800):
                    raise ValueError('GIF frame count or dimensions differ')
                for t in range(250):
                    image.seek(t); image.load()
                    if image.info.get('duration') != duration:
                        raise ValueError('GIF frame timing differs')
                    if t in [0, 62, 125, 187, 249]:
                        pixels = np.asarray(image.convert('RGB'))
                        if duration == 40:
                            png = output / f"{case['case']}_frame{case['start'] + t:04d}.png"
                            with Image.open(png) as original:
                                expected = np.asarray(original.convert('RGB').quantize(colors=128).convert('RGB'))
                            if not np.array_equal(pixels, expected):
                                raise ValueError('Decoded GIF differs from the corresponding rendered PNG')
                            decoded_selected[t] = pixels.copy()
                        elif not np.array_equal(pixels, decoded_selected[t]):
                            raise ValueError('Slow playback changed the displayed pose/frame')
            gifs.append({'file': file.name, 'frames': 250, 'duration_ms_per_frame': duration,
                         'total_duration_s': 250 * duration / 1000, 'all_frames_decoded': True})
        checks.append({'case': case['case'], 'mask': case['mask'], 'source_frames': [case['start'], case['stop']],
                       'poses_bitwise_equal_to_original_caches': True, 'timeline_pairing': 'exact',
                       'gt_matches_processed_recording': True, 'jerk_stencils': 247})
    for file, digest in p['protected_files'].items():
        if sha256(file) != digest:
            raise ValueError('Protected file changed during audit')
    result = {'status': 'passed', 'artifact_files_verified': len(manifest), 'cases': checks, 'gifs': gifs,
              'unique_source_frames': 750, 'all_panel_joint_frames': 750 * 4 * 24,
              'independent_fk_max_difference_m': max_joint_diff, 'independent_clip_summary_max_difference': max_stat_diff,
              'display_transform_is_proper_rotation': True, 'protected_files_unchanged': True,
              'network_forward_calls': 0, 'optimizer_steps': 0,
              'scope': 'Artifact identity, numerical and encoded timing checks; visual interpretation recorded separately'}
    target = output / 'independent_validation.json'
    if target.exists():
        raise FileExistsError('Do not overwrite previous validation')
    target.write_text(json.dumps(result, indent=2) + '\n')
    files = {str(f.relative_to(output)): sha256(f) for f in sorted(output.iterdir()) if f.is_file()}
    (output / 'VALIDATION_MANIFEST.json').write_text(json.dumps(files, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(2); audit(args.output.resolve())
