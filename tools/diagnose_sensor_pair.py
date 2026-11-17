"""GT-assisted upstream-rotation diagnostic and local source/provenance audit (no training)."""
import argparse
import csv
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import imuposer
from imuposer.datasets.globalModelDataset import GlobalModelDataset
from imuposer.datasets import utils as dataset_utils
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
from imuposer.config import amass_combos

from runtime import load_body_model, pose_rotations, sha256
from sensor_metrics import COMBOS, REGIONS, weighted_mean


def main(args):
    run = Path(args.run_dir)
    out = run / 'diagnostics'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    provenance = json.loads((run / 'provenance.json').read_text())
    body = load_body_model(provenance['smpl']['path'])
    # Audit the actual imported package, which may be an editable install of the original checkout.
    package = Path(imuposer.__file__).resolve().parent
    source_files = {str(p): sha256(p) for p in [
        package / 'config.py', Path(inspect.getfile(GlobalModelDataset)),
        Path(inspect.getfile(dataset_utils)), Path(inspect.getfile(IMUPoserModel)),
        package / 'models/LSTMs/RNN.py', package / 'smpl/parametricModel.py',
    ]}
    training_root = package.parents[1]
    for relative in ['scripts/2. Train/1. Train Global Model.py',
                     'scripts/1. Preprocessing/1. preprocess_all.py',
                     'scripts/1. Preprocessing/2. preprocess_all_to_imuposer_at_25fps.py']:
        path = training_root / relative
        source_files[str(path)] = sha256(path)
    audit = {'actual_imported_package': str(package), 'actual_imported_source_sha256': source_files,
        'original_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=training_root, text=True).strip(),
        'sensor_configurations': amass_combos,
        'training_implementation_facts': [
            'GlobalModelDataset selects processed files whose names do not contain dip for training.',
            'Every source motion is expanded into all 25 masks, once per combination and window, before the random train/validation split.',
            'get_dataset uses random_split with train_pct=0.9; train DataLoader uses shuffle=True and no custom sampler.',
            'For a complete pre-split dataset each mask has exactly equal sample and valid-frame exposure; after splitting the exact counts need retained split indices or reconstruction.',
            'Training entry sets seed_everything(42, workers=True). Run metadata records --combo_id global and --num_workers 0.',
            'combo_id global selects the 60-dimensional model input; it does not restrict GlobalModelDataset to the global mask.',
        ],
        'historical_uncertainties': [
            'No archived dataloader indices, per-mask batch counts, or complete source snapshot was found in the run files.',
            'Current source plus checkpoint/run metadata support balanced enumeration; exact historical per-mask training exposure is not independently established.',
            'The random sample split can share motion windows across train/validation masks; validation is not a held-out-motion protocol.',
        ],
        'input_mapping': {'processed_indices': ['left forearm/wrist proxy', 'right forearm/wrist proxy', 'left thigh/pocket proxy', 'right thigh/pocket proxy', 'head'],
            'raw_dip_indices_in_preprocessor': [7, 8, 9, 10, 0, 2],
            'sixth_input_discarded': True, 'gt_segments_for_orientation_proxy': [18, 19, 1, 2, 15],
            'warning': 'DIP segment IMUs are proxies for consumer-device locations; pose ground truth is IMU-derived, not independent optical ground truth.'}}
    (out / 'training_source_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    rows = []
    with np.load(run / 'predictions.npz', allow_pickle=False) as data, torch.inference_mode():
        for idx, (seq, start, frames) in enumerate(zip(data['sequence_ids'], data['start_frames'], data['frames'])):
            sl = slice(data['offsets'][idx], data['offsets'][idx + 1])
            gt = pose_rotations(torch.from_numpy(data['target_pose'][sl]))
            _, gt_pos = body.forward_kinematics(gt, calc_mesh=False)
            for combo in COMBOS:
                pred = pose_rotations(torch.from_numpy(data[f'pose_{combo}'][sl]))
                for correction, ids in [('none', []), ('gt_root', [0]), ('gt_root_torso', REGIONS['torso'])]:
                    corrected = pred.clone()
                    corrected[:, ids] = gt[:, ids]
                    _, pos = body.forward_kinematics(corrected, calc_mesh=False)
                    error = (pos - gt_pos).norm(dim=-1) * 100
                    rows.append({'sequence_id': int(seq), 'start_frame': int(start), 'frames': int(frames),
                        'combo': combo, 'correction': correction, 'mpjpe_cm': error.mean().item(),
                        'arms_cm': error[:, REGIONS['arms']].mean().item(),
                        'right_arm_cm': error[:, REGIONS['right_arm']].mean().item()})
    with (out / 'upstream_rotation_per_window.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for combo in COMBOS:
        for correction in ['none', 'gt_root', 'gt_root_torso']:
            subset = [r for r in rows if r['combo'] == combo and r['correction'] == correction]
            summary.append({'combo': combo, 'correction': correction,
                **{k: float(weighted_mean([r[k] for r in subset], [r['frames'] for r in subset])) for k in ['mpjpe_cm', 'arms_cm', 'right_arm_cm']}})
    (out / 'upstream_rotation_summary.json').write_text(json.dumps({
        'warning': 'GT-assisted diagnostic, not an achievable inference result or additive causal attribution. Corrected root/torso rotations are replaced with GT; arm-local rotations are unchanged.',
        'corrected_joint_ids': {'gt_root': [0], 'gt_root_torso': REGIONS['torso']},
        'overall': summary,
        'focus_cases': [r for r in rows if (r['sequence_id'], r['start_frame']) in [(18, 750), (8, 625), (8, 750)]]}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    main(parser.parse_args())
