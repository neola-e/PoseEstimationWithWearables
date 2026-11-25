"""Render audited IK results and the explicit paper-convention reconciliation."""
import argparse
import csv
import json
import math
from pathlib import Path
MODELS = ('network_final', 'network_final_ik')
METRICS = [('mpjre_deg', 'MPJRE', '°'), ('mpjpe_cm', 'MPJPE (joints)', 'cm'), ('mpjve_cm', 'MPJVE (mesh vertices)', 'cm'), ('jitter_100_m_s3', 'Jitter', '100 m/s³')]

def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def render(run):
    p = json.loads((run / 'PROTOCOL.json').read_text())
    r = json.loads((run / 'metrics.json').read_text())
    validation = json.loads((run / 'independent_validation.json').read_text())
    if not validation['passed'] or r['coverage']['pairs'] != 456:
        raise ValueError('Complete audited run required')
    rows = [json.loads(f.read_text()) for f in sorted((run / 'rows').glob('*.json'))]
    summary = []
    for key, label, unit in METRICS:
        deltas = [r['per_mask'][mask][MODELS[1]]['metrics']['frame_weighted'][key] - r['per_mask'][mask][MODELS[0]]['metrics']['frame_weighted'][key] for mask in p['mask_order']]
        summary.append({'metric': key, 'label': label, 'unit': unit, 'raw': r['overall_mask_equal'][MODELS[0]][key], 'ik': r['overall_mask_equal'][MODELS[1]][key], **r['changes'][key], 'masks_lower': sum((d < 0 for d in deltas)), 'masks_higher': sum((d > 0 for d in deltas)), 'masks_equal': sum((d == 0 for d in deltas))})
    subjects = {}
    for subject in p['data']['subjects']:
        ids = {rec['sequence_id'] for rec in p['records'] if rec['subject'] == subject}
        subjects[subject] = {}
        for model in MODELS:
            subjects[subject][model] = {}
            for key, _, _ in METRICS:
                count = 'jerk_samples' if key == 'jitter_100_m_s3' else 'frames'
                means = []
                for mask in p['mask_order']:
                    selected = [row['models'][model] for row in rows if row['mask'] == mask and row['sequence_id'] in ids]
                    means.append(math.fsum((v['metrics'][key] * v[count] for v in selected)) / sum((v[count] for v in selected)))
                subjects[subject][model][key] = math.fsum(means) / 24
    subject_equal = {model: {key: math.fsum((sub[model][key] for sub in subjects.values())) / len(subjects) for key, _, _ in METRICS} for model in MODELS}
    reductions = {'primary_24_masks_frame_weighted': r['overall_mask_equal'], 'aux_24_masks_sequence_equal': r['overall_sequence_equal'], 'aux_12_groups_equal': r['overall_12_groups_equal'], 'aux_24_masks_subject_equal': subject_equal}
    imu = {}
    for key in ['orientation_mse_before', 'orientation_mse_after', 'orientation_deg_before', 'orientation_deg_after']:
        imu[key] = math.fsum((row['ik'][key] * row['ik']['frames'] * len(row['ik']['sensors']) for row in rows)) / sum((row['ik']['frames'] * len(row['ik']['sensors']) for row in rows))
    imu['imu_mse_increased_frames'] = sum((row['ik']['imu_mse_increased_frames'] for row in rows))
    imu['mask_frames'] = sum((rec['frames'] for rec in p['records'])) * 24
    imu['projection_frobenius_max'] = max((row['ik']['projection_frobenius_max'] for row in rows))
    imu['unoptimized_local_parameters_bitwise_unchanged'] = True
    write_csv(run / 'changes.csv', list(summary[0]), summary)
    write_csv(run / 'per_subject.csv', ['subject', 'model', *[m[0] for m in METRICS]], [{'subject': subject, 'model': model, **values} for subject, models in subjects.items() for model, values in models.items()])
    write_csv(run / 'aggregation_sensitivity.csv', ['reduction', 'model', *[m[0] for m in METRICS]], [{'reduction': name, 'model': model, **values} for name, models in reductions.items() for model, values in models.items()])
    alignment = [('Device configurations', '24 location combinations;12 display groups', 'All24 masks;collapse by wrist/pocket/head counts into12 groups', 'Verified', 'Three single-location,five double,four triple classes;overall weighting separate'), ('DIP split', 'First8 train,last2 test', 'Final s01–s08,two epochs162 updates;s09/s10 locked reporting only', 'Verified', 'Budget selected on development,not these test scores'), ('Sampling/scaling', '25Hz;acc/30;global orientations', 'Original25Hz preprocessing,60D channels,missing-zero', 'Verified', 'Entrywise matrix interpolation;separate IK projection'), ('Wrist orientation mapping', 'Wrist/hand devices;optimize shoulders/elbows', 'Public synthesis ji_mask constrains forearms18/19,optimizes16/18 or17/19', 'Source supported', 'Retain raw DIP [7,8,9,10,0];training-domain check without GT fitting about1.4–3.4 degrees,head0.37'), ('Head/pocket DOFs', 'Analogous head/hip refinement;full indices not listed', 'Head12/15;left pocket0/1;right pocket0/2', 'Reconstruction convention', 'Exact author index lists unavailable'), ('Objective/budget', 'Global IMU orientation,MSE,Adam,10 steps/frame', 'Available orientation-matrix MSE only,10 steps,joint constraints', 'Implemented and verified', 'No GT,positions,future predictions or hidden sensors'), ('IK learning rate', 'Not specified in IMUPoser;cites AvatarPoser', 'Fixed1e-3 from cited paper', 'Reconstruction convention', 'AvatarPoser public default1e-1/five steps,position objective;not IMUPoser implementation'), ('Rotation parameterization', 'Unspecified', 'Local right-multiplicative rotvec increments,FP64 optimize,FP32 output', 'Reconstruction convention', 'Zero increments,SO(3);unoptimized local entries retained'), ('Output/state', 'Details unspecified', 'Independent Adam/frame;post-update10 output,no fallback/best iterate', 'Reconstruction convention', 'Actual-data batch1/256 recheck;no added temporal filter'), ('Online window', '125 history;also1-sample overlap', 'F03 stride1,last output,short-prefix startup', 'Author details unresolved', 'Public evaluation loop unavailable;literal overlap not proof of stride1'), ('Preprocessing future access', 'Five-frame mean,direction unspecified', 'Retain centered acc mean,inherited filling/interpolation', 'Disclosed difference', 'Two-frame/80ms acceleration lookahead;IK adds no future access'), ('MPJRE', 'Global joint-rotation error,degrees', '24-joint FK global geodesic;local error separate', 'Verified', 'No root-rotation alignment'), ('MPJPE', 'All-joint cm error,pelvis aligned', '24 joints,pelvis translation alignment,root rotation retained', 'Verified', 'Not Procrustes alignment'), ('MPJVE', 'All SMPL vertex cm error,pelvis aligned', '6,890 vertices;separate from joint velocity', 'Verified', 'Cited AvatarPoser MPJVE means velocity;that definition not imported'), ('Jitter', 'Mean predicted jerk;table100m/s³', '25³ third differences,within-recording T−3 weights,divide100', 'Verified', 'No GT-jerk subtraction or cross-recording differences'), ('Body shape', 'Main text does not specify betas', 'Fixed mean male beta0,same as F03', 'Author details unresolved', 'Public processed DIP stores all1;does not replace validated primary setting'), ('Mesh/translation', 'SMPL;no global translation prediction', 'Inherited FK default pose-blendshape off,zero trajectory', 'Source convention', 'Author measurement switch not fully published'), ('Aggregation', 'Overall mean across configurations;12 display groups', 'Primary frame/T−3 within masks,then equal24 masks', 'Author details unresolved', 'Also report sequence-equal,12-group-equal,subject-equal sensitivity'), ('AMASS recipe', 'Described63h,24 masks,80 epochs', 'Historical Epoch59 from25 masks,audited corpus/early stop', 'Known training difference', 'No AMASS retraining here;strict recipe separately planned'), ('DIP epoch selection/padding', 'Internal selection/tail rules incomplete', 'Frozen development K2;final restarts Epoch59;retain training padding loss', 'Explicit convention', 'Evaluation excludes padding;test scores do not change training'), ('Consumer/live system', 'Separate consumer dataset/hardware demonstration', 'This task only DIP offline metrics/IK', 'Phase2', 'CPU batch throughput is not device latency or consumer accuracy')]
    write_csv(run / 'paper_alignment.csv', ['item', 'paper', 'current', 'status', 'residual'], [dict(zip(['item', 'paper', 'current', 'status', 'residual'], row)) for row in alignment])
    report = {'primary': summary, 'per_subject': subjects, 'aggregation_sensitivity': reductions, 'imu_constraint_diagnostics': imu, 'paper_table2_reference': p['paper_table2_reference'], 'paper_table2_differences_after_ik': {key: r['overall_mask_equal'][MODELS[1]][key] - p['paper_table2_reference'][key] for key, _, _ in METRICS}, 'strict_reproduction': False, 'interpretation': 'Paper-guided reconstructed orientation IK; unknown original settings remain explicit, no test tuning.'}
    (run / 'report_summary.json').write_text(json.dumps(report, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.8), constrained_layout=True)
    for ax, row, title in zip(axes, summary, ['MPJRE', 'MPJPE', 'MPJVE (vertices)', 'Jitter']):
        bars = ax.bar([0, 1], [row['raw'], row['ik']], color=['#66788a', '#1b8d88'], width=0.62)
        ax.set_xticks([0, 1], ['No IK', 'IMU-only IK'])
        ax.set_ylabel(row['unit'])
        ax.set_title(title)
        ax.set_ylim(0, max(row['raw'], row['ik']) * 1.25)
        ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Final DIP model: fixed 10-step IMU-only IK, all24 masks')
    fig.savefig(run / 'ik_four_metrics.png', dpi=180)
    fig.savefig(run / 'ik_four_metrics.pdf')
    plt.close(fig)
    lines = ['# IMU-only IK: final DIP model comparison', '', 'Same final two-epoch weights,s09/s10,19 recordings,24 masks.', '', '| Metric | No IK | IMU-only IK | Relative decrease | Masks lower |', '|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['label']} ({row['unit']}) | {row['raw']:.5f} | {row['ik']:.5f} | {row['relative_decrease_percent']:+.2f}% | {row['masks_lower']}/24 |")
    lines.extend(['', 'Positive means decrease,negative increase. Jitter describes smoothness;MPJVE always means mesh-vertex position error.', '', 'Ten Adam steps,lr1e-3,orientation-matrix MSE;no GT constraints/calibration,weight changes or test tuning. Head/hip indices,learning rate and rotation parameterization are frozen reconstruction conventions,not a recovered line-by-line author implementation.', '', '![IK metrics](ik_four_metrics.png)', ''])
    (run / 'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    render(parser.parse_args().run)
