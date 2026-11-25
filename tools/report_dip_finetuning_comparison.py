"""Render an audited, complete DIP before/after comparison without inference."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODELS = ['amass_epoch59', 'dip_final_2ep']
METRICS = [('mpjre_deg', 'MPJRE', 'deg'), ('mpjpe_cm', 'MPJPE', 'cm'), ('mpjve_cm', 'MPJVE', 'cm'), ('jitter_100_m_s3', 'Jitter', '100 m/s³')]

def render(run):
    protocol = json.loads((run / 'PROTOCOL.json').read_text())
    result = json.loads((run / 'metrics.json').read_text())
    validation = json.loads((run / 'independent_validation.json').read_text())
    if not validation['passed'] or result['coverage']['pairs'] != 456:
        raise ValueError('A complete, independently audited comparison is required')
    if validation['independent_aggregate'] != result['overall_mask_equal']:
        for model in MODELS:
            for key, _, _ in METRICS:
                if abs(validation['independent_aggregate'][model][key] - result['overall_mask_equal'][model][key]) > 1e-10:
                    raise ValueError('Validation and report metrics differ')
    masks = protocol['mask_order']
    fields = ['metric', 'unit', 'before', 'after', 'after_minus_before', 'improvement_percent', 'masks_improved', 'masks_worsened', 'masks_equal']
    rows = []
    subject_rows = []
    pair_rows = [json.loads((run / 'rows' / f"{mask}_seq{rec['sequence_id']:02d}.json").read_text()) for mask in masks for rec in protocol['records']]
    for key, label, unit in METRICS:
        differences = [result['per_mask'][mask][MODELS[1]]['metrics']['frame_weighted'][key] - result['per_mask'][mask][MODELS[0]]['metrics']['frame_weighted'][key] for mask in masks]
        rows.append({'metric': label, 'unit': unit, 'before': result['overall_mask_equal'][MODELS[0]][key], 'after': result['overall_mask_equal'][MODELS[1]][key], **result['changes'][key], 'masks_improved': sum((value < 0 for value in differences)), 'masks_worsened': sum((value > 0 for value in differences)), 'masks_equal': sum((value == 0 for value in differences))})
    for subject in protocol['data']['subjects']:
        ids = {rec['sequence_id'] for rec in protocol['records'] if rec['subject'] == subject}
        for model in MODELS:
            values = {}
            for key, _, _ in METRICS:
                weight = 'jerk_samples' if key == 'jitter_100_m_s3' else 'frames'
                mask_means = []
                for mask in masks:
                    selected = [row['models'][model] for row in pair_rows if row['mask'] == mask and row['sequence_id'] in ids]
                    mask_means.append(sum((row['metrics'][key] * row[weight] for row in selected)) / sum((row[weight] for row in selected)))
                values[key] = sum(mask_means) / len(mask_means)
            subject_rows.append({'subject': subject, 'model': model, **values})
    with (run / 'comparison_with_changes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (run / 'per_subject.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['subject', 'model', *[key for key, _, _ in METRICS]])
        writer.writeheader()
        writer.writerows(subject_rows)
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.7), constrained_layout=True)
    for ax, row in zip(axes, rows):
        bars = ax.bar([0, 1], [row['before'], row['after']], color=['#66788a', '#1b8d88'], width=0.62)
        ax.set_xticks([0, 1], ['AMASS\nEpoch 59', 'DIP\n2 epochs'])
        ax.set_title(row['metric'])
        ax.set_ylabel(row['unit'])
        ax.set_ylim(0, max(row['before'], row['after']) * 1.24)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('DIP test: s09/s10, all 24 masks; lower is better')
    fig.savefig(run / 'four_metrics.png', dpi=180)
    fig.savefig(run / 'four_metrics.pdf')
    plt.close(fig)
    lines = ['# DIP fine-tuning: four metrics before and after', '', 'Test:s09/s10,19 recordings,24,078 frames,all24 masks. Within masks,frame-weight spatial errors and T−3-weight jitter;then equally average masks.', '', '| Metric | Before | After two epochs | Relative improvement | Improved masks |', '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['metric']} ({row['unit']}) | {row['before']:.5f} | {row['after']:.5f} | {row['improvement_percent']:+.2f}% | {row['masks_improved']}/24 |")
    lines.extend(['', 'Relative improvement=(before-after)/before×100%;positive means lower. MPJVE is mesh-vertex position error. Jitter is predicted-joint third difference,in100m/s³.', '', f"Inference device:{protocol['inference'].get('execution_device', 'cuda:0')}. Rolling125,stride1,last valid output,prefix startup;retain filtering,no IK.", '', 'Independent audit passes:456 paired caches/all GT,1,155,744 predicted frames;independent reduction/eight fixed FK-chunk rechecks pass.', '', 'Descriptive locked-model test comparison;two subjects do not establish significance. Historical25-mask initialization,window boundaries,future filtering and no IK remain paper-protocol differences.', '', '![Four metrics](four_metrics.png)', ''])
    (run / 'RESULTS.md').write_text('\n'.join(lines))
    (run / 'report_summary.json').write_text(json.dumps({'metrics': rows, 'per_subject': subject_rows}, indent=2) + '\n')
    print(json.dumps(rows, indent=2))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    render(parser.parse_args().run)
