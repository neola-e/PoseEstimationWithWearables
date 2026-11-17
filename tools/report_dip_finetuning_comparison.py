"""Render an audited, complete DIP before/after comparison without inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS = ['amass_epoch59', 'dip_final_2ep']
METRICS = [
    ('mpjre_deg', 'MPJRE', 'deg'),
    ('mpjpe_cm', 'MPJPE', 'cm'),
    ('mpjve_cm', 'MPJVE', 'cm'),
    ('jitter_100_m_s3', 'Jitter', '100 m/s³'),
]


def render(run):
    protocol = json.loads((run/'PROTOCOL.json').read_text())
    result = json.loads((run/'metrics.json').read_text())
    validation = json.loads((run/'independent_validation.json').read_text())
    if not validation['passed'] or result['coverage']['pairs'] != 456:
        raise ValueError('A complete, independently audited comparison is required')
    if validation['independent_aggregate'] != result['overall_mask_equal']:
        # Independent summation can differ at floating-point roundoff.
        for model in MODELS:
            for key, _, _ in METRICS:
                if abs(validation['independent_aggregate'][model][key] - result['overall_mask_equal'][model][key]) > 1e-10:
                    raise ValueError('Validation and report metrics differ')
    masks = protocol['mask_order']
    fields = ['metric', 'unit', 'before', 'after', 'after_minus_before',
              'improvement_percent', 'masks_improved', 'masks_worsened', 'masks_equal']
    rows = []
    subject_rows = []
    pair_rows = [json.loads((run/'rows'/f'{mask}_seq{rec["sequence_id"]:02d}.json').read_text())
                 for mask in masks for rec in protocol['records']]
    for key, label, unit in METRICS:
        differences = [result['per_mask'][mask][MODELS[1]]['metrics']['frame_weighted'][key]
                       - result['per_mask'][mask][MODELS[0]]['metrics']['frame_weighted'][key]
                       for mask in masks]
        rows.append({'metric': label, 'unit': unit,
                     'before': result['overall_mask_equal'][MODELS[0]][key],
                     'after': result['overall_mask_equal'][MODELS[1]][key],
                     **result['changes'][key],
                     'masks_improved': sum(value < 0 for value in differences),
                     'masks_worsened': sum(value > 0 for value in differences),
                     'masks_equal': sum(value == 0 for value in differences)})
    for subject in protocol['data']['subjects']:
        ids = {rec['sequence_id'] for rec in protocol['records'] if rec['subject'] == subject}
        for model in MODELS:
            values = {}
            for key, _, _ in METRICS:
                weight = 'jerk_samples' if key == 'jitter_100_m_s3' else 'frames'
                mask_means = []
                for mask in masks:
                    selected = [row['models'][model] for row in pair_rows
                                if row['mask'] == mask and row['sequence_id'] in ids]
                    mask_means.append(sum(row['metrics'][key]*row[weight] for row in selected)
                                      / sum(row[weight] for row in selected))
                values[key] = sum(mask_means)/len(mask_means)
            subject_rows.append({'subject': subject, 'model': model, **values})
    with (run/'comparison_with_changes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (run/'per_subject.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['subject', 'model', *[key for key, _, _ in METRICS]])
        writer.writeheader()
        writer.writerows(subject_rows)
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.7), constrained_layout=True)
    for ax, row in zip(axes, rows):
        bars = ax.bar([0, 1], [row['before'], row['after']], color=['#66788a', '#1b8d88'], width=.62)
        ax.set_xticks([0, 1], ['AMASS\nEpoch 59', 'DIP\n2 epochs'])
        ax.set_title(row['metric'])
        ax.set_ylabel(row['unit'])
        ax.set_ylim(0, max(row['before'], row['after'])*1.24)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('DIP test: s09/s10, all 24 masks; lower is better')
    fig.savefig(run/'four_metrics.png', dpi=180)
    fig.savefig(run/'four_metrics.pdf')
    plt.close(fig)
    lines = ['# DIP 微调前后四指标', '',
             '测试集 s09/s10，共 19 条录制、24,078 帧，全部 24 个 mask。'
             '每个 mask 内空间误差按帧数加权，jitter 按 T−3 加权，再对 mask 等权平均。', '',
             '| 指标 | 微调前 | 两轮微调后 | 相对改善 | 改善组合数 |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["metric"]} ({row["unit"]}) | {row["before"]:.5f} | {row["after"]:.5f} '
                     f'| {row["improvement_percent"]:+.2f}% | {row["masks_improved"]}/24 |')
    lines.extend(['', '相对改善为 (前−后)/前×100%；正值为下降。MPJVE 是网格顶点位置误差。'
                  'Jitter 是预测关节位置的三阶差分，表格单位 100 m/s³。', '',
                  f'推理设备：{protocol["inference"].get("execution_device", "cuda:0")}。'
                  '125 帧滚动窗口、步长 1、取最后有效帧、前缀起步；保持原滤波，无 IK。', '',
                  '独立核查通过：456 个配对缓存及所有 GT，1,155,744 个预测帧；'
                  '独立重算汇总和固定 8 次 FK 分块重算通过。', '',
                  '本结果是固定模型的描述性测试比较；两名测试者不足以支撑统计显著性结论。'
                  '尚存 AMASS 25-mask 初始化、窗口边界/未来滤波与无 IK 等论文口径差异。', '',
                  '![Four metrics](four_metrics.png)', ''])
    (run/'RESULTS.md').write_text('\n'.join(lines))
    (run/'report_summary.json').write_text(json.dumps({'metrics': rows, 'per_subject': subject_rows}, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    render(parser.parse_args().run)
