"""Compare inference semantics on reserved DIP development recordings, not test.

Freeze PROTOCOL.json before prediction. Save paired predictions, per-output
dependencies, four metrics, independent single-window checks, and source hashes.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np
import torch
import imuposer
from imuposer import math
from imuposer.config import amass_combos

from eval_wrist_pocket_dev import fingerprint, resample
from paper_metrics import PoseMetricEvaluator, aggregate_sequences, rotation_angles_deg
from runtime import PosePredictor, load_body_model, load_torch, peak_ram_mib, pose_rotations, sha256
from window_inference import dependency_trace, infer_sequence, smooth_acceleration

VARIANTS = {'chunk_centered': ('chunk', 'centered'),
            'rolling_centered': ('rolling_last', 'centered'),
            'rolling_trailing': ('rolling_last', 'trailing')}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unsmoothed_acc(row):
    """Inherited raw loading/filling/cropping/resampling, BEFORE acceleration mean.

    Trailing smoothing is causal at the resampled 25 Hz stage only. The inherited
    backward NaN filling and 60-to-25 Hz interpolation are separately disclosed.
    """
    path = Path(row['raw_path'])
    if sha256(path) != row['raw_sha256']:
        raise ValueError('Raw recording changed')
    with path.open('rb') as stream:
        raw = pickle.load(stream, encoding='latin1')
    acc = torch.from_numpy(raw['imu_acc'][:, [7, 8, 9, 10, 0, 2]]).float()
    nan_count = int(torch.isnan(acc).sum())
    for _ in range(4):
        acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
        acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
    acc = resample(acc[6:-6])
    if not torch.isfinite(acc).all():
        raise ValueError('Raw recording failed inherited preprocessing')
    return acc, nan_count


def inputs(acc, ori, sensors, scale):
    x = torch.cat([(acc[:, :5] / scale).flatten(1), ori[:, :5].float().flatten(1)], 1)
    mask = torch.zeros(60, dtype=torch.bool)
    mask[:15].view(5, 3)[sensors] = True
    mask[15:].view(5, 9)[sensors] = True
    x[:, ~mask] = 0
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--smpl-model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--combo', choices=list(amass_combos), default='lw_rw_h')
    p.add_argument('--sequences', type=int, nargs='+', help='Default: all 11 reserved recordings')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=16)
    args = p.parse_args()
    if Path(args.data).name != 'dip_train.pt':
        raise ValueError('This development comparison never loads test data')
    manifest = json.loads(Path(args.manifest).read_text())
    if sha256(args.data) != manifest['data']['sha256']:
        raise ValueError('Development container changed')
    selected = manifest['selected']
    if args.sequences is not None:
        if len(set(args.sequences)) != len(args.sequences):
            raise ValueError('Duplicate sequence IDs')
        selected = [r for r in selected if r['sequence_id'] in args.sequences]
        if len(selected) != len(args.sequences):
            raise ValueError('Only reserved development recordings may be selected')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    predictor = PosePredictor(args.checkpoint, args.device, args.threads)
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
                'checkpoint': {'path': str(Path(args.checkpoint).resolve()), 'sha256': sha256(args.checkpoint),
                               'epoch': predictor.epoch, 'global_step': predictor.global_step},
                'data': {'path': str(Path(args.data).resolve()), 'sha256': sha256(args.data)},
                'split_manifest_sha256': sha256(args.manifest), 'selected': selected,
                'scope': 'DIP development s_07/s_08; one fixed mask; AMASS initialization; no fine-tuning or IK',
                'excluded_subjects': ['s_09', 's_10'], 'sensors': amass_combos[args.combo], 'combo': args.combo,
                'fps': 25, 'window': 125, 'rolling_stride': 1, 'rolling_overlap_after_warmup': 124,
                'rolling_output': 'last valid frame only', 'startup': 'prefix, length 1..125; no repetition or future padding',
                'state': 'reset both LSTM directions for every window and recording',
                'variants': VARIANTS, 'orientation_filter': 'none; inherited elementwise interpolation retained',
                'smoothing': '5-frame acc mean; shortened edges; trailing uses [t-4,t], centered uses [t-2,t+2]',
                'causality_scope': 'rolling_trailing is causal AFTER resampling/imputation; not raw-stream certified',
                'paper_ambiguity': '1-sample overlap not silently equated to stride=1; stride=1 is an explicit interpretation of online history',
                'paper': 'https://arxiv.org/html/2304.12518v1',
                'metric_protocol': 'POSE_METRICS.md; 24 joints/6890 vertices; mean male shape; pose blend shapes off; zero translation',
                'aggregation': 'spatial frame weighted and sequence equal; jerk weighted by T-3, never across recordings',
                'smpl_model_sha256': sha256(args.smpl_model), 'device': str(predictor.device),
                'batch_size': args.batch_size, 'threads': args.threads,
                'batching': 'independent windows only; throughput timing is not live end-to-end latency',
                'torch': torch.__version__, 'imported_imuposer': imuposer.__file__}
    dump(out / 'PROTOCOL.json', protocol)
    sources = [Path(__file__), Path(__file__).with_name('window_inference.py'),
               Path(__file__).with_name('runtime.py'), Path(__file__).with_name('paper_metrics.py'),
               Path(__file__).with_name('eval_wrist_pocket_dev.py'),
               Path(__file__).parents[1] / 'tests/test_window_inference.py']
    package = Path(imuposer.__file__).parent
    sources += list(package.rglob('*.py'))
    source_hashes = {}
    for source in sources:
        rel = Path('installed_imuposer') / source.relative_to(package) if source.is_relative_to(package) else Path('team') / source.name
        dest = out / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        source_hashes[str(source)] = sha256(source)
    dump(out / 'SOURCE_HASHES.json', source_hashes)
    body = load_body_model(args.smpl_model)
    evaluator = PoseMetricEvaluator(body)
    data = load_torch(args.data)
    reports = {name: [] for name in VARIANTS}
    validation, timings, pairwise = [], [], []
    started = time.perf_counter()
    for row in selected:
        seq = row['sequence_id']
        n = row['frames']
        for field, digest in row['processed_sha256'].items():
            if fingerprint(data[field][seq]) != digest:
                raise ValueError(f'Sequence {seq} {field} changed')
        acc, ori, pose = [data[k][seq] for k in ['acc', 'ori', 'pose']]
        raw_acc, raw_nans = unsmoothed_acc(row)
        if not torch.equal(smooth_acceleration(raw_acc), acc):
            raise ValueError('Centered raw reconstruction differs from frozen processed input')
        target = math.rotation_matrix_to_r6d(pose).reshape(n, 144)
        predictions = {}
        for name, (mode, smoothing) in VARIANTS.items():
            x = inputs(acc if smoothing == 'centered' else smooth_acceleration(raw_acc, smoothing),
                       ori, amass_combos[args.combo], predictor.training_config.acc_scale)
            tick = time.perf_counter()
            def progress(done, total):
                if done % 512 == 0 or done == total:
                    print(f'seq={seq} {name}: {done}/{total} windows; {time.perf_counter()-tick:.1f}s', flush=True)
            prediction = infer_sequence(predictor, x, mode, batch_size=args.batch_size, progress=progress)
            seconds = time.perf_counter() - tick
            predictions[name] = prediction
            trace = dependency_trace(n, mode, smoothing=smoothing)
            with (out / f'seq{seq}_{name}_trace.csv').open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                writer.writeheader()
                writer.writerows(trace)
            checks = []
            for t in sorted({0, 1, min(123, n-1), min(124, n-1), min(125, n-1), min(249, n-1), n-1}):
                record = trace[t]
                single = predictor.predict(x[record['input_start']:record['input_stop']])
                expected = single[record['model_output_index']]
                error = (prediction[t] - expected).abs().max().item()
                if error > 2e-5:
                    raise ValueError('Trace reconstruction/single-window batching check failed')
                checks.append({'frame': t, 'max_abs_6d_difference': error})
            validation.append({'sequence_id': seq, 'variant': name, 'outputs': n,
                               'source_reconstruction_bitwise_equal': True,
                               'raw_acc_nan_values_before_imputation': raw_nans,
                               'single_window_checks': checks,
                               'model_future_outputs': sum(r['input_stop']-1 > r['output_frame'] for r in trace),
                               'pre_smoothing_future_outputs': sum(r['future_frames_at_25hz'] > 0 for r in trace)})
            metric = evaluator.evaluate_sequence(prediction, target, seq)
            metric.update(subject=row['subject'], source_file=row['source_file'])
            reports[name].append(metric)
            timings.append({'sequence_id': seq, 'variant': name, 'frames': n, 'inference_seconds': seconds,
                            'windows': (n+124)//125 if mode == 'chunk' else n})
            print(f'seq={seq} {name} four metrics: {json.dumps(metric["metrics"])}', flush=True)
        np.savez_compressed(out / f'seq{seq}_predictions.npz', target_pose=target.numpy(), fps=25,
                            **{f'pose_{k}': v.numpy() for k, v in predictions.items()})
        pairwise.append({'sequence_id': seq, 'frames': n,
                         'chunk_vs_rolling_mean_local_deg': rotation_angles_deg(
                             pose_rotations(predictions['chunk_centered']), pose_rotations(predictions['rolling_centered'])).mean().item(),
                         'rolling_centered_vs_trailing_mean_local_deg': rotation_angles_deg(
                             pose_rotations(predictions['rolling_centered']), pose_rotations(predictions['rolling_trailing'])).mean().item()})
        dump(out / 'PROGRESS.json', {'completed_sequences': len(pairwise), 'planned_sequences': len(selected),
                                    'elapsed_seconds': time.perf_counter()-started})
    summary = {name: aggregate_sequences(rows) for name, rows in reports.items()}
    deltas = {}
    for before, after in [('chunk_centered', 'rolling_centered'), ('rolling_centered', 'rolling_trailing')]:
        deltas[f'{after}_minus_{before}'] = {
            mode: {key: summary[after]['metrics'][mode][key] - value
                   for key, value in summary[before]['metrics'][mode].items() if value is not None}
            for mode in ['frame_weighted', 'sequence_equal']}
    dump(out / 'metrics.json', {'summary': summary, 'deltas': deltas, 'per_sequence': reports,
                                'prediction_differences': pairwise})
    dump(out / 'validation.json', {'passed': True, 'trace_single_window_atol_6d': 2e-5, 'checks': validation})
    dump(out / 'resources.json', {'elapsed_seconds': time.perf_counter()-started, 'peak_ram_mib': peak_ram_mib(),
                                 'timings': timings})
    for source, digest in source_hashes.items():
        if sha256(source) != digest:
            raise ValueError('Source changed during comparison')
    if sha256(args.checkpoint) != protocol['checkpoint']['sha256'] or sha256(args.data) != protocol['data']['sha256']:
        raise ValueError('Input asset changed during comparison')
    dump(out / 'ARTIFACT_MANIFEST.json', {str(f.relative_to(out)): sha256(f)
                                        for f in sorted(out.rglob('*')) if f.is_file()})
    print(json.dumps({'status': 'complete', 'output': str(out), 'summary': summary, 'deltas': deltas}), flush=True)


if __name__ == '__main__':
    main()
