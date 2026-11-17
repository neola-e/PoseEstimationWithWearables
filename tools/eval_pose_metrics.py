"""Compute all four pose metrics from existing NPZ predictions, without inference."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import imuposer

from imuposer.smpl.parametricModel import ParametricModel
from paper_metrics import PoseMetricEvaluator, aggregate_sequences, cache_sequences
from runtime import peak_ram_mib, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--pose-key', default='pose')
    parser.add_argument('--target-key', default='target_pose')
    parser.add_argument('--smpl-model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fps', type=float, default=25.)
    parser.add_argument('--fk-chunk', type=int, default=32)
    parser.add_argument('--pose-blendshape', action='store_true')
    parser.add_argument('--betas', type=float, nargs=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    torch.set_num_threads(2)
    input_hash, model_hash = sha256(args.input), sha256(args.smpl_model)
    started = time.perf_counter()
    with np.load(args.input, allow_pickle=False) as source:
        keys = [args.pose_key, args.target_key, 'offsets', 'lengths', 'sequence_ids', 'start_frames',
                'predicted_translation_m', 'target_translation_m']
        cache = {k: source[k] for k in keys if k in source.files}
        if 'fps' in source.files:
            fps = source['fps']
            if fps.ndim != 0 or float(fps) != args.fps:
                raise ValueError('Cache fps must be scalar and match --fps')
    sequences = cache_sequences(cache, args.pose_key, args.target_key)
    body = ParametricModel(str(args.smpl_model), use_pose_blendshape=args.pose_blendshape, device=torch.device('cpu'))
    evaluator = PoseMetricEvaluator(body, args.fps, args.fk_chunk)
    rows = []
    for sequence in sequences:
        windows = sequence.pop('source_windows')
        row = evaluator.evaluate_sequence(**sequence, shape=args.betas)
        row['source_windows'] = windows
        rows.append(row)
        print(f"Measured sequence {row['sequence_id']}: {row['frames']} frames, {row['jerk_samples']} jerk samples", flush=True)
    package = Path(imuposer.__file__).resolve().parent
    paths = [Path(__file__).resolve(), Path(__file__).with_name('paper_metrics.py'),
             Path(__file__).with_name('runtime.py'), package / 'smpl/parametricModel.py',
             package / 'math/angular.py', package / 'math/spatial.py', package / 'math/general.py']
    report = {**aggregate_sequences(rows), 'per_sequence': rows,
              'provenance': {'created_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
                             'input': {'path': str(args.input.resolve()), 'sha256': input_hash, 'pose_key': args.pose_key,
                                       'target_key': args.target_key},
                             'smpl': {'path': str(args.smpl_model.resolve()), 'sha256': model_hash},
                             'imuposer_import': str(Path(imuposer.__file__).resolve()),
                             'source_sha256': {str(p): sha256(p) for p in paths},
                             'software': {n: importlib.metadata.version(n) for n in ['torch', 'numpy']},
                             'fk_chunk': args.fk_chunk, 'shape_policy': 'shared fixed betas per pair; model mean when omitted',
                             'translation_policy': 'supplied meter translations or zero; retained for jitter',
                             'mesh_pose_blendshape_default_source': 'Inherited SMPL and TransPose evaluator default false; IMUPoser paper does not specify this flag',
                             'configuration_pooling': 'one supplied prediction configuration; no pooled device benchmark',
                             'training_performed': False, 'model_inference_performed': False},
              'seconds': time.perf_counter() - started, 'peak_process_rss_mib': peak_ram_mib(),
              'scope': 'Metric implementation check on supplied predictions; not a completed paper-protocol reproduction'}
    if sha256(args.input) != input_hash or sha256(args.smpl_model) != model_hash:
        raise ValueError('An input changed during measurement')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
