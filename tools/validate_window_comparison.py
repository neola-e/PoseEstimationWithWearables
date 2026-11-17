"""Independently audit saved per-frame intervals and legacy chunk equivalence.

Does not use window_plan/dependency_trace/infer_sequence. Run after comparison.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from runtime import sha256


def validate(run, legacy_cache):
    protocol = json.loads((run / 'PROTOCOL.json').read_text())
    manifest = json.loads((run / 'ARTIFACT_MANIFEST.json').read_text())
    for relative, digest in manifest.items():
        if sha256(run / relative) != digest:
            raise ValueError(f'Artifact hash mismatch: {relative}')
    with np.load(legacy_cache, allow_pickle=False) as cache:
        legacy = {key: cache[key] for key in cache.files}
    checked, per_sequence, future = 0, [], {}
    names = ['chunk_centered', 'rolling_centered', 'rolling_trailing']
    for row in protocol['selected']:
        seq, n = row['sequence_id'], row['frames']
        windows = np.flatnonzero(legacy['sequence_ids'] == seq)
        if len(windows) == 0:
            raise ValueError('Legacy cache is missing a sequence')
        order = sorted(windows, key=lambda w: int(legacy['start_frames'][w]))
        p, t, cursor = [], [], 0
        for w in order:
            if int(legacy['start_frames'][w]) != cursor:
                raise ValueError('Legacy windows are not contiguous')
            a, b = legacy['offsets'][w:w+2]
            p.append(legacy['pose_lw_rw_h'][a:b])
            t.append(legacy['target_pose'][a:b])
            cursor += b-a
        if cursor != n:
            raise ValueError('Legacy coverage differs')
        with np.load(run / f'seq{seq}_predictions.npz', allow_pickle=False) as cache:
            if not np.array_equal(cache['target_pose'], np.concatenate(t)):
                raise ValueError('Paired targets differ from frozen E04 baseline')
            error = float(np.abs(cache['pose_chunk_centered']-np.concatenate(p)).max())
            if error > 2e-5:
                raise ValueError('Recomputed chunks differ from frozen E04 predictions')
        per_sequence.append({'sequence_id': seq, 'frames': n, 'legacy_chunk_max_abs_6d': error})
        for name in names:
            path = run / f'seq{seq}_{name}_trace.csv'
            with path.open() as stream:
                rows = [{k:int(v) for k,v in r.items()} for r in csv.DictReader(stream)]
            if len(rows) != n:
                raise ValueError('Trace row count mismatch')
            counters = future.setdefault(name, {'outputs':0, 'model_future_outputs':0,
                                                'pre_smoothing_future_outputs':0,
                                                'max_model_future_frames':0,
                                                'max_pre_smoothing_future_frames':0})
            for frame, record in enumerate(rows):
                if name == 'chunk_centered':
                    start, stop = frame//125*125, min(frame//125*125+125,n)
                else:
                    start, stop = max(0,frame-124), frame+1
                left, right = (4,0) if name == 'rolling_trailing' else (2,2)
                acc_start, acc_stop = max(0,start-left), min(n,stop+right)
                expected = {'output_frame':frame, 'input_start':start, 'input_stop':stop,
                            'model_length':stop-start, 'model_output_index':frame-start,
                            'left_repeats_of_frame_zero':0,
                            'acc_source_start':acc_start, 'acc_source_stop':acc_stop,
                            'ori_source_start':start, 'ori_source_stop':stop,
                            'available_at_25hz_frame':max(stop,acc_stop)-1,
                            'future_frames_at_25hz':max(stop,acc_stop)-1-frame,
                            'window_id':frame//125 if name == 'chunk_centered' else frame}
                if record != expected:
                    raise ValueError(f'Trace mismatch at {seq}/{name}/{frame}')
                checked += 1
                counters['outputs'] += 1
                model_future, preprocessing_future = stop-1-frame, max(stop,acc_stop)-1-frame
                counters['model_future_outputs'] += int(model_future > 0)
                counters['pre_smoothing_future_outputs'] += int(preprocessing_future > 0)
                counters['max_model_future_frames'] = max(counters['max_model_future_frames'],model_future)
                counters['max_pre_smoothing_future_frames'] = max(counters['max_pre_smoothing_future_frames'],preprocessing_future)
    return {'passed':True, 'legacy_cache':str(legacy_cache), 'legacy_cache_sha256':sha256(legacy_cache),
            'original_run_artifact_hashes_checked':len(manifest), 'trace_rows_checked':checked,
            'interval_convention':'zero-based half-open, at pre-smoothing 25 Hz stage',
            'per_sequence':per_sequence, 'future_dependencies':future}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--legacy-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = validate(args.run,args.legacy_cache)
    result['validator_sha256'] = sha256(__file__)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
