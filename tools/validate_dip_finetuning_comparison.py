"""Audit full paired caches, independent reductions, and fixed FK-chunk rechecks.

No training or network inference. Reuses the canonical metric for fixed spot
rechecks, but independently reconstructs all aggregation and reference pairing.
"""
import argparse
import json
import math as scalar_math
from pathlib import Path

import numpy as np
import torch
from imuposer import math

from paper_metrics import PoseMetricEvaluator
from runtime import load_body_model, load_torch, sha256

MODELS = ['amass_epoch59','dip_final_2ep']
METRICS = ['mpjre_deg','mpjpe_cm','mpjve_cm','jitter_100_m_s3']


def audit(run):
    p = json.loads((run/'PROTOCOL.json').read_text())
    finished = json.loads((run/'completion.json').read_text())
    if finished['status'] != 'complete':
        raise ValueError('Comparison not complete')
    manifest = json.loads((run/'ARTIFACT_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        if sha256(run/name) != digest:
            raise ValueError(f'Artifact hash mismatch: {name}')
    if sha256(p['data']['path']) != p['data']['sha256']:
        raise ValueError('Test data changed')
    for item in [*p['models'].values(), p['smpl']]:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Checkpoint or body model changed')
    for path, digest in p['source_hashes'].items():
        if sha256(path) != digest:
            raise ValueError(f'Evaluation source changed: {path}')
    data = load_torch(p['data']['path'])
    targets = {r['sequence_id']:math.rotation_matrix_to_r6d(data['pose'][r['sequence_id']]).reshape(r['frames'],144).numpy()
               for r in p['records']}
    all_rows,checks = [],[]
    for mask in p['mask_order']:
        for rec in p['records']:
            seq,n = rec['sequence_id'],rec['frames']
            cache = run/'predictions'/f'{mask}_seq{seq:02d}.npz'
            row = json.loads((run/'rows'/f'{mask}_seq{seq:02d}.json').read_text())
            if (row['mask'] != mask or row['sequence_id'] != seq
                    or row['protocol_sha256'] != sha256(run/'PROTOCOL.json')
                    or row['cache_sha256'] != manifest[str(cache.relative_to(run))]):
                raise ValueError('Row/protocol/cache identity mismatch')
            with np.load(cache,allow_pickle=False) as arrays:
                if (str(arrays['protocol_sha256']) != sha256(run/'PROTOCOL.json')
                        or str(arrays['mask']) != mask or int(arrays['sequence_id']) != seq):
                    raise ValueError('Cached protocol, mask, or recording identity differs')
                if not np.array_equal(arrays['target_pose'],targets[seq]) or int(arrays['fps']) != 25:
                    raise ValueError('Target identity or sample rate mismatch')
                for model in MODELS:
                    if arrays[f'pose_{model}'].shape != (n,144) or not np.isfinite(arrays[f'pose_{model}']).all():
                        raise ValueError('Prediction coverage or finite-value check failed')
                    result = row['models'][model]
                    if result['frames'] != n or result['jerk_samples'] != n-3:
                        raise ValueError('Metric denominator differs from recording coverage')
            all_rows.append(row)
    stored = json.loads((run/'metrics.json').read_text())
    independent,maximum = {},0.
    for model in MODELS:
        independent[model] = {}
        for key in METRICS:
            values = []
            for mask in p['mask_order']:
                selected = [r['models'][model] for r in all_rows if r['mask']==mask]
                field = 'jerk_samples' if key=='jitter_100_m_s3' else 'frames'
                value = scalar_math.fsum(r['metrics'][key]*r[field] for r in selected)/sum(r[field] for r in selected)
                values.append(value)
            value = scalar_math.fsum(values)/len(values)
            independent[model][key] = value
            difference = abs(value-stored['overall_mask_equal'][model][key])
            maximum = max(maximum,difference)
            if difference > 1e-10:
                raise ValueError('Independent aggregation differs')
    evaluator = PoseMetricEvaluator(load_body_model(p['smpl']['path']),fk_chunk=11)
    # Choices fixed by position, independent of observed accuracy: first/last
    # mask and first/last test recording, both models, 8 recomputations total.
    for mask in [p['mask_order'][0],p['mask_order'][-1]]:
        for rec in [p['records'][0],p['records'][-1]]:
            seq = rec['sequence_id']
            row = next(r for r in all_rows if r['mask']==mask and r['sequence_id']==seq)
            with np.load(run/'predictions'/f'{mask}_seq{seq:02d}.npz',allow_pickle=False) as arrays:
                for model in MODELS:
                    recomputed = evaluator.evaluate_sequence(arrays[f'pose_{model}'],arrays['target_pose'],seq)
                    diff = {key:abs(recomputed['metrics'][key]-row['models'][model]['metrics'][key]) for key in METRICS}
                    if max(diff.values()) > 1e-4:
                        raise ValueError('Fixed FK-chunk spot check differs by more than 1e-4')
                    checks.append({'mask':mask,'sequence_id':seq,'model':model,'absolute_metric_differences':diff})
    return {'passed':True,'validator_sha256':sha256(__file__),'original_artifacts_verified':len(manifest),
            'paired_caches_checked':len(all_rows),'prediction_frames_checked':sum(r['frames'] for r in p['records'])*24*2,
            'all_targets_bitwise_equal':True,'independent_aggregate_max_abs_difference':maximum,
            'independent_aggregate':independent,'fk_chunk_rechecks':checks,'recheck_atol':1e-4,
            'scope':'Cache/reference/reduction and fixed measurement checks; no inference or model selection'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    result=audit(args.run)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
