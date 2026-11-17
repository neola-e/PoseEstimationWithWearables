"""Independent full-input/aggregation audit plus fixed solver and metric rechecks."""
import argparse
import json
import math as scalar_math
from pathlib import Path

import numpy as np
import torch
from imu_only_ik import IKSettings, OPTIMIZED_BY_SENSOR, refine_pose
from paper_metrics import PoseMetricEvaluator
from runtime import load_body_model, load_torch, sha256

MODELS=('network_final','network_final_ik')
PRIMARY=('mpjre_deg','mpjpe_cm','mpjve_cm','jitter_100_m_s3')


def validate(run):
    p=json.loads((run/'PROTOCOL.json').read_text());protocol_sha=sha256(run/'PROTOCOL.json')
    completed=json.loads((run/'completion.json').read_text())
    if completed['status']!='complete' or completed['coverage']['pairs']!=456:raise ValueError('Full run required')
    manifest=json.loads((run/'ARTIFACT_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        if sha256(run/name)!=digest:raise ValueError('Artifact changed: '+name)
    for path,digest in p['protected_files'].items():
        if sha256(path)!=digest:raise ValueError('Protected source/data/checkpoint changed: '+path)
    data=load_torch(p['data']['path']);base=Path(p['baseline_run'])
    original=json.loads((base/'ARTIFACT_MANIFEST.json').read_text());rows=[];unchanged_joints=0
    for mask in p['mask_order']:
        sensors=p['masks'][mask];fixed=[j for j in range(24) if j not in {k for s in sensors for k in OPTIMIZED_BY_SENSOR[s]}]
        absent=[s for s in range(5) if s not in sensors]
        for rec in p['records']:
            seq,n=rec['sequence_id'],rec['frames'];name=f'{mask}_seq{seq:02d}'
            inp=run/'inputs'/f'{name}.npz';out=run/'refined'/f'{name}.npz'
            baseline=base/'predictions'/f'{name}.npz';baseline_row=base/'rows'/f'{name}.json'
            row=json.loads((run/'rows'/f'{name}.json').read_text());rows.append(row)
            if (row['protocol_sha256']!=protocol_sha or row['input_sha256']!=sha256(inp)
                    or row['refined_sha256']!=sha256(out) or row['sequence_id']!=seq or row['mask']!=mask):
                raise ValueError('Refinement row identity/hash mismatch')
            for source,key in [(baseline,'baseline_cache_sha256'),(baseline_row,'baseline_row_sha256')]:
                if sha256(source)!=row[key] or row[key]!=original[str(source.relative_to(base))]:
                    raise ValueError('Baseline source hash mismatch')
            if row['models'][MODELS[0]]!=json.loads(baseline_row.read_text())['models']['dip_final_2ep']:
                raise ValueError('Raw metric row changed')
            with np.load(inp,allow_pickle=False) as inputs,np.load(out,allow_pickle=False) as outputs,np.load(baseline,allow_pickle=False) as raw:
                if set(inputs.files)!={'predicted_pose','imu_orientations','sensor_ids','sequence_id','mask','protocol_sha256','baseline_cache_sha256'}:
                    raise ValueError('Standalone solver input contains unexpected/GT fields')
                if (not np.array_equal(inputs['predicted_pose'],raw['pose_dip_final_2ep'])
                        or inputs['sensor_ids'].tolist()!=sensors
                        or not np.array_equal(inputs['imu_orientations'][:,sensors],data['ori'][seq][:,sensors].numpy())
                        or np.any(inputs['imu_orientations'][:,absent]!=0)):
                    raise ValueError('Available measurement/mask/prediction differs')
                if (str(outputs['protocol_sha256'])!=protocol_sha or str(outputs['input_sha256'])!=sha256(inp)
                        or outputs['pose'].shape!=(n,144) or not np.isfinite(outputs['pose']).all()
                        or outputs['loss_history'].shape!=(n,11) or not np.isfinite(outputs['loss_history']).all()):
                    raise ValueError('Refinement shape/iteration/finite-value check failed')
                if not np.array_equal(outputs['pose'].reshape(n,24,6)[:,fixed],raw['pose_dip_final_2ep'].reshape(n,24,6)[:,fixed]):
                    raise ValueError('Unoptimized local joint changed')
                unchanged_joints+=n*len(fixed)
            for model in MODELS:
                if row['models'][model]['frames']!=n or row['models'][model]['jerk_samples']!=n-3:
                    raise ValueError('Coverage/derivative denominator differs')
    stored=json.loads((run/'metrics.json').read_text());independent={};maximum=0.
    for model in MODELS:
        independent[model]={}
        for key in PRIMARY:
            mask_values=[]
            for mask in p['mask_order']:
                selected=[r['models'][model] for r in rows if r['mask']==mask]
                weight='jerk_samples' if key=='jitter_100_m_s3' else 'frames'
                value=scalar_math.fsum(r['metrics'][key]*r[weight] for r in selected)/sum(r[weight] for r in selected)
                mask_values.append(value)
            value=scalar_math.fsum(mask_values)/24;independent[model][key]=value
            maximum=max(maximum,abs(value-stored['overall_mask_equal'][model][key]))
    if maximum>1e-10:raise ValueError('Independent reduction differs')
    evaluator=PoseMetricEvaluator(load_body_model(p['smpl']['path']),fps=25,fk_chunk=11)
    checks=[]
    for mask in [p['mask_order'][0],p['mask_order'][-1]]:
        for rec in [p['records'][0],p['records'][-1]]:
            seq,n=rec['sequence_id'],rec['frames'];name=f'{mask}_seq{seq:02d}'
            row=next(r for r in rows if r['mask']==mask and r['sequence_id']==seq)
            with np.load(run/'inputs'/f'{name}.npz',allow_pickle=False) as inputs,np.load(run/'refined'/f'{name}.npz',allow_pickle=False) as outputs,np.load(base/'predictions'/f'{name}.npz',allow_pickle=False) as raw:
                indices=np.array(sorted({0,1,min(124,n-1),min(125,n-1),n-1}))
                settings=IKSettings(iterations=10,learning_rate=p['ik']['learning_rate'],frame_batch=1)
                repeated=refine_pose(inputs['predicted_pose'][indices],inputs['imu_orientations'][indices],inputs['sensor_ids'].tolist(),settings)
                pose_diff=float(np.max(np.abs(repeated['pose'].numpy()-outputs['pose'][indices])))
                if pose_diff>2e-7:raise ValueError('Independent-frame recheck differs')
                metric_diff={}
                for model,pose in [(MODELS[0],raw['pose_dip_final_2ep']),(MODELS[1],outputs['pose'])]:
                    remeasured=evaluator.evaluate_sequence(pose,raw['target_pose'],seq)
                    metric_diff[model]={key:abs(remeasured['metrics'][key]-row['models'][model]['metrics'][key]) for key in PRIMARY}
                    if max(metric_diff[model].values())>1e-4:raise ValueError('Fixed FK-chunk metric recheck differs')
                checks.append({'mask':mask,'sequence_id':seq,'solver_frames':indices.tolist(),
                               'solver_max_abs_pose6d_difference':pose_diff,'metric_absolute_differences':metric_diff})
    return {'passed':True,'original_artifacts_verified':len(manifest),'paired_caches_checked':len(rows),
            'standalone_inputs_have_no_gt':True,'all_active_imu_inputs_bitwise_equal':True,
            'all_unavailable_channels_zero':True,'unoptimized_joint_frames_bitwise_checked':unchanged_joints,
            'baseline_predictions_and_rows_preserved':True,'protected_files_unchanged':True,
            'independent_aggregate_max_abs_difference':maximum,'independent_aggregate':independent,
            'fixed_solver_and_fk_rechecks':checks,'solver_atol':2e-7,'metric_atol':1e-4,
            'validator_sha256':sha256(__file__),'scope':'No training, hyperparameter selection, or GT-assisted constraints'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);result=validate(args.run)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
