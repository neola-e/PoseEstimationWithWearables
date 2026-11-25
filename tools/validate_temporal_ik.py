"""Independent coverage/reduction/causal-window and fixed FK checks for I02."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from imu_only_ik import OPTIMIZED_BY_SENSOR
from temporal_ik import TemporalIKSettings,refine_temporal
from paper_metrics import PoseMetricEvaluator
from runtime import load_body_model,sha256

MODELS=['raw','framewise_ik','temporal_ik']
KEYS=['mpjre_deg','mpjpe_cm','mpjve_cm','jitter_100_m_s3']


def audit(root):
    run=root/'test';p=json.loads((run/'PROTOCOL.json').read_text());protocol_sha=sha256(run/'PROTOCOL.json')
    if json.loads((run/'completion.json').read_text())['pairs']!=456:raise ValueError('Full test required')
    manifest=json.loads((run/'ARTIFACT_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        if sha256(run/name)!=digest:raise ValueError('Temporal test artifact changed: '+name)
    for path,digest in p['protected_files'].items():
        if sha256(path)!=digest:raise ValueError('Protected input/source/checkpoint changed: '+path)
    if sha256(p['data']['path'])!=p['data']['sha256']:raise ValueError('Original test tensor container changed')
    calibration=root/'calibration';cm=json.loads((calibration/'metrics.json').read_text())['overall']
    targets={k:cm['raw'][k] for k in KEYS[:3]}
    targets['mpjve_cm']-=.5*max(cm['raw']['mpjve_cm']-cm['framewise_ik']['mpjve_cm'],0)
    ranked=[]
    for weight in json.loads((calibration/'PROTOCOL.json').read_text())['grid']:
        value=cm['lambda_'+str(weight)]
        violation=sum(max(0,(value[k]-targets[k])/max(abs(targets[k]),1e-12)) for k in targets)
        eligible=violation<=1e-12 and value['jitter_100_m_s3']<cm['framewise_ik']['jitter_100_m_s3']
        ranked.append((not eligible,violation,value['jitter_100_m_s3'],weight))
    if min(ranked)[-1]!=p['settings']['jerk_weight']:raise ValueError('Test weight did not follow development-only choice')
    old=Path(p['baseline_ik_run']);rawroot=Path(p['baseline_raw_run']);rows=[];frozen_joint_frames=0
    for mask in p['mask_order']:
        fixed=[j for j in range(24) if j not in {q for s in p['masks'][mask] for q in OPTIMIZED_BY_SENSOR[s]}]
        for rec in p['records']:
            seq,n=rec['sequence_id'],rec['frames'];name=f'{mask}_seq{seq:02d}'
            inp=old/'inputs'/f'{name}.npz';out=run/'refined'/f'{name}.npz';source=old/'rows'/f'{name}.json'
            row=json.loads((run/'rows'/f'{name}.json').read_text());rows.append(row)
            if (row['input_sha256']!=sha256(inp) or row['output_sha256']!=sha256(out)
                    or row['baseline_row_sha256']!=sha256(source) or row['protocol_sha256']!=protocol_sha):raise ValueError('Pair identity/hash mismatch')
            original=json.loads(source.read_text())
            if row['models']['raw']!=original['models']['network_final'] or row['models']['framewise_ik']!=original['models']['network_final_ik']:
                raise ValueError('Old baseline metric row was replaced')
            with np.load(inp,allow_pickle=False) as a,np.load(out,allow_pickle=False) as b:
                if any('target' in k or 'gt'==k for k in a.files):raise ValueError('GT present in solver input')
                if sha256(rawroot/'predictions'/f'{name}.npz')!=str(a['baseline_cache_sha256']):
                    raise ValueError('Original paired target/prediction cache changed')
                if (b['pose'].shape!=(n,144) or not np.isfinite(b['pose']).all() or b['loss_history'].shape!=(n,11,3)
                        or not np.isfinite(b['loss_history']).all() or str(b['protocol_sha256'])!=protocol_sha):raise ValueError('Invalid output coverage/values')
                if not np.array_equal(a['predicted_pose'].reshape(n,24,6)[:,fixed],b['pose'].reshape(n,24,6)[:,fixed]):raise ValueError('Frozen local joint changed')
                frozen_joint_frames+=n*len(fixed)
            for value in row['models'].values():
                if value['frames']!=n or value['jerk_samples']!=n-3:raise ValueError('Metric denominator changed')
    expected={(m,r['sequence_id']) for m in p['mask_order'] for r in p['records']}
    identities=[(r['mask'],r['sequence_id']) for r in rows]
    if len(identities)!=456 or len(set(identities))!=456 or set(identities)!=expected:raise ValueError('Missing/extra/duplicate pair')
    result=json.loads((run/'metrics.json').read_text());independent={};maximum=0.
    for model in MODELS:
        independent[model]={}
        for key in KEYS:
            means=[]
            for mask in p['mask_order']:
                selected=[row['models'][model] for row in rows if row['mask']==mask]
                count='jerk_samples' if key=='jitter_100_m_s3' else 'frames'
                means.append(math.fsum(r['metrics'][key]*r[count] for r in selected)/sum(r[count] for r in selected))
            value=math.fsum(means)/24;independent[model][key]=value
            maximum=max(maximum,abs(value-result['overall'][model][key]))
    if maximum>1e-10:raise ValueError('Independent summary differs')
    evaluator=PoseMetricEvaluator(load_body_model(p['smpl']['path']),fps=25,fk_chunk=11)
    offsets=torch.tensor(p['offsets_m'],dtype=torch.float64);settings=TemporalIKSettings(**{**p['settings'],'frame_batch':1})
    checks=[]
    for mask in [p['mask_order'][0],p['mask_order'][-1]]:
        for rec in [p['records'][0],p['records'][-1]]:
            seq,n=rec['sequence_id'],rec['frames'];name=f'{mask}_seq{seq:02d}'
            with np.load(old/'inputs'/f'{name}.npz',allow_pickle=False) as a,np.load(run/'refined'/f'{name}.npz',allow_pickle=False) as b,np.load(rawroot/'predictions'/f'{name}.npz',allow_pickle=False) as reference:
                frames=sorted({0,1,3,4,min(124,n-1),min(125,n-1),n-1});differences=[]
                for t in frames:
                    first=max(0,t-settings.window+1)
                    rebuilt=refine_temporal(a['predicted_pose'][first:t+1],a['imu_orientations'][first:t+1],a['sensor_ids'].tolist(),offsets,settings)['pose'][-1].numpy()
                    differences.append(float(np.max(np.abs(rebuilt-b['pose'][t]))))
                if max(differences)>2e-7:raise ValueError('Past-only scalar-window recheck differs')
                recomputed=evaluator.evaluate_sequence(b['pose'],reference['target_pose'],seq)
                row=next(r for r in rows if r['mask']==mask and r['sequence_id']==seq)['models']['temporal_ik']
                metric_differences={k:abs(recomputed['metrics'][k]-row['metrics'][k]) for k in KEYS}
                if max(metric_differences.values())>1e-4:raise ValueError('Fixed FK measurement differs')
                checks.append({'mask':mask,'sequence_id':seq,'output_frames':frames,'input_support':'[max(0,t-4),t], no future',
                               'maximum_pose_difference':max(differences),'metric_differences':metric_differences})
    return {'passed':True,'artifacts_verified':len(manifest),'pairs':len(rows),'no_gt_solver_inputs':True,
            'old_baseline_rows_and_protected_files_unchanged':True,'frozen_joint_frames_checked':frozen_joint_frames,
            'development_selection_reproduced':True,'independent_aggregate_max_abs_difference':maximum,
            'independent_aggregate':independent,'fixed_causal_window_and_fk_checks':checks,
            'validator_sha256':sha256(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);value=audit(args.run);args.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
