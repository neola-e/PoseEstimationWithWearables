"""Freeze and evaluate IMU-only orientation refinement of the locked DIP model.

Uses existing F03 predictions; no neural inference, training, or model selection.
Standalone IK input/output files contain no target poses. GT is read separately
by the metric coordinator after refinement and cannot enter the solver.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time
import uuid
import os

import numpy as np
import torch

from imu_only_ik import IKSettings, PARENTS, SENSOR_JOINTS, OPTIMIZED_BY_SENSOR, refine_pose
from paper_metrics import PoseMetricEvaluator, aggregate_sequences
from runtime import load_body_model, load_torch, sha256
from train_dip_finetuning import atomic_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/local_dip/dip_finetune_comparison_cpu_v1'
EVIDENCE = ROOT/'outputs/local_dip/ik_refinement_audit_20261116'
MODELS = ('network_final', 'network_final_ik')
PRIMARY = ('mpjre_deg', 'mpjpe_cm', 'mpjve_cm', 'jitter_100_m_s3')


def arrays_write(path, **arrays):
    temporary=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream,**arrays)
    os.replace(temporary,path)


def group_name(sensors):
    return f'w{sum(s in [0,1] for s in sensors)}_p{sum(s in [2,3] for s in sensors)}_h{int(4 in sensors)}'


def verify(run):
    p=json.loads((run/'PROTOCOL.json').read_text())
    for relative,digest in json.loads((run/'PREPARE_MANIFEST.json').read_text()).items():
        if sha256(run/relative)!=digest:raise ValueError('Preparation artifact changed: '+relative)
    for path,digest in p['protected_files'].items():
        if sha256(path)!=digest:raise ValueError('Protected source/data/model changed: '+path)
    return p


def prepare(run):
    if run.exists():raise FileExistsError(run)
    torch.set_num_threads(2)
    baseline=json.loads((BASE/'PROTOCOL.json').read_text())
    completion=json.loads((BASE/'completion.json').read_text())
    validation=json.loads((BASE/'independent_validation.json').read_text())
    if completion['status']!='complete' or not validation['passed'] or completion['coverage']['pairs']!=456:
        raise ValueError('A complete validated F03 baseline is required')
    original_manifest=json.loads((BASE/'ARTIFACT_MANIFEST.json').read_text())
    for name,digest in original_manifest.items():
        if sha256(BASE/name)!=digest:raise ValueError('Baseline artifact changed: '+name)
    sources=copy.deepcopy(baseline['source_hashes'])
    for name in ['tools/imu_only_ik.py','tools/eval_imu_only_ik.py','tools/audit_imu_only_ik.py',
                 'tools/validate_imu_only_ik.py','tests/test_imu_only_ik.py','tests/test_ik_aggregation.py']:
        sources[str(ROOT/name)]=sha256(ROOT/name)
    protected={**sources,baseline['data']['path']:baseline['data']['sha256'],
               baseline['smpl']['path']:baseline['smpl']['sha256']}
    for model in baseline['models'].values():protected[model['path']]=model['sha256']
    training=ROOT/'outputs/local_dip/dip_finetuning_v1_20261114/PROTOCOL.json'
    protected[str(training)]=sha256(training)
    for name in ['PROTOCOL.json','completion.json','metrics.json','ARTIFACT_MANIFEST.json','independent_validation.json']:
        protected[str(BASE/name)]=sha256(BASE/name)
    body=load_body_model(baseline['smpl']['path'])
    if tuple(-1 if x is None else x for x in body.parent)!=PARENTS:raise ValueError('SMPL topology differs')
    p={'format':'dip-imu-only-ik-v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
       'baseline_run':str(BASE),'baseline_protocol_sha256':sha256(BASE/'PROTOCOL.json'),
       'baseline_original_manifest_sha256':sha256(BASE/'ARTIFACT_MANIFEST.json'),
       'checkpoint':baseline['models']['dip_final_2ep'],'training_protocol_sha256':sha256(training),
       'model_order':list(MODELS),'data':baseline['data'],'smpl':baseline['smpl'],
       'records':baseline['records'],'mask_order':baseline['mask_order'],'masks':baseline['masks'],
       'network_inference':baseline['inference'],'metrics':baseline['metrics'],
       'ik':{**asdict(IKSettings()),'sensor_segment_joints':list(SENSOR_JOINTS),
             'optimized_by_sensor':[list(j) for j in OPTIMIZED_BY_SENSOR],
             'parameterization':'right-multiplicative local SO(3) rotation-vector increments, float64; output float32',
             'objective':'per-frame mean squared 3x3 global orientation residual over available sensors; summed across frames for backward',
             'adam':{'betas':[.9,.999],'eps':1e-8,'weight_decay':0,'foreach':False,'fused':False},
             'targets':'positive full-rank provided orientation matrices projected to SO(3) by SVD; IK-only, original input unchanged',
             'selection':'final tenth update, no early stopping, best-iterate choice, rollback or GT calibration',
             'state':'fresh zero increments and Adam moments for every frame; simultaneous available constraints',
             'priors':'prediction initialization and frozen unselected local joints; no extra loss or joint limits',
             'temporal_information':'prediction[t] plus available orientation[t] only; no added smoothing or cross-frame state'},
       'primary_aggregation':'same F03: within-mask frame/T-3 weighting then equal average of24 masks',
       'auxiliary_aggregation':['sequence-equal within mask','12 mirror-collapsed group means and group-equal overall',
                                'subject-specific frame-weighted means; subject-equal overall'],
       'paper_table2_reference':dict(zip(PRIMARY,[23.9,9.7,12.1,1.9])),
       'paper_comparison_status':'Descriptive only; unknown author IK settings and aggregation prevent strict numerical equivalence',
       'known_conventions':{'learning_rate':'1e-3 from cited AvatarPoser paper, NOT recovered IMUPoser parameter; AvatarPoser code default differs',
                            'head_hip_selection':'sensed segment and immediate parent; unpublished index list reconstructed',
                            'rotation_parameterization_and_projection':'explicit implementation choices, not specified by IMUPoser',
                            'window_and_aggregation':'F03 preserved; author overlap/startup/exact reduction not recovered'},
       'runtime':{'torch':torch.__version__,'parts':2,'threads_per_part':2,'network_inference_performed':False},
       'protected_files':protected,'sources':sources,'training_performed':False,'test_tuning_performed':False,
       'evidence':{'imuposer_paper':'https://arxiv.org/html/2304.12518v1#S4.SS4',
                   'imuposer_commit':'a0e9d33b9c57d6737d711185b62084e17684b0e2',
                   'avatarposer_paper':'https://static.siplab.org/papers/eccv2022-avatarposer.pdf',
                   'avatarposer_commit':'0f97cb27fa7ad97ceb9c02e4d068d613eb315ba0',
                   'mapping_issue':'https://github.com/FIGLAB/IMUPoser/issues/16'}}
    for path,digest in protected.items():
        if sha256(path)!=digest:raise ValueError('Protected pre-run file changed: '+path)
    run.mkdir(parents=True)
    atomic_json(run/'PROTOCOL.json',p)
    for i,(path,digest) in enumerate(sources.items()):
        source=Path(path);destination=run/'source_snapshot'/f'{i:02d}_{source.name}'
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source,destination)
    shutil.copytree(EVIDENCE,run/'evidence')
    atomic_json(run/'PREPARE_MANIFEST.json',{str(f.relative_to(run)):sha256(f) for f in sorted(run.rglob('*')) if f.is_file()})
    print(json.dumps({'status':'prepared','run':str(run),'pairs':456,'network_inference':False,'ik_settings':p['ik']}),flush=True)


def reduce_rows(rows,p):
    expected={(mask,rec['sequence_id']) for mask in p['mask_order'] for rec in p['records']}
    identities=[(r['mask'],r['sequence_id']) for r in rows]
    if len(set(identities))!=len(identities) or set(identities)!=expected:
        raise ValueError('Incomplete, extra, or duplicate IK metric coverage')
    lengths={r['sequence_id']:r['frames'] for r in p['records']}
    per_mask={}
    for mask in p['mask_order']:
        per_mask[mask]={}
        for model in MODELS:
            selected=[r['models'][model] for r in rows if r['mask']==mask]
            for row in selected:
                if row['frames']!=lengths[int(row['sequence_id'])] or row['jerk_samples']!=max(row['frames']-3,0):
                    raise ValueError('Recording or jitter denominator changed')
            per_mask[mask][model]=aggregate_sequences(selected)
    overall={model:{key:float(np.mean([per_mask[m][model]['metrics']['frame_weighted'][key] for m in p['mask_order']]))
                    for key in PRIMARY} for model in MODELS}
    references=[per_mask[m][model]['metrics']['frame_weighted']['reference_jitter_m_s3'] for m in p['mask_order'] for model in MODELS]
    if any(x!=references[0] for x in references):raise ValueError('GT jitter/coverage changed')
    groups={}
    for mask,sensors in p['masks'].items():groups.setdefault(group_name(sensors),[]).append(mask)
    per_group={group:{'masks':masks,'models':{model:{key:float(np.mean([per_mask[m][model]['metrics']['frame_weighted'][key] for m in masks]))
                                               for key in PRIMARY} for model in MODELS}} for group,masks in sorted(groups.items())}
    group_equal={model:{key:float(np.mean([g['models'][model][key] for g in per_group.values()])) for key in PRIMARY} for model in MODELS}
    changes={key:{'after_minus_before':overall[MODELS[1]][key]-overall[MODELS[0]][key],
                  'relative_decrease_percent':(overall[MODELS[0]][key]-overall[MODELS[1]][key])/overall[MODELS[0]][key]*100}
             for key in PRIMARY}
    return {'coverage':{'masks':len(p['mask_order']),'sequences':len(p['records']),'pairs':len(rows),
                         'frames_per_mask':sum(lengths.values())},
            'overall_mask_equal':overall,'changes':changes,'per_mask':per_mask,'mirror_groups':per_group,
            'overall_12_groups_equal':group_equal,'reference_jitter_m_s3':references[0],
            'overall_sequence_equal':{model:{key:float(np.mean([per_mask[m][model]['metrics']['sequence_equal'][key] for m in p['mask_order']]))
                                            for key in PRIMARY} for model in MODELS}}


def summarize(run,p):
    rows=[json.loads(f.read_text()) for f in sorted((run/'rows').glob('*.json'))]
    result=reduce_rows(rows,p)
    baseline=json.loads((Path(p['baseline_run'])/'metrics.json').read_text())['overall_mask_equal']['dip_final_2ep']
    if result['overall_mask_equal'][MODELS[0]]!=baseline:raise ValueError('Raw baseline changed')
    atomic_json(run/'metrics.json',result)
    import csv
    for name,fields,items in [
        ('comparison.csv',['model',*PRIMARY],[{'model':model,**result['overall_mask_equal'][model]} for model in MODELS]),
        ('per_mask.csv',['mask','model',*PRIMARY],[{'mask':mask,'model':model,**{k:result['per_mask'][mask][model]['metrics']['frame_weighted'][k] for k in PRIMARY}}
                                                 for mask in p['mask_order'] for model in MODELS]),
        ('per_group.csv',['group','mask_count','model',*PRIMARY],[{'group':group,'mask_count':len(g['masks']),'model':model,**g['models'][model]}
                                                                for group,g in result['mirror_groups'].items() for model in MODELS])]:
        with (run/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(items)
    atomic_json(run/'completion.json',{'status':'complete','protocol_sha256':sha256(run/'PROTOCOL.json'),
                                      'coverage':result['coverage'],'overall':result['overall_mask_equal'],
                                      'changes':result['changes'],'training_performed':False,'network_inference_performed':False})
    return result


def execute(run,part):
    p=verify(run);protocol_sha=sha256(run/'PROTOCOL.json')
    for name in ['inputs','refined','rows']:(run/name).mkdir(exist_ok=True)
    lock=(run/f'.part{part}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(2)
    data=load_torch(p['data']['path']);body=load_body_model(p['smpl']['path'])
    evaluator=PoseMetricEvaluator(body,fps=25,fk_chunk=32)
    settings=IKSettings(**{k:p['ik'][k] for k in ['iterations','learning_rate','frame_batch','device']})
    base=Path(p['baseline_run']);baseline_manifest=json.loads((base/'ARTIFACT_MANIFEST.json').read_text())
    masks=[m for i,m in enumerate(p['mask_order']) if i%2==part]
    start=time.perf_counter();done=0;total=len(masks)*len(p['records'])
    for mask in masks:
        ids=p['masks'][mask]
        for rec in p['records']:
            seq,n=rec['sequence_id'],rec['frames'];name=f'{mask}_seq{seq:02d}'
            input_file=run/'inputs'/f'{name}.npz';output_file=run/'refined'/f'{name}.npz';row_file=run/'rows'/f'{name}.json'
            if row_file.exists():
                row=json.loads(row_file.read_text())
                if row['protocol_sha256']!=protocol_sha or row['input_sha256']!=sha256(input_file) or row['refined_sha256']!=sha256(output_file):
                    raise ValueError('Resumed IK result changed')
                done+=1;continue
            source=base/'predictions'/f'{name}.npz';raw_row=base/'rows'/f'{name}.json'
            for f in [source,raw_row]:
                if sha256(f)!=baseline_manifest[str(f.relative_to(base))]:raise ValueError('Baseline pair changed')
            with np.load(source,allow_pickle=False) as arrays:
                prediction=arrays['pose_dip_final_2ep'].copy()
            if not input_file.exists():
                orientation=torch.zeros(n,5,3,3,dtype=torch.float32)
                orientation[:,ids]=data['ori'][seq][:,ids].float()
                arrays_write(input_file,predicted_pose=prediction,imu_orientations=orientation.numpy(),
                             sensor_ids=np.array(ids,dtype=np.int64),sequence_id=seq,mask=mask,protocol_sha256=protocol_sha,
                             baseline_cache_sha256=sha256(source))
            with np.load(input_file,allow_pickle=False) as arrays:
                if (str(arrays['protocol_sha256'])!=protocol_sha or not np.array_equal(arrays['predicted_pose'],prediction)
                        or not np.array_equal(arrays['imu_orientations'][:,ids],data['ori'][seq][:,ids].numpy())):
                    raise ValueError('Standalone IK input differs from frozen data')
                result=refine_pose(arrays['predicted_pose'],arrays['imu_orientations'],arrays['sensor_ids'].tolist(),settings)
            arrays_write(output_file,pose=result['pose'].numpy(),loss_history=result['loss_history'].numpy(),
                         orientation_angles_deg=result['orientation_angles_deg'].numpy(),protocol_sha256=protocol_sha,
                         input_sha256=sha256(input_file),sequence_id=seq,mask=mask)
            # Only the metric stage reads GT. The standalone IK stage above has
            # already completed and saved its output without any target poses.
            with np.load(source,allow_pickle=False) as arrays:target=arrays['target_pose'].copy()
            refined=evaluator.evaluate_sequence(result['pose'],target,seq)
            original=json.loads(raw_row.read_text())['models']['dip_final_2ep']
            if original['settings']!=refined['settings']:raise ValueError('Raw and IK measurement settings differ')
            row={'mask':mask,'sequence_id':seq,'protocol_sha256':protocol_sha,
                 'input_sha256':sha256(input_file),'refined_sha256':sha256(output_file),
                 'baseline_cache_sha256':sha256(source),'baseline_row_sha256':sha256(raw_row),
                 'models':{MODELS[0]:original,MODELS[1]:refined},'ik':result['summary']}
            atomic_json(row_file,row);done+=1
            atomic_json(run/f'part{part}_progress.json',{'status':'running','completed':done,'planned':total,
                        'elapsed_seconds':time.perf_counter()-start,'mask':mask,'sequence_id':seq,
                        'updated_utc':datetime.now(timezone.utc).isoformat()})
            print(f'part{part} {done}/{total} {mask} seq{seq} IMU MSE {result["summary"]["orientation_mse_before"]:.7f} -> {result["summary"]["orientation_mse_after"]:.7f}',flush=True)
    atomic_json(run/f'part{part}_progress.json',{'status':'complete','completed':done,'planned':total,'elapsed_seconds':time.perf_counter()-start})
    with (run/'.finalization.lock').open('a') as final_lock:
        fcntl.flock(final_lock,fcntl.LOCK_EX)
        if all((run/f'part{i}_progress.json').exists() and json.loads((run/f'part{i}_progress.json').read_text())['status']=='complete' for i in [0,1]):
            verify(run);result=summarize(run,p)
            atomic_json(run/'ARTIFACT_MANIFEST.json',{str(f.relative_to(run)):sha256(f) for f in sorted(run.rglob('*')) if f.is_file() and f.name!='ARTIFACT_MANIFEST.json'})
            print('IK EVALUATION COMPLETE '+json.dumps(result['overall_mask_equal']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--part',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.output)
    else:execute(args.output,args.part)
