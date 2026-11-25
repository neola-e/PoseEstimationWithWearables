"""Independent Temporal IK experiment: development-only weight choice, fixed test.

Calibration uses the development epoch2 (trained s01–s06) on fixed central
96-frame panels of s07/s08 and12 mirror-group representatives. Test uses the
unchanged final first-eight epoch2 and all24 masks/19 recordings. No retraining.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from imuposer import math
from imu_only_ik import IKSettings,refine_pose,PARENTS
from temporal_ik import TemporalIKSettings,refine_temporal,world_kinematics
from eval_imu_only_ik import arrays_write,group_name
from paper_metrics import PoseMetricEvaluator,aggregate_sequences
from runtime import PosePredictor,load_body_model,load_torch,sha256
from train_dip_finetuning import atomic_json

ROOT=Path(__file__).resolve().parents[1]
I01=ROOT/'outputs/local_dip/dip_imu_only_ik_v1_20261116'
GRID=[.001,.01,.1,1.]
PRIMARY=['mpjre_deg','mpjpe_cm','mpjve_cm','jitter_100_m_s3']
DEV_CHECKPOINT=ROOT/'outputs/local_dip/dip_dev_v1/checkpoints/epoch_002_03a6585efa03.ckpt'
DEV_SHA='4515e13ae95ee5ec49ff90d941c81149ec4c583ba293aadf21abe17de699ba16'


def offsets_from_body(body):
    joint,_=body.get_zero_pose_joint_and_vertex()
    offsets=joint.double().clone()
    for j in range(1,24):offsets[j]=joint[j].double()-joint[PARENTS[j]].double()
    return offsets


@torch.inference_mode()
def predict_interval(predictor,features,first,stop,batch_size=64):
    if not 0<=first<stop<=len(features):raise ValueError('Invalid prediction interval')
    outputs=[]
    for cursor in range(first,stop,batch_size):
        times=list(range(cursor,min(stop,cursor+batch_size)))
        windows=[features[max(0,t-124):t+1] for t in times];lengths=[len(w) for w in windows]
        batch=torch.zeros(len(windows),max(lengths),60)
        for i,window in enumerate(windows):batch[i,:len(window)]=window
        y=predictor.predict(batch,lengths)
        outputs.append(y[torch.arange(len(times)),torch.tensor(lengths)-1])
    return torch.cat(outputs)


def source_hashes():
    return {str(ROOT/name):sha256(ROOT/name) for name in ['tools/temporal_ik.py','tools/experiment_temporal_ik.py',
            'tests/test_temporal_ik.py','tests/test_temporal_experiment.py']}


def verify_files(p):
    for path,digest in p['protected_files'].items():
        if sha256(path)!=digest:raise ValueError('Protected experiment file changed: '+path)


def aggregate(rows,masks,models):
    per_mask={mask:{model:aggregate_sequences([r['models'][model] for r in rows if r['mask']==mask])
                    for model in models} for mask in masks}
    overall={model:{key:float(np.mean([per_mask[mask][model]['metrics']['frame_weighted'][key] for mask in masks]))
                    for key in PRIMARY} for model in models}
    return {'overall':overall,'per_mask':per_mask}


def select_weight(metrics):
    raw,ind=metrics['raw'],metrics['framewise_ik']
    targets={k:raw[k] for k in PRIMARY[:3]}
    targets['mpjve_cm']=raw['mpjve_cm']-.5*max(raw['mpjve_cm']-ind['mpjve_cm'],0)
    scored=[]
    for weight in GRID:
        value=metrics['lambda_'+str(weight)]
        violation=sum(max(0,(value[k]-targets[k])/max(abs(targets[k]),1e-12)) for k in targets)
        jitter_ok=value['jitter_100_m_s3']<ind['jitter_100_m_s3']
        eligible=violation<=1e-12 and jitter_ok
        scored.append({'weight':weight,'eligible':eligible,'spatial_violation':violation,
                       'jitter':value['jitter_100_m_s3'],'metrics':value})
    chosen=min(scored,key=lambda x:(not x['eligible'],x['spatial_violation'],x['jitter'],x['weight']))
    return {'chosen_weight':chosen['weight'],'eligible':chosen['eligible'],'targets':targets,'candidates':scored,
            'rule':'Prefer all3 errors no worse than raw and retain >=50% framewise-IK vertex gain, with jitter lower than framewise IK. Among eligible minimize jitter; if none, minimize normalized spatial violation then jitter. Positive weights only; no test selection.'}


def calibrate(output):
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);run=output/'calibration';run.mkdir();(run/'inputs').mkdir();(run/'rows').mkdir()
    parent=json.loads((I01/'PROTOCOL.json').read_text());manifest_path=ROOT/'outputs/local_dip/dip_finetuning_v1_20261114/data_manifest.json'
    manifest=json.loads(manifest_path.read_text());records=[r for r in manifest['records'] if r['role']=='validation']
    if len(records)!=11 or {r['subject'] for r in records}!={'s_07','s_08'}:raise ValueError('Development subject split differs')
    if sha256(DEV_CHECKPOINT)!=DEV_SHA:raise ValueError('Development checkpoint changed')
    representatives={}
    for mask,ids in parent['masks'].items():representatives.setdefault(group_name(ids),mask)
    masks=list(representatives.values())
    if len(masks)!=12:raise ValueError('Expected12 representatives')
    torch.set_num_threads(2);body=load_body_model(parent['smpl']['path']);offsets=offsets_from_body(body)
    # Geometry-only integration check against the canonical implementation.
    from imu_only_ik import exp_map
    local=exp_map(torch.arange(8*24*3,dtype=torch.float64).reshape(8,24,3)*1e-4)
    _,pos=world_kinematics(local,offsets)
    reference=body.forward_kinematics(local.float(),calc_mesh=False)[1].double()
    if (pos-reference).abs().max()>2e-6:raise ValueError('Temporal skeleton differs from canonical mean-body FK')
    # Do not even hash/open the test tensor container during calibration.
    prior_protected={path:digest for path,digest in parent['protected_files'].items() if path!=parent['data']['path']}
    protected={**prior_protected,**source_hashes(),str(DEV_CHECKPOINT):DEV_SHA,
               str(manifest_path):sha256(manifest_path),manifest['data_path']:manifest['data_sha256']}
    p={'format':'temporal-ik-calibration-v1','created_utc':datetime.now(timezone.utc).isoformat(),
       'scope':'New experiment; only s07/s08 calibration, no s09/s10 loading or test-driven choice',
       'checkpoint':{'path':str(DEV_CHECKPOINT),'sha256':DEV_SHA,'training_subjects':'s01–s06','completed_epoch':2},
       'data':{'path':manifest['data_path'],'sha256':manifest['data_sha256']},'records':records,
       'mask_order':masks,'masks':{m:parent['masks'][m] for m in masks},'smpl':parent['smpl'],
       'offsets_m':offsets.tolist(),'geometry_max_abs_difference_m':float((pos-reference).abs().max()),
       'temporal_defaults':asdict(TemporalIKSettings()),'grid':GRID,
       'selection_rule':'Among positive weights, require all3 errors no worse than raw, >=50% framewise vertex gain, and jitter below framewise. Minimize jitter among eligible; if none minimize summed normalized spatial violation, then jitter, then smaller weight.',
       'calibration_sampling':'central96 consecutive frames per recording;4 earlier pose outputs supply temporal history; each neural output has original125-frame history',
       'window_loss':'sum valid per-frame orientation MSE +lambda*sum valid four-frame stencils mean((D3 joint positions/0.01m)^2)',
       'output':'last frame of past-only5-frame window; prefix startup; fresh Adam/window; no added future or post-filter',
       'aggregation':'equal mean over12 fixed representative masks, frame/T-3 within mask',
       'protected_files':protected,'sources':source_hashes(),
       'test_success_criteria':'All3 test errors below raw; retain >=50% I01 vertex gain; jitter below I01. Separately report whether jitter also falls below raw.',
       'validation_plan':'first/last masks and first/last recordings; frames0,1,3,4,124,125,last; independent past-only scalar windows, FK chunk11, all inputs/denominators/hashes'}
    verify_files(p);atomic_json(run/'PROTOCOL.json',p)
    (run/'source_snapshot').mkdir()
    for i,path in enumerate(p['sources']):shutil.copy2(path,run/'source_snapshot'/f'{i}_{Path(path).name}')
    data=load_torch(p['data']['path']);predictor=PosePredictor(DEV_CHECKPOINT,'cpu',8)
    evaluator=PoseMetricEvaluator(body,fps=25,fk_chunk=32);rows=[];start=time.perf_counter()
    for mask in masks:
        ids=p['masks'][mask]
        for rec in records:
            seq,n=rec['sequence_id'],rec['frames'];first=max(0,(n-96)//2);stop=min(n,first+96);context=max(0,first-4)
            x=torch.cat([(data['acc'][seq][:,:5]/30).flatten(1),data['ori'][seq][:,:5].float().flatten(1)],1)
            keep=torch.zeros(60,dtype=torch.bool);keep[:15].reshape(5,3)[ids]=True;keep[15:].reshape(5,9)[ids]=True;x[:,~keep]=0
            torch.set_num_threads(8);pred=predict_interval(predictor,x,context,stop).numpy()
            ori=torch.zeros(stop-context,5,3,3);ori[:,ids]=data['ori'][seq][context:stop,ids]
            name=f'{mask}_seq{seq:02d}';input_file=run/'inputs'/f'{name}.npz'
            arrays_write(input_file,predicted_pose=pred,imu_orientations=ori.numpy(),sensor_ids=np.array(ids),
                         sequence_id=seq,source_start=context,metric_start=first-context,source_stop=stop)
            torch.set_num_threads(2);start_metric=first-context;target=math.rotation_matrix_to_r6d(data['pose'][seq][first:stop]).reshape(-1,144)
            independent=refine_pose(pred,ori,ids,IKSettings(frame_batch=128))['pose']
            zero=refine_temporal(pred,ori,ids,offsets,TemporalIKSettings(jerk_weight=0))['pose']
            if (zero-independent).abs().max()>2e-7:raise ValueError('Zero temporal weight does not reproduce I01 solver')
            measured={'raw':evaluator.evaluate_sequence(pred[start_metric:],target,seq),
                      'framewise_ik':evaluator.evaluate_sequence(independent[start_metric:],target,seq)}
            for weight in GRID:
                result=refine_temporal(pred,ori,ids,offsets,TemporalIKSettings(jerk_weight=weight))
                measured['lambda_'+str(weight)]=evaluator.evaluate_sequence(result['pose'][start_metric:],target,seq)
            row={'mask':mask,'sequence_id':seq,'subject':rec['subject'],'source_file':rec['source_file'],
                 'first':first,'stop':stop,'context_start':context,'input_sha256':sha256(input_file),
                 'zero_weight_max_abs_pose_difference':float((zero-independent).abs().max()),'models':measured}
            atomic_json(run/'rows'/f'{name}.json',row);rows.append(row)
            print(f'calibration {len(rows)}/132 {mask} seq{seq}',flush=True)
    result=aggregate(rows,masks,['raw','framewise_ik',*['lambda_'+str(w) for w in GRID]])
    selection=select_weight(result['overall']);selection['calibration_protocol_sha256']=sha256(run/'PROTOCOL.json')
    selection['test_data_opened']=False;selection['elapsed_seconds']=time.perf_counter()-start
    verify_files(p);atomic_json(run/'metrics.json',result);atomic_json(run/'selection.json',selection)
    atomic_json(run/'completion.json',{'status':'complete','pairs':len(rows),'chosen_weight':selection['chosen_weight'],'test_data_opened':False})
    atomic_json(run/'ARTIFACT_MANIFEST.json',{str(f.relative_to(run)):sha256(f) for f in sorted(run.rglob('*')) if f.is_file() and f.name!='ARTIFACT_MANIFEST.json'})
    print('CALIBRATION COMPLETE '+json.dumps(selection),flush=True)


def prepare_test(output):
    calibration=output/'calibration';selection=json.loads((calibration/'selection.json').read_text())
    c=json.loads((calibration/'PROTOCOL.json').read_text());verify_files(c)
    if not (calibration/'completion.json').exists() or selection['test_data_opened']:raise ValueError('Development-only selection required')
    expected=select_weight(json.loads((calibration/'metrics.json').read_text())['overall'])
    if expected['chosen_weight']!=selection['chosen_weight']:raise ValueError('Selection rule mismatch')
    for name,digest in json.loads((calibration/'ARTIFACT_MANIFEST.json').read_text()).items():
        if sha256(calibration/name)!=digest:raise ValueError('Calibration artifact changed')
    parent=json.loads((I01/'PROTOCOL.json').read_text());run=output/'test';run.mkdir(exist_ok=False)
    protected={**c['protected_files']}
    for name in ['PROTOCOL.json','metrics.json','selection.json','ARTIFACT_MANIFEST.json']:protected[str(calibration/name)]=sha256(calibration/name)
    for name in ['PROTOCOL.json','ARTIFACT_MANIFEST.json','metrics.json','independent_validation.json']:protected[str(I01/name)]=sha256(I01/name)
    p={'format':'temporal-ik-fixed-test-v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
       'baseline_ik_run':str(I01),'baseline_raw_run':parent['baseline_run'],'checkpoint':parent['checkpoint'],
       'data':parent['data'],'smpl':parent['smpl'],'records':parent['records'],'masks':parent['masks'],'mask_order':parent['mask_order'],
       'offsets_m':c['offsets_m'],'settings':asdict(TemporalIKSettings(jerk_weight=selection['chosen_weight'])),
       'calibration_selection':selection,'calibration_path':str(calibration),
       'loss_and_causality':{k:c[k] for k in ['window_loss','output','validation_plan','test_success_criteria']},
       'metric_protocol':parent['metrics'],'aggregation':parent['primary_aggregation'],
       'protected_files':protected,'sources':source_hashes(),'new_experiment':True,'old_baselines_overwritten':False}
    verify_files(p);atomic_json(run/'PROTOCOL.json',p)
    print(json.dumps({'status':'prepared_fixed_test','settings':p['settings']}),flush=True)


def summarize_test(run,p):
    models=['raw','framewise_ik','temporal_ik'];rows=[json.loads(f.read_text()) for f in sorted((run/'rows').glob('*.json'))]
    expected={(m,r['sequence_id']) for m in p['mask_order'] for r in p['records']}
    if len(rows)!=456 or {(r['mask'],r['sequence_id']) for r in rows}!=expected:raise ValueError('Test coverage mismatch')
    result=aggregate(rows,p['mask_order'],models)
    parent=json.loads((I01/'metrics.json').read_text())['overall_mask_equal']
    if result['overall']['raw']!=parent['network_final'] or result['overall']['framewise_ik']!=parent['network_final_ik']:
        raise ValueError('Frozen baseline aggregates changed')
    raw,ind,temp=[result['overall'][m] for m in models]
    result['vertex_gain_retained_fraction']=(raw['mpjve_cm']-temp['mpjve_cm'])/(raw['mpjve_cm']-ind['mpjve_cm'])
    result['jitter_drop_vs_framewise_percent']=(ind['jitter_100_m_s3']-temp['jitter_100_m_s3'])/ind['jitter_100_m_s3']*100
    result['jitter_change_vs_raw_percent']=(temp['jitter_100_m_s3']-raw['jitter_100_m_s3'])/raw['jitter_100_m_s3']*100
    result['predeclared_success']=all(temp[k]<raw[k] for k in PRIMARY[:3]) and result['vertex_gain_retained_fraction']>=.5 and temp['jitter_100_m_s3']<ind['jitter_100_m_s3']
    result['also_below_raw_jitter']=temp['jitter_100_m_s3']<raw['jitter_100_m_s3']
    atomic_json(run/'metrics.json',result)
    import csv
    for filename,fields,values in [
      ('comparison.csv',['model',*PRIMARY],[{'model':m,**result['overall'][m]} for m in models]),
      ('per_mask.csv',['mask','model',*PRIMARY],[{'mask':mask,'model':m,**{k:result['per_mask'][mask][m]['metrics']['frame_weighted'][k] for k in PRIMARY}} for mask in p['mask_order'] for m in models])]:
        with (run/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
    atomic_json(run/'completion.json',{'status':'complete','pairs':456,'settings':p['settings'],'overall':result['overall'],
                                      'predeclared_success':result['predeclared_success'],'also_below_raw_jitter':result['also_below_raw_jitter']})
    return result


def run_test(output,part):
    run=output/'test';p=json.loads((run/'PROTOCOL.json').read_text());verify_files(p)
    if (run/'completion.json').exists():
        print('Already complete; no files changed.',flush=True);return
    torch.set_num_threads(2)
    lock=(run/f'.part{part}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for folder in ['refined','rows']:(run/folder).mkdir(exist_ok=True)
    body=load_body_model(p['smpl']['path']);evaluator=PoseMetricEvaluator(body,fps=25,fk_chunk=32)
    settings=TemporalIKSettings(**p['settings']);offsets=torch.tensor(p['offsets_m'],dtype=torch.float64)
    original=json.loads((I01/'ARTIFACT_MANIFEST.json').read_text());digest=sha256(run/'PROTOCOL.json')
    done=0;start=time.perf_counter()
    for mask in [m for i,m in enumerate(p['mask_order']) if i%2==part]:
        for rec in p['records']:
            seq=rec['sequence_id'];name=f'{mask}_seq{seq:02d}';input_file=I01/'inputs'/f'{name}.npz'
            source_row=I01/'rows'/f'{name}.json';out=run/'refined'/f'{name}.npz';rowfile=run/'rows'/f'{name}.json'
            for source in [input_file,source_row]:
                if sha256(source)!=original[str(source.relative_to(I01))]:raise ValueError('I01 input changed')
            if rowfile.exists():
                row=json.loads(rowfile.read_text())
                if row['protocol_sha256']!=digest or row['output_sha256']!=sha256(out):raise ValueError('Resumed output changed')
                done+=1;continue
            with np.load(input_file,allow_pickle=False) as a:
                refined=refine_temporal(a['predicted_pose'],a['imu_orientations'],a['sensor_ids'].tolist(),offsets,settings)
            arrays_write(out,pose=refined['pose'].numpy(),loss_history=refined['loss_history'].numpy(),
                         sequence_id=seq,mask=mask,protocol_sha256=digest,input_sha256=sha256(input_file))
            with np.load(Path(p['baseline_raw_run'])/'predictions'/f'{name}.npz',allow_pickle=False) as a:target=a['target_pose'].copy()
            old=json.loads(source_row.read_text());new=evaluator.evaluate_sequence(refined['pose'],target,seq)
            models={'raw':old['models']['network_final'],'framewise_ik':old['models']['network_final_ik'],'temporal_ik':new}
            if any(v['settings']!=new['settings'] for v in models.values()):raise ValueError('Metric settings differ')
            atomic_json(rowfile,{'mask':mask,'sequence_id':seq,'protocol_sha256':digest,'input_sha256':sha256(input_file),
                                'output_sha256':sha256(out),'baseline_row_sha256':sha256(source_row),'models':models,'temporal':refined['summary']})
            done+=1;atomic_json(run/f'part{part}_progress.json',{'status':'running','completed':done,'planned':228,'elapsed_seconds':time.perf_counter()-start,'mask':mask,'sequence_id':seq})
            print(f'temporal part{part} {done}/228 {mask} seq{seq}',flush=True)
    atomic_json(run/f'part{part}_progress.json',{'status':'complete','completed':done,'planned':228,'elapsed_seconds':time.perf_counter()-start})
    with (run/'.finalization.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if all((run/f'part{i}_progress.json').exists() and json.loads((run/f'part{i}_progress.json').read_text())['status']=='complete' for i in [0,1]):
            verify_files(p);result=summarize_test(run,p)
            atomic_json(run/'ARTIFACT_MANIFEST.json',{str(f.relative_to(run)):sha256(f) for f in sorted(run.rglob('*')) if f.is_file() and f.name!='ARTIFACT_MANIFEST.json'})
            print('TEMPORAL TEST COMPLETE '+json.dumps(result['overall']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['calibrate','prepare-test','run-test']);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--part',type=int,choices=[0,1],default=0);args=parser.parse_args()
    if args.action=='calibrate':calibrate(args.output)
    elif args.action=='prepare-test':prepare_test(args.output)
    else:run_test(args.output,args.part)
