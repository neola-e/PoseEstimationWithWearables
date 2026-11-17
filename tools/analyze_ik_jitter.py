"""Attribute I01 jerk increases and measure rotation/correction high frequencies.

All derivatives stay within recordings. Contributions add exactly to the frozen
24-mask,24-joint,T-3 weighted jitter difference. Frequency bands are descriptive
signal measurements, not ground-truth noise labels.
"""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from paper_metrics import local_rotations
from runtime import load_body_model,sha256
from sensor_metrics import JOINT_NAMES
from train_dip_finetuning import atomic_json

ROOT=Path(__file__).resolve().parents[1]
IK=ROOT/'outputs/local_dip/dip_imu_only_ik_v1_20261116'
OVERVIEW=Path('/mnt/d/01_Downloads/DIPIMUandOthers/DIP_IMU_and_Others/overview_of_published_data.txt')
CATEGORIES={1:'upper_body',2:'lower_body',3:'locomotion',4:'freestyle',5:'interaction'}


def third_difference(x):return x[3:]-3*x[2:-1]+3*x[1:-2]-x[:-3]


def band_energy(signal,fps=25.,cutoff=5.):
    """Integrated one-sided Hann periodogram; remove a linear trend per channel."""
    x=np.asarray(signal,dtype=np.float64)
    if len(x)<4:raise ValueError('Need at least4 samples')
    time=np.arange(len(x),dtype=np.float64);time-=time.mean()
    shape=(len(x),)+((1,)*(x.ndim-1));t=time.reshape(shape)
    x=x-x.mean(axis=0,keepdims=True)
    x=x-t*(x*t).sum(axis=0,keepdims=True)/(time*time).sum()
    window=np.hanning(len(x));spectrum=np.fft.rfft(x*window.reshape(shape),axis=0)
    power=np.abs(spectrum)**2/(len(x)*(window*window).sum())
    weights=np.ones(len(power));weights[1:]=2
    if len(x)%2==0:weights[-1]=1
    power*=weights.reshape((len(power),)+((1,)*(x.ndim-1)))
    frequencies=np.fft.rfftfreq(len(x),1/fps)
    return power[frequencies>=cutoff].sum(axis=(0,-1)),power.sum(axis=(0,-1))


def rotation_vectors(matrices):
    r=np.asarray(matrices)
    return Rotation.from_matrix(r.reshape(-1,3,3)).as_rotvec().reshape(*r.shape[:-2],3)


def rotations_and_positions(body,pose,chunk=32):
    local=local_rotations(pose)
    global_r,positions=[],[]
    with torch.inference_mode():
        for first in range(0,len(local),chunk):
            a,b=body.forward_kinematics(local[first:first+chunk],tran=torch.zeros(min(chunk,len(local)-first),3),calc_mesh=False)
            global_r.append(a.double().numpy());positions.append(b.double().numpy())
    return local.double().numpy(),np.concatenate(global_r),np.concatenate(positions)


def analyze_pair(body,raw,refined):
    local0,global0,pos0=rotations_and_positions(body,raw)
    local1,global1,pos1=rotations_and_positions(body,refined)
    jerk0=third_difference(pos0)*25**3;jerk1=third_difference(pos1)*25**3
    norm0=np.linalg.norm(jerk0,axis=-1);norm1=np.linalg.norm(jerk1,axis=-1)
    local_velocity0=rotation_vectors(np.swapaxes(local0[:-1],-1,-2)@local0[1:])*25
    local_velocity1=rotation_vectors(np.swapaxes(local1[:-1],-1,-2)@local1[1:])*25
    global_velocity0=rotation_vectors(np.swapaxes(global0[:-1],-1,-2)@global0[1:])*25
    global_velocity1=rotation_vectors(np.swapaxes(global1[:-1],-1,-2)@global1[1:])*25
    correction=rotation_vectors(np.swapaxes(local0,-1,-2)@local1)
    raw_hf,raw_all=band_energy(local_velocity0);ik_hf,ik_all=band_energy(local_velocity1)
    corr_hf,corr_all=band_energy(correction)
    correction_step=np.linalg.norm(np.diff(correction,axis=0),axis=-1)*180/np.pi
    correction_jump=(np.abs(np.diff(correction,axis=0))>.015).any(-1)
    per_joint=[]
    for j in range(24):
        per_joint.append({'joint':j,'joint_name':JOINT_NAMES[j],
                          'raw_jitter_100_m_s3':norm0[:,j].mean()/100,
                          'ik_jitter_100_m_s3':norm1[:,j].mean()/100,
                          'jitter_delta_sum_m_s3':float((norm1[:,j]-norm0[:,j]).sum()),
                          'local_step_raw_deg':np.linalg.norm(local_velocity0[:,j],axis=-1).mean()/25*180/np.pi,
                          'local_step_ik_deg':np.linalg.norm(local_velocity1[:,j],axis=-1).mean()/25*180/np.pi,
                          'global_step_raw_deg':np.linalg.norm(global_velocity0[:,j],axis=-1).mean()/25*180/np.pi,
                          'global_step_ik_deg':np.linalg.norm(global_velocity1[:,j],axis=-1).mean()/25*180/np.pi,
                          'correction_step_deg':float(correction_step[:,j].mean()),
                          'correction_large_jumps':int(correction_jump[:,j].sum()),
                          'local_velocity_hf_raw_power':float(raw_hf[j]),'local_velocity_total_raw_power':float(raw_all[j]),
                          'local_velocity_hf_ik_power':float(ik_hf[j]),'local_velocity_total_ik_power':float(ik_all[j]),
                          'correction_hf_power':float(corr_hf[j]),'correction_total_power':float(corr_all[j])})
    return per_joint,{'jerk_norm_raw':norm0.astype(np.float32),'jerk_norm_ik':norm1.astype(np.float32),
                      'correction_rotvec':correction.astype(np.float32),
                      'local_step_raw_deg':(np.linalg.norm(local_velocity0,axis=-1)/25*180/np.pi).astype(np.float32),
                      'local_step_ik_deg':(np.linalg.norm(local_velocity1,axis=-1)/25*180/np.pi).astype(np.float32),
                      'global_step_raw_deg':(np.linalg.norm(global_velocity0,axis=-1)/25*180/np.pi).astype(np.float32),
                      'global_step_ik_deg':(np.linalg.norm(global_velocity1,axis=-1)/25*180/np.pi).astype(np.float32),
                      'correction_step_deg':correction_step.astype(np.float32)}


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def summarize(run,p):
    import math
    records=p['records'];rows=[json.loads(f.read_text()) for f in sorted((run/'rows').glob('*.json'))]
    if len(rows)!=456:raise ValueError('Full456-pair diagnosis required')
    denominator=24*24*sum(rec['frames']-3 for rec in records)*100
    pairs=[];joint_rows=[]
    for row in rows:
        meta={k:row[k] for k in ['mask','sequence_id','subject','source_file','category','frames','jerk_samples']}
        delta_sum=math.fsum(j['jitter_delta_sum_m_s3'] for j in row['joints'])
        pairs.append({**meta,'raw_jitter':row['raw_jitter'],'ik_jitter':row['ik_jitter'],
                      'mean_increase':row['ik_jitter']-row['raw_jitter'],
                      'overall_contribution':delta_sum/denominator,'peak_output_frame':row['peak_output_frame'],
                      'peak_joint':row['peak_joint'],'peak_single_joint_increase_m_s3':row['peak_single_joint_increase_m_s3']})
        for j in row['joints']:joint_rows.append({**meta,**j,'overall_contribution':j['jitter_delta_sum_m_s3']/denominator})
    table_groups={}
    for name,fields in [('by_joint',['joint','joint_name']),('by_mask',['mask']),('by_sequence',['sequence_id','subject','source_file','category']),
                        ('by_category',['category']),('by_joint_category',['joint','joint_name','category'])]:
        groups={}
        for j in joint_rows:groups.setdefault(tuple(j[k] for k in fields),[]).append(j)
        output=[]
        for key,values in groups.items():
            n=sum(v['jerk_samples'] for v in values)
            raw=math.fsum(v['raw_jitter_100_m_s3']*v['jerk_samples'] for v in values)/n
            ik=math.fsum(v['ik_jitter_100_m_s3']*v['jerk_samples'] for v in values)/n
            item={**dict(zip(fields,key)),'raw_jitter':raw,'ik_jitter':ik,'mean_increase':ik-raw,
                  'overall_contribution':math.fsum(v['overall_contribution'] for v in values)}
            for field in ['local_step_raw_deg','local_step_ik_deg','global_step_raw_deg','global_step_ik_deg','correction_step_deg',
                          'local_velocity_hf_raw_power','local_velocity_hf_ik_power','correction_hf_power','correction_total_power']:
                item[field]=math.fsum(v[field]*(v['frames']-1) for v in values)/sum(v['frames']-1 for v in values)
            item['correction_large_jumps']=sum(v['correction_large_jumps'] for v in values)
            output.append(item)
        output.sort(key=lambda x:x['overall_contribution'],reverse=True);write_csv(run/(name+'.csv'),output);table_groups[name]=output
    pairs.sort(key=lambda x:x['overall_contribution'],reverse=True);write_csv(run/'by_pair.csv',pairs)
    write_csv(run/'joint_pair_details.csv',joint_rows)
    contribution=math.fsum(v['overall_contribution'] for v in pairs)
    expected=p['baseline_delta']
    if abs(contribution-expected)>1e-10:raise ValueError('Jitter attribution does not sum to canonical baseline change')
    summary={'coverage':{'pairs':len(rows),'masks':24,'recordings':19},'overall_increase':contribution,
             'canonical_baseline_increase':expected,'attribution_max_abs_difference':abs(contribution-expected),
             'rankings':table_groups,'top_pairs':pairs[:20],
             'frequency_definition':'5–12.5Hz local angular-velocity/correction power; per-record linear detrend and Hann window; high frequency is not inherently noise',
             'category_definition':'File category prefix from local dataset overview + DIP AppendixF Table6; not per-frame action labels'}
    atomic_json(run/'summary.json',summary);return summary


def execute(run):
    if run.exists():raise FileExistsError(run)
    torch.set_num_threads(2)
    p=json.loads((IK/'PROTOCOL.json').read_text());original=json.loads((IK/'ARTIFACT_MANIFEST.json').read_text())
    for name,digest in original.items():
        if sha256(IK/name)!=digest:raise ValueError('I01 artifact changed: '+name)
    baseline=json.loads((IK/'metrics.json').read_text())
    run.mkdir(parents=True);(run/'rows').mkdir();(run/'time_series').mkdir()
    protocol={'format':'ik-jitter-attribution-v1','created_utc':datetime.now(timezone.utc).isoformat(),
              'ik_run':str(IK),'ik_manifest_sha256':sha256(IK/'ARTIFACT_MANIFEST.json'),
              'baseline_run':p['baseline_run'],'smpl':p['smpl'],'records':p['records'],'mask_order':p['mask_order'],
              'baseline_delta':baseline['changes']['jitter_100_m_s3']['after_minus_before'],
              'overview':{'path':str(OVERVIEW),'sha256':sha256(OVERVIEW)},
              'activity_source':'https://arxiv.org/pdf/1810.04703 AppendixF Table6; overview maps prefixes to categories',
              'categories':CATEGORIES,'fps':25,'high_frequency_cutoff_hz':5,
              'source_sha256':sha256(__file__),'derivative_boundaries':'within whole recordings, support[t,t+4)',
              'contributions':'sum signed per-time per-joint norm increases / (24 masks*24 joints*24021 time samples*100)',
              'scope':'Descriptive test-set error analysis; no IK tuning or model selection'}
    atomic_json(run/'PROTOCOL.json',protocol);shutil.copy2(__file__,run/'source_snapshot.py')
    body=load_body_model(p['smpl']['path']);base=Path(p['baseline_run']);done=0
    for mask in p['mask_order']:
        for rec in p['records']:
            seq,n=rec['sequence_id'],rec['frames'];name=f'{mask}_seq{seq:02d}'
            with np.load(base/'predictions'/f'{name}.npz',allow_pickle=False) as raw,np.load(IK/'refined'/f'{name}.npz',allow_pickle=False) as refined:
                joints,arrays=analyze_pair(body,raw['pose_dip_final_2ep'],refined['pose'])
            old=json.loads((IK/'rows'/f'{name}.json').read_text())
            means={model:np.mean([j[key] for j in joints]) for model,key in [('network_final','raw_jitter_100_m_s3'),('network_final_ik','ik_jitter_100_m_s3')]}
            for model,value in means.items():
                if abs(value-old['models'][model]['metrics']['jitter_100_m_s3'])>1e-10:
                    raise ValueError('Joint-only FK diagnosis differs from canonical metric')
            delta=arrays['jerk_norm_ik'].astype(np.float64)-arrays['jerk_norm_raw'].astype(np.float64)
            t,j=np.unravel_index(np.argmax(delta),delta.shape)
            row={'mask':mask,'sequence_id':seq,'subject':rec['subject'],'source_file':rec['source_file'],
                 'category':CATEGORIES[int(rec['source_file'][:2])],'frames':n,'jerk_samples':n-3,
                 'raw_jitter':means['network_final'],'ik_jitter':means['network_final_ik'],'joints':joints,
                 'peak_output_frame':int(t+3),'peak_joint':int(j),'peak_single_joint_increase_m_s3':float(delta[t,j]),
                 'raw_cache_sha256':sha256(base/'predictions'/f'{name}.npz'),
                 'ik_cache_sha256':sha256(IK/'refined'/f'{name}.npz')}
            np.savez_compressed(run/'time_series'/f'{name}.npz',**arrays)
            atomic_json(run/'rows'/f'{name}.json',row);done+=1
            if done%24==0:print(f'diagnostic {done}/456',flush=True)
    summary=summarize(run,protocol)
    atomic_json(run/'completion.json',{'status':'complete','pairs':done,'overall_increase':summary['overall_increase']})
    atomic_json(run/'ARTIFACT_MANIFEST.json',{str(f.relative_to(run)):sha256(f) for f in sorted(run.rglob('*')) if f.is_file() and f.name!='ARTIFACT_MANIFEST.json'})
    print('DIAGNOSIS COMPLETE '+json.dumps({'increase':summary['overall_increase'],'top_joint':summary['rankings']['by_joint'][0],'top_sequence':summary['rankings']['by_sequence'][0],'top_category':summary['rankings']['by_category'][0]}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    execute(parser.parse_args().output)
