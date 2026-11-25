"""Controlled pose-component counterfactuals for I01 jitter; not new IK recipes."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from analyze_ik_jitter import rotations_and_positions,third_difference
from runtime import load_body_model,sha256
from train_dip_finetuning import atomic_json

VARIANTS={'restore_pelvis':[0],'restore_hips':[1,2],'restore_arms':[16,17,18,19],'restore_head':[12,15]}


def main(run):
    if run.exists():raise FileExistsError(run)
    root=Path(__file__).resolve().parents[1];ik=root/'outputs/local_dip/dip_imu_only_ik_v1_20261116'
    p=json.loads((ik/'PROTOCOL.json').read_text());base=Path(p['baseline_run'])
    torch.set_num_threads(2);body=load_body_model(p['smpl']['path']);rows=[]
    run.mkdir(parents=True)
    atomic_json(run/'PROTOCOL.json',{'source_sha256':sha256(__file__),'ik_protocol_sha256':sha256(ik/'PROTOCOL.json'),
                                    'scope':'Controlled pose-component removal/retention, not constrained inference; norm effects are not additive',
                                    'variants':{**VARIANTS,'pelvis_only':[0]}})
    for mask in p['mask_order']:
        for rec in p['records']:
            seq,n=rec['sequence_id'],rec['frames'];name=f'{mask}_seq{seq:02d}'
            with np.load(base/'predictions'/f'{name}.npz',allow_pickle=False) as a,np.load(ik/'refined'/f'{name}.npz',allow_pickle=False) as b:
                raw=a['pose_dip_final_2ep'].reshape(n,24,6);ref=b['pose'].reshape(n,24,6)
                values={}
                for variant,joints in {**VARIANTS,'pelvis_only':[0]}.items():
                    pose=raw.copy() if variant=='pelvis_only' else ref.copy()
                    pose[:,joints]=ref[:,joints] if variant=='pelvis_only' else raw[:,joints]
                    _,_,positions=rotations_and_positions(body,pose.reshape(n,144))
                    values[variant]=np.linalg.norm(third_difference(positions)*25**3,axis=-1).mean()/100
            rows.append({'mask':mask,'sequence_id':seq,'jerk_samples':n-3,'values':values})
        print('counterfactual mask '+mask,flush=True)
    means={key:sum(row['values'][key]*row['jerk_samples'] for row in rows)/(24*24021) for key in rows[0]['values']}
    baseline=json.loads((ik/'metrics.json').read_text())['overall_mask_equal']
    atomic_json(run/'results.json',{'overall':means,'baseline':baseline,'rows':rows,
                                   'interpretation':'Removing/retaining a component probes the kinematic source; nonlinear norm effects must not be summed'})
    atomic_json(run/'ARTIFACT_MANIFEST.json',{str(f.relative_to(run)):sha256(f) for f in sorted(run.rglob('*')) if f.is_file() and f.name!='ARTIFACT_MANIFEST.json'})
    print(json.dumps(means),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
