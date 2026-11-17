"""Exploratory fixed-case shape/blendshape sensitivity; never changes primary IK."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from paper_metrics import PoseMetricEvaluator
from runtime import load_body_model,sha256

KEYS=['mpjre_deg','mpjpe_cm','mpjve_cm','jitter_100_m_s3']


def audit(run):
    p=json.loads((run/'PROTOCOL.json').read_text());base=Path(p['baseline_run']);cases=[]
    for mask in [p['mask_order'][0],p['mask_order'][-1]]:
        for rec in [p['records'][0],p['records'][-1]]:
            seq=rec['sequence_id'];name=f'{mask}_seq{seq:02d}'
            row=json.loads((run/'rows'/f'{name}.json').read_text())
            with np.load(base/'predictions'/f'{name}.npz',allow_pickle=False) as original,np.load(run/'refined'/f'{name}.npz',allow_pickle=False) as refined:
                for setting,shape,blend in [('shared_beta_ones',torch.ones(10),False),('pose_blendshape_on',None,True)]:
                    body=load_body_model(p['smpl']['path']);body.use_pose_blendshape=blend
                    evaluator=PoseMetricEvaluator(body,fps=25,fk_chunk=32)
                    for model,pose in [('network_final',original['pose_dip_final_2ep']),('network_final_ik',refined['pose'])]:
                        measured=evaluator.evaluate_sequence(pose,original['target_pose'],seq,shape=shape)
                        difference={key:measured['metrics'][key]-row['models'][model]['metrics'][key] for key in KEYS}
                        if abs(difference['mpjre_deg'])>1e-8:raise ValueError('Shape/mesh flag changed angular error')
                        if blend and max(abs(difference[k]) for k in ['mpjpe_cm','jitter_100_m_s3'])>1e-8:
                            raise ValueError('Pose blendshape unexpectedly changed joint metric')
                        cases.append({'mask':mask,'sequence_id':seq,'frames':rec['frames'],'model':model,'setting':setting,
                                      'metrics':{k:measured['metrics'][k] for k in KEYS},'change_from_primary':difference})
    return {'scope':'Exploratory sensitivity on four fixed whole-recording mask pairs, not a full24-mask score or inferred author setting',
            'case_selection':'First/last masks and first/last recordings; same fixed positions as independent validator',
            'primary_protocol_unchanged':True,'checkpoint_and_predictions_unchanged':True,'no_calibration_or_tuning':True,
            'source_sha256':sha256(__file__),'cases':cases}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(2);result=audit(args.run)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
