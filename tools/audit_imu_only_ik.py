"""Audit inherited train-side orientation/shape conventions without fitting to GT."""
import argparse
from pathlib import Path
import json

import torch
from imu_only_ik import SENSOR_NAMES, SENSOR_JOINTS, global_rotations, project_orientations, angular_distance_deg
from runtime import load_torch, sha256

ROOT=Path(__file__).resolve().parents[1]


def audit():
    torch.set_num_threads(2)
    frozen=ROOT/'outputs/local_dip/dip_finetuning_v1_20261114/data_manifest.json'
    manifest=json.loads(frozen.read_text());path=Path(manifest['data_path'])
    if path.name!='dip_train.pt' or sha256(path)!=manifest['data_sha256']:
        raise ValueError('Only the audited first-eight training pool is allowed')
    data=load_torch(path)
    orientation=torch.cat([x[:,:5].double() for x in data['ori']])
    target=project_orientations(orientation)
    orth=torch.linalg.matrix_norm(orientation.transpose(-1,-2)@orientation-torch.eye(3,dtype=torch.float64),ord='fro')
    determinant=torch.linalg.det(orientation)
    distance=torch.linalg.matrix_norm(target-orientation,ord='fro')
    errors=[];by_subject={}
    for rec in manifest['records']:
        seq=rec['sequence_id'];ids=torch.arange(0,len(data['pose'][seq]),25)
        predicted_segment=global_rotations(data['pose'][seq][ids].double())[:,SENSOR_JOINTS]
        actual=project_orientations(data['ori'][seq][ids,:5].double())
        value=angular_distance_deg(predicted_segment,actual)
        errors.append(value);by_subject.setdefault(rec['subject'],[]).append(value)
    errors=torch.cat(errors)
    shapes=sorted({tuple(float(v) for v in x.reshape(-1)) for x in data['shape']})
    return {'purpose':'Source/measurement sanity only; no calibration, parameter fitting, solver selection, or test-data access',
            'data_path':str(path),'data_sha256':sha256(path),'manifest_sha256':sha256(frozen),
            'sequences':len(data['ori']),'frames':len(orientation),
            'provided_shape_betas_unique':[list(v) for v in shapes],
            'current_metric_shape_betas':[0.]*10,
            'shape_conclusion':'Public processed DIP data uses provided betas; F03 canonical measurement deliberately uses shared mean male shape. Author evaluation betas not published.',
            'orientation_handling':'Original entrywise-interpolated matrices retained in dataset/network input; SVD projection only for IK target',
            'projection':{name:{'segment_joint':SENSOR_JOINTS[i],
                                'orthogonality_frobenius_mean':orth[:,i].mean().item(),
                                'orthogonality_frobenius_max':orth[:,i].max().item(),
                                'fraction_orthogonality_error_above_1e_4':(orth[:,i]>1e-4).double().mean().item(),
                                'determinant_min':determinant[:,i].min().item(),
                                'projection_frobenius_mean':distance[:,i].mean().item(),
                                'projection_frobenius_max':distance[:,i].max().item()} for i,name in enumerate(SENSOR_NAMES)},
            'train_only_mapping_sanity':{'sampling':'every25th frame of each first-eight recording, fixed in advance',
                                       'samples':len(errors),'no_gt_derived_calibration_applied':True,
                                       'mean_segment_orientation_error_deg':dict(zip(SENSOR_NAMES,errors.mean(0).tolist())),
                                       'per_subject_mean_deg':{s:dict(zip(SENSOR_NAMES,torch.cat(v).mean(0).tolist())) for s,v in by_subject.items()}},
            'public_mapping':{'raw_DIP_indices':[7,8,9,10,0],'SMPL_segment_joints':list(SENSOR_JOINTS),
                              'stale_leaf_joints_not_used':[20,21,7,8,12],
                              'author_clarification_status':'Issue16 contains another user question; no author confirmation recovered'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=audit();args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
