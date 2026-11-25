"""Explicit CPU-only inference fallback for the same locked paired evaluation.

Separate output/protocol from the prepared GPU run. Both models share the same
CPU batching and canonical two-thread CPU metrics. Stops at a recording boundary
if the preferred ordinary-terminal GPU pipeline starts producing predictions.
No training or changes to the prepared GPU protocol/source are performed.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np
import torch

from compare_dip_finetuning import (MODEL_NAMES, cache_path, row_path, read_row,
    verify_protocol, metric_initializer, measure_pair, summarize)
from dip_finetuning import sensor_mask
from runtime import PosePredictor, load_torch, sha256
from train_dip_finetuning import atomic_json
from window_inference import infer_sequence
from imuposer import math


def prepare(source,output):
    p=copy.deepcopy(verify_protocol(source))
    output.mkdir(parents=True,exist_ok=False)
    p['parent_gpu_protocol_sha256']=sha256(source/'PROTOCOL.json')
    p['preferred_gpu_run']=str(source.resolve())
    p['inference']['batch_size']=64
    p['inference']['execution_device']='cpu'
    p['runtime']={'torch':torch.__version__,'execution_route':'Explicit CPU fallback within Codex',
                  'inference_threads_per_part':8,'metric_threads_per_part':2,'parts':2,
                  'primary_route_policy':'Pause CPU parts once preferred GPU run has prediction progress'}
    p['source_hashes'][str(Path(__file__).resolve())]=sha256(__file__)
    atomic_json(output/'PROTOCOL.json',p)
    shutil.copytree(source/'source_snapshot',output/'source_snapshot')
    shutil.copy2(__file__,output/'source_snapshot/compare_dip_finetuning_cpu.py')
    atomic_json(output/'PREPARE_MANIFEST.json',{str(f.relative_to(output)):sha256(f) for f in sorted(output.rglob('*')) if f.is_file()})


def gpu_started(p):
    path=Path(p['preferred_gpu_run'])/'progress.json'
    if not path.exists():
        return False
    state=json.loads(path.read_text())
    return state.get('prediction_pairs_available',0)>0


def execute(run,part):
    p=verify_protocol(run)
    protocol_sha=sha256(run/'PROTOCOL.json')
    for name in ['rows','predictions']:
        (run/name).mkdir(exist_ok=True)
    lock=(run/f'.cpu_part{part}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metric_initializer(p['smpl']['path'],p['metrics']['fk_chunk'])
    predictors={name:PosePredictor(v['path'],'cpu',8) for name,v in p['models'].items()}
    data=load_torch(p['data']['path'])
    masks=[mask for i,mask in enumerate(p['mask_order']) if i%2==part]
    total=len(masks)*len(p['records']);done=0;start=time.perf_counter()
    for mask in masks:
        keep=sensor_mask(p['masks'][mask])
        for rec in p['records']:
            if gpu_started(p):
                atomic_json(run/f'part{part}_progress.json',{'status':'paused_for_preferred_gpu_run','completed':done,'planned':total})
                print(f'CPU part{part}: pausing because the preferred GPU pipeline started.',flush=True)
                return
            seq,n=rec['sequence_id'],rec['frames']
            if read_row(run,mask,seq,protocol_sha):
                done+=1;continue
            path=cache_path(run,mask,seq)
            target=math.rotation_matrix_to_r6d(data['pose'][seq]).reshape(n,144)
            if not path.exists():
                x=torch.cat([(data['acc'][seq][:,:5]/30).flatten(1),data['ori'][seq][:,:5].float().flatten(1)],1)
                x[:,~keep]=0
                torch.set_num_threads(8)
                tick=time.perf_counter()
                predictions={name:infer_sequence(model,x,mode='rolling_last',window=125,startup='prefix',batch_size=64).numpy()
                             for name,model in predictors.items()}
                temporary=path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
                with temporary.open('xb') as stream:
                    np.savez_compressed(stream,target_pose=target.numpy(),sequence_id=seq,mask=mask,
                        protocol_sha256=protocol_sha,fps=25,**{f'pose_{name}':v for name,v in predictions.items()})
                os.replace(temporary,path)
                print(f'CPU part{part} inferred {mask} seq{seq} ({n}x2 frames) in {time.perf_counter()-tick:.1f}s',flush=True)
            with np.load(path,allow_pickle=False) as saved:
                if str(saved['protocol_sha256'])!=protocol_sha or not np.array_equal(saved['target_pose'],target.numpy()):
                    raise ValueError('CPU cache reference/protocol differs')
            torch.set_num_threads(2)
            row=measure_pair(str(path),protocol_sha)
            atomic_json(row_path(run,mask,seq),row)
            done+=1
            atomic_json(run/f'part{part}_progress.json',{'status':'running','completed':done,'planned':total,
                        'elapsed_seconds':time.perf_counter()-start,'last_mask':mask,'last_sequence_id':seq})
            print(f'CPU part{part} metrics {done}/{total}',flush=True)
    atomic_json(run/f'part{part}_progress.json',{'status':'complete','completed':done,'planned':total,'elapsed_seconds':time.perf_counter()-start})
    with (run/'.finalization.lock').open('a') as final_lock:
        fcntl.flock(final_lock,fcntl.LOCK_EX)
        if all((run/f'part{i}_progress.json').exists() and json.loads((run/f'part{i}_progress.json').read_text())['status']=='complete' for i in [0,1]):
            verify_protocol(run)
            summarize(run,p)
            atomic_json(run/'resources.json',{'inference_device':'cpu','parts':2,'inference_threads_per_part':8,'metric_threads_per_part':2})
            atomic_json(run/'ARTIFACT_MANIFEST.json',{str(f.relative_to(run)):sha256(f) for f in sorted(run.rglob('*')) if f.is_file() and f.name!='ARTIFACT_MANIFEST.json'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--part',type=int,choices=[0,1],default=0)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.source,args.output)
    else:
        execute(args.output,args.part)


if __name__=='__main__':
    main()
