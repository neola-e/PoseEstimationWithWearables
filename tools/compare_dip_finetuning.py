"""Freeze and run the paired 24-mask DIP test comparison, without training.

GPU inference runs in the ordinary WSL terminal. Spawned CPU workers use the
unchanged, validated paper_metrics implementation. Per-recording caches/rows
allow restarting without recomputing completed predictions or measurements.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import fcntl
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import torch
import imuposer
from imuposer import math

from dip_finetuning import sensor_mask, validate_masks
from eval_wrist_pocket_dev import fingerprint, raw_inputs
from paper_metrics import PoseMetricEvaluator, aggregate_sequences
from runtime import PosePredictor, load_body_model, load_torch, peak_ram_mib, sha256
from train_dip_finetuning import atomic_json, final_budget, read_committed
from window_inference import infer_sequence

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'outputs/local_dip/dip_finetuning_v1_20261114'
FINAL = ROOT/'outputs/local_dip/dip_final_v1'
MODEL_NAMES = ['amass_epoch59','dip_final_2ep']
PRIMARY = ['mpjre_deg','mpjpe_cm','mpjve_cm','jitter_100_m_s3']
EXPECTED_TEST_SHA = '352ce4f1312fbc7d496ce3d3d3c7287f01389624b75296ce4e349868b958b475'
_EVALUATOR = None


def source_inventory():
    paths = [Path(__file__),ROOT/'tools/runtime.py',ROOT/'tools/window_inference.py',
             ROOT/'tools/paper_metrics.py',ROOT/'tools/dip_finetuning.py',
             ROOT/'tools/eval_wrist_pocket_dev.py',ROOT/'tools/train_dip_finetuning.py',
             ROOT/'tests/test_compare_dip_finetuning.py']
    paths += sorted(Path(imuposer.__file__).parent.rglob('*.py'))
    return {str(p.resolve()):sha256(p) for p in paths}


def verify_protocol(run):
    for name,digest in json.loads((run/'PREPARE_MANIFEST.json').read_text()).items():
        if sha256(run/name) != digest:
            raise ValueError(f'Prepared evaluation artifact changed: {name}')
    p = json.loads((run/'PROTOCOL.json').read_text())
    if p['format'] != 'dip-paired-four-metrics-v1' or p['model_order'] != MODEL_NAMES:
        raise ValueError('Unexpected comparison protocol')
    for path,digest in p['source_hashes'].items():
        if sha256(path) != digest:
            raise ValueError(f'Comparison source changed: {path}')
    for model in p['models'].values():
        if sha256(model['path']) != model['sha256']:
            raise ValueError('Frozen comparison checkpoint changed')
    if sha256(p['data']['path']) != p['data']['sha256'] or sha256(p['smpl']['path']) != p['smpl']['sha256']:
        raise ValueError('Comparison input data/body changed')
    return p


def prepare(args):
    frozen = json.loads((FROZEN/'PROTOCOL.json').read_text())
    final_info = json.loads((FINAL/'RUN.json').read_text())
    completion = json.loads((FINAL/'completion.json').read_text())
    state,saved = read_committed(FINAL)
    k,selection = final_budget(final_info['selection']['run'],sha256(FROZEN/'PROTOCOL.json'),30)
    if (k != 2 or state['stage'] != 'final' or saved['epoch'] != 1 or saved['global_step'] != 162
            or not saved['eligible_for_final_reporting'] or saved['kind'] != 'dip_final_fixed_budget'
            or completion['status'] != 'complete' or completion['test_data_opened']
            or completion['epochs_completed'] != 2 or completion['global_step'] != 162
            or completion['last_checkpoint'] != state['last_checkpoint']
            or final_info['selection'] != selection
            or saved['training_protocol_sha256'] != sha256(FROZEN/'PROTOCOL.json')):
        raise ValueError('Final model does not match the fixed two-epoch first-eight protocol')
    for name,digest in final_info['execution_sources'].items():
        if sha256(name) != digest:
            raise ValueError('Training source no longer matches the final run')
    if sha256(frozen['checkpoint']['path']) != frozen['checkpoint']['sha256']:
        raise ValueError('Original initialization changed')
    data_path = Path(frozen['checkpoint']['path']).parents[2]/'data/processed_imuposer_25fps/dip_test.pt'
    if sha256(data_path) != EXPECTED_TEST_SHA:
        raise ValueError('Test container differs from the audited version')
    torch.set_num_threads(2)
    data = load_torch(data_path)
    lookup = {}
    for i in range(len(data['acc'])):
        key = tuple(fingerprint(data[field][i]) for field in ['acc','ori','pose'])
        if key in lookup:
            raise ValueError('Duplicate processed test content')
        lookup[key] = i
    records,excluded = [],[]
    raw_root = Path(frozen['checkpoint']['path']).parents[2]/'data/raw/DIP_IMU'
    for subject in ['s_09','s_10']:
        paths = sorted((raw_root/subject).glob('*.pkl'))
        if not paths:
            raise FileNotFoundError(f'Missing raw test identity files for {subject}')
        for path in paths:
            meta = {'subject':subject,'source_file':path.name,'raw_path':str(path.resolve()),'raw_sha256':sha256(path)}
            values = raw_inputs(path)
            if values is None:
                excluded.append({**meta,'reason':'Fails inherited finite-value preprocessing'})
                continue
            key = tuple(fingerprint(values[field]) for field in ['acc','ori','pose'])
            if key not in lookup:
                raise ValueError('Raw test motion has no exact processed match')
            records.append({**meta,'sequence_id':lookup[key],'frames':len(values['acc']),
                            'processed_sha256':dict(zip(['acc','ori','pose'],key))})
            print(f'Test identity: {subject}/{path.name} -> sequence {lookup[key]}',flush=True)
    if sorted(r['sequence_id'] for r in records) != list(range(19)) or sum(r['frames'] for r in records) != 24078:
        raise ValueError('Expected full test coverage: 19 recordings, 24078 frames')
    validate_masks(frozen['masks'])
    run = Path(args.output)
    run.mkdir(parents=True,exist_ok=False)
    p = {'format':'dip-paired-four-metrics-v1','frozen_utc':datetime.now(timezone.utc).isoformat(),
         'model_order':MODEL_NAMES,'models':{
             MODEL_NAMES[0]:frozen['checkpoint'],
             MODEL_NAMES[1]:{'path':str((FINAL/state['last_checkpoint']['path']).resolve()),
                             'sha256':state['last_checkpoint']['sha256'],'completed_epochs':2,'global_step':162}},
         'training_protocol_sha256':sha256(FROZEN/'PROTOCOL.json'),
         'final_run_id':final_info['run_id'],'final_budget_selection':selection,
         'data':{'path':str(data_path),'sha256':EXPECTED_TEST_SHA,'subjects':['s_09','s_10'],
                 'sequences':19,'frames_per_mask':24078,'jerk_samples_per_mask':24021},
         'records':sorted(records,key=lambda x:x['sequence_id']),'raw_exclusions':excluded,
         'masks':frozen['masks'],'mask_order':frozen['mask_order'],
         'inference':{'window':125,'stride':1,'output':'last valid frame','startup':'available prefix',
                      'state':'reset every window and recording','batch_size':256,
                      'filter':'Inherited centered acceleration mean; no extra smoothing',
                      'fps':25,'feature_dim':60,'acc_scale':30,'precision':'FP32, TF32 disabled, no AMP'},
         'metrics':{'implementation':'unchanged tools/paper_metrics.py','device':'CPU',
                    'fk_chunk':32,'joints':24,'vertices':6890,'pose_blendshape':False,
                    'shape':'same mean male SMPL','spatial_alignment':'pelvis translation only',
                    'trajectory_translation':'zero for both','jitter':'whole-recording third difference; no boundary crossing',
                    'primary':PRIMARY,'aggregation':'Frame-weight spatial and T-3-weight jerk within each mask; equal mean over all24 masks',
                    'refinement':'none for either model'},
         'smpl':frozen['smpl_model'],'source_hashes':source_inventory(),
         'scope':'Locked before/after test comparison. No training, checkpoint selection, IK, or tuning from test scores.',
         'known_limits':['AMASS initialization had25 masks','Paper overlap/startup wording not fully recovered',
                         'Centered filter and inherited filling/interpolation use future information','No IMU-only IK'],
         'runtime':{'torch':torch.__version__,'cpu_metric_workers':2,'threads_per_process':2,
                    'gpu_execution_route':'ordinary WSL Terminal under user-selected unchanged sandbox permissions'}}
    atomic_json(run/'PROTOCOL.json',p)
    package = Path(imuposer.__file__).parent
    for name in p['source_hashes']:
        source = Path(name)
        relative = Path('team')/source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('installed_imuposer')/source.relative_to(package)
        dest = run/'source_snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    atomic_json(run/'PREPARE_MANIFEST.json',{str(f.relative_to(run)):sha256(f) for f in sorted(run.rglob('*')) if f.is_file()})
    print(json.dumps({'status':'prepared','output':str(run),'models':p['models'],'pairs':24*19,'predicted_frames':2*24*24078}),flush=True)


def metric_initializer(smpl_path,fk_chunk):
    global _EVALUATOR
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    _EVALUATOR = PoseMetricEvaluator(load_body_model(smpl_path),fps=25,fk_chunk=fk_chunk)


def measure_pair(cache_path,protocol_sha):
    with np.load(cache_path,allow_pickle=False) as cache:
        if str(cache['protocol_sha256']) != protocol_sha:
            raise ValueError('Cache belongs to a different protocol')
        target = cache['target_pose']
        seq,mask = int(cache['sequence_id']),str(cache['mask'])
        result = {model:_EVALUATOR.evaluate_sequence(cache[f'pose_{model}'],target,seq) for model in MODEL_NAMES}
    return {'mask':mask,'sequence_id':seq,'protocol_sha256':protocol_sha,
            'cache_sha256':sha256(cache_path),'models':result}


def aggregate_pairs(rows,p):
    expected = {(mask,r['sequence_id']) for mask in p['mask_order'] for r in p['records']}
    observed = [(r['mask'],r['sequence_id']) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError('Missing/extra/duplicate mask-sequence results')
    lengths = {r['sequence_id']:r['frames'] for r in p['records']}
    per_mask = {}
    for mask in p['mask_order']:
        per_mask[mask] = {}
        for model in MODEL_NAMES:
            selected = [r['models'][model] for r in rows if r['mask']==mask]
            for row in selected:
                if row['frames'] != lengths[int(row['sequence_id'])] or row['jerk_samples'] != max(row['frames']-3,0):
                    raise ValueError('Per-recording frame or derivative coverage changed')
            per_mask[mask][model] = aggregate_sequences(selected)
    global_metrics = {model:{key:float(np.mean([per_mask[m][model]['metrics']['frame_weighted'][key]
                                               for m in p['mask_order']])) for key in PRIMARY} for model in MODEL_NAMES}
    reference = [per_mask[m][model]['metrics']['frame_weighted']['reference_jitter_m_s3'] for m in p['mask_order'] for model in MODEL_NAMES]
    if any(v != reference[0] for v in reference):
        raise ValueError('Ground-truth trajectory/coverage differs between configurations')
    changes = {}
    for key in PRIMARY:
        before,after = [global_metrics[model][key] for model in MODEL_NAMES]
        changes[key] = {'after_minus_before':after-before,
                        'improvement_percent':None if before==0 else (before-after)/before*100}
    return {'overall_mask_equal':global_metrics,'changes':changes,'per_mask':per_mask,
            'reference_jitter_m_s3':reference[0],
            'coverage':{'masks':len(p['mask_order']),'recordings_per_mask':len(p['records']),
                        'frames_per_mask':sum(lengths.values()),'pairs':len(rows)}}


def row_path(run,mask,seq):
    return run/'rows'/f'{mask}_seq{seq:02d}.json'


def cache_path(run,mask,seq):
    return run/'predictions'/f'{mask}_seq{seq:02d}.npz'


def read_row(run,mask,seq,protocol_sha):
    path = row_path(run,mask,seq)
    if not path.exists():
        return None
    row = json.loads(path.read_text())
    if (row['protocol_sha256'] != protocol_sha or row['mask'] != mask or row['sequence_id'] != seq
            or sha256(cache_path(run,mask,seq)) != row['cache_sha256']):
        raise ValueError('Saved metric row or prediction cache changed')
    return row


def summarize(run,p):
    digest = sha256(run/'PROTOCOL.json')
    rows = [read_row(run,mask,rec['sequence_id'],digest) for mask in p['mask_order'] for rec in p['records']]
    if any(r is None for r in rows):
        raise ValueError('Metric comparison is not complete')
    result = aggregate_pairs(rows,p)
    atomic_json(run/'metrics.json',result)
    with (run/'comparison.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['model',*PRIMARY])
        writer.writeheader()
        for model in MODEL_NAMES:
            writer.writerow({'model':model,**result['overall_mask_equal'][model]})
    with (run/'per_mask.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=['mask','model',*PRIMARY])
        writer.writeheader()
        for mask in p['mask_order']:
            for model in MODEL_NAMES:
                v = result['per_mask'][mask][model]['metrics']['frame_weighted']
                writer.writerow({'mask':mask,'model':model,**{key:v[key] for key in PRIMARY}})
    atomic_json(run/'completion.json',{'status':'complete','protocol_sha256':digest,
                'models':p['models'],'coverage':result['coverage'],'training_performed':False,
                'checkpoint_selection_performed':False,'overall':result['overall_mask_equal'],'changes':result['changes']})
    print('COMPARISON COMPLETE: '+json.dumps(result['overall_mask_equal']),flush=True)
    return result


def run_comparison(args):
    run = Path(args.output)
    p = verify_protocol(run)
    if (run/'completion.json').exists():
        return summarize(run,p)
    protocol_sha = sha256(run/'PROTOCOL.json')
    torch.set_num_threads(2)
    if not args.metrics_only and not torch.cuda.is_available():
        raise ValueError('GPU inference requires the ordinary WSL terminal; no automatic CPU fallback')
    for folder in ['rows','predictions']:
        (run/folder).mkdir(exist_ok=True)
    data = load_torch(p['data']['path'])
    for record in p['records']:
        for field,digest in record['processed_sha256'].items():
            if fingerprint(data[field][record['sequence_id']]) != digest:
                raise ValueError('Test recording identity changed')
    predictors = {} if args.metrics_only else {name:PosePredictor(v['path'],'cuda:0',2) for name,v in p['models'].items()}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pending,completed,predicted = {},0,0
    started = time.perf_counter()
    def save_progress():
        atomic_json(run/'progress.json',{'completed_metric_pairs':completed,'planned_pairs':len(p['masks'])*len(p['records']),
                    'prediction_pairs_available':predicted,'elapsed_seconds':time.perf_counter()-started})
    def collect(future):
        nonlocal completed
        row = future.result()
        atomic_json(row_path(run,row['mask'],row['sequence_id']),row)
        completed += 1
        del pending[future]
        print(f'Metrics {completed}/456: {row["mask"]} seq{row["sequence_id"]}',flush=True)
        save_progress()
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=2,mp_context=context,initializer=metric_initializer,
                             initargs=(p['smpl']['path'],p['metrics']['fk_chunk'])) as pool:
        for mask in p['mask_order']:
            keep = sensor_mask(p['masks'][mask])
            for rec in p['records']:
                seq,n = rec['sequence_id'],rec['frames']
                cache = cache_path(run,mask,seq)
                old_row = read_row(run,mask,seq,protocol_sha)
                if old_row:
                    completed += 1;predicted += 1
                    continue
                target = math.rotation_matrix_to_r6d(data['pose'][seq]).reshape(n,144)
                if not cache.exists():
                    if args.metrics_only:
                        raise FileNotFoundError(f'Inference cache is not ready: {cache}')
                    x = torch.cat([(data['acc'][seq][:,:5]/30).flatten(1),data['ori'][seq][:,:5].float().flatten(1)],1)
                    x[:,~keep] = 0
                    predictions = {}
                    tick = time.perf_counter()
                    for name,predictor in predictors.items():
                        predictions[name] = infer_sequence(predictor,x,mode='rolling_last',window=125,
                                                          startup='prefix',batch_size=p['inference']['batch_size']).numpy()
                    temporary = cache.with_name(f'.{cache.name}.{uuid.uuid4().hex}.tmp')
                    with temporary.open('xb') as stream:
                        np.savez_compressed(stream,target_pose=target.numpy(),sequence_id=seq,mask=mask,
                            protocol_sha256=protocol_sha,fps=25,**{f'pose_{k}':v for k,v in predictions.items()})
                    os.replace(temporary,cache)
                    print(f'Inference {mask} seq{seq}: {n} frames x2; {time.perf_counter()-tick:.1f}s',flush=True)
                with np.load(cache,allow_pickle=False) as saved:
                    if (str(saved['protocol_sha256']) != protocol_sha or int(saved['sequence_id']) != seq
                            or str(saved['mask']) != mask or not np.array_equal(saved['target_pose'],target.numpy())
                            or any(saved[f'pose_{name}'].shape != (n,144) for name in MODEL_NAMES)):
                        raise ValueError('Prediction cache pairing/protocol mismatch')
                predicted += 1
                future = pool.submit(measure_pair,str(cache),protocol_sha)
                pending[future] = (mask,seq)
                for finished in list(pending):
                    if finished.done():
                        collect(finished)
                save_progress()
        for future in as_completed(list(pending)):
            collect(future)
    verify_protocol(run)
    result = summarize(run,p)
    atomic_json(run/'resources.json',{'elapsed_seconds':time.perf_counter()-started,'peak_parent_ram_mib':peak_ram_mib(),
                'inference_device':'cuda:0' if not args.metrics_only else 'cache only','metrics_device':'CPU, two workers x two threads'})
    atomic_json(run/'ARTIFACT_MANIFEST.json',{str(f.relative_to(run)):sha256(f) for f in sorted(run.rglob('*'))
                                            if f.is_file() and f.name!='ARTIFACT_MANIFEST.json'})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','summarize'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metrics-only',action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    elif args.action == 'run':
        with (args.output/'.comparison.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another process is running this comparison') from None
            run_comparison(args)
    else:
        summarize(args.output,verify_protocol(args.output))


if __name__ == '__main__':
    main()
