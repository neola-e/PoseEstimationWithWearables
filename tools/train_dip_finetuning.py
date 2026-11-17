"""Full DIP fine-tuning under the frozen V1 protocol; CUDA required by the CLI.

Development: first six subjects train, s07/s08 validate, 30 epochs, best val loss.
Final: reset Epoch59 and train first eight for K selected development epochs.
Epoch checkpoints are immutable; STATE.json commits the last/best pointers.
No test container is opened, and no frozen source/protocol file is modified.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import copy
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import logging
import math
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid

import torch
from torch.utils.data import ConcatDataset, DataLoader
import imuposer
from imuposer.smpl.parametricModel import ParametricModel

from dip_finetuning import (DipFineTuneDataset, collate, combined_loss, make_optimizer,
                           restore_rng, rng_state, seed_all, validate_masks)
from eval_wrist_pocket_dev import fingerprint
from runtime import PosePredictor, load_torch, peak_ram_mib, sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT/'outputs/local_dip/dip_finetuning_v1_20261114'
FORMAT = 'imuposer-dip-full-training-v1'


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def contained_file(run, relative):
    path = (Path(run)/relative).resolve()
    if not path.is_relative_to(Path(run).resolve()) or not path.is_file():
        raise ValueError('Checkpoint path must name a file inside its run directory')
    return path


def read_committed(run):
    state = json.loads((Path(run)/'STATE.json').read_text())
    for key in ['last_checkpoint', 'best_checkpoint']:
        pointer = state.get(key)
        if pointer and sha256(contained_file(run,pointer['path'])) != pointer['sha256']:
            raise ValueError(f'Committed {key} hash mismatch')
    saved = load_torch(contained_file(run,state['last_checkpoint']['path']),trusted=True)
    if (saved.get('format') != FORMAT or saved['run_id'] != state['run_id']
            or saved['epoch'] != state['epoch'] or saved['global_step'] != state['global_step']):
        raise ValueError('Committed state/checkpoint identity mismatch')
    return state, saved


def point_alias(run, name, relative):
    alias = Path(run)/name
    if alias.exists() and not alias.is_symlink():
        raise FileExistsError(f'Refusing to replace a non-symlink checkpoint: {alias}')
    temporary = alias.with_name(f'.{name}.{uuid.uuid4().hex}.tmp')
    os.symlink(relative,temporary)
    os.replace(temporary,alias)


def materialize_views(run, state, history):
    point_alias(run,'last.ckpt',state['last_checkpoint']['path'])
    if state.get('best_checkpoint'):
        point_alias(run,'best.ckpt',state['best_checkpoint']['path'])
    if state['stage'] == 'final' and state['epoch']+1 == state['budget_epochs']:
        point_alias(run,'final.ckpt',state['last_checkpoint']['path'])
    atomic_json(Path(run)/'history.json',history)
    stream = io.StringIO()
    writer = csv.DictWriter(stream,fieldnames=list(history[0]))
    writer.writeheader()
    writer.writerows(history)
    temporary = Path(run)/f'.history.{uuid.uuid4().hex}.tmp'
    temporary.write_text(stream.getvalue())
    os.replace(temporary,Path(run)/'history.csv')


def commit_epoch(run, capsule, previous_state=None):
    """Write a new immutable checkpoint, then atomically publish its pointers.

    A crash before STATE.json leaves an uncommitted checkpoint, which resume
    ignores. A crash afterward only needs aliases/history regenerated. No
    checkpoint bytes are overwritten or deleted, including orphan files.
    """
    run = Path(run)
    folder = run/'checkpoints'
    folder.mkdir(exist_ok=True)
    relative = f'checkpoints/epoch_{capsule["epoch"]+1:03d}_{uuid.uuid4().hex[:12]}.ckpt'
    destination = run/relative
    temporary = destination.with_suffix('.tmp')
    try:
        with temporary.open('xb') as stream:
            torch.save(capsule,stream)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)
    pointer = {'path':relative,'sha256':sha256(destination)}
    best = previous_state.get('best_checkpoint') if previous_state else None
    if capsule['stage'] == 'development' and capsule['best_epoch'] == capsule['epoch']:
        best = pointer
    state = {'run_id':capsule['run_id'],'stage':capsule['stage'],
             'budget_epochs':capsule['budget_epochs'],'epoch':capsule['epoch'],
             'global_step':capsule['global_step'],'best_epoch':capsule['best_epoch'],
             'best_score':capsule['best_score'],'last_checkpoint':pointer,'best_checkpoint':best,
             'training_protocol_sha256':capsule['training_protocol_sha256']}
    atomic_json(run/'STATE.json',state)
    materialize_views(run,state,capsule['history'])
    return state


def improved(score, best_score):
    if not math.isfinite(score):
        raise ValueError('Non-finite validation score')
    # Strict comparison preserves the earlier epoch on an exact tie.
    return best_score is None or score < best_score


def run_epoch(model, body, loader, optimizer=None, progress=None, loss_function=combined_loss):
    training = optimizer is not None
    model.train(training)
    totals = {key:0.0 for key in ['combined_loss','pose_mse','joint_mse_m2']}
    denominator, valid, batches = 0, 0, 0
    started = time.perf_counter()
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch_index,batch in enumerate(loader,1):
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss,values = loss_function(model,body,batch)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss')
            if training:
                loss.backward()
                parameters = list(model.parameters())
                if any(p.grad is None for p in parameters):
                    raise ValueError('A trainable parameter has no gradient')
                if not torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all().item():
                    raise ValueError('Non-finite gradient; no optimizer update applied')
                optimizer.step()
                if not torch.stack([torch.isfinite(p).all() for p in parameters]).all().item():
                    raise ValueError('Non-finite model parameter after update')
            padded = int(batch[0].shape[0]*batch[0].shape[1])
            if padded < 1 or values['valid_frames']+values['padded_frames'] != padded:
                raise ValueError('Loss denominator differs from padded batch B*T')
            for key in totals:
                if not math.isfinite(values[key]):
                    raise ValueError('Non-finite loss component')
                totals[key] += values[key]*padded
            denominator += padded
            valid += values['valid_frames']
            batches += 1
            if progress:
                progress(batch_index,len(loader),values)
    if not denominator:
        raise ValueError('Empty epoch')
    return {**{k:v/denominator for k,v in totals.items()},'valid_frames':valid,
            'padded_frames':denominator-valid,'batches':batches,
            'seconds':time.perf_counter()-started}


def load_frozen(folder):
    folder = Path(folder)
    p = json.loads((folder/'PROTOCOL.json').read_text())
    if p['protocol_id'] != 'DIP_FINETUNING_V1':
        raise ValueError('Expected frozen DIP_FINETUNING_V1')
    for name,digest in json.loads((folder/'FREEZE_MANIFEST.json').read_text()).items():
        if sha256(folder/name) != digest:
            raise ValueError(f'Frozen artifact changed: {name}')
    sources = json.loads((folder/'SOURCE_HASHES.json').read_text())
    for name,digest in sources.items():
        if sha256(name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    if sha256(folder/'SOURCE_HASHES.json') != p['source_hashes_sha256']:
        raise ValueError('Frozen source inventory changed')
    if sha256(folder/'data_manifest.json') != p['data_manifest_sha256']:
        raise ValueError('Data manifest changed')
    manifest = json.loads((folder/'data_manifest.json').read_text())
    data_path = Path(manifest['data_path'])
    if data_path.name != 'dip_train.pt' or sha256(data_path) != manifest['data_sha256']:
        raise ValueError('Training data changed or points at test data')
    for asset in ['checkpoint','smpl_model']:
        if sha256(p[asset]['path']) != p[asset]['sha256']:
            raise ValueError(f'Frozen {asset} changed')
    validate_masks(p['masks'])
    if list(p['masks']) != p['mask_order']:
        raise ValueError('Mask order changed')
    data = load_torch(data_path)
    for record in manifest['records']:
        for key,digest in record['processed_sha256'].items():
            if fingerprint(data[key][record['sequence_id']]) != digest:
                raise ValueError('Sequence identity changed')
    train = DipFineTuneDataset(data,manifest['records'],'train',p['masks'])
    validation = DipFineTuneDataset(data,manifest['records'],'validation',p['masks'])
    a,b = ({r['sequence_id'] for r in ds.windows} for ds in [train,validation])
    if a & b or {r['subject'] for r in train.windows} != set(p['development']['train_subjects']) or {r['subject'] for r in validation.windows} != set(p['development']['validation_subjects']):
        raise ValueError('Subject split differs from frozen protocol')
    for role,ds in [('train',train),('validation',validation)]:
        if len(ds) != manifest['statistics'][role]['masked_samples']:
            raise ValueError('Dataset sample count differs')
    return p,manifest,sources,train,validation


def final_budget(development_run, protocol_sha256, maximum_epochs):
    run = Path(development_run)
    run_info = json.loads((run/'RUN.json').read_text())
    state,saved = read_committed(run)
    if (run_info['stage'] != 'development' or state['stage'] != 'development'
            or state['training_protocol_sha256'] != protocol_sha256
            or state['epoch']+1 != maximum_epochs or saved['budget_epochs'] != maximum_epochs
            or len(saved['history']) != maximum_epochs or not state.get('best_checkpoint')):
        raise ValueError('Final training requires a fully completed development run under the same protocol')
    scores = [r['validation_loss'] for r in saved['history']]
    best_epoch = min(range(len(scores)),key=lambda i:scores[i])
    best = load_torch(contained_file(run,state['best_checkpoint']['path']),trusted=True)
    if (state['best_epoch'] != best_epoch or state['best_score'] != scores[best_epoch]
            or best['epoch'] != best_epoch or not best['selection_eligible']
            or best['kind'] != 'dip_development_training'):
        raise ValueError('Development selection metadata is inconsistent')
    return best_epoch+1,{'run':str(run.resolve()),'state_sha256':sha256(run/'STATE.json'),
                         'best_checkpoint':state['best_checkpoint'],'selected_epochs':best_epoch+1,
                         'validation_loss':scores[best_epoch]}


def execution_sources(frozen_sources):
    result = dict(frozen_sources)
    for path in [Path(__file__),ROOT/'tests/test_train_dip_finetuning.py']:
        result[str(path.resolve())] = sha256(path)
    return result


def log_setup(run):
    logger = logging.getLogger('dip_training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(message)s')
    for handler in [logging.StreamHandler(sys.stdout),logging.FileHandler(Path(run)/'train.log',mode='a')]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def execute(args):
    if args.threads < 1 or args.progress_every < 1:
        raise ValueError('threads and progress-every must be positive')
    torch.set_num_threads(args.threads)
    p,manifest,frozen_sources,train,validation = load_frozen(args.protocol_dir)
    protocol_sha = sha256(Path(args.protocol_dir)/'PROTOCOL.json')
    budget,selection = p['development']['maximum_epochs'],None
    if args.stage == 'final':
        if not args.development_run:
            raise ValueError('--stage final requires --development-run')
        budget,selection = final_budget(args.development_run,protocol_sha,budget)
    elif args.development_run:
        raise ValueError('--development-run is only used for the final stage')
    dataset = train if args.stage == 'development' else ConcatDataset([train,validation])
    plan = {'stage':args.stage,'epochs':budget,'batch_size':p['development']['batch_size'],
            'train_samples':len(dataset),'validation_samples':len(validation) if args.stage=='development' else 0,
            'train_batches':math.ceil(len(dataset)/p['development']['batch_size']),
            'test_data_opened':False,'protocol_sha256':protocol_sha,'selection':selection}
    if args.check_only:
        print(json.dumps({'status':'configuration_and_data_verified','plan':plan,
                          'cuda_available':torch.cuda.is_available(),'training_started':False},indent=2))
        return 0
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required. Run this command in the ordinary WSL terminal; CPU fallback is disabled.')
    device = torch.device(args.device)
    run = Path(args.output).resolve()
    source_hashes = execution_sources(frozen_sources)
    resume_info = None
    if args.resume:
        if not run.is_dir() or not (run/'RUN.json').is_file():
            raise ValueError('--resume requires an existing full-training run')
        # Validate the run header before creating a lock or appending logs. In
        # particular, never add files to the old engineering-smoke directory.
        resume_info = json.loads((run/'RUN.json').read_text())
        if (resume_info.get('kind') != 'dip_full_training' or resume_info['stage'] != args.stage
                or resume_info['training_protocol_sha256'] != protocol_sha or resume_info['budget_epochs'] != budget
                or resume_info['execution_sources'] != source_hashes or resume_info['threads'] != args.threads
                or resume_info['torch'] != torch.__version__ or resume_info['device'] != str(device)
                or resume_info['selection'] != selection):
            raise ValueError('Resume configuration/source differs from recorded full-training run')
    else:
        run.mkdir(parents=True,exist_ok=False)
    with (run/'.training.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process already owns this training run') from None
        logger = log_setup(run)
        saved,state = None,None
        if args.resume:
            info = resume_info
            if (run/'STATE.json').exists():
                state,saved = read_committed(run)
                if saved['training_protocol_sha256'] != protocol_sha or saved['execution_sources'] != source_hashes:
                    raise ValueError('Resume checkpoint protocol/source mismatch')
            run_id = info['run_id']
        else:
            run_id = uuid.uuid4().hex
            info = {'kind':'dip_full_training','run_id':run_id,'created_utc':datetime.now(timezone.utc).isoformat(),
                    'argv':sys.argv,'stage':args.stage,'budget_epochs':budget,'training_protocol_sha256':protocol_sha,
                    'protocol_directory':str(Path(args.protocol_dir).resolve()),'execution_sources':source_hashes,
                    'threads':args.threads,'torch':torch.__version__,'python':platform.python_version(),
                    'device':str(device),'gpu':torch.cuda.get_device_name(device),'plan':plan,'selection':selection,
                    'loader_generator_seeds':{'train':p['seed'],'validation':p['seed']+1},
                    'data_manifest_sha256':p['data_manifest_sha256'],
                    'final_subject_role_change':args.stage=='final','test_data_opened':False}
            for source in source_hashes:
                source = Path(source)
                package = Path(imuposer.__file__).parent
                relative = Path('team')/source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path('installed_imuposer')/source.relative_to(package)
                target = run/'source_snapshot'/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
            for name in ['PROTOCOL.json','data_manifest.json']:
                shutil.copy2(Path(args.protocol_dir)/name,run/name)
            atomic_json(run/'RUN.json',info)
        checkpoint = contained_file(run,state['last_checkpoint']['path']) if state else p['checkpoint']['path']
        predictor = PosePredictor(checkpoint,str(device),args.threads)
        optimizer = make_optimizer(predictor.model,p['optimizer'])
        body = ParametricModel(p['smpl_model']['path'],device=device)
        seed_all(p['seed'])
        train_generator = torch.Generator().manual_seed(p['seed'])
        val_generator = torch.Generator().manual_seed(p['seed']+1)
        train_loader = DataLoader(dataset,batch_size=p['development']['batch_size'],shuffle=True,num_workers=0,
                                  drop_last=False,collate_fn=collate,generator=train_generator)
        val_loader = DataLoader(validation,batch_size=p['development']['batch_size'],shuffle=False,num_workers=0,
                                drop_last=False,collate_fn=collate,generator=val_generator) if args.stage=='development' else None
        epoch,step,history,best_epoch,best_score = 0,0,[],None,None
        if saved:
            optimizer.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]))
            restore_rng(saved['rng_state'])
            train_generator.set_state(saved['loader_states']['train'])
            val_generator.set_state(saved['loader_states']['validation'])
            epoch,step = saved['epoch']+1,saved['global_step']
            history,best_epoch,best_score = saved['history'],saved['best_epoch'],saved['best_score']
            materialize_views(run,state,history)
        logger.info('stage=%s device=%s GPU=%s epochs=%s batch=%s lr=%s train=%s validation=%s start_epoch=%s',
                    args.stage,device,torch.cuda.get_device_name(device),budget,p['development']['batch_size'],
                    p['optimizer']['lr'],len(dataset),len(validation) if val_loader else 0,epoch+1)
        free,total = torch.cuda.mem_get_info(device)
        logger.info('GPU memory currently free=%.0f MiB total=%.0f MiB; frozen batch is unchanged',free/2**20,total/2**20)
        try:
            for epoch in range(epoch,budget):
                def progress(batch,total,values):
                    if batch % args.progress_every == 0 or batch == total:
                        logger.info('epoch=%s/%s train_batch=%s/%s loss=%.8f',epoch+1,budget,batch,total,values['combined_loss'])
                training = run_epoch(predictor.model,body,train_loader,optimizer,progress)
                step += training['batches']
                validating = run_epoch(predictor.model,body,val_loader) if val_loader else None
                score = validating['combined_loss'] if validating else None
                if validating and improved(score,best_score):
                    best_epoch,best_score = epoch,score
                torch.cuda.synchronize(device)
                row = {'epoch':epoch+1,'global_step':step,'train_loss':training['combined_loss'],
                       'train_pose_mse':training['pose_mse'],'train_joint_mse_m2':training['joint_mse_m2'],
                       'validation_loss':score,'validation_pose_mse':validating['pose_mse'] if validating else None,
                       'validation_joint_mse_m2':validating['joint_mse_m2'] if validating else None,
                       'train_valid_frames':training['valid_frames'],'train_padding_frames':training['padded_frames'],
                       'validation_valid_frames':validating['valid_frames'] if validating else 0,
                       'validation_padding_frames':validating['padded_frames'] if validating else 0,
                       'train_seconds':training['seconds'],'validation_seconds':validating['seconds'] if validating else 0,
                       'best_epoch':None if best_epoch is None else best_epoch+1,'best_score':best_score,
                       'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/2**20}
                history.append(row)
                for name,digest in source_hashes.items():
                    if sha256(name) != digest:
                        raise ValueError('Execution source changed during training; no new checkpoint committed')
                cfg = copy.deepcopy(predictor.training_config)
                cfg.device,cfg.mkdir = device,False
                capsule = {'format':FORMAT,'run_id':run_id,'stage':args.stage,'budget_epochs':budget,
                           'state_dict':predictor.model.state_dict(),'hyper_parameters':{'config':cfg},
                           'epoch':epoch,'global_step':step,'optimizer_states':[optimizer.state_dict()],
                           'lr_schedulers':[],'rng_state':rng_state(),
                           'loader_states':{'train':train_generator.get_state(),'validation':val_generator.get_state()},
                           'sampler_state':{'completed_epoch':epoch,'next_epoch':epoch+1,'next_batch':0,'shuffle':True},
                           'history':history,'best_epoch':best_epoch,'best_score':best_score,
                           'training_validation_metric':score,'training_protocol_sha256':protocol_sha,
                           'execution_sources':source_hashes,'pretrained_epoch':p['checkpoint']['epoch'],
                           'pretrained_global_step':p['checkpoint']['global_step'],
                           'kind':'dip_development_training' if validating else 'dip_final_fixed_budget',
                           'selection_eligible':args.stage=='development',
                           'eligible_for_final_reporting':args.stage=='final' and epoch+1==budget}
                state = commit_epoch(run,capsule,state)
                logger.info('EPOCH COMPLETE %s/%s train=%.8f validation=%s best_epoch=%s best_loss=%s',
                            epoch+1,budget,training['combined_loss'],score,row['best_epoch'],best_score)
                atomic_json(run/'progress.json',{'completed_epochs':epoch+1,'budget_epochs':budget,
                            'global_step':step,'best_epoch':row['best_epoch'],'best_score':best_score})
            completion = {'status':'complete','stage':args.stage,'epochs_completed':budget,'global_step':step,
                          'best_epoch':None if best_epoch is None else best_epoch+1,'best_score':best_score,
                          'selected_epochs_for_final':None if best_epoch is None else best_epoch+1,
                          'training_protocol_sha256':protocol_sha,'test_data_opened':False,'peak_ram_mib':peak_ram_mib(),
                          'last_checkpoint':state['last_checkpoint'],'best_checkpoint':state['best_checkpoint']}
            atomic_json(run/'completion.json',completion)
            logger.info('TRAINING COMPLETE: %s',json.dumps(completion))
            return 0
        except (KeyboardInterrupt,Exception) as error:
            reason = 'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
            atomic_json(run/'failure.json',{'status':reason,'error':repr(error),'epoch_in_progress':epoch+1,
                        'last_committed_epoch':state['epoch']+1 if state else 0,
                        'recovery':'Run the same command with --resume; only complete committed epochs are restored',
                        'batch_size_unchanged':p['development']['batch_size']})
            logger.exception('Training stopped. Complete checkpoints remain intact. Use --resume for this run.')
            if isinstance(error,torch.cuda.OutOfMemoryError):
                logger.error('CUDA OOM: free unrelated GPU memory before retrying. Batch/lr/precision were NOT changed.')
            return 130 if isinstance(error,KeyboardInterrupt) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-dir',type=Path,default=DEFAULT_PROTOCOL)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['development','final'],default='development')
    parser.add_argument('--development-run',type=Path)
    parser.add_argument('--device',choices=['cuda:0'],default='cuda:0')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--progress-every',type=int,default=10)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--check-only',action='store_true',help='Verify source/data/plan only; no output directory or training')
    args = parser.parse_args()
    try:
        return execute(args)
    except Exception as error:
        parser.exit(2,f'ERROR: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
