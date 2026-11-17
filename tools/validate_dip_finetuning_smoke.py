"""Fresh-process audit of the saved smoke checkpoint and paired predictions.

Reconstruct the probe directly from processed data, without the new dataset,
collate, loss, or checkpoint training helpers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from imuposer import math

from runtime import PosePredictor, load_torch, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.protocol_dir
    p = json.loads((root/'PROTOCOL.json').read_text())
    for folder, filename in [(root, 'FREEZE_MANIFEST.json'), (root/'smoke', 'ARTIFACT_MANIFEST.json')]:
        for name, digest in json.loads((folder/filename).read_text()).items():
            if sha256(folder/name) != digest:
                raise ValueError(f'Artifact changed: {folder/name}')
    run = json.loads((root/'smoke/RUN.json').read_text())
    results = json.loads((root/'smoke/results.json').read_text())
    if not results['passed'] or results['formal_training_launched'] or results['selection_eligible']:
        raise ValueError('Wrong run kind or incomplete run')
    checkpoint = Path(results['checkpoint'])
    saved = load_torch(checkpoint, trusted=True)
    if saved['global_step'] != 3 or saved['epoch'] != -1 or saved['selection_eligible']:
        raise ValueError('Wrong checkpoint metadata')
    if saved['training_protocol_sha256'] != sha256(root/'PROTOCOL.json'):
        raise ValueError('Checkpoint protocol mismatch')
    optimizer = saved['optimizer_states'][0]
    if len(optimizer['state']) != 20 or len(optimizer['param_groups']) != 1:
        raise ValueError('Missing Adam parameter state')
    for state in optimizer['state'].values():
        if (state['step'].item() != 3 or not torch.isfinite(state['exp_avg']).all()
                or not torch.isfinite(state['exp_avg_sq']).all() or (state['exp_avg_sq'] < 0).any()):
            raise ValueError('Invalid Adam counters or moments')
    group = optimizer['param_groups'][0]
    if group['lr'] != p['optimizer']['lr'] or tuple(group['betas']) != tuple(p['optimizer']['betas']):
        raise ValueError('Saved optimizer recipe differs')
    all_samples = run['sample_plan']
    if len(all_samples) != 24 or {s['mask'] for s in all_samples} != set(p['masks']):
        raise ValueError('Mask coverage differs')
    manifest = json.loads((root/'data_manifest.json').read_text())
    data_path = Path(manifest['data_path'])
    if data_path.name != 'dip_train.pt' or sha256(data_path) != manifest['data_sha256']:
        raise ValueError('Unexpected data container')
    data = load_torch(data_path)
    by_seq = {r['sequence_id']: r for r in manifest['records']}
    if any(by_seq[s['sequence_id']]['role'] != 'train' or s['subject'] not in p['development']['train_subjects'] for s in all_samples):
        raise ValueError('Reserved development examples entered training')
    samples = all_samples[:8]
    lengths = [s['frames'] for s in samples]
    x = torch.zeros(8,max(lengths),60)
    target = torch.zeros(8,max(lengths),144)
    for i, s in enumerate(samples):
        seq, start, n = [s[k] for k in ['sequence_id','start_frame','frames']]
        acc = torch.zeros(n,5,3)
        ori = torch.zeros(n,5,3,3)
        sensors = p['masks'][s['mask']]
        acc[:,sensors] = data['acc'][seq][start:start+n,sensors]/30
        ori[:,sensors] = data['ori'][seq][start:start+n,sensors].float()
        x[i,:n] = torch.cat([acc.flatten(1),ori.flatten(1)],1)
        target[i,:n] = math.rotation_matrix_to_r6d(data['pose'][seq][start:start+n]).reshape(n,144)
    with np.load(root/'smoke/probe_predictions.npz',allow_pickle=False) as cache:
        before, after, reloaded = [cache[k].copy() for k in ['pose_before','pose_after','pose_reloaded']]
        if not np.array_equal(cache['lengths'],lengths) or not np.array_equal(cache['target_pose'],target.numpy()):
            raise ValueError('Probe lengths or targets differ after independent reconstruction')
    device = run['device']
    fine = PosePredictor(checkpoint,device=device,threads=run['threads'])
    predicted = fine.predict(x,lengths).numpy()
    if not np.array_equal(predicted,after) or not np.array_equal(predicted,reloaded):
        raise ValueError('Fresh-process prediction differs from saved smoke cache')
    if sha256(p['checkpoint']['path']) != p['checkpoint']['sha256']:
        raise ValueError('AMASS initialization changed')
    initial = PosePredictor(p['checkpoint']['path'],device=device,threads=run['threads'])
    initial_prediction = initial.predict(x,lengths).numpy()
    if not np.array_equal(initial_prediction,before):
        raise ValueError('Fresh-process initial prediction differs')
    changes = {k: (v-saved['state_dict'][k]).abs().max().item() for k,v in initial.model.state_dict().items()}
    if sum(v > 0 for v in changes.values()) != results['parameter_tensors_changed']:
        raise ValueError('Reported parameter changes differ')
    output = {'passed':True,'validator_sha256':sha256(__file__),
              'training_protocol_sha256':sha256(root/'PROTOCOL.json'), 'checkpoint_sha256':sha256(checkpoint), 'device':device,
              'optimizer_states_checked':len(optimizer['state']), 'optimizer_steps':3,
              'independent_probe_examples':8,'independent_target_and_length_match':True,
              'fresh_process_initial_prediction_max_abs_difference':float(np.abs(initial_prediction-before).max()),
              'fresh_process_reloaded_prediction_max_abs_difference':float(np.abs(predicted-after).max()),
              'changed_parameter_tensors':sum(v > 0 for v in changes.values()),
              'max_abs_parameter_change_from_amass':max(changes.values()),
              'train_subjects':run['data']['train_subjects'],
              'no_validation_examples_in_training':True,'test_data_opened':False,
              'scope':'Independent artifact/loader/inference verification on the recorded device; not a model accuracy evaluation'}
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output),flush=True)


if __name__ == '__main__':
    main()
