"""Full-loop reductions, immutable checkpoint commits, and shuffled-epoch replay.

These tests use tiny synthetic CPU regressors, not the real Epoch59/DIP smoke
or a formal training run. Actual launcher data checking is a separate dry run.
"""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from dip_finetuning import assert_tree_equal, collate, restore_rng, rng_state, seed_all
from runtime import sha256
from train_dip_finetuning import (FORMAT, atomic_json, commit_epoch, final_budget, improved,
                                  execute, materialize_views, read_committed, run_epoch)


def loss_fn(model,body,batch):
    x,y,lengths,_=batch
    loss=(model(x)-y).square().mean()
    valid=int(lengths.sum())
    return loss,{'combined_loss':loss.item(),'pose_mse':loss.item(),'joint_mse_m2':0.,
                 'valid_frames':valid,'padded_frames':x.shape[0]*x.shape[1]-valid}


def tiny_model():
    return nn.Sequential(nn.Linear(60,8),nn.ReLU(),nn.Dropout(.3),nn.Linear(8,144))


def loaders():
    g=torch.Generator().manual_seed(19)
    data=[(torch.randn(i%4+1,60,generator=g),torch.randn(i%4+1,144,generator=g),{'id':i}) for i in range(13)]
    train_g=torch.Generator().manual_seed(42)
    val_g=torch.Generator().manual_seed(43)
    return (DataLoader(data,batch_size=4,shuffle=True,collate_fn=collate,generator=train_g),
            DataLoader(data,batch_size=5,shuffle=False,collate_fn=collate,generator=val_g),train_g,val_g)


def capsule(model,optimizer,epoch,history,train_g=None,val_g=None,budget=3):
    scores=[r['validation_loss'] for r in history]
    best=min(range(len(scores)),key=lambda i:scores[i])
    return {'format':FORMAT,'run_id':'unit-run','stage':'development','budget_epochs':budget,
            'epoch':epoch,'global_step':4*(epoch+1),'state_dict':model.state_dict(),
            'optimizer_states':[optimizer.state_dict()],'rng_state':rng_state(),
            'loader_states':{'train':train_g.get_state() if train_g else None,
                             'validation':val_g.get_state() if val_g else None},
            'history':copy.deepcopy(history),'best_epoch':best,'best_score':scores[best],
            'training_protocol_sha256':'unit-protocol','kind':'dip_development_training','selection_eligible':True}


class FullFineTuningLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_validation_weights_unequal_padded_batch_sizes(self):
        model=nn.Linear(60,144)
        with torch.no_grad():
            model.weight.zero_();model.bias.zero_()
        a=(torch.ones(2,1,60),torch.ones(2,1,144),torch.tensor([1,1]),[{},{}])
        b=(torch.ones(1,3,60),torch.full((1,3,144),3.),torch.tensor([3]),[{}])
        result=run_epoch(model,None,[a,b],loss_function=loss_fn)
        self.assertAlmostEqual(result['combined_loss'],(2*1+3*9)/5)
        self.assertNotEqual(result['combined_loss'],(1+9)/2)

    def test_validation_disables_gradients_dropout_and_weight_updates(self):
        model=tiny_model()
        train,val,_,_=loaders()
        initial=copy.deepcopy(model.state_dict())
        observed=[]
        def checking(model,body,batch):
            observed.append((model.training,torch.is_grad_enabled()))
            return loss_fn(model,body,batch)
        first=run_epoch(model,None,val,loss_function=checking)
        second=run_epoch(model,None,val,loss_function=checking)
        self.assertEqual(first['combined_loss'],second['combined_loss'])
        self.assertTrue(all(value==(False,False) for value in observed))
        assert_tree_equal(initial,model.state_dict())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_exact_validation_tie_keeps_earlier_checkpoint(self):
        best=None;chosen=None
        for epoch,score in enumerate([2.,1.,1.,1.1]):
            if improved(score,best):
                chosen,best=epoch,score
        self.assertEqual(chosen,1)
        with self.assertRaises(ValueError):
            improved(float('nan'),best)

    def test_nonfinite_training_loss_is_rejected_before_optimizer_update(self):
        model=tiny_model();opt=torch.optim.Adam(model.parameters(),lr=3e-4)
        train,_,_,_=loaders()
        initial=copy.deepcopy(model.state_dict())
        def nonfinite(model,body,batch):
            loss,values=loss_fn(model,body,batch)
            return loss*float('nan'),values
        with self.assertRaises(ValueError):
            run_epoch(model,None,train,opt,loss_function=nonfinite)
        assert_tree_equal(initial,model.state_dict())
        self.assertFalse(opt.state)

    def test_cuda_unavailable_refuses_training_before_output_creation(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            (folder/'PROTOCOL.json').write_text('{}')
            output=folder/'never_created'
            args=SimpleNamespace(threads=2,progress_every=10,protocol_dir=folder,
                                 stage='development',development_run=None,check_only=False,
                                 output=output,device='cuda:0',resume=False)
            recipe={'development':{'maximum_epochs':30,'batch_size':256}}
            with patch('train_dip_finetuning.load_frozen',return_value=(recipe,{}, {},[0]*5,[0]*2)), \
                 patch('torch.cuda.is_available',return_value=False):
                with self.assertRaisesRegex(ValueError,'CPU fallback is disabled'):
                    execute(args)
            self.assertFalse(output.exists())

    def test_resume_rejects_smoke_run_without_touching_its_directory(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            (folder/'PROTOCOL.json').write_text('{}')
            output=folder/'existing_smoke'
            output.mkdir()
            (output/'RUN.json').write_text('{"kind":"engineering_smoke_only"}')
            args=SimpleNamespace(threads=2,progress_every=10,protocol_dir=folder,
                                 stage='development',development_run=None,check_only=False,
                                 output=output,device='cuda:0',resume=True)
            recipe={'development':{'maximum_epochs':30,'batch_size':256}}
            with patch('train_dip_finetuning.load_frozen',return_value=(recipe,{}, {},[0]*5,[0]*2)), \
                 patch('torch.cuda.is_available',return_value=True), \
                 patch('train_dip_finetuning.execution_sources',return_value={}):
                with self.assertRaisesRegex(ValueError,'recorded full-training run'):
                    execute(args)
            self.assertEqual(sorted(p.name for p in output.iterdir()),['RUN.json'])

    def test_checkpoint_commit_preserves_bytes_and_recovers_missing_views(self):
        model=tiny_model();opt=torch.optim.Adam(model.parameters())
        history=[{'epoch':1,'validation_loss':1.}]
        with tempfile.TemporaryDirectory() as name:
            run=Path(name)
            first=commit_epoch(run,capsule(model,opt,0,history))
            first_path=run/first['last_checkpoint']['path']
            first_hash=sha256(first_path)
            history.append({'epoch':2,'validation_loss':1.})
            second=commit_epoch(run,capsule(model,opt,1,history),first)
            self.assertEqual(second['best_checkpoint'],first['last_checkpoint'])
            self.assertEqual(sha256(first_path),first_hash)
            # A leftover uncommitted file must never become the resume state.
            torch.save({'not':'committed'},run/'checkpoints/orphan.ckpt')
            (run/'last.ckpt').unlink();(run/'history.csv').unlink()
            state,saved=read_committed(run)
            self.assertEqual(state['epoch'],1)
            materialize_views(run,state,saved['history'])
            self.assertTrue((run/'last.ckpt').is_symlink())
            self.assertEqual((run/'last.ckpt').resolve(),(run/second['last_checkpoint']['path']).resolve())
            self.assertEqual(len(json.loads((run/'history.json').read_text())),2)

    def test_committed_checkpoint_hash_tamper_is_rejected(self):
        model=tiny_model();opt=torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as name:
            run=Path(name)
            state=commit_epoch(run,capsule(model,opt,0,[{'epoch':1,'validation_loss':1.}]))
            path=run/state['last_checkpoint']['path']
            with path.open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaises(ValueError):
                read_committed(run)

    def test_shuffled_dropout_adam_epoch_resume_matches_uninterrupted(self):
        seed_all(42)
        model=tiny_model();opt=torch.optim.Adam(model.parameters(),lr=3e-4,foreach=False)
        train,val,tg,vg=loaders()
        train_metrics=run_epoch(model,None,train,opt,loss_function=loss_fn)
        val_metrics=run_epoch(model,None,val,loss_function=loss_fn)
        history=[{'epoch':1,'train_loss':train_metrics['combined_loss'],'validation_loss':val_metrics['combined_loss']}]
        with tempfile.TemporaryDirectory() as name:
            run=Path(name)
            commit_epoch(run,capsule(model,opt,0,history,tg,vg))
            _,saved=read_committed(run)
            expected_train=run_epoch(model,None,train,opt,loss_function=loss_fn)
            expected_val=run_epoch(model,None,val,loss_function=loss_fn)
            expected_weights=copy.deepcopy(model.state_dict())
            expected_opt=copy.deepcopy(opt.state_dict());expected_rng=rng_state()
            resumed_model=tiny_model()
            resumed_model.load_state_dict(saved['state_dict'])
            resumed_opt=torch.optim.Adam(resumed_model.parameters(),lr=3e-4,foreach=False)
            resumed_opt.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]))
            rtrain,rval,rtg,rvg=loaders()
            restore_rng(saved['rng_state'])
            rtg.set_state(saved['loader_states']['train']);rvg.set_state(saved['loader_states']['validation'])
            actual_train=run_epoch(resumed_model,None,rtrain,resumed_opt,loss_function=loss_fn)
            actual_val=run_epoch(resumed_model,None,rval,loss_function=loss_fn)
            for key in ['combined_loss','pose_mse','valid_frames','padded_frames','batches']:
                self.assertEqual(expected_train[key],actual_train[key])
                self.assertEqual(expected_val[key],actual_val[key])
            assert_tree_equal(expected_weights,resumed_model.state_dict())
            assert_tree_equal(expected_opt,resumed_opt.state_dict())
            assert_tree_equal(expected_rng,rng_state())
            assert_tree_equal(tg.get_state(),rtg.get_state());assert_tree_equal(vg.get_state(),rvg.get_state())

    def test_final_budget_requires_completed_development_and_selects_completed_epochs(self):
        model=tiny_model();opt=torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as name:
            run=Path(name)
            atomic_json(run/'RUN.json',{'stage':'development'})
            history=[];state=None
            for epoch,score in enumerate([2.,1.,1.]):
                history.append({'epoch':epoch+1,'validation_loss':score})
                state=commit_epoch(run,capsule(model,opt,epoch,history),state)
                if epoch<2:
                    with self.assertRaises(ValueError):
                        final_budget(run,'unit-protocol',3)
            k,selection=final_budget(run,'unit-protocol',3)
            self.assertEqual(k,2)  # Zero-based best epoch1 means TWO completed epochs.
            self.assertEqual(selection['validation_loss'],1.)
            with self.assertRaises(ValueError):
                final_budget(run,'different-protocol',3)


if __name__ == '__main__':
    unittest.main()
