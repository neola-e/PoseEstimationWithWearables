"""Data isolation, public-loader parity, padded-loss arithmetic, and RNG replay."""
import copy
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn
from imuposer.config import amass_combos
from imuposer.datasets.globalModelDatasetFineTuneDIP import GlobalModelDatasetFineTuneDIP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from dip_finetuning import (DipFineTuneDataset, MASKS, assert_tree_equal, collate,
                           combined_loss, restore_rng, rng_state, seed_all, validate_masks)


def data_and_records():
    data = {'acc':[], 'ori':[], 'pose':[]}
    records = []
    for i, n in enumerate([126, 5, 4]):
        data['acc'].append(torch.arange(n*18).reshape(n,6,3).float()+i)
        data['ori'].append(torch.eye(3).expand(n,6,3,3).clone())
        pose = torch.eye(3).expand(n,24,3,3).clone()
        pose[:,1] = torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        data['pose'].append(pose)
        records.append({'sequence_id':i,'frames':n,'subject':['s_01','s_02','s_07'][i],
                        'role':'train' if i < 2 else 'validation'})
    return data, records


class ConstantPose(nn.Module):
    def __init__(self):
        super().__init__()
        self.pose = nn.Parameter(torch.tensor([1.,0.,0.,0.,1.,0.]).repeat(24))

    def forward(self, x, lengths):
        return self.pose.expand(x.shape[0],x.shape[1],144)


class AnalyticBody:
    def forward_kinematics(self, r, calc_mesh=False):
        rotations = r.reshape(-1,24,3,3)
        return rotations, rotations[:,:,0]


class DipFineTuningTests(unittest.TestCase):
    def test_exact_sparse_mask_inventory_and_reject_global(self):
        validate_masks(MASKS)
        self.assertEqual([sum(len(v)==n for v in MASKS.values()) for n in [1,2,3]],[5,10,9])
        with self.assertRaises(ValueError):
            validate_masks(amass_combos)

    def test_subject_split_precedes_mask_expansion_and_rejects_leakage(self):
        data, records = data_and_records()
        train = DipFineTuneDataset(data,records,'train')
        validation = DipFineTuneDataset(data,records,'validation')
        self.assertEqual(len(train),3*24)
        self.assertEqual(len(validation),24)
        self.assertEqual({r['sequence_id'] for r in train.windows},{0,1})
        self.assertEqual({r['sequence_id'] for r in validation.windows},{2})
        wrong = copy.deepcopy(records)
        wrong[-1]['role']='train'
        with self.assertRaises(ValueError):
            DipFineTuneDataset(data,wrong,'train')
        with self.assertRaises(ValueError):
            DipFineTuneDataset(data,records,'test')

    def test_every_mask_and_tail_matches_inherited_public_loader(self):
        data, records = data_and_records()
        data={k:v[:2] for k,v in data.items()}
        new = DipFineTuneDataset(data,records[:2],'train')
        with tempfile.TemporaryDirectory() as folder:
            torch.save(data,Path(folder)/'dip_train.pt')
            config=SimpleNamespace(processed_imu_poser_25fps=Path(folder),acc_scale=30,
                                   max_sample_len=300,r6d=True,pred_joints_set=list(range(24)))
            old=GlobalModelDatasetFineTuneDIP('train',config)
            for index in range(len(new)):
                x,y,meta=new[index]
                seq=meta['sequence_id']
                base=0 if seq==0 else 2*25
                n_windows=2 if seq==0 else 1
                legacy_index=base+list(amass_combos).index(meta['mask'])*n_windows+meta['start_frame']//125
                a,b=old[legacy_index]
                self.assertTrue(torch.equal(x,a))
                self.assertTrue(torch.equal(y,b))

    def test_mask_fetch_does_not_mutate_shared_features(self):
        data, records=data_and_records()
        dataset=DipFineTuneDataset(data,records,'train')
        original=dataset.features[0].clone()
        x,_,_=dataset[0]
        x[:] = 999
        self.assertTrue(torch.equal(dataset.features[0],original))
        self.assertFalse(torch.equal(dataset[0][0],x))

    def test_inherited_padding_is_explicit_in_pose_and_joint_loss(self):
        model=ConstantPose()
        identity=model.pose.detach().clone()
        batch=collate([(torch.zeros(2,60),identity.expand(2,-1),{}),
                       (torch.zeros(1,60),identity.expand(1,-1),{})])
        loss, values=combined_loss(model,AnalyticBody(),batch)
        # Three identity frames and one zero-padded target. Both pose and joint
        # MSE have 1/3 of components equal to one on the padded frame: 1/12.
        self.assertAlmostEqual(values['pose_mse'],1/12,places=7)
        self.assertAlmostEqual(values['joint_mse_m2'],1/12,places=7)
        self.assertAlmostEqual(loss.item(),1/6,places=7)
        self.assertEqual(values['valid_frames'],3)
        self.assertEqual(values['padded_frames'],1)
        loss.backward()
        self.assertTrue(torch.isfinite(model.pose.grad).all())

    def test_python_numpy_and_torch_rng_replay(self):
        seed_all(42)
        state=rng_state()
        a=(random.random(),np.random.rand(3),torch.rand(3))
        restore_rng(state)
        b=(random.random(),np.random.rand(3),torch.rand(3))
        assert_tree_equal(a,b)
        with self.assertRaises(AssertionError):
            assert_tree_equal({'step':torch.tensor(3)}, {'step':torch.tensor(4)})
        with self.assertRaises(AssertionError):
            assert_tree_equal({'moment':torch.ones(3)}, {'moment':torch.zeros(3)})


if __name__ == '__main__':
    unittest.main()
