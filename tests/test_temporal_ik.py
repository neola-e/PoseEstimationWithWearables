import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from imu_only_ik import IKSettings,refine_pose
from temporal_ik import TemporalIKSettings,refine_temporal,discrete_jerk_loss,world_kinematics
from test_imu_only_ik import inputs,rotation_z


class TemporalIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_zero_weight_matches_frozen_framewise_algorithm(self):
        p,o=inputs(9);offsets=torch.zeros(24,3)
        for t in range(9):o[t,0]=rotation_z((-1)**t*25)
        a=refine_pose(p,o,[0],IKSettings(frame_batch=3))['pose']
        b=refine_temporal(p,o,[0],offsets,TemporalIKSettings(jerk_weight=0,frame_batch=4))['pose']
        torch.testing.assert_close(a,b,atol=1e-7,rtol=0)

    def test_future_input_perturbation_cannot_change_earlier_outputs(self):
        p,o=inputs(10);offsets=torch.randn(24,3,dtype=torch.float64)*.1;offsets[0]=0
        settings=TemporalIKSettings(jerk_weight=.1,frame_batch=3)
        a=refine_temporal(p,o,[0,2],offsets,settings)['pose']
        changed=o.clone();changed[7:]=rotation_z(70)
        b=refine_temporal(p,changed,[0,2],offsets,settings)['pose']
        self.assertTrue(torch.equal(a[:7],b[:7]))
        torch.testing.assert_close(a,refine_temporal(p,o,[0,2],offsets,TemporalIKSettings(jerk_weight=.1,frame_batch=8))['pose'],atol=1e-7,rtol=0)

    def test_no_fake_startup_or_record_boundary_penalty(self):
        position=torch.zeros(1,5,24,3,dtype=torch.float64);position[:,0]=100
        valid=torch.tensor([[False,False,True,True,True]])
        self.assertEqual(discrete_jerk_loss(position,valid,.01).item(),0)
        p,o=inputs(3);offsets=torch.zeros(24,3)
        a=refine_pose(p,o,[0])['pose']
        b=refine_temporal(p,o,[0],offsets,TemporalIKSettings())['pose']
        torch.testing.assert_close(a,b,atol=1e-7,rtol=0)

    def test_joint_kinematics_known_two_segment_chain(self):
        from imu_only_ik import PARENTS
        rotations=torch.eye(3,dtype=torch.float64).repeat(1,24,1,1)
        rotations[:,16]=rotation_z(90)
        offsets=torch.zeros(24,3,dtype=torch.float64);offsets[18,0]=1;offsets[20,0]=1
        _,p=world_kinematics(rotations,offsets)
        torch.testing.assert_close(p[0,20],torch.tensor([0.,2.,0.],dtype=torch.float64),atol=1e-12,rtol=0)

    def test_identity_and_unavailable_channels_are_unchanged(self):
        p,o=inputs(6);o[:]=torch.eye(3,dtype=torch.float64)
        offsets=torch.zeros(24,3)
        a=refine_temporal(p,o,[0],offsets,TemporalIKSettings())['pose']
        self.assertTrue(torch.equal(a,p))
        o[:,1:]=float('nan')
        b=refine_temporal(p,o,[0],offsets,TemporalIKSettings())['pose']
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(refine_temporal(p,o,[],offsets)['pose'],p))

    def test_jerk_penalty_known_cubic_and_invalid_settings(self):
        t=torch.arange(5,dtype=torch.float64)
        position=torch.zeros(1,5,24,3,dtype=torch.float64);position[:,:,:,0]=t[None,:,None]**3
        self.assertAlmostEqual(discrete_jerk_loss(position,torch.ones(1,5,dtype=torch.bool),1.).item(),24.)
        for kw in [dict(jerk_weight=-1),dict(window=3),dict(jerk_scale_m=0),dict(frame_batch=False)]:
            with self.assertRaises(ValueError):TemporalIKSettings(**kw)


if __name__=='__main__':unittest.main()
