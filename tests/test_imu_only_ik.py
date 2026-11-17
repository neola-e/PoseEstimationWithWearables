"""Numerical, topology, isolation, and frame-independence checks for orientation IK."""
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from imu_only_ik import (IKSettings, OPTIMIZED_BY_SENSOR, PARENTS, SENSOR_JOINTS,
                         exp_map, global_rotations, project_orientations, refine_pose)
from imuposer import math as pose_math


def rotation_z(degrees):
    a=math.radians(degrees)
    return torch.tensor([[math.cos(a),-math.sin(a),0.],
                         [math.sin(a), math.cos(a),0.],[0.,0.,1.]],dtype=torch.float64)


def inputs(frames=4):
    pose=torch.tensor([1.,0.,0.,0.,1.,0.]).repeat(frames,24).float()
    ori=torch.eye(3,dtype=torch.float64).repeat(frames,5,1,1)
    ori[:,0]=rotation_z(25)
    ori[:,1]=rotation_z(-18)
    ori[:,2]=rotation_z(8)
    ori[:,3]=rotation_z(12)
    ori[:,4]=rotation_z(30)
    return pose,ori


class ImuOnlyIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exp_has_known_angles_and_finite_correct_zero_derivative(self):
        for degrees in [0,1e-5,90,180,-37]:
            v=torch.tensor([0.,0.,math.radians(degrees)],dtype=torch.float64)
            torch.testing.assert_close(exp_map(v),rotation_z(degrees),atol=1e-12,rtol=0)
        zero=torch.zeros(2,3,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(exp_map,(zero,),eps=1e-6,atol=1e-8))

    def test_segment_mapping_and_global_rotation_chain(self):
        self.assertEqual(SENSOR_JOINTS,(18,19,1,2,15))
        self.assertEqual(OPTIMIZED_BY_SENSOR[0],(16,18))
        self.assertEqual(OPTIMIZED_BY_SENSOR[2],(0,1))
        r=torch.eye(3,dtype=torch.float64).repeat(1,24,1,1)
        r[:,16]=rotation_z(10);r[:,18]=rotation_z(20);r[:,20]=rotation_z(40)
        g=global_rotations(r)
        torch.testing.assert_close(g[0,18],rotation_z(30),atol=1e-12,rtol=0)
        torch.testing.assert_close(g[0,20],rotation_z(70),atol=1e-12,rtol=0)
        torch.testing.assert_close(g,pose_math.forward_kinematics_R(r,PARENTS),atol=1e-12,rtol=0)

    def test_single_wrist_constraint_decreases_and_only_selected_local_joints_change(self):
        p,o=inputs();original=p.clone();q=refine_pose(p,o,[0])
        self.assertLess(q['summary']['orientation_mse_after'],q['summary']['orientation_mse_before'])
        self.assertTrue((q['loss_history'][:,-1]<q['loss_history'][:,0]).all())
        for joint in range(24):
            changed=not torch.equal(q['pose'].view(-1,24,6)[:,joint],p.view(-1,24,6)[:,joint])
            self.assertEqual(changed,joint in (16,18))
        self.assertTrue(torch.equal(p,original))

    def test_final_update_returned_and_zero_iterations_bitwise_identity(self):
        p,o=inputs(1)
        zero=refine_pose(p,o,[0],IKSettings(iterations=0))
        self.assertTrue(torch.equal(zero['pose'],p))
        one=refine_pose(p,o,[0],IKSettings(iterations=1))
        self.assertFalse(torch.equal(one['pose'],p))
        self.assertEqual(one['loss_history'].shape,(1,2))
        self.assertLess(one['loss_history'][0,1],one['loss_history'][0,0])

    def test_matching_motion_is_fixed_and_zero_sensor_ablation_ignores_nan(self):
        p,o=inputs(2);o[:]=torch.eye(3,dtype=torch.float64)
        q=refine_pose(p,o,[0,1,4])
        torch.testing.assert_close(q['pose'],p,atol=0,rtol=0)
        self.assertEqual(q['loss_history'].max().item(),0)
        o[:]=float('nan')
        self.assertTrue(torch.equal(refine_pose(p,o,[])['pose'],p))

    def test_independent_frame_batching_and_no_future_or_optimizer_state(self):
        p,o=inputs(5)
        for t in range(5):o[t,0]=rotation_z(5+7*t)
        one=refine_pose(p,o,[0,2],IKSettings(frame_batch=1))
        batch=refine_pose(p,o,[0,2],IKSettings(frame_batch=3))
        torch.testing.assert_close(one['pose'],batch['pose'],atol=1e-7,rtol=0)
        torch.testing.assert_close(one['loss_history'],batch['loss_history'],atol=1e-14,rtol=0)
        changed=o.clone();changed[1]=rotation_z(-90)
        other=refine_pose(p,changed,[0,2],IKSettings(frame_batch=3))
        self.assertTrue(torch.equal(batch['pose'][[0,2,3,4]],other['pose'][[0,2,3,4]]))
        reordered=refine_pose(p.flip(0),o.flip(0),[2,0],IKSettings(frame_batch=2))
        torch.testing.assert_close(batch['pose'],reordered['pose'].flip(0),atol=1e-7,rtol=0)

    def test_missing_measurements_cannot_affect_available_constraint(self):
        p,o=inputs();a=refine_pose(p,o,[0])
        o[:,1:]=float('nan');b=refine_pose(p,o,[0])
        self.assertTrue(torch.equal(a['pose'],b['pose']))
        self.assertTrue(torch.equal(a['loss_history'],b['loss_history']))

    def test_joint_optimization_preserves_so3_and_does_not_update_input_gradients(self):
        p,o=inputs();p.requires_grad_()
        with torch.inference_mode():q=refine_pose(p,o,[0,2,4])
        self.assertIsNone(p.grad)
        self.assertFalse(q['pose'].requires_grad)
        r=pose_math.r6d_to_rotation_matrix(q['pose']).reshape(-1,3,3)
        torch.testing.assert_close(r.transpose(-1,-2)@r,torch.eye(3).expand_as(r),atol=2e-6,rtol=0)
        torch.testing.assert_close(torch.linalg.det(r),torch.ones(len(r)),atol=2e-6,rtol=0)

    def test_projection_has_known_polar_factor_and_rejects_invalid_active_data(self):
        expected=rotation_z(32)
        distorted=expected@torch.diag(torch.tensor([.8,.9,1.],dtype=torch.float64))
        torch.testing.assert_close(project_orientations(distorted),expected,atol=1e-12,rtol=0)
        for bad in [torch.zeros(3,3),torch.diag(torch.tensor([1.,1.,-1.])),torch.full((3,3),float('nan'))]:
            with self.assertRaises(ValueError):project_orientations(bad.double())

    def test_invalid_settings_masks_shapes_and_degenerate_pose_rejected(self):
        for kwargs in [dict(iterations=-1),dict(iterations=True),dict(learning_rate=0),
                       dict(frame_batch=0),dict(learning_rate=float('nan'))]:
            with self.assertRaises(ValueError):IKSettings(**kwargs)
        p,o=inputs()
        for ids in [[0,0],[True],[5],[-1]]:
            with self.assertRaises(ValueError):refine_pose(p,o,ids)
        with self.assertRaises(ValueError):refine_pose(p[:,:100],o,[0])
        with self.assertRaises(ValueError):refine_pose(torch.zeros_like(p),o,[0])
        with self.assertRaises(ValueError):refine_pose(p,o[:1],[0])


if __name__=='__main__':unittest.main()
