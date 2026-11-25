"""Analytic metric checks plus optional licensed-SMPL integration checks."""
import copy
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from paper_metrics import (PoseMetricEvaluator, aggregate_sequences, aligned_position_errors,
                           cache_sequences, jerk_norms, rotation_angles_deg)


def z_rotation(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=torch.float64)


class AnalyticMetricTests(unittest.TestCase):
    def test_known_global_rotation_angles_at_identity_small_right_angle_and_pi(self):
        for degrees in [0., 1e-5, 90., 180.]:
            with self.subTest(degrees=degrees):
                value = rotation_angles_deg(z_rotation(degrees)[None], torch.eye(3)[None])
                self.assertAlmostEqual(value.item(), degrees, places=9)

    def test_identical_nontrivial_rotations_are_exactly_zero(self):
        r = z_rotation(37.4).float().repeat(3, 24, 1, 1)
        self.assertEqual(float(rotation_angles_deg(r, r).max()), 0.)

    def test_rotation_rejects_scale_reflection_and_nonfinite(self):
        for r in [torch.eye(3) * 2, torch.diag(torch.tensor([1., 1., -1.])), torch.full((3, 3), float('nan'))]:
            with self.subTest(r=r), self.assertRaises(ValueError):
                rotation_angles_deg(r[None], torch.eye(3)[None])

    def test_cm_units_root_inclusion_and_vertex_measure_are_distinct(self):
        reference_j = torch.zeros(1, 2, 3, dtype=torch.float64)
        predicted_j = reference_j.clone()
        predicted_j[:, 1, 0] = .01
        reference_v = torch.zeros(1, 4, 3, dtype=torch.float64)
        predicted_v = reference_v.clone()
        predicted_v[:, :, 0] = .02
        je, ve = aligned_position_errors(predicted_j, reference_j, predicted_v, reference_v)
        self.assertAlmostEqual(je.mean().item(), .5)  # Includes the zero-error root.
        self.assertAlmostEqual(ve.mean().item(), 2.)  # Vertices, not joint or velocity error.

    def test_pelvis_alignment_removes_translation_but_not_rotation(self):
        j = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], dtype=torch.float64)
        v = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], dtype=torch.float64)
        offset = torch.tensor([[[5., -2., 3.]]])
        je, ve = aligned_position_errors(j + offset, j, v + offset, v)
        self.assertEqual(je.max().item(), 0.)
        self.assertEqual(ve.max().item(), 0.)
        je, _ = aligned_position_errors(j @ z_rotation(90).T, j, v @ z_rotation(90).T, v)
        self.assertAlmostEqual(je[0, 1].item(), math.sqrt(2) * 100, places=9)

    def test_jerk_static_linear_and_quadratic_are_zero(self):
        time = torch.arange(12, dtype=torch.float64) / 25
        for power in [0, 1, 2]:
            p = torch.zeros(12, 2, 3, dtype=torch.float64)
            p[:, :, 0] = (time ** power)[:, None]
            self.assertLess(jerk_norms(p, 25).max().item(), 1e-8)

    def test_cubic_trajectory_has_known_jerk_and_cubic_fps_scaling(self):
        time = torch.arange(12, dtype=torch.float64) / 25
        p = torch.zeros(12, 2, 3, dtype=torch.float64)
        p[:, :, 0] = (time ** 3)[:, None]
        torch.testing.assert_close(jerk_norms(p, 25), torch.full((9, 2), 6., dtype=torch.float64), atol=1e-9, rtol=0)
        torch.testing.assert_close(jerk_norms(p, 50), torch.full((9, 2), 48., dtype=torch.float64), atol=1e-8, rtol=0)

    def test_short_trajectories_have_no_jerk_samples(self):
        for n in [1, 2, 3]:
            self.assertEqual(jerk_norms(torch.zeros(n, 24, 3), 25).shape, (0, 24))
        for fps in [0, -1, float('inf'), True]:
            with self.assertRaises(ValueError):
                jerk_norms(torch.zeros(5, 24, 3), fps)

    def test_sequence_boundary_must_not_create_artificial_jerk(self):
        a, b = torch.zeros(5, 24, 3), torch.ones(5, 24, 3) * 100
        self.assertEqual(jerk_norms(a, 25).max().item(), 0.)
        self.assertEqual(jerk_norms(b, 25).max().item(), 0.)
        self.assertGreater(jerk_norms(torch.cat([a, b]), 25).mean().item(), 0.)

    def test_cache_reassembles_windows_within_recordings_only(self):
        pose = np.arange(12 * 144).reshape(12, 144)
        cache = {'p': pose, 'target_pose': pose.copy(), 'offsets': np.array([0, 4, 7, 12]),
                 'sequence_ids': np.array([30, 30, 31]), 'start_frames': np.array([0, 4, 0])}
        groups = cache_sequences(cache, 'p')
        self.assertEqual([len(g['predicted']) for g in groups], [7, 5])
        self.assertEqual([g['source_windows'] for g in groups], [2, 1])
        np.testing.assert_array_equal(groups[0]['predicted'], pose[:7])
        # offsets alone are independent segments, not an implicit continuous recording.
        cache.pop('sequence_ids'); cache.pop('start_frames')
        self.assertEqual([len(g['predicted']) for g in cache_sequences(cache, 'p')], [4, 3, 5])

    def test_cache_rejects_gaps_overlaps_bad_offsets_and_partial_metadata(self):
        pose = np.zeros((8, 144))
        base = {'p': pose, 'target_pose': pose, 'offsets': np.array([0, 4, 8]),
                'sequence_ids': np.array([1, 1]), 'start_frames': np.array([0, 4])}
        for updates in [{'start_frames': np.array([0, 5])}, {'start_frames': np.array([0, 3])},
                        {'offsets': np.array([0, 4, 9])}, {'offsets': np.array([0., 4., 8.])}]:
            with self.assertRaises(ValueError):cache_sequences({**base, **updates}, 'p')
        for key in ['offsets', 'start_frames']:
            bad = dict(base); bad.pop(key)
            with self.assertRaises(ValueError):cache_sequences(bad, 'p')
        with self.assertRaises(ValueError):
            cache_sequences({'p': pose, 'target_pose': pose, 'offsets': np.array([0, 6, 4, 8], dtype=np.uint64)}, 'p')

    def test_single_sequence_lengths_exclude_padding_and_reject_ambiguous_lengths(self):
        p = np.zeros((12, 144))
        source = {'p': p, 'target_pose': p, 'lengths': np.array([9])}
        self.assertEqual(len(cache_sequences(source, 'p')[0]['predicted']), 9)
        for length in [np.array([0]), np.array([13]), np.array([9.]), np.array([4, 5])]:
            with self.assertRaises(ValueError):cache_sequences({**source, 'lengths': length}, 'p')
        with self.assertRaises(ValueError):
            cache_sequences({**source, 'offsets': np.array([0, 12])}, 'p')

    def test_temporal_and_spatial_aggregates_use_their_own_counts(self):
        def result(identity, n, value):
            return {'sequence_id': identity, 'frames': n, 'jerk_samples': max(n - 3, 0), 'settings': {},
                    'metrics': {**{k: value for k in ['mpjre_deg', 'mpjpe_cm', 'mpjve_cm', 'local_rotation_deg']},
                                'jitter_m_s3': value if n >= 4 else None, 'reference_jitter_m_s3': value if n >= 4 else None}}
        rows = [result('a', 5, 2.), result('b', 7, 10.), result('short', 1, 100.)]
        report = aggregate_sequences(rows)
        self.assertAlmostEqual(report['metrics']['frame_weighted']['mpjpe_cm'], 180 / 13)
        self.assertAlmostEqual(report['metrics']['frame_weighted']['jitter_m_s3'], 44 / 6)
        self.assertAlmostEqual(report['metrics']['sequence_equal']['jitter_m_s3'], 6.)
        self.assertAlmostEqual(report['metrics']['frame_weighted']['jitter_100_m_s3'], 44 / 600)
        short = aggregate_sequences([result('a', 1, 0.)])
        self.assertIsNone(short['metrics']['frame_weighted']['jitter_m_s3'])


SMPL = Path(os.environ.get('IMUPOSER_SMPL', '/home/alex/research/IMUPoser/src/imuposer/smpl/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'))


@unittest.skipUnless(SMPL.is_file(), 'Optional licensed SMPL asset unavailable')
class SmplMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from imuposer.smpl.parametricModel import ParametricModel
        torch.set_num_threads(2)
        cls.body = ParametricModel(str(SMPL), device=torch.device('cpu'))

    def identity(self, n=9):
        return torch.eye(3).repeat(n, 24, 1, 1)

    def test_identical_static_pose_is_zero_for_every_metric(self):
        p = self.identity()
        result = PoseMetricEvaluator(self.body, fk_chunk=3).evaluate_sequence(p, p, 'identity')
        for value in result['metrics'].values():self.assertEqual(value, 0.)

    def test_identical_moving_pose_has_zero_errors_but_nonzero_jitter(self):
        p = self.identity()
        for i in range(len(p)):p[i, 0] = z_rotation(i * i).float()
        result = PoseMetricEvaluator(self.body).evaluate_sequence(p, p, 'moving')
        for key in ['mpjre_deg', 'mpjpe_cm', 'mpjve_cm', 'local_rotation_deg']:
            self.assertEqual(result['metrics'][key], 0.)
        self.assertGreater(result['metrics']['jitter_m_s3'], 0.)
        self.assertEqual(result['metrics']['jitter_m_s3'], result['metrics']['reference_jitter_m_s3'])

    def test_root_rotation_distinguishes_global_and_local_and_matches_analytic_geometry(self):
        t = self.identity()
        p = t.clone(); p[:, 0] = z_rotation(90).float()
        result = PoseMetricEvaluator(self.body).evaluate_sequence(p, t, 'root90')
        j, v = self.body.get_zero_pose_joint_and_vertex()
        expected_j = j.double()[:, :2].norm(dim=-1).mean().item() * math.sqrt(2) * 100
        expected_v = v.double()[:, :2].norm(dim=-1).mean().item() * math.sqrt(2) * 100
        self.assertAlmostEqual(result['metrics']['mpjre_deg'], 90., places=6)
        self.assertAlmostEqual(result['metrics']['local_rotation_deg'], 90 / 24, places=6)
        self.assertAlmostEqual(result['metrics']['mpjpe_cm'], expected_j, places=4)
        self.assertAlmostEqual(result['metrics']['mpjve_cm'], expected_v, places=4)

    def test_leaf_rotation_moves_mesh_without_moving_joint_positions(self):
        t = self.identity(); p = t.clone(); p[:, 22] = z_rotation(90).float()
        result = PoseMetricEvaluator(self.body).evaluate_sequence(p, t, 'leaf')
        self.assertLess(result['metrics']['mpjpe_cm'], 1e-5)
        self.assertGreater(result['metrics']['mpjve_cm'], .01)

    def test_chunk_size_repeatability_shared_shape_and_pose_blendshape(self):
        t = self.identity(11); p = t.clone(); p[:, 18] = z_rotation(30).float()
        a = PoseMetricEvaluator(self.body, fk_chunk=2).evaluate_sequence(p, t, 'a', shape=[.1] * 10)
        b = PoseMetricEvaluator(self.body, fk_chunk=7).evaluate_sequence(p, t, 'b', shape=[.1] * 10)
        again = PoseMetricEvaluator(self.body, fk_chunk=2).evaluate_sequence(p, t, 'a', shape=[.1] * 10)
        self.assertEqual(a, again)
        for k in a['metrics']:self.assertAlmostEqual(a['metrics'][k], b['metrics'][k], places=5)
        from imuposer.smpl.parametricModel import ParametricModel
        blend = ParametricModel(str(SMPL), use_pose_blendshape=True, device=torch.device('cpu'))
        c = PoseMetricEvaluator(blend).evaluate_sequence(p, t, 'c', shape=[.1] * 10)
        self.assertEqual(a['metrics']['mpjpe_cm'], c['metrics']['mpjpe_cm'])
        self.assertNotEqual(a['metrics']['mpjve_cm'], c['metrics']['mpjve_cm'])

    def test_translation_alignment_and_jitter_convention(self):
        p = self.identity(10)
        time = torch.arange(10, dtype=torch.float32) / 25
        translation = torch.zeros(10, 3); translation[:, 0] = time ** 3
        row = PoseMetricEvaluator(self.body).evaluate_sequence(p, p, 'translation', predicted_translation_m=translation)
        self.assertLess(row['metrics']['mpjpe_cm'], 1e-5)
        self.assertLess(row['metrics']['mpjve_cm'], 1e-5)
        self.assertAlmostEqual(row['metrics']['jitter_m_s3'], 6., places=3)
        self.assertEqual(row['metrics']['reference_jitter_m_s3'], 0.)

    def test_short_pose_sequence_returns_null_jitter_and_rejects_invalid_pose(self):
        p = self.identity(3)
        result = PoseMetricEvaluator(self.body).evaluate_sequence(p, p, 'short')
        self.assertIsNone(result['metrics']['jitter_m_s3'])
        bad = p.clone(); bad[0, 1] *= 2
        with self.assertRaises(ValueError):PoseMetricEvaluator(self.body).evaluate_sequence(bad, p, 'bad')
        with self.assertRaises(ValueError):PoseMetricEvaluator(self.body).evaluate_sequence(p, p, 'bad', shape=np.zeros((3, 10)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
