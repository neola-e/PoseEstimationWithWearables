"""Lightweight regression checks; no checkpoint or licensed data needed."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from sensor_metrics import (REGIONS, cluster_bootstrap, direction_counts, joint_errors,
                            mask_input, pair_windows, validate_coverage, weighted_mean)
from dip_subset import DipTestSubset


def row(seq, start, frames):
    return {'sequence_id': seq, 'start_frame': start, 'frames': frames}


class PairedSensorTests(unittest.TestCase):
    def test_pairs_reorder_but_require_identical_keys_and_lengths(self):
        a = [row(0, 0, 125), row(0, 125, 2)]
        self.assertEqual(pair_windows(a, a[::-1]), [(a[0], a[0]), (a[1], a[1])])
        for bad in ([row(0, 0, 124), a[1]], a[:1], a + [a[0]], [row(1, 0, 125), a[1]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pair_windows(a, bad)

    def test_coverage_partial_tail_and_failures(self):
        rows = [row(0, 0, 125), row(0, 125, 2), row(1, 0, 1)]
        self.assertEqual(validate_coverage(rows, [127, 1])['frames'], 128)
        for bad in (rows[:-1], rows + [rows[0]], [rows[0], row(0, 126, 1), rows[2]],
                    [rows[0], row(0, 125, 1), rows[2]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_coverage(bad, [127, 1])

    def test_mask_feature_layout_preserves_source(self):
        x = np.arange(1, 181, dtype=np.float32).reshape(3, 60)
        original = x.copy()
        y = mask_input(x, [0, 3, 4])
        np.testing.assert_array_equal(x, original)
        for sensor in range(5):
            for sl in (slice(3 * sensor, 3 * sensor + 3), slice(15 + 9 * sensor, 24 + 9 * sensor)):
                np.testing.assert_array_equal(y[:, sl], x[:, sl] if sensor in [0, 3, 4] else np.zeros_like(x[:, sl]))
        np.testing.assert_array_equal(mask_input(x, range(5)), x)

    def test_real_dataset_windowing_and_masking_on_synthetic_container(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dip_test.pt'
            lengths = [127, 3]
            torch.save({'acc': [torch.ones(n, 6, 3) * 30 for n in lengths],
                        'ori': [torch.eye(3).repeat(n, 6, 1, 1) for n in lengths],
                        'pose': [torch.eye(3).repeat(n, 24, 1, 1) for n in lengths]}, path)
            config = SimpleNamespace(max_sample_len=300, acc_scale=30, r6d=True, pred_joints_set=list(range(24)))
            a = DipTestSubset(path, config, None, None, 'global')
            b = DipTestSubset(path, config, None, None, 'lw_rp_h')
            self.assertEqual(validate_coverage(a.records, lengths)['frames'], 130)
            self.assertEqual(len(pair_windows(a.records, b.records)), 3)
            for i in range(len(a)):
                x, gt = a[i]
                y, target = b[i]
                np.testing.assert_array_equal(y.numpy(), mask_input(x.numpy(), [0, 3, 4]))
                torch.testing.assert_close(gt, target, rtol=0, atol=0)

    def test_frame_weighting_and_partition(self):
        self.assertAlmostEqual(weighted_mean([2, 100], [125, 1]), 350 / 126)
        joint = weighted_mean(np.array([[2] * 24, [100] * 24]), [125, 1])
        self.assertAlmostEqual(joint.mean(), 350 / 126)
        flat = sum([REGIONS[k] for k in ['arms', 'legs', 'torso', 'head']], [])
        self.assertEqual(sorted(flat), list(range(24)))
        with self.assertRaises(ValueError):
            weighted_mean([1, np.nan], [1, 1])

    def test_known_local_global_rotation_and_position(self):
        # A two-joint chain: root rotation moves the child but child local rotation stays identity.
        class Body:
            def forward_kinematics(self, r, calc_mesh=False):
                g = torch.stack([r[:, 0], r[:, 0] @ r[:, 1]], dim=1)
                p = torch.stack([torch.zeros(len(r), 3), r[:, 0, :, 0]], dim=1)
                return g, p
        gt = torch.eye(3).repeat(1, 2, 1, 1)
        pred = gt.clone()
        pred[:, 0] = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        errors = joint_errors(pred, gt, Body())
        torch.testing.assert_close(errors['local_deg'], torch.tensor([[90., 0.]]))
        torch.testing.assert_close(errors['global_deg'], torch.tensor([[90., 90.]]))
        self.assertAlmostEqual(errors['position_cm'][0, 1].item(), np.sqrt(2) * 100, places=4)

    def test_bootstrap_resamples_whole_paired_sequences(self):
        result = cluster_bootstrap([-2, 2], [100, 1], draws=1000, seed=3)
        self.assertEqual(result, cluster_bootstrap([-2, 2], [100, 1], draws=1000, seed=3))
        fixed = cluster_bootstrap([-1, -1], [100, 1], draws=10)
        self.assertEqual(fixed['frame_weighted_delta_ci95'], [-1., -1.])
        self.assertEqual(direction_counts([-1, 0, 1])['tied'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
