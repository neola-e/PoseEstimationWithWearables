"""Infrastructure regressions with synthetic data, never accuracy evidence."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from eval_wrist_pocket_dev import COMBOS, ReservedDipSubset, resample, smooth, validate_records
from analyze_wrist_pocket_dev import contrasts
from sensor_metrics import mask_input


class PlacementStudyTests(unittest.TestCase):
    def test_factorial_additive_effects_and_interaction(self):
        # baseline=10, right wrist=+2, right pocket=+3, interaction=+4
        effects = contrasts({'lw_rp_h': 13., 'rw_rp_h': 19., 'lw_lp_h': 10., 'rw_lp_h': 12.})
        self.assertEqual(effects['wrist_right_minus_left'], 4.)
        self.assertEqual(effects['pocket_right_minus_left'], 5.)
        self.assertEqual(effects['interaction'], 4.)
        self.assertEqual(effects['wrist_right_minus_left_right_pocket'], 6.)
        self.assertEqual(effects['wrist_right_minus_left_left_pocket'], 2.)

    def test_wrist_coverage_signs_are_opposite_for_left_and_right(self):
        left = contrasts({'lw_rp_h': 1., 'rw_rp_h': 5., 'lw_lp_h': 2., 'rw_lp_h': 6.})
        right = contrasts({'lw_rp_h': 6., 'rw_rp_h': 2., 'lw_lp_h': 5., 'rw_lp_h': 1.})
        self.assertEqual(left['wrist_right_minus_left'], 4.)
        self.assertEqual(right['wrist_right_minus_left'], -4.)

    def test_resampling_and_centered_smoothing_match_known_ramp(self):
        ramp = torch.arange(12, dtype=torch.float32).view(12, 1, 1)
        torch.testing.assert_close(resample(ramp).flatten(), torch.tensor([0., 2.4, 4.8, 7.2, 9.6]))
        torch.testing.assert_close(smooth(ramp).flatten(), torch.tensor([1., 1.5, 2., 3., 4., 5., 6., 7., 8., 9., 9.5, 10.]))

    def test_reservation_uses_noncontiguous_original_ids_and_short_tails(self):
        selected = [{'sequence_id': 30, 'subject': 's_07', 'source_file': 'a', 'frames': 127},
                    {'sequence_id': 40, 'subject': 's_08', 'source_file': 'b', 'frames': 1}]
        rows = [{**selected[0], 'start_frame': 0, 'frames': 125},
                {**selected[0], 'start_frame': 125, 'frames': 2},
                {**selected[1], 'start_frame': 0}]
        result = validate_records(rows, selected)
        self.assertEqual(result, {'subjects': 2, 'sequences': 2, 'windows': 3, 'frames': 128, 'partial_windows': 2})
        for bad in [rows[:-1], rows + [rows[0]], [{**rows[0], 'subject': 's_09'}, *rows[1:]],
                    [rows[0], {**rows[1], 'start_frame': 126}, rows[2]]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_records(bad, selected)

    def test_dataset_excludes_unreserved_sequences_and_all_masks_preserve_targets(self):
        lengths = [3, 127, 4, 2]
        data = {'acc': [torch.full((n, 6, 3), (i + 1.) * 30) for i, n in enumerate(lengths)],
                'ori': [torch.eye(3).repeat(n, 6, 1, 1) for n in lengths],
                'pose': [torch.eye(3).repeat(n, 24, 1, 1) for n in lengths]}
        selected = [{'sequence_id': 1, 'subject': 's_07', 'source_file': 'a', 'frames': 127},
                    {'sequence_id': 3, 'subject': 's_08', 'source_file': 'b', 'frames': 2}]
        config = SimpleNamespace(acc_scale=30, max_sample_len=300, r6d=True, pred_joints_set=list(range(24)))
        ds = ReservedDipSubset(data, selected, config)
        self.assertEqual(len(ds), 3)
        self.assertEqual({r['sequence_id'] for r in ds.records}, {1, 3})
        validate_records(ds.records, selected)
        for i in range(len(ds)):
            x, target = ds[i]
            self.assertEqual(target.shape, (ds.records[i]['frames'], 144))
            expected_acc = 2 if ds.records[i]['sequence_id'] == 1 else 4
            torch.testing.assert_close(x[:, :15], torch.full_like(x[:, :15], expected_acc))
            for sensors in COMBOS.values():
                masked = mask_input(x.numpy(), sensors)
                for sensor in range(5):
                    self.assertEqual(bool(np.any(masked[:, 3 * sensor:3 * sensor + 3])), sensor in sensors)
                    self.assertEqual(bool(np.any(masked[:, 15 + 9 * sensor:24 + 9 * sensor])), sensor in sensors)
            torch.testing.assert_close(ds[i][0], x, rtol=0, atol=0)
            torch.testing.assert_close(ds[i][1], target, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
