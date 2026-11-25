"""Synthetic infrastructure checks; these do not supply accuracy evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from dual_wrist_dev import SENSORS, aggregate, check_pairs, outcomes
from sensor_metrics import METRICS, mask_input


def row(subject, sequence, frames, left, right):
    joint = {k: np.zeros(24) for k in METRICS}
    joint['position_cm'][20:22] = [left, right]
    return {'subject': subject, 'sequence_id': sequence, 'frames': frames, 'joint_errors': joint}


class DualWristTests(unittest.TestCase):
    def test_equal_subjects_differs_from_frame_and_sequence_weighting(self):
        rows = [row('a', 1, 1, 2, 4), row('a', 2, 3, 6, 8), row('b', 3, 12, 10, 12)]
        self.assertEqual(outcomes(aggregate(rows, 'subject_equal'))['left_wrist_position_cm'], 7.5)
        self.assertEqual(outcomes(aggregate(rows, 'frame_weighted'))['left_wrist_position_cm'], 8.75)
        self.assertEqual(outcomes(aggregate(rows, 'sequence_equal'))['left_wrist_position_cm'], 6.)

    def test_worse_wrist_is_computed_after_joint_aggregation(self):
        rows = [row('a', 1, 1, 2, 10), row('b', 2, 1, 10, 2)]
        values = outcomes(aggregate(rows, 'subject_equal'))
        self.assertEqual(values['bilateral_wrist_mean_position_cm'], 6.)
        self.assertEqual(values['worse_wrist_position_cm'], 6.)
        self.assertNotEqual(values['worse_wrist_position_cm'], 10.)

    def test_dual_wrist_mask_retains_both_feature_blocks_and_preserves_input(self):
        x = np.arange(1, 121, dtype=np.float32).reshape(2, 60)
        original = x.copy()
        masked = mask_input(x, SENSORS)
        self.assertEqual(len(SENSORS), 3)
        for sensor in range(5):
            for block in [slice(3 * sensor, 3 * sensor + 3), slice(15 + 9 * sensor, 24 + 9 * sensor)]:
                np.testing.assert_array_equal(masked[:, block], x[:, block] if sensor in SENSORS else 0 * x[:, block])
        np.testing.assert_array_equal(x, original)

    def test_pairing_rejects_target_subject_recording_or_window_change(self):
        a = {'sequence_id': 30, 'start_frame': 0, 'frames': 125,
             'subject': 's_07', 'source_file': '01.pkl', 'target_sha256': 'same'}
        check_pairs([a], [dict(a)])
        for key, value in [('subject', 's_08'), ('source_file', '02.pkl'), ('target_sha256', 'changed'), ('frames', 124)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_pairs([a], [{**a, key: value}])


if __name__ == '__main__':
    unittest.main(verbosity=2)
