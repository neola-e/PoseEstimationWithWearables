"""Numerical checks for replay selection, handedness, frame timing and jerk support."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from render_final_comparison import clip_statistics, display_coordinates, fixed_cases, frame_record


class FinalReplayTests(unittest.TestCase):
    def test_fixed_selection_uses_identity_center_and_archived_peak(self):
        records = [dict(subject='s_09', source_file='03_a.pkl', sequence_id=5, frames=1207),
                   dict(subject='s_09', source_file='01_a.pkl', sequence_id=0, frames=1978),
                   dict(subject='s_10', source_file='01_a.pkl', sequence_id=9, frames=2333)]
        diagnosis = {'top_pairs': [dict(sequence_id=9, mask='rw_lp', category='upper_body', peak_output_frame=1371, peak_joint=23)]}
        selected = fixed_cases(records, diagnosis)
        self.assertEqual([(c['start'], c['stop']) for c in selected], [(478, 728), (864, 1114), (1246, 1496)])
        self.assertEqual([c['mask'] for c in selected], ['rw_lp'] * 3)

    def test_display_preserves_handedness_distance_and_y_up(self):
        basis = display_coordinates(np.eye(3))
        np.testing.assert_array_equal(np.cross(basis[0], basis[1]), basis[2])
        np.testing.assert_array_equal(display_coordinates([[0, 1, 0]]), [[0, 0, 1]])
        self.assertEqual(float(np.linalg.det(basis)), 1.)
        np.testing.assert_allclose(np.linalg.norm(display_coordinates([[1, 2, 3]])), np.sqrt(14))

    def test_all_panels_use_exact_frame_and_processed_clock(self):
        row = frame_record({'start': 1246}, 125)
        self.assertEqual(row['recording_frame'], 1371)
        self.assertEqual(row['recording_seconds'], 54.84)
        self.assertEqual(row['clip_seconds'], 5.)
        self.assertTrue(all(row[k + '_source_frame'] == 1371 for k in ['network', 'framewise_ik', 'temporal_ik', 'gt']))

    def test_known_trajectory_jerk_and_pelvis_alignment(self):
        t = np.arange(8, dtype=np.float64) / 25
        reference = np.zeros((8, 24, 3))
        motion = reference.copy(); motion[:, 1:, 0] = t[:, None] ** 3
        branches = ['network', 'framewise_ik', 'temporal_ik', 'gt']
        positions = {k: reference.copy() if k == 'gt' else motion.copy() for k in branches}
        rotations = {k: np.broadcast_to(np.eye(3), (8, 24, 3, 3)).copy() for k in branches}
        stats = clip_statistics(positions, rotations)
        self.assertEqual(stats['network']['jerk'].shape, (5, 24))
        self.assertAlmostEqual(stats['network']['jitter_100_m_s3'], .06 * 23 / 24)
        self.assertEqual(stats['gt']['mpjpe_cm'], 0.)
        self.assertEqual(stats['network']['root_error_mean_deg'], 0.)
        translated = {k: p + np.array([5., -2., 7.]) for k, p in positions.items()}
        stats2 = clip_statistics(translated, rotations)
        self.assertAlmostEqual(stats2['network']['mpjpe_cm'], stats['network']['mpjpe_cm'], places=10)
        self.assertAlmostEqual(stats2['network']['jitter_100_m_s3'], stats['network']['jitter_100_m_s3'], places=10)


if __name__ == '__main__':
    unittest.main()
