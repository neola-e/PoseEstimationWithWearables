"""Window support, online/offline equivalence, and future-perturbation tests."""
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from window_inference import (RollingPoseStream, dependency_trace, infer_sequence,
                              smooth_acceleration, window_plan)


class BidirectionalProbe:
    """Output depends on both the past and future of each VALID window frame."""
    def predict(self, x, lengths=None):
        x = torch.as_tensor(x)
        single = x.ndim == 2
        if single:
            x = x[None]
        lengths = [x.shape[1]] * len(x) if lengths is None else lengths
        result = torch.zeros(len(x), x.shape[1], 144)
        for b, n in enumerate(lengths):
            value = x[b, :n, 0]
            past = value.cumsum(0)
            future = value.flip(0).cumsum(0).flip(0)
            result[b, :n] = (past + 10 * future)[:, None]
        return result[0] if single else result


class WindowInferenceTests(unittest.TestCase):
    def setUp(self):
        self.predictor = BidirectionalProbe()
        self.x = torch.zeros(260, 60)
        self.x[:, 0] = torch.arange(1, 261)

    def test_rolling_trace_warmup_boundaries_and_last_frame(self):
        rows = dependency_trace(260, smoothing='none')
        for t, start, stop, index in [(0,0,1,0), (123,0,124,123),
                                     (124,0,125,124), (125,1,126,124), (259,135,260,124)]:
            self.assertEqual([rows[t][k] for k in ['input_start','input_stop','model_output_index']],
                             [start, stop, index])
        self.assertEqual([r['output_frame'] for r in rows], list(range(260)))
        self.assertTrue(all(r['future_frames_at_25hz'] == 0 for r in rows))

    def test_chunk_is_offline_and_short_tail_retained(self):
        y = infer_sequence(self.predictor, self.x, 'chunk', batch_size=2)
        self.assertEqual(y.shape, (260,144))
        self.assertEqual(y[0,0], 1 + 10 * sum(range(1,126)))
        self.assertEqual(y[259,0], sum(range(251,261)) + 2600)
        rows = dependency_trace(260, 'chunk', smoothing='none')
        self.assertEqual(rows[0]['future_frames_at_25hz'], 124)
        self.assertEqual(rows[250]['input_stop'], 260)
        self.assertEqual(rows[-1]['future_frames_at_25hz'], 0)

    def test_rolling_matches_analytic_history_and_ignores_future(self):
        y = infer_sequence(self.predictor, self.x, batch_size=16)
        for t in range(len(self.x)):
            expected = sum(range(max(0,t-124)+1,t+2)) + 10*(t+1)
            self.assertEqual(y[t,0], expected)
        changed = self.x.clone()
        changed[130:,0] = -999
        z = infer_sequence(self.predictor, changed, batch_size=13)
        self.assertTrue(torch.equal(y[:130], z[:130]))
        self.assertFalse(torch.equal(y[130:], z[130:]))

    def test_live_push_equivalent_to_batched_windows_and_reset(self):
        for startup in ['prefix', 'repeat_first']:
            stream = RollingPoseStream(self.predictor, startup=startup)
            online = torch.stack([stream.push(frame) for frame in self.x])
            batch = infer_sequence(self.predictor, self.x, startup=startup, batch_size=17)
            self.assertTrue(torch.equal(online, batch))
            self.assertEqual(len(stream.buffer), 125)
            stream.reset()
            self.assertEqual(stream.frames_seen, 0)
            self.assertTrue(torch.equal(stream.push(self.x[0]), batch[0]))

    def test_prefix_differs_from_repeat_startup_but_converges_at_125(self):
        a = infer_sequence(self.predictor, self.x)
        b = infer_sequence(self.predictor, self.x, startup='repeat_first')
        self.assertNotEqual(a[0,0], b[0,0])
        self.assertTrue(torch.equal(a[124:],b[124:]))
        trace = dependency_trace(260, startup='repeat_first', smoothing='none')
        self.assertEqual(trace[0]['left_repeats_of_frame_zero'],124)
        self.assertEqual(trace[0]['model_output_index'],124)
        self.assertEqual(trace[0]['input_stop'],1)

    def test_five_frame_average_edges_and_future_perturbation(self):
        x = torch.arange(10.).reshape(10,1)
        trailing = smooth_acceleration(x, 'trailing')
        centered = smooth_acceleration(x, 'centered')
        self.assertEqual(trailing[0,0], 0)
        self.assertEqual(trailing[4,0], 2)
        self.assertEqual(centered[0,0], 1)
        self.assertEqual(centered[-1,0], 8)
        z = x.clone()
        z[5:] = 1000
        self.assertTrue(torch.equal(trailing[:5], smooth_acceleration(z,'trailing')[:5]))
        self.assertFalse(torch.equal(centered[:5], smooth_acceleration(z,'centered')[:5]))
        rows = dependency_trace(10, smoothing='centered')
        self.assertEqual(rows[4]['acc_source_stop'],7)
        self.assertEqual(rows[4]['future_frames_at_25hz'],2)
        self.assertEqual(rows[-1]['future_frames_at_25hz'],0)
        rows = dependency_trace(260, smoothing='trailing')
        self.assertEqual(rows[130]['acc_source_start'],2)
        self.assertTrue(all(r['future_frames_at_25hz'] == 0 for r in rows))

    def test_recording_boundary_does_not_reuse_previous_motion(self):
        first = infer_sequence(self.predictor, self.x)
        second = infer_sequence(self.predictor, -self.x)
        self.assertTrue(torch.equal(first, -second))
        self.assertEqual(window_plan(1)[0].stop,1)

    def test_invalid_inputs_rejected(self):
        for args in [(0,), (2,'unknown'), (2,'rolling_last',0), (2,'chunk',125,'repeat_first')]:
            with self.assertRaises(ValueError):
                window_plan(*args)
        for x in [torch.zeros(0,60), torch.zeros(3,59), torch.full((1,60),float('nan'))]:
            with self.assertRaises(ValueError):
                infer_sequence(self.predictor,x)
        with self.assertRaises(ValueError):
            infer_sequence(self.predictor,self.x,batch_size=0)

    @unittest.skipUnless(os.getenv('IMUPOSER_TEST_CHECKPOINT'), 'optional trusted local checkpoint')
    def test_actual_checkpoint_live_batch_prefix_and_padding_invariance(self):
        from runtime import PosePredictor
        predictor = PosePredictor(os.environ['IMUPOSER_TEST_CHECKPOINT'])
        generator = torch.Generator().manual_seed(401)
        x = torch.randn(132,60,generator=generator) * .1
        batch = infer_sequence(predictor,x,batch_size=16)
        stream = RollingPoseStream(predictor)
        online = torch.stack([stream.push(frame) for frame in x])
        # Packed CPU LSTM GEMM shapes differ between singleton and batched
        # evaluation. Observed max FP32 difference on this deliberately random
        # input: 6.34e-5. This tolerance concerns batching, not causal support.
        live_difference = (batch-online).abs().max().item()
        print(f'checkpoint live/batch max absolute 6D difference: {live_difference:.9g}')
        self.assertTrue(torch.allclose(batch,online,atol=1e-4,rtol=0))
        changed = x.clone()
        changed[80:] += 10
        perturbed = infer_sequence(predictor,changed,batch_size=16)
        # Same batch partition removes numerical-shape differences. A future
        # perturbation must leave earlier outputs bitwise identical.
        self.assertTrue(torch.equal(batch[:80],perturbed[:80]))
        chunk = infer_sequence(predictor,x,'chunk',batch_size=2)
        independent = torch.cat([predictor.predict(x[:125]),predictor.predict(x[125:])])
        self.assertTrue(torch.allclose(chunk,independent,atol=2e-5,rtol=0))


if __name__ == '__main__':
    unittest.main()
