"""Explicit, stateless chunk and last-frame rolling inference on ONE recording.

All frame intervals in traces are zero-based and half-open. A rolling output at
t is recomputed from x[max(0,t-W+1):t+1]; no recurrent state crosses windows.
Batching independent windows is an offline throughput optimization only.
"""
from dataclasses import dataclass
from collections import deque

import torch


@dataclass(frozen=True)
class Window:
    start: int
    stop: int
    left_repeats: int = 0


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def window_plan(frames, mode='rolling_last', window=125, startup='prefix'):
    positive_integer(frames, 'frames')
    positive_integer(window, 'window')
    if mode not in ['chunk', 'rolling_last'] or startup not in ['prefix', 'repeat_first']:
        raise ValueError('Unknown mode or startup rule')
    if mode == 'chunk':
        if startup != 'prefix':
            raise ValueError('chunk uses actual length, without startup repetition')
        return [Window(s, min(s + window, frames)) for s in range(0, frames, window)]
    return [Window(max(0, t - window + 1), t + 1,
                   max(0, window - t - 1) if startup == 'repeat_first' else 0)
            for t in range(frames)]


def dependency_trace(frames, mode='rolling_last', window=125, startup='prefix', smoothing='centered'):
    """Conservative support of all channels at the pre-smoothing 25 Hz stage.

    This is NOT a claim about raw 60 Hz causality: interpolation and inherited
    bidirectional NaN filling precede this stage. Acceleration alone is filtered.
    """
    if smoothing not in ['centered', 'trailing', 'none']:
        raise ValueError('Unknown smoothing convention')
    before, after = {'centered': (2, 2), 'trailing': (4, 0), 'none': (0, 0)}[smoothing]
    rows = []
    for identity, spec in enumerate(window_plan(frames, mode, window, startup)):
        outputs = range(spec.start, spec.stop) if mode == 'chunk' else [spec.stop - 1]
        for t in outputs:
            acc_start, acc_stop = max(0, spec.start - before), min(frames, spec.stop + after)
            rows.append({'output_frame': t, 'window_id': identity,
                         'input_start': spec.start, 'input_stop': spec.stop,
                         'model_length': spec.stop - spec.start + spec.left_repeats,
                         'left_repeats_of_frame_zero': spec.left_repeats,
                         'model_output_index': t - spec.start + spec.left_repeats,
                         'acc_source_start': acc_start, 'acc_source_stop': acc_stop,
                         'ori_source_start': spec.start, 'ori_source_stop': spec.stop,
                         'available_at_25hz_frame': max(spec.stop, acc_stop) - 1,
                         'future_frames_at_25hz': max(spec.stop, acc_stop) - 1 - t})
    return rows


def smooth_acceleration(acc, mode='centered'):
    """Five-frame arithmetic mean, shortened edges; orientation is untouched."""
    x = torch.as_tensor(acc, dtype=torch.float32, device='cpu')
    if x.ndim < 2 or len(x) == 0 or not torch.isfinite(x).all():
        raise ValueError('Expected nonempty finite [T,...] acceleration')
    if mode == 'none':
        return x.clone()
    if mode not in ['centered', 'trailing']:
        raise ValueError('Unknown smoothing convention')
    # Same stack/nanmean arithmetic and edge rule as inherited smooth_avg(s=5).
    before, after = (2, 2) if mode == 'centered' else (4, 0)
    prefix = torch.full((before, *x.shape[1:]), torch.nan)
    suffix = torch.full((after, *x.shape[1:]), torch.nan)
    padded = torch.cat([prefix, x, suffix])
    return torch.stack([padded[i:i + len(x)] for i in range(5)]).nanmean(0)


class RollingPoseStream:
    """One push -> one last-frame pose; call reset at recording boundaries.

    Input is already scaled/masked 60D features. Upstream filtering is the
    caller's responsibility. Only the most recent `window` inputs are retained.
    """
    def __init__(self, predictor, window=125, startup='prefix'):
        positive_integer(window, 'window')
        if startup not in ['prefix', 'repeat_first']:
            raise ValueError('Unknown startup rule')
        self.predictor, self.window, self.startup = predictor, window, startup
        self.reset()

    def reset(self):
        self.buffer = deque(maxlen=self.window)
        self.frames_seen = 0

    def push(self, frame):
        x = torch.as_tensor(frame, dtype=torch.float32, device='cpu')
        if x.shape != (60,) or not torch.isfinite(x).all():
            raise ValueError('Expected one finite 60D frame')
        self.buffer.append(x.clone())
        self.frames_seen += 1
        features = torch.stack(list(self.buffer))
        if self.startup == 'repeat_first' and len(features) < self.window:
            features = torch.cat([features[:1].expand(self.window-len(features), -1), features])
        return self.predictor.predict(features)[-1]


@torch.inference_mode()
def infer_sequence(predictor, imu, mode='rolling_last', window=125,
                   startup='prefix', batch_size=16, progress=None):
    positive_integer(batch_size, 'batch_size')
    x = torch.as_tensor(imu, dtype=torch.float32, device='cpu')
    if x.ndim != 2 or x.shape[1] != 60 or len(x) == 0 or not torch.isfinite(x).all():
        raise ValueError('Expected nonempty finite [T,60] IMU')
    plan = window_plan(len(x), mode, window, startup)
    result = torch.empty(len(x), 144)
    for offset in range(0, len(plan), batch_size):
        specs = plan[offset:offset + batch_size]
        lengths = [s.stop - s.start + s.left_repeats for s in specs]
        batch = torch.zeros(len(specs), max(lengths), 60)
        for i, spec in enumerate(specs):
            if spec.left_repeats:
                batch[i, :spec.left_repeats] = x[0]
            batch[i, spec.left_repeats:lengths[i]] = x[spec.start:spec.stop]
        prediction = predictor.predict(batch, lengths)
        for i, spec in enumerate(specs):
            if mode == 'chunk':
                result[spec.start:spec.stop] = prediction[i, :lengths[i]]
            else:
                result[spec.stop - 1] = prediction[i, lengths[i] - 1]
        if progress is not None:
            progress(min(offset + len(specs), len(plan)), len(plan))
    return result
