"""Small DIP test subset using GlobalModelDataset's exact __getitem__ logic."""
from pathlib import Path

import torch
from imuposer.config import amass_combos
from imuposer.datasets.globalModelDataset import GlobalModelDataset

from runtime import load_torch


class DipTestSubset(GlobalModelDataset):
    def __init__(self, path, config, sequence_ids=(0,), max_frames=125, combo="global"):
        self.path = Path(path)
        if self.path.name != "dip_test.pt":
            raise ValueError("Expected dip_test.pt; train/validation files are not accepted")
        if not sequence_ids or len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError("Provide distinct sequence IDs")
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        if combo not in amass_combos:
            raise ValueError("Unknown sensor combination")
        self.sequence_ids = list(sequence_ids)
        self.max_frames = max_frames
        self.combo = combo
        super().__init__(split="test", config=config)

    def load_data(self):
        # This legacy .pt stores lists, not individually addressable sequences.
        # Deserialize the 34 MB test file once; retain only selected windows.
        data = load_torch(self.path)
        self.total_sequences = len(data["acc"])
        self.imu, self.pose, self.imu_masks, self.records = [], [], [], []
        mask = torch.zeros(60, dtype=torch.bool)
        sensors = amass_combos[self.combo]
        mask[:15].view(5, 3)[sensors] = True
        mask[15:].view(5, 9)[sensors] = True
        window = self.config.max_sample_len * 25 // 60
        for seq in self.sequence_ids:
            if not 0 <= seq < self.total_sequences:
                raise ValueError(f"Sequence ID {seq} outside [0,{self.total_sequences - 1}]")
            acc, ori, pose = (data[key][seq] for key in ("acc", "ori", "pose"))
            original = len(acc)
            if ori.shape != (original, 6, 3, 3) or acc.shape != (original, 6, 3):
                raise ValueError("Unexpected processed DIP IMU schema")
            if pose.shape != (original, 24, 3, 3):
                raise ValueError("Unexpected processed DIP pose schema")
            count = min(original, self.max_frames)
            # Exact GlobalModelDataset preprocessing. No extra normalization.
            x = torch.cat([(acc[:count, :5] / self.config.acc_scale).flatten(1),
                           ori[:count, :5].float().flatten(1)], dim=1)
            # Upstream masks at load time; the local memory-efficient dataset
            # masks at fetch time. Pre-mask once so both __getitem__ versions work.
            x[:, ~mask] = 0
            for start in range(0, count, window):
                stop = min(start + window, count)
                self.imu.append(x[start:stop])
                self.pose.append(pose[start:stop].reshape(stop - start, -1))
                self.imu_masks.append(mask)
                self.records.append({"sequence_id": seq, "start_frame": start,
                                     "frames": stop - start, "original_frames": original})


def masked_pose_mse(prediction, target, lengths):
    """Per-element 6D MSE; never include padding."""
    lengths = torch.as_tensor(lengths)
    mask = torch.arange(prediction.shape[1])[None] < lengths[:, None]
    return (prediction[mask].double() - target[mask].double()).square().mean().item()
