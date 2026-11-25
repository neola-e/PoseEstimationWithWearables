import torch
from torch.utils.data import Dataset
from imuposer import math
from imuposer.config import Config, amass_combos

class GlobalModelDataset(Dataset):
    def __init__(self, split="train", config:Config=None):
        super().__init__()

        # load the data
        self.train = split
        self.config = config
        self.data = self.load_data()
        
    def load_data(self):
        if self.train == "train":
            data_files = [x.name for x in self.config.processed_imu_poser_25fps.iterdir() if "dip" not in x.name]
        else:
            data_files = ["dip_test.pt"]

        imu = []
        pose = []
        imu_masks = []
        combo_masks = {}
        for combo, sensors in amass_combos.items():
            mask = torch.zeros(60, dtype=torch.bool)
            mask[:15].view(5, 3)[sensors] = True
            mask[15:].view(5, 9)[sensors] = True
            combo_masks[combo] = mask

        for fname in data_files:
            fdata = torch.load(self.config.processed_imu_poser_25fps / fname)

            for i in range(len(fdata["acc"])):
                # inputs
                facc = fdata["acc"][i] 
                fori = fdata["ori"][i]

                # load all the data
                glb_acc = facc.view(-1, 6, 3)[:, [0, 1, 2, 3, 4]] / self.config.acc_scale
                glb_ori = fori.view(-1, 6, 3, 3)[:, [0, 1, 2, 3, 4]]

                acc = glb_acc
                ori = glb_ori

                # outputs
                fpose = fdata["pose"][i]
                fpose = fpose.reshape(fpose.shape[0], -1)

                # clip the data
                # 25 is the data sampling rate

                window_length = self.config.max_sample_len * 25 // 60
                # Share input storage across combinations; apply masks when fetching samples.
                imu_windows = torch.split(torch.cat([acc.flatten(1), ori.float().flatten(1)], dim=1), window_length)
                pose_windows = torch.split(fpose, window_length)

                for _combo in list(amass_combos):
                    # acc N, 5, 3
                    # ori N, 5, 3, 3

                    imu.extend(imu_windows)
                    pose.extend(pose_windows)
                    imu_masks.extend([combo_masks[_combo]] * len(imu_windows))

        self.imu = imu
        self.pose = pose
        self.imu_masks = imu_masks

    def __getitem__(self, idx):
        _imu = self.imu[idx].float().clone()
        _imu[:, ~self.imu_masks[idx]] = 0
        _pose = self.pose[idx].float()

        _input = _imu
        if self.config.r6d == True:
            _output = math.rotation_matrix_to_r6d(_pose).reshape(-1, 24, 6)[:, self.config.pred_joints_set].reshape(-1, 6 * len(self.config.pred_joints_set))
        else:
            _output = _pose

        return _input, _output

    def __len__(self):
        return len(self.imu)

