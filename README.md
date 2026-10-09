# Pose Estimation With Wearables — team research code

Our team builds on [FIGLAB/IMUPoser](https://github.com/FIGLAB/IMUPoser). The original attribution, LICENSE, and research instructions are retained below. The following section documents our tested post-training additions.

## Pretrained Model — Quick Start

After this branch is merged, clone the team repository. Use Ubuntu/Linux/WSL and Python 3.10:

```bash
git clone https://github.com/neola-e/PoseEstimationWithWearables.git
cd PoseEstimationWithWearables
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m pip install -e src/
```

This installs CPU PyTorch 2.7.0, NumPy 1.23.5, Lightning 1.9.5, and the original package. No AMASS, training logs, WandB account, GPU, SMPL model, or retraining is needed for inference. The original research requirements below describe the earlier training environment; use requirements-inference.txt for this quick start.

### Obtain the checkpoint separately

**Approved download URL: pending licensing review.** Request the Epoch 55 checkpoint directly from the project maintainer through an approved team channel. No public download or GitHub Release has been created. Place it at `checkpoints/imuposer_global_best.ckpt`, or pass `--checkpoint`/set `IMUPOSER_CHECKPOINT` to its actual location. The loader supports both the original 123 MB Lightning checkpoint and the smaller portable inference-only file; they have identical network tensors and different container hashes. See docs/MODEL_CARD.md for both SHA-256 hashes and provenance. Only load trusted checkpoints: the full Lightning format contains a pickled Config object.

### Run CPU inference

```bash
python examples/example_inference.py --checkpoint checkpoints/imuposer_global_best.ckpt
python tools/infer.py --checkpoint checkpoints/imuposer_global_best.ckpt --input outputs/synthetic_imu.npz --output outputs/custom_prediction.npz --device cpu
IMUPOSER_CHECKPOINT=checkpoints/imuposer_global_best.ckpt python tests/test_post_training.py
```

The example produces **125 finite `[125,144]` pose predictions**, writes outputs/example_prediction.npz and outputs/synthetic_imu.npz, and reports CPU peak RAM. Synthetic identity orientations and arbitrary small accelerations are an **interface smoke test only**, not an accuracy demonstration. Replace the input file with your own correctly preprocessed sequence. Generated outputs and checkpoints are ignored by Git. Optional DIP/SMPL tests skip explicitly when authorized external assets are not configured.

### Custom inputs and outputs

Input NPZ: finite float32 `imu` shaped `[T,60]` or `[B,T,60]`; optional integer `lengths` shaped `[1]` or `[B]`; optional scalar `fps=25`. Tensor-only PT dictionaries use the same keys. All five world-frame acceleration XYZ triples occupy columns 0:15 and are divided by 30; all five row-major global 3×3 orientations occupy columns 15:60. Sensor order is **left wrist, right wrist, left thigh/pocket, right thigh/pocket, head**. Missing sensors have both their acceleration and orientation values zeroed.

Output `pose` is `[T,144]` or `[B,T,144]`, containing 24 local SMPL rotations in 6D (first matrix column XYZ then second column XYZ), plus lengths and fps. Trim padded frames before rotation conversion or FK; the wrapper returns zero padding. No shape or root translation is predicted. See docs/IMU_INPUT_FORMAT.md for exact preprocessing and coordinate conventions.

### Optional DIP evaluation and skeleton visualization

Acquire DIP and the male SMPL model independently under their licenses. DIP evaluation uses only the authorized processed test file; angular/MSE metrics do not need SMPL. Position metrics and skeleton rendering additionally need a licensed SMPL asset and the optional dependencies:

```bash
python -m pip install -r requirements-visualization.txt
python tools/eval_dip.py --checkpoint checkpoints/imuposer_global_best.ckpt --data data/processed_imuposer_25fps/dip_test.pt --sequences 0 1 2 --max-frames 125 --smpl-model data/external/smpl/male.pkl --example-dir outputs/local_dip --output outputs/local_dip/dip_test_metrics.json
python tools/visualize_pose.py --input outputs/local_dip/dip_prediction.npz --smpl-model data/external/smpl/male.pkl --output outputs/local_dip/predicted_vs_groundtruth.gif --frames 125
```

The renderer uses original SMPL FK and hierarchy, mean male shape, zero translation, fixed meter axes, and streamed GIF frames. Labels distinguish prediction and ground truth. These commands require the separately obtained assets at the illustrated paths; no dataset or new model asset is included in this change.

Previously evaluated **375 valid frames** yielded pose MSE **0.01605**, mean local joint-angle error **13.51°**, and root-relative MPJPE **3.26 cm**. These are three five-second prefixes, **not full-dataset benchmark results**. Pose MSE differs from the training validation metric **0.01031**, which includes joint-position loss. Full per-window values and conventions are in docs/dip_subset_metrics.json.

### Real sensors and latency

The unchanged LSTM is bidirectional and uses future frames in each supplied window; training used 125-frame windows. This is not causal zero-lookahead streaming. Centered acceleration smoothing itself adds 80 ms future context. Log synchronized physical sensors offline first, then verify units, gravity compensation, a common global frame, sensor-to-segment mounting calibration, and the exact 25 FPS feature layout. Device drift, pocket motion, and unfamiliar actions can cause domain shift. Validate an explicit lookahead/latency policy before building a real-time adapter.

## Original IMUPoser research documentation

# IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds
Click to watch the video!
<p align="center">
  <a href="https://www.youtube.com/watch?v=hgpjbKv8XFY"><img src="media/IMUPoser_github.png" alt="animated" width="100%"/></a>
</p>

Research code for IMUPoser (CHI 2023)

## Reference
Vimal Mollyn, Riku Arakawa, Mayank Goel, Chris Harrison, and Karan Ahuja. 2023. IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems (CHI '23). Association for Computing Machinery, New York, NY, USA, Article 529, 1–12.

[Paper](https://dl.acm.org/doi/10.1145/3544548.3581392) | [Website](https://www.figlab.com/research/2023/imuposer)


BibTeX Reference:

```
@inproceedings{10.1145/3544548.3581392,
author = {Mollyn, Vimal and Arakawa, Riku and Goel, Mayank and Harrison, Chris and Ahuja, Karan},
title = {IMUPoser: Full-Body Pose Estimation Using IMUs in Phones, Watches, and Earbuds},
year = {2023},
isbn = {9781450394215},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3544548.3581392},
doi = {10.1145/3544548.3581392},
booktitle = {Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems},
articleno = {529},
numpages = {12},
keywords = {sensors, inertial measurement units, mobile devices, Motion capture},
location = {Hamburg, Germany},
series = {CHI '23}
}
```

## 1. Clone (or Fork!) this repository
```
git clone https://github.com/FIGLAB/IMUPoser.git
```
 
## 2. Create a virtual environment
We recommend using conda. Tested on `Ubuntu 18.04`, with `python 3.7`.

```bash
conda create -n "imuposer" python=3.7
conda activate imuposer
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

python -m pip install -r requirements.txt
python -m pip install -e src/
```

## 3. Download training data
1. Download training data from [AMASS](https://amass.is.tue.mpg.de/index.html) and [DIP-IMU](https://dip.is.tuebingen.mpg.de/) and place in the data folder as shown below after unzipping. We use the SMPLH+G version of the AMASS dataset. Note that AMASS has been updated since our paper was published, so use the latest version for best results.

```bash
data
└── raw
    ├── AMASS
    │   ├── ACCAD
    │   ├── BioMotionLab_NTroje
    │   ├── BMLhandball
    │   ├── BMLmovi
    │   ├── CMU
    │   ├── DanceDB
    │   ├── DFaust_67
    │   ├── EKUT
    │   ├── Eyes_Japan_Dataset
    │   ├── HUMAN4D
    │   ├── HumanEva
    │   ├── KIT
    │   ├── MPI_HDM05
    │   ├── MPI_Limits
    │   ├── MPI_mosh
    │   ├── SFU
    │   ├── SSM_synced
    │   ├── TCD_handMocap
    │   ├── TotalCapture
    │   └── Transitions_mocap
    ├── DIP_IMU
    │   ├── s_01
    │   ├── s_02
    │   ├── s_03
    │   ├── s_04
    │   ├── s_05
    │   ├── s_06
    │   ├── s_07
    │   ├── s_08
    │   ├── s_09
    │   └── s_10
    └── README.md
```

## 4. Training Steps
1. Start by preprocessing the AMASS and DIP-IMU datasets [scripts/1. Preprocessing](scripts/1.%20Preprocessing). Run all files in order.
2. Train the model [scripts/2. Train/run\_combos.sh](scripts/2.%20Train/run_combos.sh)

## 5. IMUPoser Dataset
The IMUPoser Dataset is available for download [here](https://www.dropbox.com/s/myq9k0x9i5waphn/imuposer_dataset.zip?dl=0). Check out the [Dataset Walkthrough notebook](scripts/3.%20Dataset%20Walkthrough.ipynb) for a quick overview of the data. Big thanks to Justin Macey from CMU and Giorgio Becherini from the Perceiving Systems group at the Max Planck
Institute for Intelligent Systems for help with processing our motion capture data.

## Follow up Research:
- [WheelPoser: Sparse-IMU Based Body Pose Estimation for Wheelchair Users](https://github.com/axle-lab/WheelPoser) (ASSETS 2024)
- [MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices](https://spice-lab.org/projects/MobilePoser/) (UIST 2024)
- [SmartPoser: Arm Pose Estimation with a Smartphone and Smartwatch Using UWB and IMU Data](https://www.figlab.com/research/2023/smartposer) (UIST 2023)

## Disclaimer
```
THE PROGRAM IS DISTRIBUTED IN THE HOPE THAT IT WILL BE USEFUL, BUT WITHOUT ANY WARRANTY. IT IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW THE AUTHOR WILL BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF THE AUTHOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
```

## Acknowledgments
Some of the modules in this repo were inspired by the amazing [TransPose](https://github.com/Xinyu-Yi/TransPose/) github repo. If you liked IMUPoser, you should definitely check out TransPose!

## License
The IMUPoser code can only be used for research i.e., non-commercial purposes. For a commercial license, please contact Vimal Mollyn, Karan Ahuja and Chris Harrison.

## Contact
Feel free to contact [Vimal Mollyn](mailto:ms123vimal@gmail.com) for any help, questions or general feedback!
