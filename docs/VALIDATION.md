# CPU handoff validation — 2026-10-08

The proposed feature tree was tested in a fresh temporary checkout outside both working repositories, using a new Python 3.10 virtual environment. Because the feature has not been pushed, the fresh checkout used a local Git clone plus the proposed source files. The README installation, inference, test, evaluation, and visualization commands below were executed verbatim. Checkpoints and licensed DIP/SMPL assets were supplied locally; they are not part of the proposed commit.

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-inference.txt
python -m pip install -e src/
python examples/example_inference.py --checkpoint checkpoints/imuposer_global_best.ckpt
python tools/infer.py --checkpoint checkpoints/imuposer_global_best.ckpt --input outputs/synthetic_imu.npz --output outputs/custom_prediction.npz --device cpu
IMUPOSER_CHECKPOINT=checkpoints/imuposer_global_best.ckpt python tests/test_post_training.py
python -m pip install -r requirements-visualization.txt
python tools/eval_dip.py --checkpoint checkpoints/imuposer_global_best.ckpt --data data/processed_imuposer_25fps/dip_test.pt --sequences 0 1 2 --max-frames 125 --smpl-model data/external/smpl/male.pkl --example-dir outputs/local_dip --output outputs/local_dip/dip_test_metrics.json
python tools/visualize_pose.py --input outputs/local_dip/dip_prediction.npz --smpl-model data/external/smpl/male.pkl --output outputs/local_dip/predicted_vs_groundtruth.gif --frames 125
```

The fresh environment used torch 2.7.0+cpu, pytorch-lightning 1.9.5, numpy 1.23.5, and the original imuposer package installed from this checkout. Example and CLI predictions were finite float32 [125,144] and bitwise identical. Synthetic inputs test the interface only.

After installing optional dependencies and configuring licensed assets, all 13 tests passed with zero failures/errors/skips. Without optional assets, nine tests passed and four skipped explicitly. The tests cover checkpoint metadata/architecture, input validation, finite/shape/repeatability checks, sequence lengths/padding, NPZ/PT IO, rotation conversion, masked MSE, DIP preprocessing/missing sensors, short DIP evaluation, SMPL FK, and GIF frame decoding. Before optional dependencies were installed, two asset-dependent checks correctly skipped even when local asset paths existed. The first fresh run exposed that skip-condition issue; it was corrected and the complete workflow passed.

The 375-frame DIP subset reproduced pose MSE 0.016047360602627844, local joint-angle error 13.508703231811523 degrees, and root-relative MPJPE 3.258464423318704 cm. These are three 125-frame prefixes, not a complete dataset benchmark. A 125-frame, 25 FPS predicted-versus-ground-truth GIF was generated and every frame decoded successfully. Data-derived outputs remain local pending redistribution review.

Linux process peak RSS: CPU inference about 462 MiB, three-prefix DIP evaluation about 513 MiB, streamed skeleton rendering about 417 MiB, and full fresh-environment tests about 540 MiB. GNU time measured 552504 KiB maximum RSS for the successful workflow (about 540 MiB), zero swaps. These are process peaks, not a measurement of total WSL memory. The existing CUDA-enabled Conda environment used about 883 MiB for the same CPU test suite. No GPU or training was used. The legacy DIP container requires deserializing the approximately 34 MB test file once, but never expands the AMASS/global training dataset.

The original Epoch 55 and exported full Lightning files have matching SHA-256. All 20 network tensors and 125-frame synthetic predictions match the portable inference-only file exactly; both formats load through the migrated runtime. See MODEL_CARD.md for hashes. The original LICENSE, source architecture, training implementation, and research README body are unchanged in the team checkout. Only source and text documentation are proposed for Git; no model/dataset/GIF/credential binaries are added.

Lightning 1.9.5 still emits a nonfatal pkg_resources deprecation warning; setuptools is pinned for compatibility. The approved model download URL remains pending. Obtain trusted weights directly from the maintainer and licensed DIP/SMPL separately. This branch has not been pushed.
