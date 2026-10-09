"""CPU integration checks; external DIP/SMPL checks skip if assets are absent."""
import json
from importlib.util import find_spec
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np
import torch
from imuposer.math import axis_angle_to_rotation_matrix, rotation_matrix_to_r6d

from runtime import (PosePredictor, load_body_model, load_torch, peak_ram_mib,
                     pose_rotations, read_arrays, write_arrays)
from dip_subset import DipTestSubset, masked_pose_mse
from eval_dip import evaluate

CHECKPOINT = Path(os.environ.get("IMUPOSER_CHECKPOINT", ROOT / "checkpoints/imuposer_global_best.ckpt"))
DIP = Path(os.environ.get("IMUPOSER_DIP_TEST", ROOT / "data/processed_imuposer_25fps/dip_test.pt"))
SMPL = Path(os.environ.get("IMUPOSER_SMPL_MODEL", ROOT / "data/external/smpl/male.pkl"))

SMPL_READY = SMPL.is_file() and all(find_spec(name) is not None for name in ("scipy", "chumpy"))
GIF_READY = SMPL_READY and all(find_spec(name) is not None for name in ("matplotlib", "PIL"))

def synthetic_imu(frames=10):
    imu = torch.zeros(frames, 60)
    imu[:, 15:] = torch.eye(3).repeat(frames, 5, 1, 1).flatten(1)
    return imu

class RotationAndMetricTests(unittest.TestCase):
    def test_mse_excludes_padding(self):
        pred = torch.zeros(2, 4, 144)
        target = torch.ones_like(pred)
        pred[0, 2:] = 1000
        pred[1, 3:] = 1000
        self.assertEqual(masked_pose_mse(pred, target, [2, 3]), 1.0)

    def test_rotation_roundtrip_and_column_convention(self):
        rot = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        r6d = rotation_matrix_to_r6d(rot)
        torch.testing.assert_close(r6d[0], torch.tensor([1., 0., 0., 0., 0., 1.]))
        converted = pose_rotations(r6d.repeat(3, 24))
        torch.testing.assert_close(converted, rot.expand(3, 24, 3, 3))
        torch.testing.assert_close(torch.linalg.det(converted), torch.ones(3, 24))

    def test_degenerate_rotations_rejected(self):
        with self.assertRaises(ValueError):
            pose_rotations(torch.zeros(2, 144))

@unittest.skipUnless(CHECKPOINT.is_file(), "Trusted checkpoint not installed")
class PostTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = PosePredictor(CHECKPOINT, device="cpu", threads=2)

    def test_checkpoint_metadata_and_architecture(self):
        p = self.predictor
        self.assertEqual(p.epoch, 55)
        self.assertTrue(p.training_config.use_joint_loss)
        self.assertFalse(p.model.training)
        self.assertFalse(hasattr(p.model, "bodymodel"))
        self.assertTrue(p.model.dip_model.rnn.bidirectional)
        self.assertEqual(p.model.dip_model.rnn.num_layers, 2)
        self.assertEqual(p.model.dip_model.rnn.hidden_size, 512)
        self.assertEqual(p.model.dip_model.linear1.in_features, 60)
        self.assertEqual(p.model.dip_model.linear2.out_features, 144)

    def test_inference_shapes_finite_and_repeatable(self):
        x = synthetic_imu()
        y = self.predictor.predict(x)
        self.assertEqual(tuple(y.shape), (10, 144))
        self.assertTrue(torch.isfinite(y).all())
        self.assertFalse(y.requires_grad)
        torch.testing.assert_close(y, self.predictor.predict(x), rtol=0, atol=0)
        self.assertEqual(tuple(self.predictor.predict(x[None]).shape), (1, 10, 144))

    def test_invalid_shapes_and_nonfinite(self):
        for shape in [(60,), (2, 59), (0, 60), (0, 2, 60), (2, 3, 4, 60)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.predictor.predict(torch.zeros(shape))
        for value in [float("nan"), float("inf")]:
            x = synthetic_imu()
            x[0, 0] = value
            with self.assertRaises(ValueError):
                self.predictor.predict(x)

    def test_invalid_lengths(self):
        x = synthetic_imu()[None].repeat(2, 1, 1)
        for lengths in [[0, 5], [11, 5], [3], [3.2, 5], [float("nan"), 5], [[3, 5]], [True, False]]:
            with self.subTest(lengths=lengths), self.assertRaises(ValueError):
                self.predictor.predict(x, lengths)

    def test_padding_does_not_change_valid_predictions(self):
        x = synthetic_imu()[None].repeat(2, 1, 1)
        x[0, 7:] = 1000
        x[1, 5:] = -1000
        y = self.predictor.predict(x, [7, 5])
        self.assertEqual(tuple(y.shape), (2, 10, 144))
        for i, length in enumerate([7, 5]):
            torch.testing.assert_close(y[i, :length], self.predictor.predict(x[i, :length]),
                                       rtol=1e-5, atol=1e-6)
            self.assertEqual(torch.count_nonzero(y[i, length:]).item(), 0)

    def test_file_schemas_npz_and_pt(self):
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in [".npz", ".pt"]:
                path = Path(tmp) / ("input" + suffix)
                write_arrays(path, imu=synthetic_imu(), lengths=[10], fps=25)
                data = read_arrays(path)
                pred = self.predictor.predict(data["imu"], data["lengths"])
                path = Path(tmp) / ("prediction" + suffix)
                write_arrays(path, pose=pred, lengths=[10], fps=25)
                torch.testing.assert_close(torch.as_tensor(read_arrays(path)["pose"]), pred)

    @unittest.skipUnless(DIP.is_file(), "Licensed processed DIP test data not installed")
    def test_dip_preprocessing_and_missing_sensor_mask(self):
        raw = load_torch(DIP)
        for combo, sensors in [("global", [0, 1, 2, 3, 4]), ("lw_rw_h", [0, 1, 4])]:
            data = DipTestSubset(DIP, self.predictor.training_config, [0], 8, combo)
            x, target = data[0]
            acc = raw["acc"][0][:8, :5].clone() / 30
            ori = raw["ori"][0][:8, :5].float().clone()
            missing = [s for s in range(5) if s not in sensors]
            acc[:, missing] = 0
            ori[:, missing] = 0
            torch.testing.assert_close(x, torch.cat([acc.flatten(1), ori.flatten(1)], dim=1))
            # Also verifies compatibility with upstream's load-time masking.
            torch.testing.assert_close(data.imu[0], x)
            torch.testing.assert_close(target, rotation_matrix_to_r6d(raw["pose"][0][:8]).reshape(8, 144))
            self.assertEqual(len(data), 1)

    @unittest.skipUnless(DIP.is_file(), "Licensed processed DIP test data not installed")
    def test_short_dip_evaluation(self):
        data = DipTestSubset(DIP, self.predictor.training_config, [0], 8)
        body = load_body_model(SMPL) if SMPL_READY else None
        with torch.inference_mode(), tempfile.TemporaryDirectory() as tmp:
            report = evaluate(self.predictor, data, body, tmp)
            self.assertEqual(report["frames_evaluated"], 8)
            self.assertTrue(all(np.isfinite(v) for v in report["metrics"].values()))
            saved = read_arrays(Path(tmp) / "dip_prediction.npz")
            expected = np.mean((saved["pose"].astype(np.float64) - saved["target_pose"]) ** 2)
            self.assertAlmostEqual(report["metrics"]["pose_representation_mse"], expected, places=12)

    @unittest.skipUnless(SMPL_READY, "Licensed SMPL asset or optional dependencies not installed")
    def test_fk_rest_pose_and_root_rotation(self):
        body = load_body_model(SMPL)
        rot = torch.eye(3).repeat(1, 24, 1, 1)
        with torch.inference_mode():
            _, joints = body.forward_kinematics(rot, calc_mesh=False)
            torch.testing.assert_close(joints[0], body._J - body._J[:1])
            root = axis_angle_to_rotation_matrix(torch.tensor([[0., 0., 0.5]]))[0]
            rot[:, 0] = root
            _, rotated = body.forward_kinematics(rot, calc_mesh=False)
            torch.testing.assert_close(rotated[0], (root @ joints[0].T).T)

    @unittest.skipUnless(GIF_READY, "Licensed SMPL asset or visualization dependencies not installed")
    def test_streamed_gif(self):
        from PIL import Image
        from visualize_pose import render
        body = load_body_model(SMPL)
        pose = self.predictor.predict(synthetic_imu(4))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "test.gif"
            render(pose, body, output, target=pose, frames=4)
            with Image.open(output) as image:
                self.assertEqual(image.n_frames, 4)
                self.assertEqual(image.info["duration"], 40)
                self.assertEqual(image.info["loop"], 0)
                self.assertEqual(image.size, (720, 360))
                for i in range(4):
                    image.seek(i)
                    image.load()

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"tests_run": result.testsRun, "failures": len(result.failures),
              "errors": len(result.errors), "skipped": len(result.skipped),
              "successful": result.wasSuccessful(), "device": "cpu",
              "peak_process_ram_mib": peak_ram_mib()}
    if "IMUPOSER_TEST_REPORT" in os.environ:
        path = Path(os.environ["IMUPOSER_TEST_REPORT"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    sys.exit(0 if result.wasSuccessful() else 1)
