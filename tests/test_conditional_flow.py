"""Numerical, paired-data and checkpoint-resume regression checks."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from conditional_flow.data import PairedVeinDataset, infer_style
from conditional_flow.matching import euler_sample, flow_path
from conditional_flow.model import ConditionalFlow, ModelConfig
from conditional_flow.runtime import load_checkpoint
from train_flow import create_parser, train

TEST_TEMP = Path(__file__).resolve().parents[1] / "runs" / "test_temp"
TEST_TEMP.mkdir(parents=True, exist_ok=True)


class FlowMathTests(unittest.TestCase):
    def test_path_endpoints_and_velocity(self):
        clean = torch.tensor([[[[0.5]]], [[[0.25]]]])
        noise = torch.tensor([[[[-1.0]]], [[[0.75]]]])
        state, velocity = flow_path(clean, noise, torch.tensor([0.0, 1.0]))
        self.assertEqual(state[0].item(), noise[0].item())
        self.assertEqual(state[1].item(), clean[1].item())
        torch.testing.assert_close(velocity, clean-noise)

    def test_euler_integrates_complete_time_range_without_clipping(self):
        times = []

        def oracle(x, t, condition, style):
            times.append(t[0].item())
            return condition

        noise = torch.full((1, 1, 4, 4), 2.0)
        condition = torch.full_like(noise, 3.0)
        output = euler_sample(oracle, noise, condition, torch.tensor([0]), 20)
        torch.testing.assert_close(output, noise+condition)
        self.assertEqual(len(times), 20)
        self.assertEqual(times[0], 0)
        self.assertAlmostEqual(times[-1], 0.95, places=6)


class PairedDataTests(unittest.TestCase):
    def make_pair(self, root):
        image = root / "vein/casia_subject/one.png"
        condition = root / "cond/casia_subject/one.bmp"
        image.parent.mkdir(parents=True)
        condition.parent.mkdir(parents=True)
        pixels = np.tile(np.arange(32, dtype=np.uint8)*8, (32, 1))
        Image.fromarray(pixels).save(image)
        Image.fromarray(pixels).save(condition)
        return image, condition

    def test_pairing_by_relative_stem_and_normalization(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            self.make_pair(root)
            data = PairedVeinDataset(root, 32, augment=False)
            clean, condition, style = data[0]
            torch.testing.assert_close(clean, condition)
            self.assertEqual(clean.shape, (1, 32, 32))
            self.assertEqual(style, 0)
            self.assertEqual(clean.min().item(), -1)
            self.assertLessEqual(clean.max().item(), 1)

    def test_unpaired_and_ambiguous_files_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            image, condition = self.make_pair(root)
            Image.new("L", (32, 32)).save(image.with_name("missing_condition.png"))
            with self.assertRaisesRegex(ValueError, "Unpaired"):
                PairedVeinDataset(root, 32)
            Image.new("L", (32, 32)).save(image.with_suffix(".bmp"))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                PairedVeinDataset(root, 32)

    def test_style_names_and_projected_conditions(self):
        self.assertEqual(infer_style(Path("HFUT_subject/image.png")), 1)
        self.assertEqual(infer_style(Path("c1_0_3/image.png")), 3)
        with self.assertRaises(ValueError):
            infer_style(Path("unknown/image.png"))


class ResumeTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_updates_and_condition_inputs_work(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            root = Path(folder)
            parser = create_parser()
            common = ["--device", "cpu", "--image-size", "32", "--model-channels", "32",
                      "--seed", "123", "--grad-accum", "2"]
            train(parser.parse_args(common + ["--smoke-test", "--steps", "3", "--out-dir", str(root/"full")]))
            train(parser.parse_args(common + ["--smoke-test", "--steps", "2", "--out-dir", str(root/"resumed")]))
            saved = root/"resumed/flow_step_0000002.pt"
            summary = train(parser.parse_args(["--device", "cpu", "--resume", str(saved),
                                              "--steps", "3", "--out-dir", str(root/"resumed")]))
            full = load_checkpoint(root/"full/flow_step_0000003.pt")
            resumed = load_checkpoint(root/"resumed/flow_step_0000003.pt")
            self.assertEqual(summary["successful_updates_this_run"], 1)
            self.assertEqual(resumed["data_info"]["kind"], "synthetic_smoke")
            for name, weight in full["model"].items():
                self.assertTrue(torch.equal(weight, resumed["model"][name]), name)
            self.assertGreater(resumed["model"]["backbone.out.2.weight"].abs().sum().item(), 0)
            model = ConditionalFlow(ModelConfig(**resumed["model_config"]))
            model.load_state_dict(resumed["model"])
            model.eval()
            x, t, style = torch.zeros(1, 1, 32, 32), torch.tensor([0.5]), torch.tensor([0])
            with torch.no_grad():
                base = model(x, t, torch.zeros_like(x), style)
                different_condition = model(x, t, torch.ones_like(x), style)
                different_style = model(x, t, torch.zeros_like(x), torch.tensor([1]))
            self.assertGreater((base-different_condition).abs().max().item(), 0)
            self.assertGreater((base-different_style).abs().max().item(), 0)
            with self.assertRaisesRegex(ValueError, "Cannot change batch_size"):
                train(parser.parse_args(["--device", "cpu", "--resume", str(saved),
                                         "--steps", "4", "--batch-size", "2"]))

    def test_diffusion_state_dictionary_is_not_accepted_as_flow(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder:
            path = Path(folder)/"diffusion.pt"
            torch.save({"unet_weight": torch.ones(1)}, path)
            with self.assertRaisesRegex(ValueError, "Not a conditional Flow checkpoint"):
                load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
