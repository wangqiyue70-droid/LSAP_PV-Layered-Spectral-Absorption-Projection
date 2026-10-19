"""Feature gradients, rollout semantics and actual frozen-teacher integration."""
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

from conditional_flow.matching import alignment_loss, euler_sample, training_prediction
from conditional_flow.model import ConditionalFlow, FlowPrediction, ModelConfig
from conditional_flow.runtime import load_checkpoint
from conditional_flow.teacher import DEFAULT_SOURCE, DEFAULT_WEIGHTS, DinoTeacher
from train_flow import create_parser, train

ROOT = Path(__file__).resolve().parents[1]
TEST_TEMP = ROOT / "runs/test_temp"
TEST_TEMP.mkdir(parents=True, exist_ok=True)
TEACHER_READY = DEFAULT_WEIGHTS.is_file() and (DEFAULT_SOURCE/"hubconf.py").is_file()


class FeatureMechanismTests(unittest.TestCase):
    def test_cosine_alignment_is_per_patch_and_target_is_detached(self):
        prediction = torch.tensor([[[[1.0]], [[0.0]]]], requires_grad=True)
        target = torch.tensor([[[[0.0]], [[1.0]]]], requires_grad=True)
        loss = alignment_loss(prediction, target)
        self.assertAlmostEqual(loss.item(), 1.0)
        loss.backward()
        self.assertGreater(prediction.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)
        self.assertAlmostEqual(alignment_loss(target, target).item(), 0)
        self.assertAlmostEqual(alignment_loss(target, -target).item(), 2)

    def test_predicted_feedback_changes_velocity_and_does_not_backpropagate(self):
        torch.set_num_threads(4)
        torch.manual_seed(10)
        model = ConditionalFlow(ModelConfig(image_size=32, model_channels=32, dropout=0, dino_mode="feedback"))
        # The production velocity head starts at zero. A nonzero head lets this
        # test isolate whether feedback really reaches the velocity calculation.
        torch.nn.init.normal_(model.backbone.out[-1].weight, std=0.01)
        x = torch.randn(1, 1, 32, 32)
        condition, t, style = torch.randn_like(x), torch.tensor([0.5]), torch.tensor([2])
        feedback = torch.randn(1, 384, 16, 16, requires_grad=True)
        output = model.predict(x, t, condition, style, feedback)
        self.assertEqual(output.velocity.shape, x.shape)
        self.assertEqual(output.representation.shape, (1, 384, 16, 16))
        loss = output.velocity.square().mean() + alignment_loss(
            output.representation, torch.randn_like(output.representation))
        loss.backward()
        self.assertIsNone(feedback.grad)
        self.assertGreater(model.backbone.feedback_projection.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone.representation_head[-1].weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            empty = model.predict(x, t, condition, style).velocity
            zeros = model.predict(x, t, condition, style, torch.zeros_like(feedback)).velocity
            nonzero = model.predict(x, t, condition, style, feedback).velocity
        torch.testing.assert_close(empty, zeros)
        self.assertGreater((empty-nonzero).abs().max().item(), 0)

    def test_lookahead_has_no_grad_and_zero_branch_is_trained(self):
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))
                self.config = SimpleNamespace(dino_mode="feedback")
                self.calls = []

            def predict(self, x, t, condition, style, self_condition=None):
                self.calls.append((torch.is_grad_enabled(), self_condition))
                return FlowPrediction(x+self.weight, condition*self.weight)

        model = Probe()
        x, t, style = torch.ones(1, 1, 2, 2), torch.tensor([0.5]), torch.tensor([0])
        output, used = training_prediction(model, x, t, x, style, 1.0)
        self.assertEqual(used, 1)
        self.assertEqual(len(model.calls), 2)
        self.assertFalse(model.calls[0][0])
        self.assertTrue(model.calls[1][0])
        self.assertFalse(model.calls[1][1].requires_grad)
        output.velocity.sum().backward()
        self.assertIsNotNone(model.weight.grad)
        model.calls.clear()
        _, used = training_prediction(model, x, t, x, style, 0.0)
        self.assertEqual(used, 0)
        self.assertEqual(model.calls, [(True, None)])
        model.calls.clear()
        model.config.dino_mode = "align"
        training_prediction(model, x, t, x, style, 1.0)
        self.assertEqual(model.calls, [(True, None)])

    def test_sampler_rolls_features_forward_with_one_network_call_per_step(self):
        class Probe:
            config = SimpleNamespace(dino_mode="feedback")

            def __init__(self):
                self.calls = 0

            def predict(self, x, t, condition, style, self_condition=None):
                expected = self.calls
                if expected == 0:
                    assert self_condition is None
                    velocity = torch.zeros_like(x)
                else:
                    assert torch.equal(self_condition, torch.full_like(x, float(expected)))
                    assert not self_condition.requires_grad
                    velocity = self_condition
                self.calls += 1
                return FlowPrediction(velocity, torch.full_like(x, float(self.calls)))

        model = Probe()
        x = torch.zeros(1, 1, 2, 2)
        result = euler_sample(model, x, x, torch.tensor([0]), steps=3)
        self.assertEqual(model.calls, 3)
        torch.testing.assert_close(result, torch.ones_like(x))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the FP16 cache regression")
    def test_amp_lookahead_preserves_backbone_gradients_with_and_without_checkpointing(self):
        torch.set_num_threads(4)
        for activation_checkpoint in (False, True):
            with self.subTest(activation_checkpoint=activation_checkpoint):
                model = ConditionalFlow(ModelConfig(image_size=32, model_channels=32, dropout=0.1,
                                        dino_mode="feedback", activation_checkpoint=activation_checkpoint)).cuda()
                x = torch.randn(1, 1, 32, 32, device="cuda")
                t, style = torch.tensor([0.5], device="cuda"), torch.tensor([0], device="cuda")
                with torch.autocast("cuda", dtype=torch.float16):
                    result, _ = training_prediction(model, x, t, x, style, 1.0)
                    loss = result.velocity.float().square().mean() + alignment_loss(
                        result.representation, torch.randn_like(result.representation))
                loss.backward()
                block = model.backbone.input_blocks[0]
                first_conv = block.block[0] if activation_checkpoint else block[0]
                self.assertIsNotNone(first_conv.weight.grad)
                self.assertTrue(torch.isfinite(first_conv.weight.grad).all().item())
                self.assertGreater(first_conv.weight.grad.abs().sum().item(), 0)


@unittest.skipUnless(TEACHER_READY, "Run setup_dino.ps1 for official-teacher integration tests")
class OfficialTeacherTests(unittest.TestCase):
    def test_teacher_is_frozen_and_construction_preserves_rng(self):
        torch.set_num_threads(4)
        torch.manual_seed(7)
        before = torch.get_rng_state().clone()
        teacher = DinoTeacher(device="cpu")
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        teacher.train()
        self.assertFalse(teacher.training)
        self.assertFalse(teacher.encoder.training)
        clean = torch.linspace(-1, 1, 32*32).reshape(1, 1, 32, 32).requires_grad_()
        features = teacher(clean)
        self.assertEqual(features.shape, (1, 384, 16, 16))
        self.assertFalse(features.requires_grad)
        self.assertTrue(torch.isfinite(features).all().item())
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
        self.assertIsNone(clean.grad)

    def test_official_teacher_training_resume_and_teacher_free_sampling(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            root = Path(directory)
            parser = create_parser()
            common = ["--device", "cpu", "--image-size", "32", "--model-channels", "32",
                      "--dino-mode", "feedback", "--self-cond-prob", "0.5", "--seed", "77"]
            full = train(parser.parse_args(common + ["--smoke-test", "--steps", "3", "--out-dir", str(root/"full")]))
            train(parser.parse_args(common + ["--smoke-test", "--steps", "2", "--out-dir", str(root/"split")]))
            train(parser.parse_args(["--device", "cpu", "--resume", str(root/"split/flow_step_0000002.pt"),
                                     "--steps", "3", "--out-dir", str(root/"split")]))
            expected = load_checkpoint(root/"full/flow_step_0000003.pt")
            resumed = load_checkpoint(root/"split/flow_step_0000003.pt")
            self.assertTrue(full["teacher_frozen"])
            for name, weight in expected["model"].items():
                self.assertTrue(torch.equal(weight, resumed["model"][name]), name)
            self.assertGreater(expected["model"]["backbone.representation_head.2.weight"].abs().sum().item(), 0)
            # The teacher's files can be absent at inference. These deliberately
            # invalid provenance paths must not be dereferenced by the sampler.
            resumed["teacher_info"]["source"] = str(root/"teacher_source_is_absent")
            resumed["teacher_info"]["weights"] = str(root/"teacher_weights_are_absent.pth")
            portable = root/"portable.pt"
            torch.save(resumed, portable)
            from PIL import Image
            condition = root/"condition.png"
            Image.new("L", (32, 32), 160).save(condition)
            result = subprocess.run([
                sys.executable, str(ROOT/"sample_flow.py"), "--model-path", str(portable),
                "--cond-path", str(condition), "--style-id", "0", "--device", "cpu",
                "--steps", "20", "--output", str(root/"sample.png")],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue((root/"sample.json").is_file())
            # Restoring training still requires the same teacher signature.
            resumed["teacher_info"]["source"] = str(DEFAULT_SOURCE)
            resumed["teacher_info"]["weights"] = str(DEFAULT_WEIGHTS)
            resumed["teacher_info"]["signature"] = "not-the-saved-teacher"
            changed = root/"changed.pt"
            torch.save(resumed, changed)
            with self.assertRaisesRegex(ValueError, "teacher source, weights or preprocessing changed"):
                train(parser.parse_args(["--device", "cpu", "--resume", str(changed), "--steps", "4",
                                         "--out-dir", str(root/"rejected")]))


if __name__ == "__main__":
    unittest.main()
