"""Regression tests for mixed INT4/INT8 conversion and evaluation."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import mixed_precision as mp  # noqa: E402


def load_fake_quant_model():
    with open(os.path.join(ROOT, "config", "int8_sensitivity.json"),
              encoding="utf-8") as handle:
        report = json.load(handle)
    ranked = sorted(
        report["per_layer_sensitivity_leave_one_out"].items(),
        key=lambda item: item[1]["delta_acc_vs_full_int8"],
    )
    four_bit_layers = [name for name, _ in ranked[:max(1, int(len(ranked) * 0.4))]]
    model = mp.PaperInceptionCNN()
    mp.wrap_model_with_fq(model, four_bit_layers)
    model.load_state_dict(torch.load(
        os.path.join(ROOT, "models", "mixed_qat_best.pth"),
        map_location="cpu",
        weights_only=True,
    ))
    return model.eval(), four_bit_layers


class MixedPrecisionRegressionTests(unittest.TestCase):
    def test_dataset_rejects_incomplete_records(self):
        with tempfile.TemporaryDirectory() as dataset_dir:
            (Path(dataset_dir) / "100.hea").touch()
            (Path(dataset_dir) / "100.dat").touch()
            with self.assertRaisesRegex(RuntimeError, "matching .hea, .dat and .atr"):
                mp.BeatExtractor(dataset_dir).extract()

    def test_real_quant_matches_frozen_fake_quant(self):
        model, _ = load_fake_quant_model()
        torch.manual_seed(7)
        calibration = torch.randn(16, 1, mp.SIGNAL_LENGTH)
        evaluation = torch.randn(16, 1, mp.SIGNAL_LENGTH)

        mp.reset_and_enable_observers(model)
        with torch.no_grad():
            model(calibration)
        model.apply(mp.disable_observer)
        with torch.no_grad():
            expected = model(evaluation)

        real_model = copy.deepcopy(model)
        mp.convert_to_real_quant(real_model)
        mp.assert_real_quant_ranges(real_model)
        with torch.no_grad():
            actual = real_model(evaluation)

        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertEqual(sum(p.numel() for p in real_model.parameters()), 0)

        bit_counts = {4: 0, 8: 0}
        for module in real_model.modules():
            if isinstance(module, (mp.RealQuantConv1d, mp.RealQuantLinear)):
                bit_counts[module.weight_bits] += 1
                self.assertGreaterEqual(int(module.weight_int8.min()), module.quant_min)
                self.assertLessEqual(int(module.weight_int8.max()), module.quant_max)
        self.assertEqual(bit_counts, {4: 12, 8: 19})

    def test_validate_does_not_update_observers(self):
        model, _ = load_fake_quant_model()
        sample_x = torch.randn(8, 1, mp.SIGNAL_LENGTH)
        sample_y = torch.randint(0, 5, (8,))
        mp.reset_and_enable_observers(model)
        with torch.no_grad():
            model(sample_x)
        before = [
            module.act_fq.activation_post_process.min_val.clone()
            for module in model.modules()
            if isinstance(module, (mp.FakeQuantConv1d, mp.FakeQuantLinear))
        ]

        loader = DataLoader(mp.ECGBeatDataset(sample_x.numpy(), sample_y.numpy()), batch_size=4)
        mp.validate(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"))
        after = [
            module.act_fq.activation_post_process.min_val.clone()
            for module in model.modules()
            if isinstance(module, (mp.FakeQuantConv1d, mp.FakeQuantLinear))
        ]
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(before, after)))

    def test_size_estimates_use_raw_bytes(self):
        model, four_bit_layers = load_fake_quant_model()
        # Estimate on the underlying FP32 architecture, before wrapping.
        sizes = mp.estimate_deployment_sizes(mp.PaperInceptionCNN(), four_bit_layers)
        self.assertEqual(sizes["fp32_raw_bytes"], 1658 * 4)
        self.assertGreater(sizes["fp32_raw_bytes"], sizes["mixed_packed_bytes"])
        self.assertEqual(sizes["n_4bit_weights"] + sizes["n_8bit_weights"] +
                         sizes["bias_params"], 1658)


if __name__ == "__main__":
    unittest.main()
