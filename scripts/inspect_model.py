"""Print the architecture and parameter distribution of the mixed model."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mixed_precision import PaperInceptionCNN  # noqa: E402


def main():
    model = PaperInceptionCNN(num_classes=5, in_channels=1)

    print("=" * 80)
    print("MODEL PARAMETERS")
    print("=" * 80)
    print(f"{'Layer':<52} {'Shape':<18} {'Parameters':>10}")

    total = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        print(f"{name:<52} {str(list(parameter.shape)):<18} {count:>10,}")

    print("-" * 80)
    print(f"{'Total':<71} {total:>8,}")

    shapes = {}
    hooks = []

    def capture(name):
        def hook(_module, inputs, output):
            shapes[name] = (list(inputs[0].shape), list(output.shape))
        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(capture(name)))

    with torch.no_grad():
        model(torch.randn(1, 1, 260))

    for hook in hooks:
        hook.remove()

    print("\n" + "=" * 80)
    print("LEAF MODULE INPUT/OUTPUT SHAPES")
    print("=" * 80)
    for name, (input_shape, output_shape) in shapes.items():
        print(f"{name:<48} {str(input_shape):<15} -> {output_shape}")


if __name__ == "__main__":
    main()
