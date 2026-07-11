"""Task 2.1: verify WavResNet forward pass."""

import torch

from models.cm_models import WavResNet


def main() -> None:
    model = WavResNet(classes=2)
    x = torch.randn(2, 64000)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    main()
