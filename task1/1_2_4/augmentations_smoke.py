"""Task 2.4: augmentation smoke test."""

import torch

from data.augmentations import WaveformAugment


def main() -> None:
    aug = WaveformAugment()
    x = torch.randn(1, 64000)
    y = aug(x)
    print(x.shape, y.shape)


if __name__ == "__main__":
    main()
