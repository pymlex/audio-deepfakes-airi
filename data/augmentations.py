import torch
import torchaudio


def build_train_augmentation(default_sr: int = 16000):
    """Build waveform augmentation pipeline for training."""
    return torch.nn.Sequential(
        torchaudio.transforms.TimeMasking(time_mask_param=40),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
    )


class WaveformAugment:
    """Apply optional time shift and gain on waveforms."""

    def __init__(
        self,
        max_shift: int = 1600,
        gain_range: tuple[float, float] = (0.8, 1.2),
        default_sr: int = 16000,
    ):
        self.max_shift = max_shift
        self.gain_range = gain_range
        self.default_sr = default_sr

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        if shift > 0:
            waveform = torch.cat(
                [torch.zeros(1, shift), waveform[..., :-shift]],
                dim=-1,
            )
        elif shift < 0:
            waveform = torch.cat(
                [waveform[..., -shift:], torch.zeros(1, -shift)],
                dim=-1,
            )
        gain = torch.empty(1).uniform_(*self.gain_range).item()
        return waveform * gain
