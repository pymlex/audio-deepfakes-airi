import numpy as np
import torch
from torch.utils.data import Dataset

from config import DEFAULT_SR, PADDING_SEC
from utils.metrics import load_waveform, pad_or_crop


class DatasetWav(Dataset):
    """Waveform dataset for countermeasure training."""

    def __init__(
        self,
        data_frame,
        padding_sec: float = PADDING_SEC,
        default_sr: int = DEFAULT_SR,
        transform=None,
        label_col: str = "label_id",
        path_col: str = "audio_path",
    ):
        self.df = data_frame.reset_index(drop=True)
        self.padding_sec = padding_sec
        self.default_sr = default_sr
        self.transform = transform
        self.labels = self.df[label_col].values.astype(np.float32)
        self.paths = self.df[path_col].values
        self.target_samples = int(padding_sec * default_sr)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        label = self.labels[idx]
        path = self.paths[idx]
        waveform, _ = load_waveform(path, self.default_sr)
        if self.transform is not None:
            waveform = self.transform(waveform)
        waveform = pad_or_crop(waveform, self.target_samples)
        x = waveform.squeeze(0)
        return x, torch.tensor(label, dtype=torch.float32)


class SASVTrialDataset(Dataset):
    """SASV trial dataset with reference and query waveforms."""

    def __init__(
        self,
        data_frame,
        padding_sec: float = PADDING_SEC,
        default_sr: int = DEFAULT_SR,
        label_col: str = "trial_label",
    ):
        self.df = data_frame.reset_index(drop=True)
        self.padding_sec = padding_sec
        self.default_sr = default_sr
        self.labels = self.df[label_col].values
        self.ref_paths = self.df["reference_audio_path"].values
        self.query_paths = self.df["query_audio_path"].values
        self.target_samples = int(padding_sec * default_sr)
        self.label_map = {"target": 2, "nontarget": 1, "spoof": 0}

    def __len__(self) -> int:
        return len(self.df)

    def _load_padded(self, path: str) -> torch.Tensor:
        waveform, _ = load_waveform(path, self.default_sr)
        waveform = pad_or_crop(waveform, self.target_samples)
        return waveform.squeeze(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ref = self._load_padded(self.ref_paths[idx])
        query = self._load_padded(self.query_paths[idx])
        label = self.label_map[self.labels[idx]]
        return ref, query, torch.tensor(label, dtype=torch.long)
