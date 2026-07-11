from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchvision.models import resnet18


class WavResNet(nn.Module):
    """Mel-spectrogram ResNet countermeasure model."""

    def __init__(
        self,
        classes: int = 2,
        resample: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        backbone: str = "resnet18",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.resample = resample
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=resample,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_ftrs, classes),
        )
        self.network = model
        self.classes = classes

    def forward(
        self,
        x: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass from waveform to logits."""
        mels = self.fbank(x)
        mels_db = self.to_db(mels)
        mels_db = (mels_db - mels_db.mean()) / (mels_db.std() + 1e-6)
        if mels_db.dim() == 3:
            mels_db = mels_db.unsqueeze(1)
        out = self.network(mels_db)
        return out


class WavResNetBCE(nn.Module):
    """Single-logit variant for BCE training."""

    def __init__(self, **kwargs):
        super().__init__()
        kwargs["classes"] = 1
        self.base = WavResNet(**kwargs)

    def forward(
        self,
        x: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.base(x, wav_lens)


class GraphAttentionLayer(nn.Module):
    """Single-head graph attention layer."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(out_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention over time-frequency nodes."""
        h = self.proj(x)
        n = h.shape[1]
        h_i = h.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h.unsqueeze(1).expand(-1, n, -1, -1)
        e = torch.tanh(self.attn(torch.cat([h_i, h_j], dim=-1)).squeeze(-1))
        alpha = F.softmax(e, dim=-1)
        out = torch.bmm(alpha, h)
        return out


class AASISTLite(nn.Module):
    """Lightweight AASIST-inspired spectro-temporal graph model."""

    def __init__(
        self,
        classes: int = 2,
        resample: int = 16000,
        n_mels: int = 80,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.resample = resample
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=resample,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.temporal_gat = GraphAttentionLayer(64, hidden_dim)
        self.freq_gat = GraphAttentionLayer(64, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mels = self.fbank(x)
        mels_db = self.to_db(mels)
        mels_db = (mels_db - mels_db.mean()) / (mels_db.std() + 1e-6)
        if mels_db.dim() == 3:
            mels_db = mels_db.unsqueeze(1)
        feat = self.conv(mels_db)
        temporal = feat.mean(dim=2).permute(0, 2, 1)
        freq = feat.mean(dim=3).permute(0, 2, 1)
        t_out = self.temporal_gat(temporal)
        f_out = self.freq_gat(freq)
        t_pool = self.pool(t_out.transpose(1, 2)).squeeze(-1)
        f_pool = self.pool(f_out.transpose(1, 2)).squeeze(-1)
        fused = torch.cat([t_pool, f_pool], dim=-1)
        return self.classifier(fused)


class ECAPAEncoder(nn.Module):
    """Compact ECAPA-TDNN-style speaker embedding encoder."""

    def __init__(
        self,
        resample: int = 16000,
        n_mels: int = 80,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=resample,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        self.conv1 = nn.Conv1d(n_mels, 512, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(512, 512, kernel_size=3, padding=1, dilation=2)
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, padding=3, dilation=3)
        self.bn = nn.BatchNorm1d(512)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 512),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mels = self.to_db(self.fbank(x))
        if mels.dim() == 3:
            mels = mels.squeeze(1)
        h = F.relu(self.conv1(mels))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.bn(h)
        se = self.se(h).unsqueeze(-1)
        h = h * se
        h = h.mean(dim=-1)
        emb = F.normalize(self.fc(h), dim=-1)
        return emb


class SASVFusionModel(nn.Module):
    """Fusion of speaker embedding similarity and CM score for SASV."""

    def __init__(
        self,
        cm_model: nn.Module,
        embed_dim: int = 192,
    ):
        super().__init__()
        self.cm_model = cm_model
        self.encoder = ECAPAEncoder(embed_dim=embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def cm_score(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.cm_model(x)
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits.squeeze(-1))
        return F.softmax(logits, dim=-1)[:, 1]

    def forward(
        self,
        ref: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        ref_emb = self.encoder(ref)
        query_emb = self.encoder(query)
        cos_sim = F.cosine_similarity(ref_emb, query_emb, dim=-1)
        cm_ref = self.cm_score(ref)
        cm_query = self.cm_score(query)
        features = torch.stack([cos_sim, cm_ref, cm_query], dim=-1)
        return self.fusion(features).squeeze(-1)


class EvidentialHead(nn.Module):
    """Evidential deep learning output layer for Dirichlet parameters."""

    def __init__(self, in_features: int, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc(x)) + 1.0


def build_cm_model(
    name: str,
    classes: int = 2,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory for countermeasure models."""
    if name == "wav_resnet":
        return WavResNet(classes=classes, dropout=dropout)
    if name == "aasist_lite":
        return AASISTLite(classes=classes, dropout=dropout)
    raise ValueError(f"Unknown model: {name}")
