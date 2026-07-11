from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CMTrainConfig(BaseModel):
    config_path: Path | None = None
    lr: float = 1e-4
    epochs: int = 5
    batch_size: int = 32
    num_workers: int = 8
    subset_size: int | None = 4000
    val_fraction: float = 0.1
    model_name: Literal["wav_resnet", "aasist_lite"] = "wav_resnet"
    loss_type: Literal["bce", "ce"] = "ce"
    use_augmentation: bool = True
    freeze_backbone: bool = True
    class_weight_alpha: float = 0.5
    device: str = "cuda"
    checkpoint_dir: Path = Field(default_factory=lambda: Path("checkpoints"))
    output_dir: Path = Field(default_factory=lambda: Path("task1/outputs"))
    wandb_project: str = "audio-deepfakes-airi-cm"
    wandb_run_name: str = "cm_baseline"
    seed: int = 42


class SASVTrainConfig(BaseModel):
    lr: float = 1e-4
    epochs: int = 8
    batch_size: int = 64
    num_workers: int = 8
    subset_size: int | None = 8000
    device: str = "cuda"
    checkpoint_dir: Path = Field(default_factory=lambda: Path("checkpoints"))
    output_dir: Path = Field(default_factory=lambda: Path("task2/outputs"))
    seed: int = 42


class UncertaintyConfig(BaseModel):
    methods: list[str] = Field(
        default_factory=lambda: [
            "mc_dropout",
            "deep_ensemble",
            "temperature_scaling",
            "evidential",
            "entropy",
        ]
    )
    mc_samples: int = 30
    n_ensemble: int = 3
    output_dir: Path = Field(default_factory=lambda: Path("task4/outputs"))
    device: str = "cuda"


class ExplainConfig(BaseModel):
    methods: list[str] = Field(
        default_factory=lambda: [
            "saliency",
            "integrated_gradients",
            "gradcam",
            "occlusion",
            "probing",
        ]
    )
    n_samples: int = 8
    output_dir: Path = Field(default_factory=lambda: Path("task5/outputs"))
    device: str = "cuda"
