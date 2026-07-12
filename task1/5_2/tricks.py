"""Task 5.2: experiment with training tricks."""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from config import DATA_ROOT, METADATA_DIR, ROOT, TASK1_OUTPUT
from data.augmentations import WaveformAugment
from data.dataset import DatasetWav
from models.cm_models import build_cm_model
from schemas import CMTrainConfig
from utils.data import ensure_output_dirs, make_cm_df, save_json
from utils.training import test, train


TRICKS = {
    "baseline": {"use_augmentation": False, "freeze_backbone": True, "lr": 1e-4},
    "augmentation": {"use_augmentation": True, "freeze_backbone": True, "lr": 1e-4},
    "full_finetune": {"use_augmentation": True, "freeze_backbone": False, "lr": 5e-5},
    "focal_loss": {"use_augmentation": True, "freeze_backbone": False, "lr": 1e-4, "focal": True},
    "label_smoothing": {"use_augmentation": True, "freeze_backbone": False, "lr": 1e-4, "label_smoothing": 0.1},
}


class FocalLoss(torch.nn.Module):
    """Focal loss for imbalanced binary classification."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = torch.nn.functional.cross_entropy(logits, targets.long(), reduction="none")
        pt = torch.exp(-ce)
        focal = self.alpha * (1 - pt) ** self.gamma * ce
        return focal.mean()


def run_trick(name: str, trick_cfg: dict, base_cfg: CMTrainConfig) -> dict:
    """Run one trick experiment."""
    cfg = deepcopy(base_cfg)
    cfg.use_augmentation = trick_cfg.get("use_augmentation", cfg.use_augmentation)
    cfg.freeze_backbone = trick_cfg.get("freeze_backbone", cfg.freeze_backbone)
    cfg.lr = trick_cfg.get("lr", cfg.lr)
    cfg.epochs = 3
    cfg.subset_size = 3000
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    df_train, df_dev, df_eval = make_cm_df(METADATA_DIR, DATA_ROOT)
    df_train = df_train.sample(n=cfg.subset_size, random_state=cfg.seed)
    df_dev = df_dev.sample(n=600, random_state=cfg.seed)
    transform = WaveformAugment() if cfg.use_augmentation else None
    train_loader = DataLoader(
        DatasetWav(df_train, transform=transform),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )
    dev_loader = DataLoader(
        DatasetWav(df_dev),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    eval_loader = DataLoader(
        DatasetWav(df_eval),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    model = build_cm_model(cfg.model_name, classes=2).to(device)
    if cfg.freeze_backbone and hasattr(model, "network"):
        for param in model.parameters():
            param.requires_grad = False
        model.network.fc.requires_grad_(True)
    if trick_cfg.get("focal"):
        criterion = FocalLoss()
    elif trick_cfg.get("label_smoothing"):
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=trick_cfg["label_smoothing"])
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
    )
    model, _ = train(
        model,
        {"train": train_loader, "val": dev_loader},
        criterion,
        optimizer,
        num_epochs=cfg.epochs,
        device=device,
        loss_type="ce",
    )
    _, _, _, metrics = test(model, eval_loader, criterion, device, loss_type="ce")
    metrics["trick"] = name
    return metrics


def main() -> None:
    """Run all trick experiments."""
    out_dir = TASK1_OUTPUT / "5_2"
    ensure_output_dirs(out_dir)
    config_path = ROOT / "task1" / "configs" / "cm_baseline.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        base_cfg = CMTrainConfig(**yaml.safe_load(f))
    results = {}
    for name, trick_cfg in TRICKS.items():
        results[name] = run_trick(name, trick_cfg, base_cfg)
        print(name, results[name]["accuracy"], results[name].get("eer"))
    save_json(results, out_dir / "tricks_comparison.json")


if __name__ == "__main__":
    main()
