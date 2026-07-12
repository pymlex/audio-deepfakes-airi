"""Task 2.2: compare BCE vs cross-entropy training."""

import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from config import DATA_ROOT, METADATA_DIR, ROOT, TASK1_OUTPUT
from data.dataset import DatasetWav
from models.cm_models import WavResNet, WavResNetBCE
from schemas import CMTrainConfig
from utils.data import ensure_output_dirs, make_cm_df, save_json
from utils.training import test, train


def run_experiment(
    loss_type: str,
    cfg: CMTrainConfig,
    df_train,
    df_dev,
    df_eval,
    device: torch.device,
) -> dict:
    """Train and evaluate one loss variant."""
    train_ds = DatasetWav(df_train)
    dev_ds = DatasetWav(df_dev)
    eval_ds = DatasetWav(df_eval)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    if loss_type == "bce":
        model = WavResNetBCE(dropout=0.3).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        model = WavResNet(classes=2, dropout=0.3).to(device)
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ckpt = cfg.checkpoint_dir / f"loss_compare_{loss_type}.pt"
    model, history = train(
        model,
        {"train": train_loader, "val": dev_loader},
        criterion,
        optimizer,
        num_epochs=cfg.epochs,
        savename=str(ckpt),
        device=device,
        loss_type=loss_type,
    )
    _, _, y_score, metrics = test(
        model,
        eval_loader,
        criterion,
        device,
        loss_type=loss_type,
    )
    metrics["history"] = history
    return metrics


def main() -> None:
    """Compare BCE and CE on a subset."""
    out_dir = TASK1_OUTPUT / "2_2"
    ensure_output_dirs(out_dir)
    config_path = ROOT / "task1" / "configs" / "cm_baseline.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = CMTrainConfig(**raw)
    cfg.epochs = 3
    cfg.subset_size = 2000
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    df_train, df_dev, df_eval = make_cm_df(METADATA_DIR, DATA_ROOT)
    if cfg.subset_size:
        df_train = df_train.sample(n=cfg.subset_size, random_state=cfg.seed)
        df_dev = df_dev.sample(n=cfg.subset_size // 5, random_state=cfg.seed)
    results = {}
    for loss_type in ["bce", "ce"]:
        results[loss_type] = run_experiment(
            loss_type,
            cfg,
            df_train,
            df_dev,
            df_eval,
            device,
        )
    save_json(results, out_dir / "loss_comparison.json")
    print(json.dumps({k: v.get("accuracy") for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
