"""Task 4.2: main CM training and evaluation pipeline."""

import argparse
import os
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from config import DATA_ROOT, METADATA_DIR, ROOT
from data.augmentations import WaveformAugment
from data.dataset import DatasetWav
from models.cm_models import build_cm_model
from schemas import CMTrainConfig
from utils.data import (
    ensure_output_dirs,
    make_cm_df,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_score_distribution,
    plot_training_history,
    save_json,
)
from utils.training import test, train


def load_config(path: Path) -> CMTrainConfig:
    """Load YAML config into pydantic model."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return CMTrainConfig(**raw)


def main(config_path: str | None = None) -> tuple:
    """Train and test countermeasure model from config."""
    load_dotenv(ROOT / ".env")
    if config_path is None:
        config_path = str(ROOT / "task1" / "configs" / "cm_baseline.yaml")
    cfg = load_config(Path(config_path))
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_output_dirs(cfg.output_dir, cfg.checkpoint_dir)
    device = torch.device(
        cfg.device if torch.cuda.is_available() else "cpu"
    )
    df_train, df_dev, df_eval = make_cm_df(
        METADATA_DIR,
        DATA_ROOT,
        val_fraction=cfg.val_fraction,
        seed=cfg.seed,
    )
    if cfg.subset_size:
        df_train = df_train.sample(n=cfg.subset_size, random_state=cfg.seed)
        df_dev = df_dev.sample(
            n=min(cfg.subset_size // 5, len(df_dev)),
            random_state=cfg.seed,
        )
    transform = WaveformAugment() if cfg.use_augmentation else None
    train_ds = DatasetWav(df_train, transform=transform)
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
    classes = 2 if cfg.loss_type == "ce" else 1
    model = build_cm_model(cfg.model_name, classes=classes)
    model = model.to(device)
    if cfg.freeze_backbone and hasattr(model, "network"):
        for param in model.parameters():
            param.requires_grad = False
        model.network.fc.requires_grad_(True)
    alpha = cfg.class_weight_alpha
    if cfg.loss_type == "bce":
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        sc = torch.tensor([1 - alpha, alpha]).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=sc)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
    )
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.model_dump(),
        )
    ckpt_path = cfg.checkpoint_dir / f"{cfg.model_name}_best.pt"
    model, history = train(
        model,
        {"train": train_loader, "val": dev_loader},
        criterion,
        optimizer,
        num_epochs=cfg.epochs,
        scheduler=scheduler,
        savename=str(ckpt_path),
        device=device,
        loss_type=cfg.loss_type,
        use_wandb=use_wandb,
    )
    y_true, y_pred, y_score, metrics = test(
        model,
        eval_loader,
        criterion,
        device,
        loss_type=cfg.loss_type,
        savename=str(cfg.output_dir / "test_predictions.csv"),
    )
    save_json(metrics, cfg.output_dir / "test_metrics.json")
    save_json(history, cfg.output_dir / "training_history.json")
    plot_training_history(history, cfg.output_dir / "training_curves.png")
    plot_score_distribution(
        y_true,
        y_score,
        "Распределение скоров на eval",
        cfg.output_dir / "score_distribution_eval.png",
    )
    plot_roc_curve(
        y_true,
        y_score,
        "ROC eval",
        cfg.output_dir / "roc_eval.png",
    )
    plot_confusion_matrix(
        y_true,
        y_pred,
        "Confusion matrix eval",
        cfg.output_dir / "confusion_eval.png",
    )
    if use_wandb:
        import wandb

        wandb.log(metrics)
        wandb.finish()
    return y_true, y_score, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "task1" / "configs" / "cm_baseline.yaml"),
    )
    args = parser.parse_args()
    main(args.config)
