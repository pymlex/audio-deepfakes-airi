"""Task 6.2: cross-domain evaluation and uncertainty on CM."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, DATA_ROOT, METADATA_DIR, ROOT, TASK1_OUTPUT
from data.dataset import DatasetWav
from models.cm_models import build_cm_model
from utils.data import ensure_output_dirs, make_cm_df, save_json
from utils.metrics import classification_metrics
from utils.training import test


def evaluate_split(
    model: torch.nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    name: str,
) -> dict:
    """Evaluate model on a dataframe split."""
    loader = DataLoader(
        DatasetWav(df),
        batch_size=32,
        shuffle=False,
        num_workers=4,
    )
    _, _, y_score, metrics = test(
        model,
        loader,
        device=device,
        loss_type="ce",
    )
    metrics["split"] = name
    metrics["n_samples"] = len(df)
    return metrics


def resolve_checkpoint() -> tuple[Path, str]:
    """Pick best available CM checkpoint and matching architecture name."""
    for name in ("aasist_lite", "wav_resnet"):
        path = CHECKPOINT_DIR / f"{name}_best.pt"
        if path.exists():
            return path, name
    raise FileNotFoundError("No CM checkpoint in checkpoints/")


def main() -> None:
    """Cross-evaluate on available splits."""
    out_dir = TASK1_OUTPUT / "6_2"
    ensure_output_dirs(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt, model_name = resolve_checkpoint()
    model = build_cm_model(model_name, classes=2)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.to(device)
    df_train, df_dev, df_eval = make_cm_df(METADATA_DIR, DATA_ROOT)
    results = {
        "train_subset": evaluate_split(
            model,
            df_train.sample(n=min(2000, len(df_train)), random_state=42),
            device,
            "train_subset",
        ),
        "dev": evaluate_split(model, df_dev, device, "dev"),
        "eval_24": evaluate_split(model, df_eval, device, "eval_24"),
    }
    save_json(results, out_dir / "cross_domain_metrics.json")
    print(results)


if __name__ == "__main__":
    main()
