"""Task 6.1: analyse misclassified samples and score overlap."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, DATA_ROOT, METADATA_DIR, ROOT, TASK1_OUTPUT
from data.dataset import DatasetWav
from models.cm_models import build_cm_model
from utils.data import ensure_output_dirs, make_cm_df, plot_score_distribution
from utils.metrics import load_waveform


def plot_misclassified_waveforms(
    df: pd.DataFrame,
    save_dir: Path,
    n: int = 4,
) -> None:
    """Plot waveforms for misclassified samples."""
    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].head(n)
    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].head(n)
    for tag, subset in [("false_accept", fp), ("false_reject", fn)]:
        for i, row in subset.iterrows():
            wav, sr = load_waveform(row["audio_path"])
            t = np.arange(wav.shape[-1]) / sr
            fig, ax = plt.subplots(figsize=(10, 2.5))
            ax.plot(t, wav.squeeze().numpy())
            ax.set_title(
                f"{tag} score={row['y_score']:.3f} "
                f"{Path(row['audio_path']).name}"
            )
            ax.grid(alpha=0.5)
            fig.tight_layout()
            fig.savefig(save_dir / f"{tag}_{i}.png", dpi=150)
            plt.close(fig)


def main() -> None:
    """Analyse wrong predictions."""
    out_dir = TASK1_OUTPUT / "6_1"
    ensure_output_dirs(out_dir)
    pred_path = ROOT / "task1" / "outputs" / "test_predictions.csv"
    if not pred_path.exists():
        print("Run task1/4_2/main.py first")
        return
    _, _, df_eval = make_cm_df(METADATA_DIR, DATA_ROOT)
    preds = pd.read_csv(pred_path)
    df_eval = df_eval.iloc[: len(preds)].copy()
    df_eval["y_true"] = preds["y_true"].values
    df_eval["y_pred"] = preds["y_pred"].values
    df_eval["y_score"] = preds["y_score"].values
    df_eval["correct"] = df_eval["y_true"] == df_eval["y_pred"]
    wrong = df_eval[~df_eval["correct"]]
    wrong.to_csv(out_dir / "misclassified.csv", index=False)
    plot_score_distribution(
        df_eval["y_true"].values,
        df_eval["y_score"].values,
        "Скоры: все eval",
        out_dir / "score_all.png",
    )
    plot_score_distribution(
        wrong["y_true"].values,
        wrong["y_score"].values,
        "Скоры: ошибочные предсказания",
        out_dir / "score_wrong.png",
    )
    plot_misclassified_waveforms(wrong, out_dir)


if __name__ == "__main__":
    main()
