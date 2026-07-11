from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def make_cm_df(
    metadata_dir: Path,
    data_root: Path,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train, dev, and eval dataframes for countermeasure task."""
    train_path = metadata_dir / "train.csv"
    eval_path = metadata_dir / "test_track_1.csv"
    df_all = pd.read_csv(train_path)
    df_all["label_id"] = (df_all["label"] == "bonafide").astype(int)
    df_all["audio_path"] = df_all["audio_path"].apply(
        lambda p: str(data_root / p)
    )
    df_train, df_dev = train_test_split(
        df_all,
        test_size=val_fraction,
        random_state=seed,
        stratify=df_all["label_id"],
    )
    df_eval = pd.read_csv(eval_path)
    df_eval["label_id"] = df_eval["is_bonafide"].astype(int)
    df_eval["label"] = df_eval["label_id"].map(
        {1: "bonafide", 0: "spoof"}
    )
    df_eval["audio_path"] = df_eval["audio_path"].apply(
        lambda p: str(data_root / p)
    )
    df_train = df_train.reset_index(drop=True)
    df_dev = df_dev.reset_index(drop=True)
    df_eval = df_eval.reset_index(drop=True)
    return df_train, df_dev, df_eval


def balanced_split(
    df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample majority class to match minority count."""
    counts = df["label_id"].value_counts()
    n_min = counts.min()
    parts = []
    for label_id in counts.index:
        subset = df[df["label_id"] == label_id].sample(
            n=n_min,
            random_state=seed,
        )
        parts.append(subset)
    return pd.concat(parts, ignore_index=True)


def make_sasv_df(
    metadata_dir: Path,
    data_root: Path,
) -> pd.DataFrame:
    """Load SASV trial dataframe with absolute audio paths."""
    df = pd.read_csv(metadata_dir / "test_4k-track_2.csv")
    df["reference_audio_path"] = df["reference_audio_path"].apply(
        lambda p: str(data_root / p)
    )
    df["query_audio_path"] = df["query_audio_path"].apply(
        lambda p: str(data_root / p)
    )
    return df.reset_index(drop=True)


def ensure_output_dirs(*paths: Path) -> None:
    """Create output directories if missing."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    """Save dictionary as JSON."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict:
    """Load JSON file."""
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_label_distribution(
    df: pd.DataFrame,
    label_col: str,
    title: str,
    save_path: Path,
) -> None:
    """Plot class distribution bar chart."""
    counts = df[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title(title)
    ax.set_ylabel("count")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot score histograms for bona fide and spoof."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        y_score[y_true == 1],
        bins=50,
        alpha=0.6,
        label="bonafide",
        density=True,
    )
    ax.hist(
        y_score[y_true == 0],
        bins=50,
        alpha=0.6,
        label="spoof",
        density=True,
    )
    ax.set_xlabel("score")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.grid(alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["spoof", "bonafide"],
        yticklabels=["spoof", "bonafide"],
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_history(
    history: dict[str, list[float]],
    save_path: Path,
) -> None:
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.5)
    axes[1].plot(history["train_acc"], label="train")
    if "val_acc" in history:
        axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
