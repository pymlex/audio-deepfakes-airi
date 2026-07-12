"""Plot uncertainty method comparison for task 4."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import TASK4_OUTPUT
from utils.data import load_json, save_json


def main() -> None:
    out_dir = TASK4_OUTPUT
    data = load_json(out_dir / "all_uncertainty.json")
    methods = ["mc_dropout", "entropy", "deep_ensemble", "temperature_scaling"]
    labels = ["MC Dropout", "Entropy", "Deep ensemble", "Temperature scaling"]
    metrics = ["eer", "roc_auc", "balanced_accuracy"]
    titles = ["EER", "ROC-AUC", "Balanced accuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [data[m][metric] for m in methods if m in data]
        ax.bar(labels[: len(vals)], vals, color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(vals)])
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_comparison.png", dpi=150)
    plt.close(fig)
    unc_vals = []
    unc_labels = []
    if "mc_dropout" in data:
        unc_vals.append(data["mc_dropout"].get("mean_uncertainty", 0))
        unc_labels.append("MC Dropout")
    if "entropy" in data:
        unc_vals.append(data["entropy"].get("mean_entropy", 0))
        unc_labels.append("Entropy")
    if "deep_ensemble" in data:
        unc_vals.append(data["deep_ensemble"].get("mean_disagreement", 0))
        unc_labels.append("Ensemble")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(unc_labels, unc_vals, color="#8172B3")
    ax.set_title("Mean uncertainty score")
    ax.grid(alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_scores.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
