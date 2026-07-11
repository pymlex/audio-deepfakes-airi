"""Task 1.1: analyse class distribution in train, dev, and eval splits."""

from pathlib import Path

import pandas as pd

from config import DATA_ROOT, METADATA_DIR, TASK1_OUTPUT
from utils.data import ensure_output_dirs, make_cm_df, plot_label_distribution, save_json


def main() -> None:
    """Compute and plot label distributions."""
    out_dir = TASK1_OUTPUT / "1_1"
    ensure_output_dirs(out_dir)
    df_train, df_dev, df_eval = make_cm_df(METADATA_DIR, DATA_ROOT)
    stats = {}
    for name, df in [("train", df_train), ("dev", df_dev), ("eval", df_eval)]:
        counts = df["label"].value_counts().to_dict()
        stats[name] = counts
        stats[f"{name}_total"] = len(df)
        stats[f"{name}_bonafide_ratio"] = counts.get("bonafide", 0) / len(df)
        plot_label_distribution(
            df,
            "label",
            f"Распределение классов: {name}",
            out_dir / f"distribution_{name}.png",
        )
    save_json(stats, out_dir / "distribution_stats.json")
    print(stats)


if __name__ == "__main__":
    main()
