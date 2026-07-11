"""Task 1.2: display spoof and bona fide audio waveforms."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import DATA_ROOT, METADATA_DIR, TASK1_OUTPUT
from utils.data import ensure_output_dirs, make_cm_df
from utils.metrics import load_waveform


def plot_waveform(
    waveform: np.ndarray,
    sr: int,
    title: str,
    save_path: Path,
) -> None:
    """Save waveform plot."""
    t = np.arange(len(waveform)) / sr
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, waveform)
    ax.set_xlabel("time, s")
    ax.set_ylabel("amplitude")
    ax.set_title(title)
    ax.grid(alpha=0.5)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Plot example bona fide and spoof waveforms."""
    out_dir = TASK1_OUTPUT / "1_2"
    ensure_output_dirs(out_dir)
    df_train, _, _ = make_cm_df(METADATA_DIR, DATA_ROOT)
    for label in ["bonafide", "spoof"]:
        row = df_train[df_train["label"] == label].iloc[0]
        wav, sr = load_waveform(row["audio_path"])
        plot_waveform(
            wav.squeeze().numpy(),
            sr,
            f"{label}: {Path(row['audio_path']).name}",
            out_dir / f"waveform_{label}.png",
        )


if __name__ == "__main__":
    main()
