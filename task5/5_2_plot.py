"""Plot layer probing bar chart for task 5."""

from pathlib import Path

import matplotlib.pyplot as plt

from config import TASK5_OUTPUT
from utils.data import load_json


def main() -> None:
    out_dir = TASK5_OUTPUT
    probing = load_json(out_dir / "layer_probing.json")
    layers = list(probing.keys())
    vals = [probing[k] for k in layers]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(layers, vals, color="#4C72B0")
    ax.set_ylabel("linear probe accuracy")
    ax.set_xlabel("ResNet layer")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "layer_probing.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
