"""Task 5.1: train AASIST-lite SOTA model."""

import subprocess
import sys

from config import ROOT


def main() -> None:
    """Run AASIST-lite training."""
    config = ROOT / "task1" / "configs" / "aasist_lite.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "task1" / "4_2" / "main.py"),
            "--config",
            str(config),
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
