import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    """Commit and push outputs to GitHub, upload models to HF."""
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "upload_hf.py")],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        ["git", "add", "-A"],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            "Update metrics, plots, and checkpoints metadata",
        ],
        cwd=ROOT,
    )
    subprocess.run(["git", "push"], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
