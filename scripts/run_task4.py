import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_script(relpath: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(
        [sys.executable, str(ROOT / relpath)],
        check=True,
        cwd=ROOT,
        env=env,
    )


if __name__ == "__main__":
    run_script("task4/4_main.py")
