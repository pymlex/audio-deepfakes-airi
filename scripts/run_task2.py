import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_script(relpath: str, args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / relpath)]
    if args:
        cmd.extend(args)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


if __name__ == "__main__":
    run_script("task2/2_main.py", sys.argv[1:])
