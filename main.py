import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_script(relpath: str, args: list[str] | None = None) -> None:
    cmd = [sys.executable, str(ROOT / relpath)]
    if args:
        cmd.extend(args)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run(cmd, check=True, cwd=ROOT, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio deepfake detection pipeline")
    parser.add_argument(
        "--task",
        choices=["all", "1", "2", "4", "5", "setup"],
        default="all",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "task1" / "configs" / "cm_baseline.yaml"),
    )
    args = parser.parse_args()
    if args.task in ("all", "setup"):
        run_script("scripts/setup_data.py")
    if args.task in ("all", "1"):
        run_script("scripts/run_task1.py", ["--step", "all", "--config", args.config])
    if args.task in ("all", "2"):
        run_script("task2/main.py")
    if args.task in ("all", "4"):
        run_script("task4/main.py")
    if args.task in ("all", "5"):
        run_script("task5/main.py")


if __name__ == "__main__":
    main()
