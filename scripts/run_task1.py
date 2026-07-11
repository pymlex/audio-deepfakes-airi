import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 CM pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "1_1", "1_2", "2_2", "train", "sota", "tricks", "6_1", "6_2"],
        default="all",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "task1" / "configs" / "cm_baseline.yaml"),
    )
    args = parser.parse_args()
    steps = {
        "1_1": "task1/1_1_distribution.py",
        "1_2": "task1/1_2_audio_samples.py",
        "2_2": "task1/2_2_loss_compare.py",
        "train": "task1/4_2_main.py",
        "sota": "task1/5_1_sota.py",
        "tricks": "task1/5_2_tricks.py",
        "6_1": "task1/6_1_analysis.py",
        "6_2": "task1/6_2_cross_eval.py",
    }
    if args.step == "all":
        run_script("scripts/setup_data.py")
        for key in ["1_1", "train", "sota", "6_1", "6_2"]:
            if key == "train":
                run_script(steps[key], ["--config", args.config])
            else:
                run_script(steps[key])
    elif args.step == "train":
        run_script(steps[args.step], ["--config", args.config])
    else:
        run_script(steps[args.step])


if __name__ == "__main__":
    main()
