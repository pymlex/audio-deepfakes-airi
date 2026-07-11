import json
import os
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/audio-deepfakes-airi"
FILES = [
    "task1/outputs/test_metrics.json",
    "task1/outputs/test_predictions.csv",
    "task1/outputs/training_history.json",
    "task1/outputs/training_curves.png",
    "task1/outputs/roc_eval.png",
    "task1/outputs/confusion_eval.png",
    "task1/outputs/score_distribution_eval.png",
    "task2/outputs/sasv_metrics.json",
    "task2/outputs/sasv_roc.png",
    "task4/outputs/all_uncertainty.json",
    "task5/outputs/layer_probing.json",
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
sftp = c.open_sftp()
for rel in FILES:
    remote = f"{REMOTE}/{rel}"
    local = ROOT / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    print(f"pulled {rel}")
sftp.close()
c.close()

if (ROOT / "task1/outputs/test_metrics.json").exists():
    print(json.dumps(json.loads((ROOT / "task1/outputs/test_metrics.json").read_text()), indent=2))
