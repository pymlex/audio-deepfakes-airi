import paramiko
import time
from pathlib import Path

HOST = "n1.us.clorecloud.net"
PORT = 1416
USER = "root"
PASSWORD = "YdgQxoX8gQAz7eAI80"
REMOTE = "/root/audio-deepfakes-airi"
ROOT = Path(__file__).resolve().parent.parent

PULL_FILES = [
    "task1/outputs/test_metrics.json",
    "task1/outputs/training_history.json",
    "task1/outputs/training_curves.png",
    "task1/outputs/roc_eval.png",
    "task1/outputs/confusion_eval.png",
    "task1/outputs/score_distribution_eval.png",
    "task1/outputs/test_predictions.csv",
    "task1/outputs/1_6_1/score_all.png",
    "task1/outputs/1_6_1/score_wrong.png",
    "task1/outputs/1_6_1/misclassified.csv",
    "task1/outputs/1_6_2/cross_eval.json",
    "task2/outputs/sasv_metrics.json",
    "task2/outputs/sasv_roc.png",
    "task4/outputs/all_uncertainty.json",
    "task4/outputs/uncertainty_comparison.png",
    "task4/outputs/uncertainty_scores.png",
    "task5/outputs/layer_probing.json",
    "task5/outputs/layer_probing.png",
    "task5/outputs/saliency_0.png",
    "task5/outputs/ig_0.png",
    "task5/outputs/gradcam_0.png",
    "task5/outputs/occlusion_0.png",
]


def ssh_cmd(client, cmd):
    _, stdout, stderr = client.exec_command(cmd)
    return stdout.read().decode("utf-8", "replace"), stderr.read().decode("utf-8", "replace")


client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

for i in range(40):
    out, _ = ssh_cmd(
        client,
        "pgrep -f 'task1/1_4_2/main.py --config task1/configs/aasist_train.yaml' || true",
    )
    if not out.strip():
        break
    grep_out, _ = ssh_cmd(client, "grep -o 'epoch [0-9]*/10' /root/audio-deepfakes-airi/aasist_train.log | tail -1")
    log_line = grep_out.strip() or f"wait {i+1}"
    open(ROOT / "_train_wait.txt", "a", encoding="utf-8").write(f"{log_line}\n")
    time.sleep(90)

post = """
cd /root/audio-deepfakes-airi
PYTHONPATH=. .venv/bin/python task1/1_6_1/analysis.py
PYTHONPATH=. .venv/bin/python task1/1_6_2/cross_eval.py
PYTHONPATH=. .venv/bin/python task4/plot.py
PYTHONPATH=. .venv/bin/python task5/plot.py
cat task1/outputs/test_metrics.json
"""
out, err = ssh_cmd(client, post)
open(ROOT / "_post_train.txt", "w", encoding="utf-8").write(out + "\n" + err)

sftp = client.open_sftp()
for rel in PULL_FILES:
    remote = f"{REMOTE}/{rel}"
    local = ROOT / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
sftp.close()
client.close()
open(ROOT / "_pull_done.txt", "w", encoding="utf-8").write("ok")
