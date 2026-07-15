import paramiko
from pathlib import Path

HOST = "n1.us.clorecloud.net"
PORT = 1416
USER = "root"
PASSWORD = "YdgQxoX8gQAz7eAI80"
REMOTE = "/root/audio-deepfakes-airi"
ROOT = Path(__file__).resolve().parent.parent

POST = """
cd /root/audio-deepfakes-airi
PYTHONPATH=. .venv/bin/python task1/1_6_1/analysis.py
PYTHONPATH=. .venv/bin/python task1/1_6_2/cross_eval.py
PYTHONPATH=. .venv/bin/python task4/plot.py
PYTHONPATH=. .venv/bin/python task5/plot.py
ls -la task1/outputs/1_6_1/
"""

PULL = [
    "task1/outputs/test_metrics.json",
    "task1/outputs/training_history.json",
    "task1/outputs/training_curves.png",
    "task1/outputs/roc_eval.png",
    "task1/outputs/confusion_eval.png",
    "task1/outputs/score_distribution_eval.png",
    "task1/outputs/1_6_1/score_all.png",
    "task1/outputs/1_6_1/score_wrong.png",
    "task1/outputs/1_6_2/cross_eval.json",
    "task4/outputs/all_uncertainty.json",
    "task4/outputs/uncertainty_comparison.png",
    "task4/outputs/uncertainty_scores.png",
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=60)
_, stdout, stderr = client.exec_command(POST, timeout=600)
post_out = stdout.read().decode("utf-8", "replace") + stderr.read().decode("utf-8", "replace")
open(ROOT / "_post_train.txt", "w", encoding="utf-8").write(post_out)

sftp = client.open_sftp()
for rel in PULL:
    local = ROOT / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(f"{REMOTE}/{rel}", str(local))
sftp.close()
client.close()
open(ROOT / "_pull_done.txt", "w", encoding="utf-8").write("ok")
