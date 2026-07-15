import paramiko
from pathlib import Path

HOST = "n1.us.clorecloud.net"
PORT = 1416
USER = "root"
PASSWORD = "YdgQxoX8gQAz7eAI80"
REMOTE = "/root/audio-deepfakes-airi"
ROOT = Path(__file__).resolve().parent.parent

PULL = [
    "task1/outputs/test_metrics.json",
    "task1/outputs/training_history.json",
    "task1/outputs/training_curves.png",
    "task1/outputs/roc_eval.png",
    "task1/outputs/confusion_eval.png",
    "task1/outputs/score_distribution_eval.png",
    "task1/outputs/6_1/score_all.png",
    "task1/outputs/6_1/score_wrong.png",
    "task1/outputs/6_2/cross_domain_metrics.json",
    "task4/outputs/all_uncertainty.json",
    "task4/outputs/uncertainty_comparison.png",
    "task4/outputs/uncertainty_scores.png",
]

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=60)
sftp = client.open_sftp()
for rel in PULL:
    local = ROOT / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(f"{REMOTE}/{rel}", str(local))
    print("ok", rel)
sftp.close()
client.close()
