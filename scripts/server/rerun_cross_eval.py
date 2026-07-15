import paramiko
from pathlib import Path

HOST = "n1.us.clorecloud.net"
PORT = 1416
USER = "root"
PASSWORD = "YdgQxoX8gQAz7eAI80"
ROOT = Path(__file__).resolve().parent.parent

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=60)
sftp = client.open_sftp()
sftp.put(
    str(ROOT / "task1" / "6_2" / "cross_eval.py"),
    "/root/audio-deepfakes-airi/task1/6_2_cross_eval.py",
)
sftp.close()
_, stdout, stderr = client.exec_command(
    "cd /root/audio-deepfakes-airi && PYTHONPATH=. .venv/bin/python task1/6_2_cross_eval.py && cat task1/outputs/6_2/cross_domain_metrics.json",
    timeout=600,
)
out = stdout.read().decode("utf-8", "replace")
open(ROOT / "_cross_eval.txt", "w", encoding="utf-8").write(out + stderr.read().decode("utf-8", "replace"))
sftp = client.open_sftp()
sftp.get(
    "/root/audio-deepfakes-airi/task1/outputs/6_2/cross_domain_metrics.json",
    str(ROOT / "task1/outputs/6_2/cross_domain_metrics.json"),
)
sftp.close()
client.close()
