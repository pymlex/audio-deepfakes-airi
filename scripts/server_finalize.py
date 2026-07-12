import paramiko
import time

HOST = "n1.us.clorecloud.net"
PORT = 1416
USER = "root"
PASSWORD = "YdgQxoX8gQAz7eAI80"

FINALIZE = """
set -e
cd /root/audio-deepfakes-airi
while pgrep -f 'task1/4_2_main.py --config task1/configs/aasist_train.yaml' >/dev/null; do
  sleep 30
done
PYTHONPATH=. .venv/bin/python task1/6_1_analysis.py
PYTHONPATH=. .venv/bin/python task1/6_2_cross_eval.py
PYTHONPATH=. .venv/bin/python task4/4_2_plot.py
PYTHONPATH=. .venv/bin/python task5/5_2_plot.py
echo DONE_FINALIZE
cat task1/outputs/test_metrics.json
"""

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
_, stdout, stderr = client.exec_command(
    f"nohup bash -c {repr(FINALIZE)} > /root/audio-deepfakes-airi/finalize.log 2>&1 & echo PID=$!"
)
out = stdout.read().decode("utf-8", "replace")
err = stderr.read().decode("utf-8", "replace")
client.close()
open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\_finalize_start.txt", "w", encoding="utf-8").write(out + err)
