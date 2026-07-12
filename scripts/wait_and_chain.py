import json
import paramiko
import time

HOST = "n1.us.clorecloud.net"
PORT = 1416
PASSWORD = "YdgQxoX8gQAz7eAI80"
ROOT = "/root/audio-deepfakes-airi"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)

c.exec_command("kill 5803 2>/dev/null")

for i in range(20):
    _, o, _ = c.exec_command(
        f"test -f {ROOT}/task1/outputs/test_metrics.json && echo DONE || tail -3 {ROOT}/train.log",
        timeout=30,
    )
    text = o.read().decode("utf-8", errors="replace")
    with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\wait_train.txt", "a", encoding="utf-8") as f:
        f.write(f"poll {i}: {text}\n")
    if "DONE" in text:
        break
    time.sleep(60)

chain = f"""
cd {ROOT}
nohup env PYTHONPATH=. .venv/bin/python -u task2/main.py > task2.log 2>&1
wait
nohup env PYTHONPATH=. .venv/bin/python -u task4/main.py > task4.log 2>&1
wait
nohup env PYTHONPATH=. .venv/bin/python -u task5/main.py > task5.log 2>&1
wait
PYTHONPATH=. .venv/bin/python scripts/upload_hf.py
"""

_, o, _ = c.exec_command(chain, timeout=3600)
chain_out = o.read().decode("utf-8", errors="replace")

fetch = f"""
cat {ROOT}/task1/outputs/test_metrics.json 2>/dev/null
cat {ROOT}/task2/outputs/sasv_metrics.json 2>/dev/null
ls {ROOT}/checkpoints/
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
"""
_, o, _ = c.exec_command(fetch, timeout=60)
final = o.read().decode("utf-8", errors="replace")

with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\pipeline_result.txt", "w", encoding="utf-8") as f:
    f.write("CHAIN:\n" + chain_out + "\n\nFINAL:\n" + final)

c.close()
