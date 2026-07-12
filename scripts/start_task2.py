import paramiko
import time

HOST = "n1.us.clorecloud.net"
PORT = 1416
PASSWORD = "YdgQxoX8gQAz7eAI80"
ROOT = "/root/audio-deepfakes-airi"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)

chain = f"""
cd {ROOT}
: > task2.log
nohup env PYTHONPATH=. .venv/bin/python -u task2/main.py > task2.log 2>&1 &
echo PID=$!
sleep 120
nvidia-smi
ps aux | grep 2_main | grep -v grep
tail -30 task2.log
"""

_, o, e = c.exec_command(chain, timeout=180)
out = o.read().decode("utf-8", errors="replace")
err = e.read().decode("utf-8", errors="replace")

with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\gpu_verify2.txt", "w") as f:
    f.write(out)
    f.write(err)

print(out)
c.close()
