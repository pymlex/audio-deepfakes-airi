import paramiko
import time

HOST = "n1.us.clorecloud.net"
PORT = 1416
PASSWORD = "YdgQxoX8gQAz7eAI80"
ROOT = "/root/audio-deepfakes-airi"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)

start = f"""
cd {ROOT}
git pull
pkill -f 'task2/main.py' 2>/dev/null || true
nohup env PYTHONPATH=. .venv/bin/python -u task2/main.py > task2.log 2>&1 &
echo STARTED_PID=$!
"""

_, o, _ = c.exec_command(start, timeout=60)
print(o.read().decode())

time.sleep(90)

checks = [
    "nvidia-smi",
    "ps aux | grep 2_main | grep -v grep",
    f"tail -20 {ROOT}/task2.log",
]
with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\gpu_verify.txt", "w") as f:
    for cmd in checks:
        _, o, _ = c.exec_command(cmd, timeout=60)
        block = o.read().decode("utf-8", errors="replace")
        f.write(f">>> {cmd}\n{block}\n\n")
        print(block)

c.close()
