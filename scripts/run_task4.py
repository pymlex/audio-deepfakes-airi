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
: > task4_rerun.log
nohup env PYTHONPATH=. .venv/bin/python -u task4/4_main.py > task4_rerun.log 2>&1 &
echo PID=$!
for i in $(seq 1 30); do
  if grep -q deep_ensemble {ROOT}/task4/outputs/all_uncertainty.json 2>/dev/null; then echo DONE; break; fi
  if ! pgrep -f 'task4/4_main.py' >/dev/null; then tail -5 task4_rerun.log; break; fi
  sleep 10
done
ls -la task4/outputs/
tail -15 task4_rerun.log
nvidia-smi
"""

_, o, e = c.exec_command(chain, timeout=600)
out = o.read().decode("utf-8", errors="replace")

with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\task4_status.txt", "w", encoding="utf-8") as f:
    f.write(out)
    f.write(e.read().decode("utf-8", errors="replace"))

print(out[-2000:])
c.close()
