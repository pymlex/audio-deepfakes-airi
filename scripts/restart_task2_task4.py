import paramiko
import time

HOST = "n1.us.clorecloud.net"
PORT = 1416
PASSWORD = "YdgQxoX8gQAz7eAI80"
ROOT = "/root/audio-deepfakes-airi"
OUT = r"C:\Users\aleks\Downloads\audio-deepfakes-airi\restart_status.txt"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)

lines = []


def run(cmd: str, timeout: int = 3600) -> str:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode("utf-8", errors="replace")
    err = e.read().decode("utf-8", errors="replace")
    lines.append(f">>> {cmd}\n{out}{err}\n")
    return out


run(f"cd {ROOT} && git pull")
cuda = run(
    f"cd {ROOT} && .venv/bin/python -c "
    f"\"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
)
run("nvidia-smi")
run(f"tail -5 {ROOT}/train.log")

chain = f"""
cd {ROOT}
pkill -f 'task2/main.py' 2>/dev/null || true
pkill -f 'task4/main.py' 2>/dev/null || true
nohup env PYTHONPATH=. .venv/bin/python -u task2/main.py > task2.log 2>&1 &
TASK2=$!
echo "task2 pid=$TASK2"
for i in $(seq 1 90); do
  if ! kill -0 $TASK2 2>/dev/null; then break; fi
  sleep 10
done
wait $TASK2 2>/dev/null || true
echo "task2 exit=$?"
nohup env PYTHONPATH=. .venv/bin/python -u task4/main.py > task4_rerun.log 2>&1 &
TASK4=$!
echo "task4 pid=$TASK4"
wait $TASK4 2>/dev/null || true
echo "task4 exit=$?"
ls -la task2/outputs/ task4/outputs/
nvidia-smi
"""

run(chain, timeout=7200)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("CUDA: " + cuda.strip() + "\n\n")
    f.write("\n".join(lines))

c.close()
print("done")
