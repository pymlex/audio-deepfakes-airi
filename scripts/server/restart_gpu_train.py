import paramiko
import time

HOST = "n1.us.clorecloud.net"
PORT = 1416
PASSWORD = "YdgQxoX8gQAz7eAI80"
ROOT = "/root/audio-deepfakes-airi"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)

setup = f"""
cd {ROOT}
git pull --ff-only
.venv/bin/pip uninstall -y torchcodec 2>/dev/null || true
.venv/bin/pip install -r requirements.txt -q
PYTHONPATH=. .venv/bin/python scripts/setup_data.py
.venv/bin/python -c "import soundfile as sf; import torch; print('cuda', torch.cuda.is_available()); sf.read('{ROOT}/data/audio/flac_T/T_0000000011.flac'); print('audio_ok')"
rm -f train.log full_run.log
nohup env PYTHONPATH=. .venv/bin/python -u task1/1_4_2/main.py > train.log 2>&1 &
echo TRAIN_PID=$!
"""

_, o, _ = c.exec_command(setup, timeout=120)
print(o.read().decode("utf-8", errors="replace"))

for minute in [1, 2, 3]:
    time.sleep(60)
    poll = f"""
tail -15 {ROOT}/train.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ps aux | grep 'task1/1_4_2/main.py' | grep -v grep
"""
    _, o, _ = c.exec_command(poll, timeout=30)
    block = o.read().decode("utf-8", errors="replace")
    print(f"\n--- poll {minute} min ---\n{block}")
    if "epoch" in block.lower() and "traceback" not in block.lower():
        break
    if "test_metrics" in block or "val_acc" in block:
        break

c.close()
