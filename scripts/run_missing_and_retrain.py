import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
script = """
set -e
cd /root/audio-deepfakes-airi
git fetch origin
git reset --hard origin/main
PYTHONPATH=. .venv/bin/python task1/1_2/audio_samples.py
PYTHONPATH=. .venv/bin/python task4/plot.py
PYTHONPATH=. .venv/bin/python task5/plot.py
nohup env PYTHONPATH=. .venv/bin/python -u task1/4_2/main.py --config task1/configs/aasist_train.yaml > aasist_train.log 2>&1 &
echo TRAIN_PID=$!
"""
_, o, e = c.exec_command(script, timeout=60)
print(o.read().decode("utf-8", errors="replace"))
print(e.read().decode("utf-8", errors="replace"))
c.close()
