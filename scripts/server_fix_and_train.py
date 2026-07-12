import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
script = """
cd /root/audio-deepfakes-airi
git pull --ff-only
.venv/bin/pip install torchcodec soundfile -q
PYTHONPATH=. .venv/bin/python scripts/setup_data.py
nohup env PYTHONPATH=. .venv/bin/python -u task1/4_2/main.py > train.log 2>&1 &
echo started
"""
_, o, _ = c.exec_command(script)
open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\server_fix.txt", "w", encoding="utf-8").write(o.read().decode("utf-8", errors="replace"))
c.close()
