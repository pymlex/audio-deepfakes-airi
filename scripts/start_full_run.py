import paramiko

HOST = "n1.us.clorecloud.net"
PORT = 1416
PASSWORD = "YdgQxoX8gQAz7eAI80"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)

script = """
cd /root/audio-deepfakes-airi
git pull --ff-only
PYTHONPATH=. .venv/bin/python scripts/setup_data.py
nohup env PYTHONPATH=. .venv/bin/python -u main.py --task all > full_run.log 2>&1 &
echo started_pid=$!
"""

_, o, _ = c.exec_command(script)
print(o.read().decode("utf-8", errors="replace"))
c.close()
