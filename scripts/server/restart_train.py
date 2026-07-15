import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmd = "cd /root/audio-deepfakes-airi && git pull --ff-only && nohup env PYTHONPATH=. .venv/bin/python -u task1/1_4_2/main.py > train.log 2>&1 & echo ok"
_, o, _ = c.exec_command(cmd)
print(o.read().decode())
c.close()
