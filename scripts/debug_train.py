import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmd = "cd /root/audio-deepfakes-airi && PYTHONPATH=. .venv/bin/python task1/4_2/main.py 2>&1 | head -80"
_, o, e = c.exec_command(cmd, timeout=120)
out = o.read().decode("utf-8", errors="replace")
open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\train_debug.txt", "w", encoding="utf-8").write(out)
c.close()
