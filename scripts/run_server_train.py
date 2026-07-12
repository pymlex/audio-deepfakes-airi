import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmds = [
    "cd /root/audio-deepfakes-airi && git pull --ff-only",
    "cd /root/audio-deepfakes-airi && PYTHONPATH=. .venv/bin/python scripts/setup_data.py",
    "ls /root/audio-deepfakes-airi/data/audio/flac_T | head -3",
    "cd /root/audio-deepfakes-airi && nohup env PYTHONPATH=. .venv/bin/python task1/4_2/main.py > train.log 2>&1 &",
]
for cmd in cmds:
    _, o, e = c.exec_command(cmd, get_pty=True)
    out = o.read().decode("utf-8", errors="replace")
    open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\server_run.txt", "a", encoding="utf-8").write(f">>> {cmd}\n{out}\n")
c.close()
