import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmds = [
    "ls /root/audio-deepfakes-airi/data/",
    "ls /root/audio-deepfakes-airi/data/audio/ 2>/dev/null | head",
    "find /root/audio-deepfakes-airi/data -name '*.flac' | head -5",
    "head -3 /root/audio-deepfakes-airi/data/metadata/train.csv",
    "tail -50 /root/audio-deepfakes-airi/run.log",
]
for cmd in cmds:
    _, o, _ = c.exec_command(cmd)
    print(">>>", cmd)
    print(o.read().decode("utf-8", errors="replace"))
c.close()
