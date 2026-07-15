import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmds = [
    "wc -l /root/audio-deepfakes-airi/train.log 2>/dev/null",
    "cat /root/audio-deepfakes-airi/train.log 2>/dev/null",
    "ps aux | grep 'task1/1_4_2/main.py' | grep -v grep",
    "ls -la /root/audio-deepfakes-airi/checkpoints/ 2>/dev/null",
    "ls -la /root/audio-deepfakes-airi/task1/outputs/ 2>/dev/null",
]
with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\train_status.txt", "w", encoding="utf-8") as f:
    for cmd in cmds:
        _, o, e = c.exec_command(cmd)
        f.write(f">>> {cmd}\n")
        f.write(o.read().decode("utf-8", errors="replace"))
        f.write(e.read().decode("utf-8", errors="replace"))
        f.write("\n")
c.close()
