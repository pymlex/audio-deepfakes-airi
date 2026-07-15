import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmds = [
    "nvidia-smi",
    "ps aux | grep python | grep -v grep",
    "tail -25 /root/audio-deepfakes-airi/task2.log",
    "ls -la /root/audio-deepfakes-airi/task2/outputs/",
]
with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\gpu_now.txt", "w", encoding="utf-8") as f:
    for cmd in cmds:
        _, o, _ = c.exec_command(cmd, timeout=60)
        f.write(f">>> {cmd}\n{o.read().decode('utf-8', errors='replace')}\n\n")
c.close()
