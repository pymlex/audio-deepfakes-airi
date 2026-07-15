import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmds = [
    "nvidia-smi",
    "ps aux | grep 'task1/4_2_main.py' | grep -v grep",
    "tail -20 /root/audio-deepfakes-airi/train.log",
    "test -f /root/audio-deepfakes-airi/task1/outputs/test_metrics.json && cat /root/audio-deepfakes-airi/task1/outputs/test_metrics.json",
]
with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\gpu_status.txt", "w", encoding="utf-8") as f:
    for cmd in cmds:
        _, o, _ = c.exec_command(cmd, timeout=30)
        f.write(f">>> {cmd}\n")
        f.write(o.read().decode("utf-8", errors="replace"))
        f.write("\n\n")
c.close()
