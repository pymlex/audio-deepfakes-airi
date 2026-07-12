import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmds = [
    "cat /root/audio-deepfakes-airi/task1/outputs/test_metrics.json",
    "cat /root/audio-deepfakes-airi/task2/outputs/sasv_metrics.json",
    "ls /root/audio-deepfakes-airi/task4/outputs/",
    "ls /root/audio-deepfakes-airi/checkpoints/",
    "nvidia-smi",
]
with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\final_status.txt", "w", encoding="utf-8") as f:
    for cmd in cmds:
        _, o, _ = c.exec_command(cmd, timeout=60)
        f.write(f">>> {cmd}\n{o.read().decode('utf-8', errors='replace')}\n\n")
c.close()
