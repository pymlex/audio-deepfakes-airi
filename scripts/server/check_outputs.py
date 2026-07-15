import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmds = [
    "ls -la /root/audio-deepfakes-airi/task1/outputs/",
    "ls -la /root/audio-deepfakes-airi/task2/outputs/ 2>/dev/null",
    "ls -la /root/audio-deepfakes-airi/task4/outputs/ 2>/dev/null",
    "ls -la /root/audio-deepfakes-airi/task5/outputs/ 2>/dev/null",
    "tail -20 /root/audio-deepfakes-airi/task2.log 2>/dev/null",
    "tail -20 /root/audio-deepfakes-airi/task4.log 2>/dev/null",
    "tail -20 /root/audio-deepfakes-airi/task5.log 2>/dev/null",
]
with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\server_outputs.txt", "w", encoding="utf-8") as f:
    for cmd in cmds:
        _, o, _ = c.exec_command(cmd, timeout=60)
        f.write(f">>> {cmd}\n{o.read().decode('utf-8', errors='replace')}\n\n")
c.close()
