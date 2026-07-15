import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
cmds = [
    "ls /root/audio-deepfakes-airi/task5/outputs/ 2>/dev/null | head -20",
    "ls /root/audio-deepfakes-airi/task1/outputs/1_2/ 2>/dev/null",
    "ls /root/audio-deepfakes-airi/task1/outputs/6_1/ 2>/dev/null",
    "ls /root/audio-deepfakes-airi/task4/outputs/ 2>/dev/null",
    "cat /root/audio-deepfakes-airi/checkpoints/*.pt 2>/dev/null | wc -c",
    "ls -la /root/audio-deepfakes-airi/checkpoints/",
]
with open(r"C:\Users\aleks\Downloads\audio-deepfakes-airi\_server_check.txt", "w", encoding="utf-8") as f:
    for cmd in cmds:
        _, o, _ = c.exec_command(cmd)
        f.write(f">>> {cmd}\n{o.read().decode('utf-8','replace')}\n")
c.close()
