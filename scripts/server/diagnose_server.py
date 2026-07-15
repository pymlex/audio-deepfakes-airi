import paramiko

HOST = "n1.us.clorecloud.net"
PORT = 1416
PASSWORD = "YdgQxoX8gQAz7eAI80"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username="root", password=PASSWORD, timeout=30)

cmds = [
    "nvidia-smi",
    "ps aux | grep python | grep -v grep",
    "ls -la /root/audio-deepfakes-airi/*.log 2>/dev/null",
    "wc -l /root/audio-deepfakes-airi/train.log /root/audio-deepfakes-airi/full_run.log 2>/dev/null",
    "tail -60 /root/audio-deepfakes-airi/train.log 2>/dev/null",
    "tail -40 /root/audio-deepfakes-airi/full_run.log 2>/dev/null",
    "cd /root/audio-deepfakes-airi && .venv/bin/python -c \"import torch; print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')\"",
    "test -f /root/audio-deepfakes-airi/task1/outputs/test_metrics.json && cat /root/audio-deepfakes-airi/task1/outputs/test_metrics.json || echo NO_METRICS",
    "ls /root/audio-deepfakes-airi/data/audio/flac_T 2>/dev/null | head -3",
    "ls /root/audio-deepfakes-airi/checkpoints/ 2>/dev/null",
]

out_path = r"C:\Users\aleks\Downloads\audio-deepfakes-airi\diagnose_out.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for cmd in cmds:
        f.write(f"\n{'='*60}\n>>> {cmd}\n{'='*60}\n")
        _, o, e = c.exec_command(cmd, timeout=60)
        f.write(o.read().decode("utf-8", errors="replace"))
        err = e.read().decode("utf-8", errors="replace")
        if err:
            f.write(f"STDERR: {err}\n")
c.close()
print("done")
