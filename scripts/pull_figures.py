import paramiko
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/audio-deepfakes-airi"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
_, o, _ = c.exec_command(f"find {REMOTE}/task5/outputs -name '*.png'")
pngs = [p.strip().replace(REMOTE + "/", "") for p in o.read().decode().splitlines() if p.strip()]
extra = [
    "task1/outputs/1_2/waveform_bonafide.png",
    "task1/outputs/1_2/waveform_spoof.png",
    "task4/outputs/uncertainty_comparison.png",
    "task4/outputs/uncertainty_scores.png",
    "task5/outputs/layer_probing.png",
]
files = list(dict.fromkeys(pngs + extra))
sftp = c.open_sftp()
for rel in files:
    local = ROOT / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(f"{REMOTE}/{rel}", str(local))
    print(rel)
sftp.close()
_, o, _ = c.exec_command("tail -3 /root/audio-deepfakes-airi/aasist_train.log; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
open(ROOT / "_train_poll.txt", "w", encoding="utf-8").write(o.read().decode("utf-8", errors="replace"))
c.close()
