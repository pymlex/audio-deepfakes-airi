import json
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "/root/audio-deepfakes-airi"

REMOTE_FILES = [
    "task5/outputs/saliency_0.png",
    "task5/outputs/saliency_1.png",
    "task5/outputs/ig_0.png",
    "task5/outputs/ig_1.png",
    "task5/outputs/gradcam_0.png",
    "task5/outputs/gradcam_1.png",
    "task5/outputs/occlusion_0.png",
    "task5/outputs/occlusion_1.png",
    "task1/outputs/1_2/waveform_bonafide.png",
    "task1/outputs/1_2/waveform_spoof.png",
    "task1/outputs/6_1/score_all.png",
    "task1/outputs/6_1/score_wrong.png",
    "task1/outputs/6_1/misclassified.csv",
    "task4/outputs/uncertainty_comparison.png",
    "task4/outputs/uncertainty_scores.png",
    "task5/outputs/layer_probing.png",
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)
sftp = c.open_sftp()
for rel in REMOTE_FILES:
    remote = f"{REMOTE}/{rel}"
    local = ROOT / rel
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    print(f"pulled {rel}")
sftp.close()
c.close()
