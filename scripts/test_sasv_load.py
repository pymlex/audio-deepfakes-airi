import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)

cmd = """
cd /root/audio-deepfakes-airi
.venv/bin/python - <<'PY'
from config import DATA_ROOT, METADATA_DIR
from utils.data import make_sasv_df
from data.dataset import SASVTrialDataset
from utils.metrics import load_waveform

df = make_sasv_df(METADATA_DIR, DATA_ROOT)
print("rows", len(df))
ds = SASVTrialDataset(df)
for i in range(len(ds)):
    ref, query, label = ds[i]
    if i % 100 == 0:
        print("ok", i, label.item())
print("all ok", len(ds))
PY
"""
_, o, e = c.exec_command(cmd, timeout=300)
print(o.read().decode())
err = e.read().decode()
if err:
    print("STDERR:", err)
c.close()
