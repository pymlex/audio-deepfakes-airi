import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("n1.us.clorecloud.net", port=1416, username="root", password="YdgQxoX8gQAz7eAI80", timeout=30)

cmd = """
cd /root/audio-deepfakes-airi
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
from config import DATA_ROOT, METADATA_DIR
from utils.data import make_sasv_df, _is_valid_audio

df = pd.read_csv(METADATA_DIR / "test_4k-track_2.csv")
df["reference_audio_path"] = df["reference_audio_path"].apply(lambda p: str(DATA_ROOT / p))
df["query_audio_path"] = df["query_audio_path"].apply(lambda p: str(DATA_ROOT / p))
print("total", len(df))
valid_ref = df["reference_audio_path"].apply(lambda p: _is_valid_audio(Path(p)))
valid_q = df["query_audio_path"].apply(lambda p: _is_valid_audio(Path(p)))
print("valid ref", valid_ref.sum(), "valid query", valid_q.sum(), "both", (valid_ref & valid_q).sum())

import soundfile as sf
bad = []
for i, row in df[valid_ref & valid_q].head(200).iterrows():
    for col in ["reference_audio_path", "query_audio_path"]:
        p = row[col]
        info = sf.info(p)
        if info.frames < 100:
            bad.append(p)
print("soundfile ok on first 200 both-valid:", 200 - len(bad)//2)
for i, row in df[valid_ref & valid_q].iterrows():
    for col in ["reference_audio_path", "query_audio_path"]:
        p = row[col]
        info = sf.info(p)
        if info.frames < 100:
            bad.append(p)
            if len(bad) >= 5:
                break
    if len(bad) >= 5:
        break
print("bad samples:", bad[:5])
PY
"""
_, o, e = c.exec_command(cmd, timeout=120)
print(o.read().decode())
print(e.read().decode())
c.close()
