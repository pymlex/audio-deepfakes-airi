import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.auto import tqdm

from config import DATA_ROOT, HF_DATASET_REPO, ROOT


def download_archives() -> None:
    """Download and extract dataset archives from HuggingFace."""
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_dir = DATA_ROOT / "metadata"
    audio_dir = DATA_ROOT / "audio"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["metadata.zip", "flac_T.zip", "last_eval.zip"]:
        print(f"Downloading {filename}...")
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
        extract_root = DATA_ROOT if filename != "metadata.zip" else DATA_ROOT
        with zipfile.ZipFile(path, "r") as zf:
            for member in tqdm(zf.namelist(), desc=f"extract {filename}"):
                zf.extract(member, extract_root)
    meta_zip_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="metadata.zip",
        repo_type="dataset",
        token=token,
    )
    with zipfile.ZipFile(meta_zip_path, "r") as zf:
        zf.extractall(DATA_ROOT)
    meta_dir = DATA_ROOT / "metadata"
    nested = meta_dir / "metadata"
    if nested.exists() and (nested / "train.csv").exists():
        meta_dir.mkdir(parents=True, exist_ok=True)
        for csv_file in nested.glob("*.csv"):
            target = meta_dir / csv_file.name
            if target.exists():
                target.unlink()
            csv_file.rename(target)


if __name__ == "__main__":
    download_archives()
