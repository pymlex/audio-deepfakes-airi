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


def cleanup_extract_artifacts() -> None:
    """Remove macOS metadata folders from extracted archives."""
    import shutil

    macosx = DATA_ROOT / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)
    for hidden in DATA_ROOT.rglob("._*"):
        hidden.unlink(missing_ok=True)


def normalize_audio_layout() -> None:
    """Move extracted archives under data/audio to match CSV paths."""
    audio_root = DATA_ROOT / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    for name in ["flac_T", "last_eval"]:
        src = DATA_ROOT / name
        dst = audio_root / name
        if src.exists() and not dst.exists():
            src.rename(dst)
        elif src.exists() and dst.exists():
            for item in src.iterdir():
                target = dst / item.name
                if target.exists():
                    continue
                item.rename(target)
            import shutil

            shutil.rmtree(src)


def normalize_metadata_layout() -> None:
    """Flatten nested metadata directory."""
    meta_dir = DATA_ROOT / "metadata"
    nested = meta_dir / "metadata"
    if nested.exists() and (nested / "train.csv").exists():
        meta_dir.mkdir(parents=True, exist_ok=True)
        for csv_file in nested.glob("*.csv"):
            target = meta_dir / csv_file.name
            if target.exists():
                target.unlink()
            csv_file.rename(target)


def download_archives() -> None:
    """Download and extract dataset archives from HuggingFace."""
    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_dir = DATA_ROOT / "metadata"
    audio_train = DATA_ROOT / "audio" / "flac_T"
    audio_eval = DATA_ROOT / "audio" / "last_eval"
    normalize_metadata_layout()
    cleanup_extract_artifacts()
    normalize_audio_layout()
    if (
        (metadata_dir / "train.csv").exists()
        and audio_train.exists()
        and audio_eval.exists()
    ):
        print("Data already present, skipping download.")
        return
    for filename in ["metadata.zip", "flac_T.zip", "last_eval.zip"]:
        print(f"Downloading {filename}...")
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            token=token,
        )
        with zipfile.ZipFile(path, "r") as zf:
            for member in tqdm(zf.namelist(), desc=f"extract {filename}"):
                if "__MACOSX" in member or member.endswith(".DS_Store"):
                    continue
                zf.extract(member, DATA_ROOT)
    normalize_metadata_layout()
    cleanup_extract_artifacts()
    normalize_audio_layout()


if __name__ == "__main__":
    download_archives()
