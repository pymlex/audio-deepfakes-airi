import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

from config import CHECKPOINT_DIR, HF_MODEL_REPO, ROOT


def upload_model() -> None:
    """Upload best checkpoint to HuggingFace."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(HF_MODEL_REPO, repo_type="model", exist_ok=True)
    for ckpt in CHECKPOINT_DIR.glob("*.pt"):
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo=ckpt.name,
            repo_id=HF_MODEL_REPO,
            repo_type="model",
        )
        print(f"Uploaded {ckpt.name}")


if __name__ == "__main__":
    upload_model()
