from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(__import__("os").environ.get("DATA_ROOT", ROOT / "data"))


def _resolve_metadata_dir() -> Path:
    candidates = [
        DATA_ROOT / "metadata",
        DATA_ROOT / "metadata" / "metadata",
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists():
            return candidate
    return DATA_ROOT / "metadata"


METADATA_DIR = _resolve_metadata_dir()
AUDIO_DIR = DATA_ROOT / "audio"

TRAIN_CSV = METADATA_DIR / "train.csv"
TEST_TRACK1_CSV = METADATA_DIR / "test_track_1.csv"
TEST_SASV_CSV = METADATA_DIR / "test_4k-track_2.csv"

HF_DATASET_REPO = "pymlex/audio-deepfakes-airi"
HF_MODEL_REPO = "pymlex/audio-deepfakes-airi"
GITHUB_REPO = "pymlex/audio-deepfakes-airi"

DEFAULT_SR = 16000
PADDING_SEC = 4.0
N_MELS = 80
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

TASK1_OUTPUT = ROOT / "task1" / "outputs"
TASK2_OUTPUT = ROOT / "task2" / "outputs"
TASK4_OUTPUT = ROOT / "task4" / "outputs"
TASK5_OUTPUT = ROOT / "task5" / "outputs"

CHECKPOINT_DIR = ROOT / "checkpoints"
