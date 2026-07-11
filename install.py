import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GITHUBLEX_URL = "https://github.com/pymlex/githublex.git"


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def pull_repository() -> None:
    run(["git", "pull", "--ff-only"], cwd=ROOT)


def ensure_env_file() -> None:
    import shutil

    env_path = ROOT / ".env"
    example_path = ROOT / ".env.example"
    if env_path.exists() or not example_path.exists():
        return
    shutil.copy(example_path, env_path)


def install_system_packages() -> None:
    import shutil

    if sys.platform.startswith("linux") and shutil.which("apt-get"):
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "git-lfs", "ffmpeg", "libsndfile1"])
        run(["git", "lfs", "install"])


def install_dependencies() -> None:
    venv_dir = ROOT / ".venv"
    if sys.platform.startswith("win"):
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)])
    run([str(venv_python), "-m", "pip", "install", "-U", "pip"])
    run([str(venv_python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            f"git+{GITHUBLEX_URL}",
        ]
    )


def venv_python() -> Path:
    if sys.platform.startswith("win"):
        return ROOT / ".venv" / "Scripts" / "python.exe"
    return ROOT / ".venv" / "bin" / "python"


def login_github() -> None:
    from githublex import gh_login, gh_setup

    gh_setup()
    gh_login()


def login_huggingface() -> None:
    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv(ROOT / ".env")
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        login(token=token)
        return
    login()


def setup_data() -> None:
    run([str(venv_python()), str(ROOT / "scripts" / "setup_data.py")])


def install() -> None:
    pull_repository()
    ensure_env_file()
    install_system_packages()
    install_dependencies()
    login_github()
    login_huggingface()
    setup_data()


if __name__ == "__main__":
    install()
