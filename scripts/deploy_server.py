"""Deploy and run pipeline on remote GPU server."""

import argparse
import os
import sys

import paramiko


def run_remote(host: str, port: int, password: str, commands: list[str]) -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, port=port, username="root", password=password, timeout=30)
    for cmd in commands:
        print(f">>> {cmd}")
        stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
        for line in stdout:
            text = line if isinstance(line, str) else line.decode("utf-8", errors="replace")
            sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        err = stderr.read().decode()
        if err:
            print(err, file=sys.stderr)
        if stdout.channel.recv_exit_status() != 0:
            client.close()
            raise RuntimeError(f"Command failed: {cmd}")
    client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="n1.us.clorecloud.net")
    parser.add_argument("--port", type=int, default=1416)
    parser.add_argument("--password", default=os.environ.get("SSH_PASSWORD", ""))
    parser.add_argument("--task", default="all")
    args = parser.parse_args()
    hf_token = os.environ.get("HF_TOKEN", "")
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    setup_cmds = [
        "apt-get update -qq && apt-get install -y -qq python3.12-venv git-lfs ffmpeg libsndfile1 2>/dev/null || true",
        "git lfs install 2>/dev/null || true",
        "cd /root && rm -rf audio-deepfakes-airi",
        "cd /root && git clone https://github.com/pymlex/audio-deepfakes-airi.git",
        f"cd /root/audio-deepfakes-airi && echo 'HF_TOKEN={hf_token}' > .env",
        f"cd /root/audio-deepfakes-airi && echo 'GITHUB_TOKEN={gh_token}' >> .env",
        "cd /root/audio-deepfakes-airi && python3 -m venv .venv",
        "cd /root/audio-deepfakes-airi && .venv/bin/pip install -U pip -q",
        "cd /root/audio-deepfakes-airi && .venv/bin/pip install -r requirements.txt -q",
        "cd /root/audio-deepfakes-airi && PYTHONPATH=. .venv/bin/python scripts/setup_data.py",
    ]
    run_cmds = [
        "cd /root/audio-deepfakes-airi && nohup env PYTHONPATH=. .venv/bin/python main.py --task all > run.log 2>&1 & echo $!",
    ]
    run_remote(args.host, args.port, args.password, setup_cmds + run_cmds)


if __name__ == "__main__":
    main()
