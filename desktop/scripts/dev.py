"""
Dev runner with auto-restart on file changes.

Usage:
    poetry run python scripts/dev.py
"""
import sys
import subprocess
from pathlib import Path
from watchfiles import run_process


def run_app():
    subprocess.run(
        [sys.executable, "-m", "meshine_shop"],
        cwd=Path(__file__).resolve().parent.parent,
    )


if __name__ == "__main__":
    src_dir = Path(__file__).resolve().parent.parent / "meshine_shop"
    print(f"Watching {src_dir} for changes...")
    print("The app will auto-restart when you save a file.\n")
    run_process(
        src_dir,
        target=run_app,
        callback=lambda changes: print(f"\nFiles changed: {[str(c[1]) for c in changes]}"),
    )
