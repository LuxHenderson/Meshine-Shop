"""
Development runner with automatic restart on file changes.

This script watches the meshine_shop/ source directory for any file
modifications and automatically restarts the application when changes
are detected. This provides a workflow similar to "live reload" in
web development — save a file, see the result immediately.

Uses the `watchfiles` library (a fast Rust-based file watcher) instead
of polling-based alternatives, which means near-instant detection of
changes with minimal CPU usage.

Usage:
    cd desktop
    poetry run python scripts/dev.py

To stop: Ctrl+C in the terminal.
"""

import sys
import subprocess
from pathlib import Path
from watchfiles import run_process


def run_app():
    """Launch the Meshine Shop application as a subprocess.

    Using subprocess.run (rather than importing and calling main() directly)
    ensures a completely fresh Python process on each restart. This avoids
    issues with Qt's application singleton — QApplication can only be
    instantiated once per process, so restarting via subprocess is the
    cleanest way to handle hot-reloading a Qt app.
    """
    subprocess.run(
        [sys.executable, "-m", "meshine_shop"],
        cwd=Path(__file__).resolve().parent.parent,
    )


if __name__ == "__main__":
    # Resolve the source directory to watch — the meshine_shop package.
    src_dir = Path(__file__).resolve().parent.parent / "meshine_shop"
    print(f"Watching {src_dir} for changes...")
    print("The app will auto-restart when you save a file.\n")

    # run_process watches the directory and calls run_app() initially,
    # then kills and restarts it whenever a file change is detected.
    # The callback prints which files triggered the restart for debugging.
    run_process(
        src_dir,
        target=run_app,
        callback=lambda changes: print(f"\nFiles changed: {[str(c[1]) for c in changes]}"),
    )
