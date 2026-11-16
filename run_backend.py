"""
run_backend.py — Launch the FastAPI backend on this machine.

Resolves paths dynamically from this file's location so it runs from any
platform (Linux, macOS, Windows) without modification.  On hosts where
Python's user-site directory is enabled, the launcher restarts itself with
``-s`` before importing the application so packages outside ``.venv`` cannot
override the project's tested dependencies.
"""
import os
import site
import sys
from pathlib import Path

# Project root is the directory that contains this file.
_PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_isolated_runtime() -> None:
    """Restart with user-site packages disabled before importing the app."""
    if site.ENABLE_USER_SITE is not True:
        return

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    argv = [
        sys.executable,
        "-s",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    os.execve(sys.executable, argv, env)
    raise RuntimeError("os.execve returned while isolating the Python runtime")


if __name__ == "__main__":
    _ensure_isolated_runtime()

# Make `backend` importable and run from the project root.
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

from backend.main import app  # noqa: E402  (intentional: after sys.path mutation)
from uvicorn import Config, Server  # noqa: E402

config = Config(app=app, host="127.0.0.1", port=8000, log_level="info")
Server(config).run()
