#!/usr/bin/env python3
"""Build a standalone executable for the optical trap calibration app."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "optical_trap_calibration.py"


def main() -> None:
    if not APP_PATH.exists():
        raise FileNotFoundError(f"Application source not found: {APP_PATH}")

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "OpticalTrapCalibration",
        "--noconsole",
        str(APP_PATH),
    ]

    if sys.platform == "darwin":
        cmd.insert(5, "--windowed")
    elif sys.platform.startswith("win"):
        cmd.insert(5, "--windowed")

    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print("\nStandalone package created in the dist/ directory.")


if __name__ == "__main__":
    main()
