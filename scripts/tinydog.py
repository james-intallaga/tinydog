#!/usr/bin/env python3
"""Run the source-tree bootstrap without requiring prior installation."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_arithmetic.bootstrap import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
