#!/usr/bin/env python3
"""Run create_ground_truth (adds src to path for development runs)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nd2_analyzer.create_ground_truth import main

if __name__ == "__main__":
    sys.exit(main())
