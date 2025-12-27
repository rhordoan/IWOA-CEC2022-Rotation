"""
Backward-compatible wrapper.

Historically this repo used `benchmarks/run_cec2022.py` as the main entrypoint.
For publication-ready, deterministic experiments with structured outputs, use:

  python benchmarks/experiments.py --dims 20 --runs 30
"""

import os
import sys

# Ensure project root is importable when executing this file directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.experiments import main


if __name__ == "__main__":
    raise SystemExit(main())