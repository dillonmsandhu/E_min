#!/usr/bin/env python3
"""
Backward-compatibility shim for scripts/sweep_pipeline.py.
The actual implementation is now located at scripts/pipeline/sweep_pipeline.py.
"""
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.pipeline.sweep_pipeline import *
from scripts.pipeline.sweep_pipeline import ALGO_REGISTRY, main

if __name__ == "__main__":
    main()
