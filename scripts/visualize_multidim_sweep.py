#!/usr/bin/env python3
"""
Backward-compatibility shim for scripts/visualize_multidim_sweep.py.
The actual implementation is now located at scripts/e_optimization/visualize_multidim_sweep.py.
"""
import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.e_optimization.visualize_multidim_sweep import *
from scripts.e_optimization.visualize_multidim_sweep import generate_multidim_analysis_pdf, main

if __name__ == "__main__":
    main()
