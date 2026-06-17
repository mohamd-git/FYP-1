"""pytest configuration: make `src` importable and seed RNGs for determinism."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

random.seed(0)
try:
    import numpy as np

    np.random.seed(0)
except Exception:
    pass
