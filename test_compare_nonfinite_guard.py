# -*- coding: utf-8 -*-
"""Checks that equivalence comparisons do not silently ignore NaN/Inf."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dev"))
import numpy as np
from compare import compare

base = {
    "obs": np.zeros((1, 4), dtype=np.float32),
    "action": np.zeros((1, 4), dtype=np.int64),
    "logp": np.zeros(1, dtype=np.float32),
    "value": np.zeros(1, dtype=np.float32),
    "reward": np.zeros(1, dtype=np.float32),
    "done": np.zeros(1, dtype=np.bool_),
    "adv": np.zeros(1, dtype=np.float32),
    "returns": np.zeros(1, dtype=np.float32),
    "ep": 0.0,
    "steps": 4,
    "metrics": {"blue_kills": 0.0},
}
assert compare([base], [base], label="finite-self-check")
bad = dict(base)
bad["obs"] = np.asarray([[np.nan, 0, 0, 0]], dtype=np.float32)
assert not compare([bad], [bad], label="nan-must-fail"), "NaN was incorrectly accepted as equivalent"
bad2 = dict(base)
bad2["metrics"] = {"blue_kills": float("inf")}
assert not compare([bad2], [bad2], label="inf-metric-must-fail"), "Inf metric was incorrectly accepted"
print("NON-FINITE COMPARISON GUARD PASSED")
