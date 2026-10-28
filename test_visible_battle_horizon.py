# -*- coding: utf-8 -*-
"""Test that visible fight checks respect a configurable policy-decision horizon."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G


def main():
    start = 1000
    repeat = G.ACTION_REPEAT
    for horizon in (240, 480, 720, 960, 1200):
        limit = start + horizon * repeat
        assert not G.visible_test_horizon_reached(start, limit - 1, horizon), (horizon, "early stop")
        assert G.visible_test_horizon_reached(start, limit, horizon), (horizon, "late stop")
        assert G.visible_test_horizon_reached(start, limit + 10, horizon), (horizon, "missed stop")
    assert not G.visible_test_horizon_reached(start, start + 10, 240)
    print("VISIBLE BATTLE HORIZON PASSED: 240/480/720/960/1200 decisions, exact ACTION_REPEAT cutoff")


if __name__ == "__main__":
    main()
