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
    for horizon in (240, 450, 480, 720, 900, 960, 1200, 1350, 1800, 2700, 3600, 5400):
        limit = start + horizon * repeat
        assert not G.visible_test_horizon_reached(start, limit - 1, horizon), (horizon, "early stop")
        assert G.visible_test_horizon_reached(start, limit, horizon), (horizon, "late stop")
        assert G.visible_test_horizon_reached(start, limit + 10, horizon), (horizon, "missed stop")
    assert not G.visible_test_horizon_reached(start, start + 10, 240)
    assert G.parse_decision_horizon("2700 (180 с)") == 2700
    assert G.parse_decision_horizon("5400 (360 с)") == 5400
    assert G.parse_decision_horizon("720") == 720  # legacy config
    assert G.parse_decision_horizon("invalid", 2700) == 2700
    assert G.GPU_RECORD_BUDGET_CHOICES[-3:] == ("1 440 000", "2 160 000", "2 880 000")
    print("VISIBLE BATTLE HORIZON PASSED: extended 16–360s presets, legacy config, exact ACTION_REPEAT cutoff")


if __name__ == "__main__":
    main()
