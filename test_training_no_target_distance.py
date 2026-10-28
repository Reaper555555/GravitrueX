# -*- coding: utf-8 -*-
"""Smoke test for reward learning without an explicit target-distance shaping term."""
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gravitruex_ai as G


def main():
    old_range = G.TRAINING_DESIRED_RANGE
    try:
        G.TRAINING_DESIRED_RANGE = 0.0
        env = G.Arena(seed=19028)
        env.reset(1, 1)
        policy = G.ActorCritic(seed=19028, width=32, depth=2, device_mode="cpu")
        result = G._collect_episode_rollout(
            policy, env, 1, 1, max_decisions=12, opponent_mode="heuristic"
        )
        records, reward, outcome, blue_n, red_n, physics_steps, metrics = result
        assert records, "No PPO records were collected with target-distance shaping disabled"
        assert math.isfinite(float(reward)), f"Non-finite total reward: {reward}"
        assert all(math.isfinite(float(record.get("reward", 0.0))) for record in records), "Non-finite per-record reward"
        assert physics_steps > 0, "No physics steps were completed"
        assert G.parse_decision_horizon("2700 (180 с)") == 2700
        print("NO-TARGET-DISTANCE TRAINING SMOKE PASSED")
    finally:
        G.TRAINING_DESIRED_RANGE = old_range


if __name__ == "__main__":
    main()
