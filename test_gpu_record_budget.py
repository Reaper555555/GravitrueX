# -*- coding: utf-8 -*-
"""Check GPU rollout record cap uses the actual team composition."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G


def make_batch(count: int, blue: int, red: int, mode: str, horizon: int = 480):
    policy = G.ActorCritic(seed=100 + count + blue + red, width=32, depth=2, device_mode="cpu")
    batch_cls = G._load_gpu_batch_class()
    batch = batch_cls(engine=G, policy=policy, count=count, blue_n=blue, red_n=red,
                      opponent_mode=mode, max_decisions=horizon, device="cpu", seed=77,
                      use_cuda_graph=False, max_records=120_000)
    return batch


def main():
    h_1v1 = make_batch(100, 1, 1, "heuristic")
    if h_1v1.max_decisions != 480:
        raise AssertionError(f"1v1 heuristic should retain full 480-decision horizon, got {h_1v1.max_decisions}")
    h_5v5 = make_batch(100, 5, 5, "heuristic")
    if h_5v5.max_decisions != 480:
        raise AssertionError(f"5v5 heuristic should retain full 480-decision horizon, got {h_5v5.max_decisions}")
    if h_5v5.E >= h_5v5.requested_count:
        raise AssertionError("5v5 should reduce arena count, not silently shorten the selected horizon")
    s_1v1 = make_batch(100, 1, 1, "selfplay")
    long_5v5 = make_batch(100, 5, 5, "heuristic", horizon=720)
    if long_5v5.max_decisions != 720:
        raise AssertionError(f"5v5 should retain requested 720-decision horizon, got {long_5v5.max_decisions}")
    if long_5v5.E >= long_5v5.requested_count:
        raise AssertionError("720-decision 5v5 should reduce arena count under record budget")
    if s_1v1.max_decisions != 480:
        raise AssertionError(f"1v1 self-play should retain full 480-decision horizon, got {s_1v1.max_decisions}")
    # Upper bound on emitted records remains within the requested cap for each case.
    cases = ((h_1v1, 1, 480, "heuristic 1v1"), (h_5v5, 5, 480, "heuristic 5v5"),
             (s_1v1, 2, 480, "self-play 1v1"), (long_5v5, 5, 720, "heuristic 5v5, horizon 720"))
    for batch, slots, expected_horizon, label in cases:
        upper_bound = batch.max_decisions * batch.E * slots
        if upper_bound > 120_000:
            raise AssertionError(f"{label}: record cap exceeded: {upper_bound}")
        if batch.max_decisions != expected_horizon:
            raise AssertionError(f"{label}: requested horizon {expected_horizon} was changed to {batch.max_decisions}")
    print("GPU RECORD BUDGET PASSED")
    for batch, slots, expected_horizon, label in cases:
        print(f"{label}: {batch.E}/{batch.requested_count} arenas, {batch.max_decisions} decisions, upper-bound records {batch.max_decisions*batch.E*slots:,}")


if __name__ == "__main__":
    main()
