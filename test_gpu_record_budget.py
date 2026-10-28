# -*- coding: utf-8 -*-
"""Check GPU rollout record cap uses the actual team composition."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G


def make_batch(count: int, blue: int, red: int, mode: str, horizon: int = 480,
               random_compositions: bool = False, max_records: int = 120_000, seed: int = 77):
    policy = G.ActorCritic(seed=100 + count + blue + red, width=32, depth=2, device_mode="cpu")
    batch_cls = G._load_gpu_batch_class()
    batch = batch_cls(engine=G, policy=policy, count=count, blue_n=blue, red_n=red,
                      random_compositions=random_compositions,
                      opponent_mode=mode, max_decisions=horizon, device="cpu", seed=seed,
                      use_cuda_graph=False, max_records=max_records)
    return batch


class StopAfterChecks:
    """A tiny stand-in for threading.Event that interrupts long tests deterministically."""
    def __init__(self, checks: int):
        self.checks = int(checks)
        self.calls = 0

    def is_set(self) -> bool:
        self.calls += 1
        return self.calls > self.checks


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
    # The preflight calculation must match the exact seeded composition sampling
    # performed by GPUVectorArenaBatch, otherwise OOM retries can repeat the same
    # overlarge effective tensor shape many times.
    for mode, trained_slots in (("heuristic", 1), ("selfplay", 2)):
        batch = make_batch(100, 1, 1, mode, horizon=480, random_compositions=True,
                           max_records=120_000, seed=77)
        expected = G.gpu_record_budget_limited_envs(
            100, 1, 1, True, mode, 480, 120_000, 77
        )
        if batch.E != expected:
            raise AssertionError(f"{mode} random composition budget estimate {expected} != batch E {batch.E}")
        expected_slots = int(sum(batch.blue_counts)) + (
            int(sum(batch.red_counts)) if mode == "selfplay" else 0
        )
        if batch.record_train_slots != expected_slots:
            raise AssertionError(
                f"{mode} compact record slots {batch.record_train_slots} != present trainable ships {expected_slots}"
            )

    # Early-ending arenas should not reserve the full max_decisions horizon.
    dynamic = make_batch(2, 2, 2, "heuristic", horizon=720, max_records=10_000)
    results = dynamic.collect(StopAfterChecks(300))
    if not 0 < dynamic.last_record_buffer_capacity < dynamic.max_decisions:
        raise AssertionError(
            f"dynamic trajectory buffers should stay below full horizon: "
            f"{dynamic.last_record_buffer_capacity}/{dynamic.max_decisions}"
        )
    if dynamic.record_slots != 2 or dynamic.obs_slots.numel() != 2:
        raise AssertionError(
            f"heuristic 2v2 should observe/store two blue slots, got "
            f"record_slots={dynamic.record_slots}, obs_slots={dynamic.obs_slots.numel()}"
        )
    if dynamic.record_train_slots != 4:
        raise AssertionError(f"two 2v2 arenas should have four trainable ship columns, got {dynamic.record_train_slots}")
    if not results or not all(len(row[0]) > 0 for row in results):
        raise AssertionError("dynamic storage returned empty training experiences")

    print("GPU RECORD BUDGET / DYNAMIC STORAGE PASSED")
    for batch, slots, expected_horizon, label in cases:
        print(f"{label}: {batch.E}/{batch.requested_count} arenas, {batch.max_decisions} decisions, upper-bound records {batch.max_decisions*batch.E*slots:,}")
    print(f"dynamic buffer: {dynamic.last_record_buffer_capacity}/{dynamic.max_decisions} decisions, "
          f"{dynamic.record_slots} trainable slots, {dynamic.last_record_store_bytes:,} bytes")


if __name__ == "__main__":
    main()
