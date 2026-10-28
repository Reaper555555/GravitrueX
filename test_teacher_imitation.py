# -*- coding: utf-8 -*-
"""CPU regression test for heuristic-teacher labels and auxiliary PPO loss."""
from __future__ import annotations
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import gravitruex_ai as G


def main():
    policy = G.ActorCritic(seed=1907, width=32, depth=2, device_mode="cpu")
    env = G.Arena(seed=1908)
    env.reset(1, 1, seed=1908)
    records, _, _, _, _, steps, _ = G._collect_episode_rollout(
        policy, env, 1, 1, max_decisions=16, opponent_mode="heuristic")
    assert steps > 0 and records
    assert any(bool(r.get("teacher_mask")) for r in records), "heuristic-mode CPU rollouts must attach teacher targets"
    for r in records:
        if r.get("teacher_mask"):
            target = np.asarray(r["teacher_action"])
            assert target.shape == (4,)
            assert np.all(target[:3] >= 0) and np.all(target[:3] <= 2)
            assert int(target[3]) in (0, 1)
    packed = G.PackedExperience.from_records(records)
    assert packed.teacher_action.shape == packed.action.shape
    assert bool(packed.teacher_mask.any())
    loss = policy.update_ppo(packed, epochs=2, batch_size=32, device="cpu", teacher_coef=0.08)
    assert math.isfinite(loss)
    assert policy.last_imitation_coef == 0.08
    assert math.isfinite(policy.last_imitation_loss) and policy.last_imitation_loss > 0

    # Self-play uses the heuristic only as an annealed warm-start label, while both
    # sides remain policy-controlled and contribute PPO records.
    env2 = G.Arena(seed=1909)
    env2.reset(1, 1, seed=1909)
    selfplay_records, *_ = G._collect_episode_rollout(
        policy, env2, 1, 1, max_decisions=8, opponent_mode="selfplay")
    assert selfplay_records and all(bool(r.get("teacher_mask")) for r in selfplay_records)
    assert abs(G.teacher_coef_for_update(0, "selfplay") - 0.12) < 1e-8
    assert abs(G.teacher_coef_for_update(300, "selfplay") - 0.08) < 1e-8
    assert abs(G.teacher_coef_for_update(600, "selfplay") - 0.04) < 1e-8
    assert G.teacher_coef_for_update(900, "selfplay") == 0.0
    assert G.teacher_coef_for_update(0, "model") == 0.0
    assert G.shot_reward_value(0.90, 0.50) > 0.0
    assert G.shot_reward_value(0.0, 0.50) < 0.0

    # The tensor-vectorized path must preserve teacher labels through packing too.
    import torch
    GPUVectorArenaBatch = G._load_gpu_batch_class()
    batch_policy = G.ActorCritic(seed=1910, width=32, depth=2, device_mode="cpu")
    batch = GPUVectorArenaBatch(engine=sys.modules["gravitruex_ai"], policy=batch_policy,
                                count=2, blue_n=1, red_n=1, opponent_mode="heuristic",
                                max_decisions=8, device="cpu", seed=1910,
                                use_cuda_graph=False, max_records=1000)
    results = batch.collect()
    assert len(results) == 2
    for rollout in results:
        assert 0.0 <= float(rollout[6].get("policy_fire_rate", -1.0)) <= 1.0
        assert 0.0 <= float(rollout[6].get("teacher_fire_rate", -1.0)) <= 1.0
    xp = G.PackedExperience.concatenate([row[0] for row in results])
    assert len(xp) > 0 and bool(xp.teacher_mask.any()), "GPU-batch path lost the teacher mask"
    loss2 = batch_policy.update_ppo(xp, epochs=1, batch_size=32, device="cpu", teacher_coef=0.06)
    assert math.isfinite(loss2) and batch_policy.last_imitation_loss > 0
    # Self-play tensor batches must carry heuristic labels for BOTH teams but still
    # sample policy actions on both teams.
    sp_batch = GPUVectorArenaBatch(engine=sys.modules["gravitruex_ai"], policy=batch_policy,
                                   count=2, blue_n=1, red_n=1, opponent_mode="selfplay",
                                   max_decisions=8, device="cpu", seed=1911,
                                   use_cuda_graph=False, max_records=1000)
    sp_results = sp_batch.collect()
    sp_exp = G.PackedExperience.concatenate([row[0] for row in sp_results])
    assert len(sp_exp) > 0 and bool(sp_exp.teacher_mask.any())
    print("TEACHER/COMBAT PASS: CPU + tensor-batch labels; self-play bootstrap labels both sides; fire reward; PPO auxiliary loss finite")


if __name__ == "__main__":
    main()
