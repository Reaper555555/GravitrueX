"""CPU-safe smoke tests for the additive Experimental Learning Lab."""
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import gravitruex_ai as engine
import gravitruex_lab as lab


def main():
    assert lab._self_test() == 0

    # Reward profile is selectable and baseline values still match the old Stage 6.14 coefficients.
    old_mode = engine.TRAINING_REWARD_MODE
    try:
        engine.TRAINING_REWARD_MODE = "baseline"
        base = engine.training_reward_coefficients()
        assert math.isclose(base["damage_dealt"], engine.REWARD_DAMAGE_DEALT)
        assert math.isclose(base["damage_taken"], engine.REWARD_DAMAGE_TAKEN)
        assert math.isclose(base["kill"], engine.REWARD_PER_KILL)
        assert math.isclose(base["death"], engine.REWARD_DEATH)
        assert base["terminal_win"] == 24.0 and base["team_progress"] == 0.0
        engine.TRAINING_REWARD_MODE = "team"
        team = engine.training_reward_coefficients()
        assert team["terminal_win"] > base["terminal_win"]
        assert team["team_progress"] > 0 and team["kill"] < base["kill"]
    finally:
        engine.TRAINING_REWARD_MODE = old_mode
    print("  OK reward profiles: baseline preserved; team reward is distinct")

    # Exercise the actual tensor-batched path on real PyTorch CPU tensors. This is not a CUDA benchmark.
    from gravitruex_gpu_batch import GPUVectorArenaBatch
    old_prior = engine.ACTION_PRIORS_MODE
    try:
        engine.ACTION_PRIORS_MODE = "off"
        for arch in ("shared_residual", "separate_critic"):
            for profile in ("baseline", "team"):
                engine.TRAINING_REWARD_MODE = profile
                policy = engine.ActorCritic(seed=19, width=64, depth=2, device_mode="cpu", architecture=arch)
                batch = GPUVectorArenaBatch(
                    engine=engine, policy=policy, count=1, blue_n=1, red_n=1,
                    random_compositions=False, opponent_mode="heuristic", max_decisions=8,
                    device="cpu", seed=29, use_cuda_graph=False, max_records=1000,
                )
                out = batch.collect()
                assert out and len(out[0][0]) > 0
                assert np.isfinite(out[0][0].reward).all() and np.isfinite(out[0][0].value).all()
                print(f"  OK tensor batch {arch}/{profile}: {len(out[0][0])} records")
    finally:
        engine.ACTION_PRIORS_MODE = old_prior
        engine.TRAINING_REWARD_MODE = old_mode

    # Checkpoint schema is backward-safe for the standard architecture and refuses accidental critic mismatch.
    with tempfile.TemporaryDirectory(prefix="gravitruex-lab-checkpoint-") as td:
        path = Path(td) / "sample.npz"
        shared = engine.ActorCritic(seed=3, width=64, depth=2, device_mode="cpu")
        sample = np.zeros((2, engine.OBS_SIZE), dtype=np.float32)
        expected, _ = shared.forward(sample)
        shared.save(path)
        restored = engine.ActorCritic(seed=4, width=64, depth=2, device_mode="cpu")
        assert restored.load(path)
        actual, _ = restored.forward(sample)
        for key in expected:
            assert np.allclose(expected[key], actual[key])
        separate = engine.ActorCritic(seed=5, width=64, depth=2, device_mode="cpu", architecture="separate_critic")
        assert not separate.load(path)
    print("  OK checkpoint architecture metadata prevents silent mismatch")

    # Short smoke evaluations exercise the real scalar arena for each team size, both starting sides.
    policy = engine.ActorCritic(seed=7, width=64, depth=2, device_mode="cpu")
    prior_mode = engine.ACTION_PRIORS_MODE
    try:
        engine.ACTION_PRIORS_MODE = "off"
        for size in range(1, 6):
            row = lab.evaluate_policy(policy, size, seeds=1, max_decisions=8, seed_base=5000)
            assert row["team_size"] == size and row["games"] == 2
            assert row["wins"] + row["losses"] + row["unresolved"] == row["games"]
            assert math.isfinite(row["damage_difference"])
            print(f"  OK evaluation smoke {size}v{size}: {row['games']} games")
    finally:
        engine.ACTION_PRIORS_MODE = prior_mode

    # Planner should lower arena count as horizon increases, never lower horizon itself.
    env480 = lab.planned_env_count(5, 5, True, "selfplay", 360000, 480, 192)
    env600 = lab.planned_env_count(5, 5, True, "selfplay", 360000, 600, 192)
    assert env480 == 75 and env600 == 60
    print("EXPERIMENTAL LEARNING LAB TESTS PASSED")


if __name__ == "__main__":
    main()
