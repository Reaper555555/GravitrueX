# -*- coding: utf-8 -*-
"""Regression test for built-in policy variants, checkpoint identity, and PPO updates."""
from __future__ import annotations
import tempfile
import hashlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G


def run_policy_update(architecture: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    policy = G.ActorCritic(seed=seed, width=32, depth=2, device_mode="cpu", architecture=architecture)
    observations = rng.normal(0.0, 0.15, size=(16, G.OBS_SIZE)).astype(np.float32)
    logits, values = policy.forward(observations)
    assert all(np.isfinite(np.asarray(logits[k])).all() for k in ("turn", "thrust", "strafe", "fire"))
    assert values is not None and values.shape == (16,) and np.isfinite(values).all()
    _, logps, values, actions = policy.predict_batch(observations, deterministic=False)
    records = []
    for i in range(len(observations)):
        records.append({
            "obs": observations[i], "action": actions[i], "logp": float(logps[i]),
            "return": float(rng.normal()), "adv": float(rng.normal()), "value": float(values[i]),
            "teacher_action": actions[i], "teacher_mask": False,
        })
    loss = policy.update_ppo(records, epochs=1, batch_size=16, device="cpu", grad_diagnostics=True)
    assert np.isfinite(loss), f"{architecture}: nonfinite PPO loss {loss}"
    assert policy.last_optimizer_steps >= 1, f"{architecture}: optimizer did not step"
    assert np.isfinite(policy.last_policy_grad_norm) and np.isfinite(policy.last_value_grad_norm)
    with tempfile.TemporaryDirectory(prefix="gravitruex-variant-test-") as tmp:
        path = Path(tmp) / f"{architecture}.npz"
        policy.save(path)
        assert G.checkpoint_network_type(path) == architecture
        loaded = G.ActorCritic(seed=seed + 1, width=32, depth=2, device_mode="cpu", architecture=architecture)
        assert loaded.load(path), f"{architecture}: own checkpoint failed to load"
        other = "separate_critic" if architecture == "shared_residual" else "shared_residual"
        incompatible = G.ActorCritic(seed=seed + 2, width=32, depth=2, device_mode="cpu", architecture=other)
        assert not incompatible.load(path), f"checkpoint architecture mismatch unexpectedly loaded: {other}"


def main() -> None:
    # Point model helpers at an isolated temporary directory. This never touches
    # user checkpoints or the regular project's GravitrueX/models directory.
    with tempfile.TemporaryDirectory(prefix="gravitruex-model-catalog-") as tmp:
        old_root = G.MODEL_ROOT
        try:
            G.MODEL_ROOT = Path(tmp) / "models"
            G.ensure_builtin_models()
            expected = {
                "builtin_simple": "shared_residual",
                "builtin_team": "separate_critic",
                "builtin_split_critic": "separate_critic",
            }
            for model_id, architecture in expected.items():
                weight = G.model_weights_path(model_id)
                assert weight.exists(), f"missing built-in weights: {model_id}"
                meta = G.read_model_meta(model_id)
                assert meta.get("architecture") == architecture, (model_id, meta.get("architecture"))
                assert meta.get("default_reward_mode") in {"baseline", "team"}
                check = G.ActorCritic(seed=8, width=meta["width"], depth=meta["depth"],
                                      device_mode="cpu", architecture=architecture)
                assert check.load(weight), f"built-in checkpoint does not match declared architecture: {model_id}"
            assert G.read_model_meta("builtin_simple")["default_reward_mode"] == "baseline"
            assert G.read_model_meta("builtin_team")["default_reward_mode"] == "team"
            simple_path = G.model_weights_path("builtin_simple")
            before = hashlib.sha256(simple_path.read_bytes()).hexdigest()
            G.write_model_meta("builtin_simple", {"custom_note": "preserve-me"})
            G.ensure_builtin_models()
            after = hashlib.sha256(simple_path.read_bytes()).hexdigest()
            assert before == after, "startup overwrote existing built-in weights"
            assert G.read_model_meta("builtin_simple").get("custom_note") == "preserve-me"
        finally:
            G.MODEL_ROOT = old_root

    run_policy_update("shared_residual", 3101)
    run_policy_update("separate_critic", 3102)
    # Exercise the actual vector-tensor rollout path for the new split critic on
    # the CPU tensor backend. This covers its policy+value tensor branches without
    # pretending CUDA was available on the build machine.
    old_priors, old_reward = G.ACTION_PRIORS_MODE, G.TRAINING_REWARD_MODE
    try:
        G.ACTION_PRIORS_MODE = "soft"
        G.TRAINING_REWARD_MODE = "team"
        split_policy = G.ActorCritic(seed=3103, width=32, depth=2, device_mode="cpu",
                                     architecture="separate_critic")
        batch_cls = G._load_gpu_batch_class()
        batch = batch_cls(engine=G, policy=split_policy, count=2, blue_n=1, red_n=1,
                          opponent_mode="selfplay", max_decisions=8, device="cpu", seed=3104,
                          use_cuda_graph=False, max_records=10_000)
        rollout = batch.collect()
        assert len(rollout) == 2 and all(len(item[0]) > 0 for item in rollout),             "separate critic failed CPU tensor-batch rollout"
        assert all(np.isfinite(float(item[1])) for item in rollout)
    finally:
        G.ACTION_PRIORS_MODE, G.TRAINING_REWARD_MODE = old_priors, old_reward
    old_reward_mode = G.TRAINING_REWARD_MODE
    try:
        G.TRAINING_REWARD_MODE = "baseline"
        baseline_win = G.training_reward_coefficients()["terminal_win"]
        G.TRAINING_REWARD_MODE = "team"
        team_win = G.training_reward_coefficients()["terminal_win"]
        assert team_win > baseline_win, "team reward profile does not prioritize team victory"
    finally:
        G.TRAINING_REWARD_MODE = old_reward_mode
    print("MODEL VARIANTS PASSED: built-ins, non-destructive startup, checkpoint identity, PPO updates, split-critic tensor rollout, team reward profile")


if __name__ == "__main__":
    main()
