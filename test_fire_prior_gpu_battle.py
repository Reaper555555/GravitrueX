"""Regression checks for CPU/GPU fire-prior parity and strict GPU-only visible inference."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import gravitruex_ai as G


def main():
    # A firing window: weapon ready, enemy present, near weapon range, aligned lead intercept.
    x = np.zeros((4, G.OBS_SIZE), dtype=np.float32)
    x[:, 5] = 0.0       # cooldown ready
    x[:, 57] = 0.2      # at least one living enemy (count / 5)
    x[:, 64] = 0.25     # target distance / 1800
    x[:, 68] = 1.0      # lead-angle cosine = perfectly aligned
    bias = G.fire_prior_bias_numpy(x)
    assert bias.shape == (4,)
    assert np.all(bias > 7.9) and np.all(bias <= 8.0)
    # The prior must not fire blindly when the basic gating condition is absent.
    no_target = x.copy(); no_target[:, 57] = 0.0
    no_target_bias = G.fire_prior_bias_numpy(no_target)
    assert np.all(no_target_bias == 0.0)
    not_ready = x.copy(); not_ready[:, 5] = 1.0
    assert np.all(G.fire_prior_bias_numpy(not_ready) == 0.0)
    out_of_range = x.copy(); out_of_range[:, 64] = 0.8
    assert np.all(G.fire_prior_bias_numpy(out_of_range) == 0.0)
    misaligned = x.copy(); misaligned[:, 68] = 0.8
    assert np.all(G.fire_prior_bias_numpy(misaligned) == 0.0)

    # Navigation prior: lead-error sign should favor the matching turn direction.
    nav_obs = x[:2].copy()
    nav_obs[0, 67], nav_obs[0, 68] = 0.5, 0.866
    nav_obs[1, 67], nav_obs[1, 68] = -0.5, 0.866
    nav_base = np.zeros((2, 3), dtype=np.float32)
    nav_logits = G.apply_target_turn_prior_numpy(nav_base, nav_obs)
    assert nav_logits[0, 2] > nav_logits[0, 0] and nav_logits[1, 0] > nav_logits[1, 2]

    # Imminent projectile: collision-course features (90..93) and relative bullet
    # vectors (48..55) should bias strafe away from the predicted pass line.
    threat = x[:1].copy()
    threat[0, 8], threat[0, 9] = 0.25, 0.0  # target bearing is approximately east
    threat[0, 12], threat[0, 13] = 0.0, 1.0 # target bearing error = zero; heading east
    threat[0, 48], threat[0, 49] = 0.10, 0.10
    threat[0, 50], threat[0, 51] = -0.50, 0.0
    threat[0, 90], threat[0, 91] = 0.20, 0.10
    threat[0, 92], threat[0, 93] = 1.0, 1.0  # second slot empty
    dodge_logits = G.apply_dodge_prior_numpy(np.zeros((1, 3), dtype=np.float32), threat)
    assert dodge_logits[0, 0] > dodge_logits[0, 1] and dodge_logits[0, 0] > dodge_logits[0, 2], dodge_logits
    no_threat_dodge = G.apply_dodge_prior_numpy(np.zeros((1, 3), dtype=np.float32), x[:1])
    assert np.allclose(no_threat_dodge, 0.0)

    policy = G.ActorCritic(seed=613, width=32, depth=2, device_mode="cpu")
    # Remove all learned fire logits so this test isolates the shared prior.
    for key in ("Wfire", "bfire", "G_Wfire", "G_bfire"):
        policy.params[key].fill(0.0)
    logits, _ = policy.forward(x)
    assert logits["fire"].shape == (4, 1)
    assert np.all(logits["fire"][:, 0] > 7.9)
    actions, _, _, array_actions = policy.predict_batch(x, deterministic=True)
    assert array_actions.shape == (4, 4)
    assert np.all(array_actions[:, 3] == 1), "good firing windows should not collapse to never-fire"
    _, _, _, no_target_actions = policy.predict_batch(no_target, deterministic=True)
    assert np.all(no_target_actions[:, 3] == 0), "neutral zero fire logits should not fire when no enemy exists"

    # Legacy pacifist weights can have a very negative fire bias in both heads.
    # The clean, weapon-ready firing window must still produce a shot, otherwise
    # hundreds of thousands of old self-play episodes can remain silent forever.
    pacifist = G.ActorCritic(seed=614, width=32, depth=2, device_mode="cpu")
    for key in ("Wfire", "G_Wfire"):
        pacifist.params[key].fill(0.0)
    pacifist.params["bfire"].fill(-12.0)
    pacifist.params["G_bfire"].fill(-12.0)
    p_logits, _ = pacifist.forward(x)
    assert np.all(p_logits["fire"][:, 0] >= 2.19), p_logits["fire"]
    _, _, _, p_actions = pacifist.predict_batch(x, deterministic=True)
    assert np.all(p_actions[:, 3] == 1), "hard clean-shot window should recover negative legacy fire logits"
    _, _, _, p_no_target_actions = pacifist.predict_batch(no_target, deterministic=True)
    assert np.all(p_no_target_actions[:, 3] == 0), "fire floor must not trigger when no enemy exists"

    import torch
    xb = torch.as_tensor(x)
    torch_bias = G.fire_prior_bias_torch(torch, xb).cpu().numpy()
    assert np.allclose(bias, torch_bias, atol=1e-6), (bias, torch_bias)

    assert np.allclose(
        G.apply_target_turn_prior_torch(torch, torch.as_tensor(nav_base), torch.as_tensor(nav_obs)).cpu().numpy(),
        G.apply_target_turn_prior_numpy(nav_base, nav_obs), atol=1e-6)
    assert np.allclose(
        G.apply_dodge_prior_torch(torch, torch.zeros((1, 3)), torch.as_tensor(threat)).cpu().numpy(),
        G.apply_dodge_prior_numpy(np.zeros((1, 3), dtype=np.float32), threat), atol=1e-6)
    torch_obj, device = policy._ensure_torch("cpu")
    tensor_logits, _ = policy._torch_forward(torch_obj, torch.as_tensor(x, dtype=torch.float32))
    np_logits, _ = policy.forward(x)
    assert tensor_logits["fire"].shape == (4,)
    assert np.allclose(tensor_logits["fire"].detach().cpu().numpy(), np_logits["fire"][:, 0], atol=1e-5)
    # Strongly negative legacy weights must behave identically in PPO's Torch path.
    pac_torch, _ = pacifist._ensure_torch("cpu")
    pac_tlogits, _ = pacifist._torch_forward(pac_torch, torch.as_tensor(x, dtype=torch.float32))
    pac_nlogits, _ = pacifist.forward(x)
    assert np.allclose(pac_tlogits["fire"].detach().cpu().numpy(), pac_nlogits["fire"][:, 0], atol=1e-5)
    assert np.all(pac_tlogits["fire"].detach().cpu().numpy() >= 2.19)

    # Visible battle must never silently switch a requested GPU action pass to NumPy.
    policy.allow_gpu_inference = True
    policy.require_gpu_inference = True
    policy.inference_mode = "gpu"
    with patch.object(torch.cuda, "is_available", return_value=False):
        try:
            policy.predict_batch(x, deterministic=True)
        except RuntimeError as exc:
            assert "CPU fallback is disabled" in str(exc) or "GPU-only inference failed" in str(exc)
        else:
            raise AssertionError("strict GPU inference fell back to CPU instead of raising")
    print("FIRE WINDOW FLOOR / PACIFIST RECOVERY / TARGETING + DODGE PRIORS / CPU-TORCH PARITY / STRICT GPU BATTLE INFERENCE PASS")

if __name__ == "__main__":
    main()
