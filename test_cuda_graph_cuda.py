# -*- coding: utf-8 -*-
"""CUDA Graph preflight over several rollout sizes, using the real model profile.

No weights are changed or saved. Set GX_CUDA_GRAPH_TEST_ARENAS (e.g. 2,64,256)
 or GX_CUDA_GRAPH_TEST_PROFILE (e.g. 1536,4) to customise the test.
"""
from __future__ import annotations
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GRAVITRUEX_GPU_FAST"] = "1"

try:
    import torch
except Exception as exc:
    print(f"SKIP: PyTorch unavailable: {exc}")
    raise SystemExit(0)
if not torch.cuda.is_available():
    print("SKIP: CUDA is not available; run with the NVIDIA/CUDA launcher.")
    raise SystemExit(0)

import numpy as np
import gravitruex_ai as G


def same_device(actual, expected):
    actual, expected = torch.device(actual), torch.device(expected)
    if actual.type != expected.type:
        return False
    if actual.type == "cuda":
        current = torch.cuda.current_device()
        ai = current if actual.index is None else actual.index
        ei = current if expected.index is None else expected.index
        return ai == ei
    return actual == expected


def finite_experience(row, index):
    packed, total_reward, _outcome, _bn, _rn, _steps, metrics = row
    for field in ("obs", "logp", "value", "reward", "adv", "returns"):
        values = np.asarray(getattr(packed, field))
        if not np.isfinite(values).all():
            raise FloatingPointError(f"arena result {index}: non-finite {field}")
    if not math.isfinite(float(total_reward)):
        raise FloatingPointError(f"arena result {index}: non-finite total reward")
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)) and not math.isfinite(float(value)):
            raise FloatingPointError(f"arena result {index}: non-finite metric {key}")


def main():
    profile = os.environ.get("GX_CUDA_GRAPH_TEST_PROFILE", "1536,4")
    try:
        width, depth = [int(x.strip()) for x in profile.split(",", 1)]
    except Exception as exc:
        raise ValueError("GX_CUDA_GRAPH_TEST_PROFILE must be 'width,depth', e.g. 1536,4") from exc
    raw_sizes = os.environ.get("GX_CUDA_GRAPH_TEST_ARENAS", "2,64,256")
    sizes = []
    for part in raw_sizes.split(","):
        n = int(part.strip())
        if n < 1 or n > 2048:
            raise ValueError(f"arena count must be 1..2048, got {n}")
        if n not in sizes:
            sizes.append(n)
    if not sizes:
        raise ValueError("GX_CUDA_GRAPH_TEST_ARENAS cannot be empty")

    print(f"CUDA Graph preflight: device={torch.cuda.get_device_name(0)} profile={width}x{depth} arenas={sizes}")
    policy = G.ActorCritic(seed=867, width=width, depth=depth, device_mode="cpu")
    candidates = []
    try:
        for path in G.MODEL_ROOT.glob("*/weights_*_latest.npz"):
            try:
                if G.checkpoint_architecture(path) == (width, depth):
                    candidates.append(path)
            except Exception:
                pass
    except Exception:
        pass
    weights_path = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    loaded = bool(policy.load(weights_path)) if weights_path else False
    print(f"Policy checkpoint: {weights_path if loaded else 'fresh compatible test weights'}; large residual active={policy._large_residual_active()}")

    batch_cls = G._load_gpu_batch_class()
    failures = []
    for count in sizes:
        batch = None
        try:
            batch = batch_cls(
                engine=G, policy=policy, count=count, blue_n=1, red_n=1,
                opponent_mode="heuristic", max_decisions=8, device="cuda",
                seed=867 + count, use_cuda_graph=True,
            )
            assert abs(batch._bullet_ttl_value - float(G.BULLET_TTL)) < 1e-7
            if not same_device(batch._bullet_ttl_spawn_values.device, batch.device):
                raise AssertionError(f"TTL tensor device mismatch: {batch._bullet_ttl_spawn_values.device} vs {batch.device}")

            # Force a transition during warmup, then ensure snapshot/restore fully recovers it.
            batch.hp[0, 0] = 0.0
            batch.shield[0, 0] = 0.0
            batch.alive[0, 0] = True
            batch.active_env[0] = True
            snapshot = batch._snapshot_state()
            graph = batch._build_cuda_graph()
            if graph is None:
                raise RuntimeError(
                    f"CUDA Graph capture failed for {count} arenas: "
                    f"{getattr(batch, 'last_cuda_graph_error', None)}\n"
                    f"{getattr(batch, 'last_cuda_graph_traceback', None)}"
                )
            torch.cuda.synchronize()
            changed = []
            for name, expected in snapshot.items():
                actual = getattr(batch, name)
                if not torch.equal(actual, expected):
                    changed.append(name)
            if changed:
                raise AssertionError("Warmup did not restore state fields: " + ", ".join(changed))
            if not bool(batch.alive[0, 0].item()):
                raise AssertionError("alive state was not restored")

            # Restore a live ship in-place; then collect() reuses the graph captured above.
            batch.hp[0, 0] = G.MAX_HP
            batch.shield[0, 0] = G.MAX_SHIELD
            batch.alive[0, 0] = True
            batch.active_env[:] = True
            results = batch.collect()
            torch.cuda.synchronize()
            if not batch.last_cuda_graph_used:
                raise AssertionError("collect() did not replay the captured graph")
            if not results:
                raise AssertionError("rollout produced no episode records")
            for idx, row in enumerate(results):
                finite_experience(row, idx)
            print(f"PASS arenas={count}: capture+restore+replay; episodes={len(results)} records={sum(len(r[0]) for r in results)} device_s={getattr(batch, 'last_cuda_device_seconds', 0.0):.3f}")
        except Exception as exc:
            failures.append((count, exc))
            print(f"FAIL arenas={count}: {type(exc).__name__}: {exc}")
        finally:
            if batch is not None:
                del batch
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
    if failures:
        print("CUDA GRAPH PREFLIGHT FAILED for: " + ", ".join(str(n) for n, _ in failures))
        return 1
    print(f"CUDA GRAPH PREFLIGHT PASSED ({len(sizes)} sizes); weights were not modified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
