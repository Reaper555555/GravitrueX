# -*- coding: utf-8 -*-
"""Run one real CPU PPO training cycle in a temporary model library.

This test is deliberately isolated: it does not touch the user's saved weights.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="gravitruex_smoke_") as temp:
        root = Path(temp)
        G.SAVE_DIR = root / "GravitrueX"
        G.MODEL_ROOT = G.SAVE_DIR / "models"
        G.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        model_id = "smoke_stage64"
        save_dir = G.model_dir_for(model_id)
        save_dir.mkdir(parents=True, exist_ok=True)
        base = G.ActorCritic(seed=901, width=32, depth=2, device_mode="cpu")
        trainer = G.Trainer(
            base_policy=base,
            fixed_blue=1,
            fixed_red=1,
            random_compositions=False,
            save_dir=save_dir,
            workers=1,
            opponent_mode="heuristic",
            device_mode="cpu",
            batch_size=128,
            epochs=1,
            model_id=model_id,
            model_name="Temporary CPU Smoke Test",
        )
        trainer.start()
        deadline = time.monotonic() + 45.0
        observed_updates = 0
        while time.monotonic() < deadline:
            stats = trainer.get_stats()
            observed_updates = int(stats.get("ppo_updates", 0))
            if observed_updates >= 1:
                trainer.stop()
                break
            if not trainer.is_alive():
                break
            time.sleep(0.025)
        trainer.stop()
        trainer.join(timeout=15.0)
        if trainer.is_alive():
            raise RuntimeError("Trainer thread did not stop within 15 seconds")
        stats = trainer.get_stats()
        if int(stats.get("ppo_updates", 0)) < 1:
            raise RuntimeError("No PPO update completed. Last message: " + str(stats.get("message")))
        weights = save_dir / f"weights_{model_id}_latest.npz"
        report = save_dir / "training_report.txt"
        training_stats = save_dir / "training_stats.json"
        for path in (weights, report, training_stats):
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"Expected training output missing or empty: {path.name}")
        restored = G.ActorCritic(seed=902, width=32, depth=2, device_mode="cpu")
        if not restored.load(weights):
            raise RuntimeError("Could not reload saved smoke-test weights")
        print("TRAINING CPU SMOKE PASSED")
        print(f"updates={stats.get('ppo_updates')} episodes={stats.get('episodes')} steps={stats.get('steps')}")
        print(f"weights_bytes={weights.stat().st_size} report_bytes={report.stat().st_size}")
        print("temporary outputs were deleted; user's model library was untouched")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
