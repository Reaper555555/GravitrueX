"""Regression tests for selectable training team size and truthful rollout telemetry."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="grav-compose-telemetry-") as td:
        save_dir = Path(td) / "model"
        base = G.ActorCritic(seed=7, width=32, depth=2, device_mode="cpu")
        trainer = G.Trainer(
            base_policy=base, fixed_blue=3, fixed_red=4, random_compositions=False,
            save_dir=save_dir, workers=1, opponent_mode="heuristic", device_mode="cpu",
            batch_size=128, epochs=1, gpu_batch_envs=2048,
            gpu_only_rollout=True, model_id="telemetry_test", model_name="Telemetry Test",
            max_decisions=720, gpu_record_budget=540_000,
        )
        # Emulate the values produced by a GPU rollout without needing CUDA.
        trainer._last_cycle_gpu_only_mode = True
        trainer._last_cycle_hybrid_mode = False
        trainer._gpu_batch_requested_envs = 2048
        trainer._last_gpu_record_budget = 540_000
        trainer._last_gpu_max_decisions = 720
        trainer._last_gpu_oom_fallback = ""
        trainer._last_gpu_device_seconds = 4.7
        trainer._last_gpu_host_overhead_seconds = 0.3
        trainer._last_gpu_graph_requested = True
        trainer._last_gpu_graph_used = False
        trainer._last_gpu_graph_error = "test graph fallback"
        trainer._last_cpu_rollout_episodes = 0
        trainer._last_cpu_rollout_records = 0
        trainer._last_cpu_rollout_steps = 0
        trainer._last_gpu_rollout_episodes = 128
        trainer._last_gpu_rollout_records = 539_700
        trainer._last_gpu_rollout_steps = 2_000_000
        trainer.stats.update({
            "episodes": 128, "steps": 2_000_000, "ppo_updates": 1,
            "gpu_rollout_envs_requested": 2048, "gpu_rollout_envs_attempted": 300,
            "gpu_rollout_envs": 300, "gpu_rollout_envs_budget_reduced": 1748,
            "gpu_cuda_graph_status": "eager fallback: test graph fallback",
            "gpu_rollout_oom_fallback": "", "gpu_max_decisions_last": 720,
            "ppo_records_last": 539_700, "last_loss": 0.1,
        })
        trainer._write_training_cycle_log(
            record_count=539_700, loss=0.1, rollout_elapsed=5.0, ppo_elapsed=1.2,
            batch_steps=2_000_000, results=[], cpu_seconds=0.0, gpu_seconds=5.0,
            cpu_wait_seconds=0.0,
        )
        report = trainer.report_path.read_text(encoding="utf-8")
        row = json.loads(trainer.metrics_jsonl_path.read_text(encoding="utf-8").splitlines()[-1])
        assert "GPU batch: requested 2048 → actual 300" in report
        assert "GPU rollout: арен запрошено 2048 · реально в батче 300" in report
        assert "CUDA Graph:" in report and "eager" in report
        assert row["gpu_batch_requested_envs"] == 2048
        assert row["gpu_batch_actual_envs"] == 300
        assert row["gpu_batch_budget_reduced_arenas"] == 1748
        assert row["gpu_record_budget"] == 540_000
        assert row["gpu_max_decisions_per_arena"] == 720

        # A CPU-only cycle must not claim GPU arenas were active just because a GPU
        # arena count is configured in the UI for a future GPU mode.
        trainer._last_cycle_gpu_only_mode = False
        trainer._last_cycle_hybrid_mode = False
        trainer._last_gpu_rollout_episodes = 0
        trainer._last_gpu_rollout_records = 0
        trainer._last_gpu_rollout_steps = 0
        trainer._last_cpu_rollout_episodes = 12
        trainer._last_cpu_rollout_records = 14_400
        trainer._last_cpu_rollout_steps = 57_600
        trainer.stats.update({"gpu_rollout_envs": 0, "gpu_rollout_envs_budget_reduced": 0})
        trainer._write_training_cycle_log(
            record_count=14_400, loss=0.2, rollout_elapsed=1.4, ppo_elapsed=0.12,
            batch_steps=57_600, results=[], cpu_seconds=1.4, gpu_seconds=0.0,
            cpu_wait_seconds=0.0,
        )
        rows = trainer.metrics_jsonl_path.read_text(encoding="utf-8").splitlines()
        cpu_row = json.loads(rows[-1])
        assert cpu_row["cpu_seconds"] == 1.4
        assert cpu_row["gpu_seconds"] == 0.0
        assert cpu_row["gpu_batch_requested_envs"] == 0
        assert cpu_row["gpu_batch_actual_envs"] == 0
        print("TRAINING COMPOSITION / ROLLOUT TELEMETRY PASSED: actual vs requested arenas, record-cap reduction, CUDA Graph status, CPU/GPU collection clocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
