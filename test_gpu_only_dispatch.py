"""Regression test for the GPU-only training route using the tensor simulator's CPU backend."""
import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gravitruex_ai as engine
from gravitruex_ai import ActorCritic, Trainer
from gravitruex_gpu_batch import GPUVectorArenaBatch


def main():
    with tempfile.TemporaryDirectory(prefix="gravitruex-gpu-only-test-") as td:
        policy = ActorCritic(seed=101, width=32, depth=2, device_mode="cpu")
        trainer = Trainer(policy, fixed_blue=1, fixed_red=1, random_compositions=False,
                          save_dir=Path(td), workers=1, opponent_mode="heuristic",
                          device_mode="cpu", batch_size=16, epochs=1,
                          gpu_only_rollout=True, gpu_batch_envs=2,
                          model_id="test", model_name="GPU-only test")

        def fake_gpu_batch(seed=None):
            t0 = time.perf_counter()
            batch = GPUVectorArenaBatch(engine=engine, policy=trainer.policy, count=2,
                                        blue_n=1, red_n=1, opponent_mode="heuristic",
                                        max_decisions=8, device="cpu", seed=seed or 101)
            results = batch.collect(trainer.stop_event)
            trainer._batch_gpu_device_seconds = 0.0
            trainer._last_gpu_batch_seconds = max(0.001, time.perf_counter() - t0)
            return results

        trainer._collect_gpu_batch = fake_gpu_batch
        original_update = trainer.policy.update_ppo
        def update_once_then_stop(*args, **kwargs):
            value = original_update(*args, **kwargs)
            trainer.stop_event.set()
            return value
        trainer.policy.update_ppo = update_once_then_stop
        trainer._run_batches(None)
        stats = trainer.get_stats()
        assert stats.get("gpu_only_rollout") is True
        assert stats.get("rollout_physics_device") == "CUDA"
        assert stats.get("cpu_rollout_episodes_last") == 0
        assert stats.get("gpu_rollout_episodes_last") == 2
        assert stats.get("cpu_rollout_records_last") == 0
        assert stats.get("gpu_rollout_records_last", 0) > 0
        report = (Path(td) / "training_report.txt").read_text(encoding="utf-8")
        assert "только GPU-физика" in report
        assert "физика=CUDA" in report
        assert "CPU: 0 эп." in report
        print("GPU-only route: PASS (CPU test backend; report/device labels and rollout accounting verified)")

if __name__ == "__main__":
    main()
