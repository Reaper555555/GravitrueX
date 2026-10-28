"""The CPU sidecar must not delay a completed CUDA batch for slow episodes."""
import sys
import time
import threading
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as e


def main():
    tmp = Path(tempfile.mkdtemp(prefix="grav_hybrid_cutoff_"))
    old_root = e.MODEL_ROOT
    old_rollout = e._collect_episode_rollout
    old_worker_stop = e._WORKER_STOP_EVENT
    old_worker_cycle = e._WORKER_CYCLE_EVENT
    old_worker_policy = e._WORKER_POLICY
    e.MODEL_ROOT = tmp / "models"
    trainer = e.Trainer(e.ActorCritic(seed=5, width=32, depth=2, device_mode="cpu"),
                        1, 1, False, tmp, workers=1, opponent_mode="heuristic",
                        device_mode="cpu", batch_size=32, epochs=1,
                        hybrid_gpu_rollout=True, gpu_batch_envs=1,
                        model_id="cutoff", model_name="cutoff")
    trainer.cpu_workers_active = 1
    trainer.mp_cycle_event = threading.Event()
    e._WORKER_STOP_EVENT = threading.Event()
    e._WORKER_CYCLE_EVENT = trainer.mp_cycle_event
    e._WORKER_POLICY = None

    def deliberately_slow_rollout(policy, env, blue_n, red_n, max_decisions=480,
                                  stop_event=None, opponent_mode="heuristic", opponent_policy=None):
        n = 0
        while n < 1000 and not stop_event.is_set():
            time.sleep(0.005)
            n += 1
        # Represents an incomplete/partial episode at cycle cutoff. The coordinator
        # must discard it rather than mixing it into the on-policy update.
        records = [{"partial": True}] * max(0, n)
        return records, 0.0, None, blue_n, red_n, n * e.ACTION_REPEAT, {}

    e._collect_episode_rollout = deliberately_slow_rollout
    gpu_calls = {"n": 0}
    def fake_gpu(seed=None):
        gpu_calls["n"] += 1
        time.sleep(0.20)
        records = [{"gpu": True}] * 12
        return [(records, 0.0, None, 1, 1, 120,
                 {"rollout_backend": "GPU tensor physics"})]
    trainer._collect_gpu_batch = fake_gpu

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        started = time.perf_counter()
        first = trainer._collect_hybrid_async(pool)
        first_elapsed = time.perf_counter() - started
        assert first_elapsed < 1.5, f"slow CPU episode delayed GPU batch: {first_elapsed:.2f}s"
        assert len(first) == 1 and first[0][6].get("rollout_backend") == "GPU tensor physics"
        assert trainer._last_hybrid_cpu_episodes == 0, "partial CPU rollout must be discarded"
        assert trainer.mp_cycle_event.is_set(), "cycle cutoff event not signalled"

        # A second consecutive zero-yield cycle should disable the CPU sidecar and
        # continue with the GPU-only collector rather than repeat wasted work.
        started = time.perf_counter()
        second = trainer._collect_hybrid_async(pool)
        second_elapsed = time.perf_counter() - started
        assert second_elapsed < 1.5, f"second slow CPU episode delayed GPU batch: {second_elapsed:.2f}s"
        assert len(second) == 1
        assert trainer.gpu_only_rollout is True, "zero-yield hybrid did not back off to GPU-only"
        assert trainer.hybrid_gpu_rollout is False
        assert "за два GPU-цикла" in trainer.auto_rollout_selection_note
        print("HYBRID ASYNC CUTOFF PASS")
        print(f"first cycle {first_elapsed:.3f}s; second cycle {second_elapsed:.3f}s; CPU cutoff responsive")
    finally:
        trainer.stop()
        pool.shutdown(wait=True, cancel_futures=True)
        if trainer._gpu_rollout_executor is not None:
            trainer._gpu_rollout_executor.shutdown(wait=True, cancel_futures=True)
        e._collect_episode_rollout = old_rollout
        e._WORKER_STOP_EVENT = old_worker_stop
        e._WORKER_CYCLE_EVENT = old_worker_cycle
        e._WORKER_POLICY = old_worker_policy
        e.MODEL_ROOT = old_root
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
