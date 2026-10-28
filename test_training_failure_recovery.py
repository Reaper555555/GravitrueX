# -*- coding: utf-8 -*-
"""Regression tests for recoverable device errors and visible fatal trainer errors."""
from __future__ import annotations
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G


def make_trainer(root: Path, model_id: str = "fault_test", device: str = "auto") -> G.Trainer:
    base = G.ActorCritic(seed=71, width=32, depth=2, device_mode="cpu")
    folder = root / "models" / model_id
    folder.mkdir(parents=True, exist_ok=True)
    return G.Trainer(base, 1, 1, False, folder, workers=1,
                     opponent_mode="heuristic", device_mode=device,
                     batch_size=32, epochs=1, model_id=model_id, model_name="Fault test")


def test_rng_integer_supports_numpy_and_python_generators():
    generators = (
        G.np.random.default_rng(1234),
        G.random.Random(1234),
    )
    for rng in generators:
        values = [G._rng_integer(rng, 1, 2**31 - 1) for _ in range(16)]
        assert all(1 <= value < 2**31 - 1 for value in values), values
    print("PASS: PPO seed generation supports NumPy Generator and Python Random")


def test_ppo_retry_cpu():
    with tempfile.TemporaryDirectory(prefix="gx_ppo_fallback_") as temp:
        trainer = make_trainer(Path(temp), device="cuda")
        calls = []
        def fake_update(records, epochs, batch_size, device, teacher_coef=0.0):
            calls.append(device)
            if len(calls) == 1:
                raise RuntimeError("CUDA error: simulated out of memory")
            return 0.125
        trainer.policy.update_ppo = fake_update
        loss, elapsed = trainer._update_ppo_with_fallback([{"dummy": True}])
        assert loss == 0.125, loss
        assert len(calls) == 2 and calls == ["cuda", "cpu"], calls
        assert trainer.device_mode == "cpu"
        assert trainer.policy.device_mode == "cpu"
        assert trainer.runtime_log_path.exists(), "fallback event was not persisted"
        assert "retrying the same batch on CPU" in trainer.runtime_log_path.read_text(encoding="utf-8")
        print("PASS: failed auto/CUDA PPO update retries the same batch on CPU")


def test_gpu_rollout_fallback_resets_flags():
    with tempfile.TemporaryDirectory(prefix="gx_rollout_fallback_") as temp:
        trainer = make_trainer(Path(temp))
        trainer.gpu_only_rollout = True
        trainer.hybrid_gpu_rollout = True
        trainer.rollout_gpu_inference = True
        trainer.auto_rollout_balance = True
        try:
            raise RuntimeError("simulated GPU batch launch error")
        except RuntimeError as exc:
            trainer._switch_rollout_to_cpu(exc, "test phase")
        assert not trainer.gpu_only_rollout
        assert not trainer.hybrid_gpu_rollout
        assert not trainer.rollout_gpu_inference
        assert not trainer.auto_rollout_balance
        assert trainer.stats["phase"] == "GPU rollout недоступен; продолжаем на CPU"
        assert "simulated GPU batch launch error" in trainer.runtime_log_path.read_text(encoding="utf-8")
        print("PASS: failed GPU rollout disables GPU collection and records the cause")


def test_gpu_only_batch_failure_continues_cpu_training():
    with tempfile.TemporaryDirectory(prefix="gx_gpu_to_cpu_") as temp:
        trainer = make_trainer(Path(temp), model_id="gpu_to_cpu")
        trainer.gpu_only_rollout = True
        trainer._last_cycle_gpu_only_mode = True
        trainer._collect_gpu_batch = lambda _seed=None: (_ for _ in ()).throw(
            RuntimeError("simulated CUDA batch allocation failure"))
        metrics = {
            "blue_damage": 1.0, "red_damage": 0.5, "blue_shots": 2.0,
            "blue_hit_rate": 0.5, "blue_kills": 1.0, "red_kills": 0.0,
            "blue_survivors": 1.0, "red_survivors": 0.0,
        }
        record = {
            "obs": G.np.zeros(G.OBS_SIZE, dtype=G.np.float32),
            "action": G.np.asarray([1, 1, 1, 0], dtype=G.np.int64),
            "logp": 0.0, "value": 0.0, "reward": 1.0,
            "adv": 1.0, "return": 1.0,
        }
        cpu_results = [([record], 1.0, 0, 1, 1, 16, metrics)]
        trainer._collect_cpu_rollouts = lambda _executor, _batch_n: cpu_results
        trainer.policy.update_ppo = lambda records, epochs, batch_size, device, teacher_coef=0.0: 0.25
        def end_after_cycle(*_args, **_kwargs):
            trainer.stop_event.set()
        trainer._write_training_cycle_log = end_after_cycle
        trainer._run_batches(None)
        stats = trainer.get_stats()
        assert not trainer.gpu_only_rollout
        assert int(stats.get("episodes", 0)) == 1
        assert int(stats.get("ppo_updates", 0)) == 1
        assert "simulated CUDA batch allocation failure" in str(stats.get("gpu_rollout_fallback_reason"))
        print("PASS: GPU-only rollout error falls back to CPU and completes a PPO update")


def test_fatal_error_report_survives_thread_exit():
    with tempfile.TemporaryDirectory(prefix="gx_crash_report_") as temp:
        root = Path(temp)
        old_save, old_models = G.SAVE_DIR, G.MODEL_ROOT
        G.SAVE_DIR, G.MODEL_ROOT = root / "GravitrueX", root / "GravitrueX" / "models"
        G.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        try:
            trainer = make_trainer(root / "library", model_id="crash_test", device="cpu")
            def fail_run_batches(_executor=None):
                raise RuntimeError("reproducible fatal marker")
            trainer._run_batches = fail_run_batches
            original_print_exc = G.traceback.print_exc
            G.traceback.print_exc = lambda: None
            try:
                trainer.run()
            finally:
                G.traceback.print_exc = original_print_exc
            assert trainer.fatal_error and "reproducible fatal marker" in trainer.fatal_error
            assert trainer.crash_report_path.exists()
            report = trainer.crash_report_path.read_text(encoding="utf-8")
            assert "reproducible fatal marker" in report
            assert "TRACEBACK:" in report
            stats = trainer.stats_path.read_text(encoding="utf-8")
            assert "fatal_error" in stats
            print("PASS: fatal trainer exception is saved to crash report and training stats")
        finally:
            G.SAVE_DIR, G.MODEL_ROOT = old_save, old_models


def main():
    test_rng_integer_supports_numpy_and_python_generators()
    test_ppo_retry_cpu()
    test_gpu_rollout_fallback_resets_flags()
    test_gpu_only_batch_failure_continues_cpu_training()
    test_fatal_error_report_survives_thread_exit()
    print("TRAINING FAILURE RECOVERY TESTS PASSED")


if __name__ == "__main__":
    main()
