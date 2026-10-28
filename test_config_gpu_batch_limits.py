"""Configuration round-trip and bounded adaptive GPU arena batch tests."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G


def main():
    assert G.GPU_MAX_ARENAS == 4096
    assert G.GPU_ROLLOUT_RECORD_BUDGET == 360000
    assert G.next_gpu_arena_fallback(4096) == 3840
    assert G.next_gpu_arena_fallback(3072) == 2816
    assert G.next_gpu_arena_fallback(3840) == 3584
    assert G.next_gpu_arena_fallback(2560) == 2304
    assert G.next_gpu_arena_fallback(2048) == 1792
    assert G.next_gpu_arena_fallback(1536) == 1280
    assert G.next_gpu_arena_fallback(1024) == 768
    assert G.next_gpu_arena_fallback(256) == 128
    assert G.next_gpu_arena_fallback(1) is None
    assert G.is_cuda_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 12 MiB"))
    assert not G.is_cuda_oom_error(RuntimeError("randrange attribute error"))

    # The final episode must be rejected atomically rather than exceeding a cap.
    assert G.episode_fits_record_budget(7680, 480, 8192)
    assert G.episode_fits_record_budget(8160, 32, 8192)
    assert not G.episode_fits_record_budget(8160, 480, 8192)
    assert not G.episode_fits_record_budget(8192, 1, 8192)

    with tempfile.TemporaryDirectory(prefix="grav-config-test-") as td:
        path = Path(td) / "gravitruex_config.json"
        settings = {
            "active_model_id": "main",
            "models": {"active": "main", "blue": "main", "red": "heuristic",
                       "training_opponent": "selfplay"},
            "battle": {"blue_n": 2, "red_n": 3, "speed": 4.0, "view_zoom": 1.2},
            "training": {"model_profile": "Экстремальная 3072×6 (~47,5 млн)",
                         "device": "CUDA обязательно", "rollout_mode": "Гибрид: CPU + GPU-параллельные бои",
                         "cpu_workers": 6, "gpu_arenas": 2560, "ppo_batch": "4096",
                         "ppo_epochs": "3", "random_compositions": True},
            "variant": {"new_model_name": "Экспериментальный вариант"},
        }
        G.save_app_config(path, settings)
        restored = G.load_app_config(path)
        assert restored == settings, (restored, settings)
        path.write_text("{broken json", encoding="utf-8")
        assert G.load_app_config(path) is None
    print("CONFIG ROUNDTRIP / GPU BATCH LIMITS / STRICT RECORD CAP PASS")


if __name__ == "__main__":
    main()
