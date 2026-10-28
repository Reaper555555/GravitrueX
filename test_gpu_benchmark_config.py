import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as gx


class GPUBenchmarkConfigTests(unittest.TestCase):
    def test_presets_and_repeats_are_consistent(self):
        self.assertEqual(gx.GPU_BENCHMARK_REPEATS, 2)
        self.assertEqual(len(gx.GPU_BENCHMARK_COUNTS), 17)
        self.assertEqual(gx.GPU_BENCHMARK_COUNTS[0], 1)
        self.assertEqual(gx.GPU_BENCHMARK_COUNTS[-1], 256)
        self.assertEqual(gx.GPU_BENCHMARK_COUNTS, sorted(set(gx.GPU_BENCHMARK_COUNTS)))
        self.assertEqual(gx.GPU_ARENA_PRESET_MAP["256 — стресс-тест"], 256)
        self.assertIsNone(gx.GPU_ARENA_PRESET_MAP["Ручное значение"])

    def test_report_row_formatter(self):
        row = {
            "arenas": 16, "run": 2, "episodes": 16, "steps": 6144, "records": 768,
            "wall": 0.75, "device": 0.68, "rate": 8192.0, "alloc": 450.0,
            "reserved": 620.0, "gpu": 73.0, "mem": 28.0, "vram": 1900.0,
            "temp": 62.0, "power": 101.0, "status": "OK",
        }
        text = gx.GravitrueXApp._format_gpu_benchmark_row(row)
        self.assertIn("16", text)
        self.assertIn("8,192", text)
        self.assertIn("OK", text)

    def test_gpu_monitor_csv_parse_includes_time_and_vram(self):
        monitor = gx.PerformanceMonitor.__new__(gx.PerformanceMonitor)
        monitor._gpu_exe = "nvidia-smi-test"
        mocked = SimpleNamespace(returncode=0,
            stdout="NVIDIA GeForce RTX 5060, GPU-test-uuid, 60, 23, 1916, 8151, 59, 86.61, 1800\n",
            stderr="")
        with patch("subprocess.run", return_value=mocked):
            result = monitor._sample_gpu()
        self.assertEqual(result["gpu_name"], "NVIDIA GeForce RTX 5060")
        self.assertEqual(result["gpu_util"], 60.0)
        self.assertEqual(result["gpu_mem_util"], 23.0)
        self.assertEqual(result["vram_used"], 1916.0)
        self.assertEqual(result["vram_total"], 8151.0)
        self.assertGreater(result["gpu_sampled_at"], 0)

    def test_monitor_retries_minimal_query_if_optional_query_fails(self):
        monitor = gx.PerformanceMonitor.__new__(gx.PerformanceMonitor)
        monitor._gpu_exe = "nvidia-smi-test"
        responses = [
            SimpleNamespace(returncode=1, stdout="", stderr="field unsupported"),
            SimpleNamespace(returncode=0,
                stdout="NVIDIA GeForce RTX 5060, GPU-test-uuid, 42, 18, 1700, 8151, 57\n", stderr=""),
        ]
        with patch("subprocess.run", side_effect=responses) as mock_run:
            result = monitor._sample_gpu()
        self.assertEqual(mock_run.call_count, 2)
        self.assertEqual(result["gpu_util"], 42.0)
        self.assertEqual(result["vram_total"], 8151.0)
        self.assertIsNone(result["gpu_power_w"])


if __name__ == "__main__":
    unittest.main()
