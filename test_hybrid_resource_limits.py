"""Deterministic tests for the balanced CPU+GPU rollout resource guard."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gravitruex_ai import resolve_hybrid_limits

GIB = 1024 ** 3

def main():
    # The user's 16-thread / low-free-RAM scenario should reserve host headroom.
    tight = resolve_hybrid_limits(8, 1024, logical_cpus=16, available_ram_bytes=int(2.5 * GIB))
    assert tight["active_workers"] == 2, tight
    assert tight["cpu_record_cap"] == 8192, tight
    assert tight["reserved_logical_cpus"] == 4, tight

    # With ample memory, cap at four lower-priority producers and 24k records.
    roomy = resolve_hybrid_limits(16, 1024, logical_cpus=16, available_ram_bytes=8 * GIB)
    assert roomy["active_workers"] == 4, roomy
    assert roomy["cpu_record_cap"] == 24000, roomy

    # Smaller CPUs keep at least one producer while reserving CPU for dispatch/UI.
    small_cpu = resolve_hybrid_limits(8, 512, logical_cpus=4, available_ram_bytes=8 * GIB)
    assert small_cpu["active_workers"] == 1, small_cpu
    assert small_cpu["reserved_logical_cpus"] == 2, small_cpu

    # When telemetry is absent, limits stay conservative and deterministic.
    unknown = resolve_hybrid_limits(16, 512, logical_cpus=16, available_ram_bytes=None)
    assert unknown["active_workers"] == 4, unknown
    assert unknown["cpu_record_cap"] == 12288, unknown
    print("HYBRID RESOURCE LIMITS PASS")
    print("low RAM:", tight)
    print("roomy RAM:", roomy)

if __name__ == "__main__":
    main()
