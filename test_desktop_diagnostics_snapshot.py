"""Compact desktop diagnostic target, overwrite behavior, and speed-metric semantics."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G


def main():
    with tempfile.TemporaryDirectory(prefix="grav-desktop-log-") as td:
        home = Path(td)
        desktop = home / "Desktop"
        desktop.mkdir()
        path = G.resolve_desktop_diagnostic_path(home=home, env={})
        assert path == desktop / G.DESKTOP_DIAGNOSTIC_FILENAME, path
        onedrive_desktop = home / "OneDrive" / "Desktop"
        onedrive_desktop.mkdir(parents=True)
        redirected = G.resolve_desktop_diagnostic_path(home=home, env={"OneDrive": str(home / "OneDrive")})
        assert redirected == onedrive_desktop / G.DESKTOP_DIAGNOSTIC_FILENAME, redirected
        G.write_compact_diagnostic_snapshot(path, "snapshot A\ncycles=2")
        first_size = path.stat().st_size
        G.write_compact_diagnostic_snapshot(path, "snapshot B\ncycles=2")
        text = path.read_text(encoding="utf-8")
        assert "snapshot B" in text and "snapshot A" not in text
        assert path.stat().st_size <= first_size + 10
        assert not path.with_name(path.name + ".tmp").exists()

    metric = G.cycle_throughput(480000, 2.4, 3.0)
    assert round(metric["rollout_steps_per_sec"]) == 200000
    assert round(metric["end_to_end_steps_per_sec"]) == 88889
    assert metric["end_to_end_steps_per_sec"] < metric["rollout_steps_per_sec"]
    assert G.health_bar_color(0.9) == "#8be0ca"
    assert G.health_bar_color(0.5) == "#ffc777"
    assert G.health_bar_color(0.1) == "#ff8794"
    print("DESKTOP DIAGNOSTIC OVERWRITE / THROUGHPUT SEMANTICS / HP BAR COLORS PASS")


if __name__ == "__main__":
    main()
