"""Desktop Tk smoke check for Stage 6.13; uses an isolated temporary copy of the app."""
from __future__ import annotations
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path


def main() -> int:
    source_root = Path(__file__).resolve().parents[1]
    try:
        import tkinter as tk
    except Exception as exc:
        print(f"SKIP: tkinter unavailable: {exc}")
        return 0

    with tempfile.TemporaryDirectory(prefix="gravitruex_ui_smoke_") as temp:
        tmp = Path(temp)
        for name in ("gravitruex_ai.py", "gravitruex_gpu_batch.py", "gravitruex_gpu_batch_stage4_original.py", "gravitruex_config.json"):
            src = source_root / name
            if src.exists():
                shutil.copy2(src, tmp / name)
        sys.path.insert(0, str(tmp))
        spec = importlib.util.spec_from_file_location("gravitruex_ui_smoke_module", tmp / "gravitruex_ai.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load isolated app module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        try:
            root = tk.Tk()
        except Exception as exc:
            print(f"SKIP: no usable desktop display for Tk UI smoke: {exc}")
            return 0
        try:
            root.withdraw()
            app = module.GravitrueXApp(root)
            # Keep this smoke test isolated from the host's actual Desktop.
            app.desktop_diagnostic_path = tmp / "Desktop" / module.DESKTOP_DIAGNOSTIC_FILENAME
            app._last_desktop_diagnostic_write = 0.0
            root.deiconify()
            root.update_idletasks()
            assert app.ui_tab == "battle"
            assert app._battle_page.winfo_manager() == "pack"
            assert not app._training_page.winfo_manager()
            app._switch_ui_tab("training")
            root.update_idletasks()
            assert app.ui_tab == "training"
            assert app._training_page.winfo_manager() == "pack"
            assert len(app.training_columns) == 3
            assert len(app.training_summary_columns) == 2
            assert len(app.training_kpi_vars) == 6
            assert len(app.training_cycle_vars) == 2
            assert hasattr(app, "test_battle_button")
            # Training-only composition/budget controls must exist and round-trip through config.
            assert hasattr(app, "training_blue_counter_row")
            assert hasattr(app, "training_red_counter_row")
            assert hasattr(app, "training_record_budget_combo")
            app.train_blue_n.set(3)
            app.train_red_n.set(4)
            app.train_record_budget.set("540 000")
            training_cfg = app._collect_settings_payload()["training"]
            assert training_cfg["blue_n"] == 3
            assert training_cfg["red_n"] == 4
            assert training_cfg["gpu_record_budget"] == "540 000"
            app.train_blue_n.set(1)
            app.train_red_n.set(1)
            app.train_record_budget.set("360 000")
            assert app._left_host.winfo_manager() == ""
            assert app._right_host.winfo_manager() == "pack"
            assert len(app.training_columns) == 3
            assert len(app.training_summary_columns) == 2
            assert app._left_host.winfo_manager() == ""
            assert app._right_host.winfo_manager() == "pack"
            app._draw()
            root.update_idletasks()
            # Compact KPI card text must not become one unbroken, clipped line.
            step_lines = app.training_kpi_vars["steps"].get().splitlines()
            assert len(step_lines) == 3, step_lines
            assert all(len(line) <= 20 for line in step_lines), step_lines
            app._write_desktop_diagnostics(force=True)
            assert app.desktop_diagnostic_path.exists()
            diagnostic_text = app.desktop_diagnostic_path.read_text(encoding="utf-8")
            assert "КОМПАКТНЫЙ ДИАГНОСТИЧЕСКИЙ СНИМОК" in diagnostic_text
            # The compact diagnostic includes the split timers and CUDA Graph state
            # whenever a trainer is active; the idle snapshot above has no trainer block.
            source_text = (tmp / "gravitruex_ai.py").read_text(encoding="utf-8")
            assert "CPU collect=" in source_text
            assert "GPU collect=" in source_text
            assert "CUDA Graph:" in source_text
            assert len(app.canvas.find_all()) > 5
            training_text = [app.canvas.itemcget(item, "text") for item in app.canvas.find_all() if "text" in app.canvas.itemconfigure(item)]
            assert any("ОБУЧЕНИЕ PPO" in text for text in training_text)
            app._switch_ui_tab("battle")
            app._draw()
            root.update_idletasks()
            assert app._battle_page.winfo_manager() == "pack"
            assert app._left_host.winfo_manager() == "pack"
            assert len(app.canvas.find_all()) > 5
            battle_text = [app.canvas.itemcget(item, "text") for item in app.canvas.find_all() if "text" in app.canvas.itemconfigure(item)]
            assert not any("ОБУЧЕНИЕ PPO" in text for text in battle_text)
            print("UI LAYOUT SMOKE PASSED: global tabs, 3-column training dashboard, KPI/last-two-cycles, test-battle button, battle first frame")
        finally:
            try:
                root.destroy()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
