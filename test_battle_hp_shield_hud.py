"""Ensure simple colored HP/shield bars are drawn for living battle ships."""
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import gravitruex_ai as G

class FakeCanvas:
    def __init__(self): self.calls=[]
    def __getattr__(self, name):
        if name.startswith("create_"):
            def call(*args, **kwargs): self.calls.append((name,args,kwargs)); return len(self.calls)
            return call
        raise AttributeError(name)

def main():
    ship = SimpleNamespace(sid=1, alive=True, team=0, heading=0.4, x=10.0, y=20.0, hp=50.0, shield=19.0)
    arena = SimpleNamespace(ships=[ship], bullets=[], blue_n=1, red_n=0, sim_time=1.0, round_over=False, winner=None,
                            alive_ships=lambda team=None: [ship] if team in (None,0) else [])
    app = SimpleNamespace(canvas=FakeCanvas(), arena=arena, view_zoom=1.0, camera_x=0.0, camera_y=0.0,
                           display_fps=60.0, game_steps_per_sec=500.0, paused=False)
    app._update_camera=lambda: None
    app._draw_world_grid=lambda *args: None
    app._screen_points=lambda *args, **kwargs: [(320.0,240.0)]
    G.GravitrueXApp._draw_arena(app,640,480)
    rects=[kwargs for name,args,kwargs in app.canvas.calls if name=="create_rectangle"]
    fills={r.get("fill") for r in rects}
    assert "#63d9ff" in fills, fills
    assert "#ffc777" in fills, fills
    assert len(rects)>=3, len(rects)
    print("BATTLE HP/SHIELD BAR DRAW TEST PASS")

if __name__ == "__main__": main()
