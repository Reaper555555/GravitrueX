#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GravitrueX Neural Arena — маленькая 2D-арена с обучаемым ИИ.

Dependency: numpy. GUI и векторная графика используют стандартный tkinter.
Run: python gravitruex_ai.py
Self-test: python gravitruex_ai.py --self-test
"""
from __future__ import annotations

import inspect
import json
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Avoid each rollout worker spawning a full BLAS thread pool (several workers would
# oversubscribe the CPU and often run slower). Users can override these env vars.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

# ------------------------- GPU arena backend selection ------------------------
# Stage 5 ships a much faster vectorized CUDA arena in gravitruex_gpu_batch.py.
# The stage 4 implementation is kept next to it as a reference and can be
# selected with GRAVITRUEX_GPU_FAST=0 for an A/B comparison.
GPU_FAST_PATH = os.environ.get("GRAVITRUEX_GPU_FAST", "1").strip().lower() not in {"0", "false", "no"}
# Upper bound for the number of records produced by one GPU rollout. A bigger
# arena count produces a lot more records per decision, so the number of
# decisions per episode is reduced to keep the observation buffer in VRAM.
# More record history reduces truncation: 120k records often ended 1v1 bouts before
# a kill could occur. At 8-GB GPUs, OOM fallback still lowers the arena batch.
GPU_ROLLOUT_RECORD_BUDGET = int(os.environ.get("GRAVITRUEX_GPU_RECORD_BUDGET", "720000"))
# Independent vectorized environments, not literal CUDA streams. 4096 is a guarded
# next step for an 8-GB RTX 5060; OOM fallback retries descend to smaller shapes.
# The larger record budget preserves more decisions per duel (critical for learning to fire).
GPU_MAX_ARENAS = 4096
GPU_ARENA_FALLBACK_SIZES = (4096, 3840, 3584, 3328, 3072, 2816, 2560, 2304, 2048, 1792, 1536, 1280, 1024,
                            768, 512, 384, 256, 128, 64, 32, 16, 8, 4, 2, 1)
APP_CONFIG_VERSION = 1
APP_BUILD_LABEL = "Stage 6.19 — Long Horizon & Adaptive Range"
DESKTOP_DIAGNOSTIC_FILENAME = "GravitrueX_диагностика.txt"


def _load_gpu_batch_class():
    if GPU_FAST_PATH:
        from gravitruex_gpu_batch import GPUVectorArenaBatch
    else:
        from gravitruex_gpu_batch_stage4_original import GPUVectorArenaBatch
    return GPUVectorArenaBatch


# ---------------------------- Simulation constants ----------------------------
WORLD_SCALE = 5.0
WORLD_W = 1000.0 * WORLD_SCALE
WORLD_H = 700.0 * WORLD_SCALE
# Keep the legacy observation normalization for the first 64 features so older
# policies can be migrated without changing the meaning of their inputs.
OBS_POS_X = 500.0
OBS_POS_Y = 350.0
CAMERA_BASE_WIDTH = 2400.0
FPS = 60
DISPLAY_FPS_TARGET = 60
DISPLAY_FRAME_MS = 1000.0 / DISPLAY_FPS_TARGET
TRAINING_UI_FPS_TARGET = 10
DT = 1.0 / FPS
ACTION_REPEAT = 4                 # One policy decision per four physics ticks.
POLICY_HZ = FPS / ACTION_REPEAT
HORIZON_CHOICES = ("240 (16 с)", "450 (30 с)", "480 (32 с)", "720 (48 с)", "900 (60 с)",
                   "960 (64 с)", "1200 (80 с)", "1350 (90 с)", "1800 (120 с)",
                   "2700 (180 с)", "3600 (240 с)", "5400 (360 с)")
GPU_RECORD_BUDGET_CHOICES = ("180 000", "360 000", "540 000", "720 000",
                             "1 440 000", "2 160 000", "2 880 000")
MAX_SPEED = 205.0
ACCEL_FORWARD = 235.0
ACCEL_STRAFE = 195.0
TURN_RATE = 2.95                  # radians / second
DRAG = 0.16                       # weak drag; movement has inertia
SHIP_RADIUS = 10.0
MAX_HP = 100.0
MAX_SHIELD = 38.0
SHIELD_REGEN = 5.0
SHIELD_DELAY = 1.6
BULLET_SPEED = 455.0
BULLET_DAMAGE = 10.0
BULLET_RADIUS = 2.8
BULLET_TTL = 2.15
FIRE_COOLDOWN = 0.48
# Combat-oriented dense rewards. Damage/kills dominate; a well-aimed shot has a
# small positive exploration signal while blind fire has a tiny cost, not a large
# incentive to keep the weapon silent.
REWARD_DAMAGE_DEALT = 0.10
REWARD_DAMAGE_TAKEN = 0.035
REWARD_PER_KILL = 13.0
REWARD_DEATH = 7.0
REWARD_SHOT_AIM_BONUS = 0.08
REWARD_SHOT_BASE_COST = 0.004
REWARD_STEP_COST = 0.001
OBS_BASE_SIZE = 64
OBS_EXTRA_SIZE = 32
OBS_SIZE = OBS_BASE_SIZE + OBS_EXTRA_SIZE
HIDDEN = 128
ARCH_PROFILES = {
    "Экономная 768×3 (~1,3 млн)": (768, 3),
    "Тактическая 1536×4 (~7,3 млн)": (1536, 4),
    "Большая 2048×5 (~17 млн)": (2048, 5),
    "Экстремальная 3072×6 (~47,5 млн)": (3072, 6),
    "Максимальная 4096×6 (~84 млн)": (4096, 6),
}
N_ACTION_HEADS = (3, 3, 3, 2)     # turn, forward/back, strafe, fire

# Training behavior profiles used by the main application, serial CPU rollout,
# process workers, and the vectorized CUDA rollout. Checkpoint defaults select
# presets; runtime values are scoped/restored around each rollout where possible.
ACTION_PRIORS_MODE = os.environ.get("GRAVITRUEX_ACTION_PRIORS", "full").strip().lower()
if ACTION_PRIORS_MODE not in {"full", "soft", "off"}:
    ACTION_PRIORS_MODE = "full"
TRAINING_REWARD_MODE = os.environ.get("GRAVITRUEX_REWARD_PROFILE", "baseline").strip().lower()
if TRAINING_REWARD_MODE not in {"baseline", "team"}:
    TRAINING_REWARD_MODE = "baseline"
# Reward-shaping distance target. This influences training rewards only; it does
# not impose a hard movement constraint on the ship.
TRAINING_DESIRED_RANGE = float(os.environ.get("GRAVITRUEX_DESIRED_RANGE", "480"))


def training_reward_coefficients() -> dict:
    """Reward settings shared by scalar and CUDA-batch training paths."""
    if TRAINING_REWARD_MODE == "team":
        # The individual ship still gets some damage/kill credit, but it is no longer
        # profitable to ignore team survival for a personal kill. Shared team progress
        # is distributed across current teammates and terminal victory dominates at 1–5v1–5.
        return {
            "damage_dealt": 0.035, "damage_taken": 0.020, "kill": 3.5, "death": 3.5,
            "enemy_dead": 3.0, "ally_dead": 3.0, "team_progress": 0.025,
            "terminal_win": 80.0, "terminal_kill": 3.0, "terminal_survivor": 8.0,
            "terminal_damage_div": 80.0, "terminal_damage_cap": 3.0, "terminal_bias": -0.5,
        }
    return {
        "damage_dealt": REWARD_DAMAGE_DEALT, "damage_taken": REWARD_DAMAGE_TAKEN,
        "kill": REWARD_PER_KILL, "death": REWARD_DEATH,
        "enemy_dead": 1.8, "ally_dead": 2.4, "team_progress": 0.0,
        "terminal_win": 24.0, "terminal_kill": 8.0, "terminal_survivor": 6.0,
        "terminal_damage_div": 65.0, "terminal_damage_cap": 3.0, "terminal_bias": -1.5,
    }

BLUE = "#63d9ff"
BLUE_DARK = "#1a86b4"
RED = "#ff806e"
RED_DARK = "#ba433e"
BG = "#07111f"
GRID = "#112337"
TEXT = "#dce8f5"
MUTED = "#8195aa"
ACCENT = "#8be0ca"

# The save folder sits next to this script, even if the script is launched
# from another working directory (e.g. a desktop shortcut).
SCRIPT_DIR = Path(__file__).resolve().parent
APP_CONFIG_PATH = SCRIPT_DIR / "gravitruex_config.json"
SAVE_DIR = SCRIPT_DIR / "GravitrueX"
WEIGHTS_PATH = SAVE_DIR / "policy_weights.npz"
STATS_PATH = SAVE_DIR / "training_stats.json"

# Set by the CPU-only launcher to guarantee no CUDA inference/training is selected.
FORCE_CPU = os.environ.get("GRAVITRUEX_FORCE_CPU", "0").strip().lower() in {"1", "true", "yes"}


def resolve_desktop_diagnostic_path(home: Optional[Path] = None, env: Optional[dict] = None) -> Path:
    """Find the Windows Desktop (including OneDrive redirection) for one compact snapshot log."""
    env = os.environ if env is None else env
    home_path = Path(home) if home is not None else Path.home()
    candidates = []
    for key in ("OneDriveCommercial", "OneDriveConsumer", "OneDrive"):
        base = env.get(key)
        if base:
            candidates.append(Path(base) / "Desktop")
    profile = env.get("USERPROFILE")
    if profile:
        candidates.append(Path(profile) / "Desktop")
    candidates.extend((home_path / "Desktop", home_path / "OneDrive" / "Desktop"))
    for candidate in candidates:
        try:
            if candidate.is_dir():
                return candidate / DESKTOP_DIAGNOSTIC_FILENAME
        except OSError:
            continue
    # Prefer the conventional Desktop path; the writer creates it if necessary.
    return home_path / "Desktop" / DESKTOP_DIAGNOSTIC_FILENAME


def write_compact_diagnostic_snapshot(path: Path, text: str) -> None:
    """Atomically overwrite one small snapshot; never append an unbounded diagnostic log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(str(text).rstrip() + "\n", encoding="utf-8")
    os.replace(temp, path)


def cycle_throughput(physics_steps: int, rollout_seconds: float, ppo_seconds: float) -> dict:
    """Separate raw rollout throughput from end-to-end throughput including PPO update."""
    steps = max(0, int(physics_steps))
    rollout = max(0.001, float(rollout_seconds))
    ppo = max(0.0, float(ppo_seconds))
    return {
        "rollout_steps_per_sec": steps / rollout,
        "end_to_end_steps_per_sec": steps / max(0.001, rollout + ppo),
        "rollout_plus_ppo_seconds": rollout + ppo,
    }


def health_bar_color(ratio: float) -> str:
    """Color code for the tiny in-world hull-health indicator."""
    ratio = clamp(float(ratio), 0.0, 1.0)
    return "#8be0ca" if ratio >= 0.66 else ("#ffc777" if ratio >= 0.33 else "#ff8794")


def _compact_file_status(path) -> str:
    """Small checkpoint status string for the desktop diagnostics snapshot."""
    try:
        path = Path(path)
        if not path.exists():
            return f"{path.name}: отсутствует"
        stat = path.stat()
        stamp = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
        return f"{path.name}, {stat.st_size:,} bytes, mtime={stamp}"
    except Exception as exc:
        return f"чекпоинт недоступен ({type(exc).__name__})"


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _read_jsonl_tail(path: Path, limit: int = 10) -> list:
    """Read only the most recent compact records needed for diagnostics."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 256_000), os.SEEK_SET)
            raw = handle.read().decode("utf-8", errors="replace").splitlines()
        rows = []
        for line in raw[-max(1, int(limit)) * 2:]:
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
            except Exception:
                continue
        return rows[-max(1, int(limit)): ]
    except Exception:
        return []


def _available_ram_bytes() -> Optional[int]:
    """Return currently available system RAM when psutil is installed."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _cuda_memory_snapshot() -> dict:
    """Sample PyTorch CUDA allocator counters separately from system VRAM usage."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gib = float(1024 ** 3)
        return {
            "allocated_gib": round(float(torch.cuda.memory_allocated()) / gib, 3),
            "reserved_gib": round(float(torch.cuda.memory_reserved()) / gib, 3),
            "peak_allocated_gib": round(float(torch.cuda.max_memory_allocated()) / gib, 3),
            "free_gib": round(float(free_bytes) / gib, 3),
            "total_gib": round(float(total_bytes) / gib, 3),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:160]}


def resolve_hybrid_limits(requested_workers: int, gpu_envs: int,
                          logical_cpus: Optional[int] = None,
                          available_ram_bytes: Optional[int] = None) -> dict:
    """Choose conservative hybrid limits while reserving CPU for CUDA dispatch/UI.

    The GPU batch stays on the main application's CUDA context. CPU rollout workers
    are separate processes and are intentionally capped/lowered in priority so they
    do not starve Python's CUDA dispatcher or the UI. A RAM-aware record cap prevents
    large Python trajectory dictionaries from consuming most of a 16-GB machine.
    """
    logical = max(1, int(logical_cpus or os.cpu_count() or 4))
    requested = max(1, int(requested_workers))
    reserve = max(2, int(math.ceil(logical * 0.25)))
    cpu_slots = max(1, logical - reserve)
    worker_ceiling = max(1, min(4, cpu_slots // 2))
    active = min(requested, worker_ceiling)

    record_cap = min(24_000, max(4_096, max(1, int(gpu_envs)) * 24))
    ram_gb = None
    if available_ram_bytes is not None:
        ram_gb = max(0.0, float(available_ram_bytes) / (1024 ** 3))
        if ram_gb < 1.5:
            # When <1.5 GiB remain, a CPU sidecar is counterproductive: GPU
            # rollout also stages packed trajectories through pinned host memory.
            active = 0
            record_cap = min(record_cap, 4_096)
        elif ram_gb < 3.0:
            active = min(active, 2)
            record_cap = min(record_cap, 8_192)
        elif ram_gb < 5.0:
            active = min(active, 3)
            record_cap = min(record_cap, 16_000)
    return {
        "requested_workers": requested,
        "active_workers": max(0, active),
        "logical_cpus": logical,
        "reserved_logical_cpus": min(reserve, max(0, logical - 1)),
        "available_ram_gb": ram_gb,
        "cpu_record_cap": int(record_cap),
        "worker_priority_lowered": True,
    }


def _lower_rollout_worker_priority() -> bool:
    """Best-effort lower priority for hybrid-only CPU producers."""
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            return bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000))  # BELOW_NORMAL_PRIORITY_CLASS
        if os.name == "posix":
            os.nice(3)
            return True
    except Exception:
        pass
    return False


def checkpoint_architecture(path: Path) -> Tuple[int, int]:
    """Read optional metadata from newer checkpoints; old NPZ files use default profile."""
    try:
        with np.load(path, allow_pickle=False) as data:
            width = int(np.asarray(data["__meta_width"]).reshape(-1)[0]) if "__meta_width" in data else 1536
            depth = int(np.asarray(data["__meta_depth"]).reshape(-1)[0]) if "__meta_depth" in data else 4
        if width not in (768, 1024, 1536, 2048, 3072, 4096) or not 2 <= depth <= 8:
            return 1536, 4
        return width, depth
    except Exception:
        return 1536, 4


def checkpoint_network_type(path: Path) -> str:
    """Read policy architecture from checkpoint; legacy weights are shared-residual."""
    try:
        with np.load(path, allow_pickle=False) as data:
            value = str(data["__meta_architecture"].reshape(-1)[0]) if "__meta_architecture" in data else "shared_residual"
        return value if value in {"shared_residual", "separate_critic"} else "shared_residual"
    except Exception:
        return "shared_residual"


def network_architecture_label(architecture: str) -> str:
    return {
        "shared_residual": "Простая PPO: общий actor/critic",
        "separate_critic": "Раздельный critic: actor учится без градиента value-сети",
    }.get(str(architecture), str(architecture))


MODEL_ROOT = SAVE_DIR / "models"
DEFAULT_MODEL_ID = "main"
SELFPLAY_LABEL = "Self-play: обучаемая модель с обеих сторон"
HEURISTIC_LABEL = "Эвристический ИИ"


def model_dir_for(model_id: str) -> Path:
    """Return the directory for one named model/checkpoint family."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(model_id or ""))
    if not safe_id:
        raise ValueError("Некорректный ID модели")
    return MODEL_ROOT / safe_id


def model_weights_path(model_id: str) -> Path:
    return model_dir_for(model_id) / f"weights_{model_id}_latest.npz"


def model_stats_path(model_id: str) -> Path:
    return model_dir_for(model_id) / "training_stats.json"


def model_meta_path(model_id: str) -> Path:
    return model_dir_for(model_id) / "model.json"


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def save_app_config(path: Path, settings: dict) -> None:
    """Atomically persist UI selections and training/combat settings, not model weights."""
    envelope = {
        "schema_version": APP_CONFIG_VERSION,
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "settings": settings if isinstance(settings, dict) else {},
    }
    _atomic_json(Path(path), envelope)


def load_app_config(path: Path) -> Optional[dict]:
    """Read a compatible settings file; invalid/missing files leave defaults untouched."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            return None
        version = int(payload.get("schema_version", 1))
        settings = payload.get("settings", payload)
        if version > APP_CONFIG_VERSION or not isinstance(settings, dict):
            return None
        return settings
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def episode_fits_record_budget(current_records: int, incoming_records: int, cap: int) -> bool:
    """A completed episode is accepted only if its whole record bundle fits the cap."""
    current_records = max(0, int(current_records))
    incoming_records = max(0, int(incoming_records))
    cap = max(0, int(cap))
    return incoming_records > 0 and current_records + incoming_records <= cap


def next_gpu_arena_fallback(count: int) -> Optional[int]:
    """Return the next smaller supported arena batch after CUDA out-of-memory."""
    count = max(1, int(count))
    return next((size for size in GPU_ARENA_FALLBACK_SIZES if size < count), None)


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Only retry actual device-memory errors; don't mask unrelated training bugs."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    message = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in message and any(token in message for token in ("cuda", "gpu", "cudnn", "cublas"))


def visible_test_horizon_reached(start_physics_step: int, current_physics_step: int,
                                 decision_horizon: int) -> bool:
    """Whether a visible evaluation match has consumed its policy-decision budget.

    Decisions are held for ACTION_REPEAT scalar physics ticks. Reaching this limit
    stops the test as unresolved; it must never be scored as a win or loss.
    """
    horizon = max(0, int(decision_horizon))
    elapsed = max(0, int(current_physics_step) - int(start_physics_step))
    return elapsed >= horizon * ACTION_REPEAT


def parse_decision_horizon(value: object, fallback: int = 2700) -> int:
    """Parse decision count from values like '2700 (180 с)'; legacy numeric config values remain valid."""
    match = re.match(r"\s*(\d+)", str(value or ""))
    if not match:
        return int(fallback)
    return int(clamp(int(match.group(1)), 120, 6000))


def read_model_meta(model_id: str) -> dict:
    path = model_meta_path(model_id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_model_meta(model_id: str, updates: dict) -> dict:
    data = read_model_meta(model_id)
    data.update(updates)
    data["id"] = model_id
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _atomic_json(model_meta_path(model_id), data)
    return data


def ensure_model_library() -> None:
    """Migrate the former single checkpoint into a named model folder once.

    Packaged starter metadata carries bundle_seed=true. On the first run after an
    upgrade, an existing legacy checkpoint is treated as user data regardless of
    zip/file timestamps. Once imported (or on a clean install), the seed flag is
    cleared, preventing stale legacy files from replacing later trained models.
    """
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    main_dir = model_dir_for(DEFAULT_MODEL_ID)
    main_dir.mkdir(parents=True, exist_ok=True)
    named_weights = model_weights_path(DEFAULT_MODEL_ID)
    named_stats = model_stats_path(DEFAULT_MODEL_ID)
    meta_path = model_meta_path(DEFAULT_MODEL_ID)
    prior_meta = read_model_meta(DEFAULT_MODEL_ID) if meta_path.exists() else {}
    bundled_seed = bool(prior_meta.get("bundle_seed", False))

    legacy_weights_newer = (WEIGHTS_PATH.exists() and named_weights.exists() and
                            WEIGHTS_PATH.stat().st_mtime > named_weights.stat().st_mtime + 1.0)
    migrated_legacy_weights = False
    if WEIGHTS_PATH.exists() and (not named_weights.exists() or legacy_weights_newer or bundled_seed):
        shutil.copy2(WEIGHTS_PATH, named_weights)
        migrated_legacy_weights = True

    legacy_stats_newer = (STATS_PATH.exists() and named_stats.exists() and
                          STATS_PATH.stat().st_mtime > named_stats.stat().st_mtime + 1.0)
    migrated_legacy_stats = False
    if STATS_PATH.exists() and (not named_stats.exists() or legacy_stats_newer or bundled_seed):
        try:
            shutil.copy2(STATS_PATH, named_stats)
            migrated_legacy_stats = True
        except OSError:
            pass

    # Write metadata when creating a catalogue or when a legacy file was imported.
    # On a clean packaged install, clear the one-time seed marker immediately.
    clear_seed_on_clean_install = bundled_seed and not WEIGHTS_PATH.exists() and not STATS_PATH.exists()
    if not meta_path.exists() or migrated_legacy_weights or migrated_legacy_stats or clear_seed_on_clean_install:
        width, depth = checkpoint_architecture(named_weights if named_weights.exists() else WEIGHTS_PATH)
        legacy_stats = {}
        try:
            with open(named_stats, "r", encoding="utf-8") as f:
                legacy_stats = json.load(f)
        except Exception:
            pass
        prior_meta = read_model_meta(DEFAULT_MODEL_ID) if meta_path.exists() else prior_meta
        now = datetime.now().isoformat(timespec="seconds")
        episodes = int(legacy_stats.get("episodes", prior_meta.get("episodes", 0)) or 0)
        steps = int(legacy_stats.get("steps", prior_meta.get("steps", 0)) or 0)
        imported_legacy = bool(prior_meta.get("legacy_imported", False) or migrated_legacy_weights or migrated_legacy_stats)
        _atomic_json(meta_path, {
            **prior_meta, "id": DEFAULT_MODEL_ID, "name": prior_meta.get("name", "Основная модель"),
            "created_at": prior_meta.get("created_at", now), "updated_at": now,
            "width": width, "depth": depth, "episodes": episodes, "steps": steps,
            "trained": bool(prior_meta.get("trained", False) or episodes > 0),
            "training_history_known": True, "bundle_seed": False, "legacy_imported": imported_legacy,
            "parent_model": prior_meta.get("parent_model"),
            "description": (
                "Импортирована из старой одиночной папки GravitrueX; несовпадающая старая архитектура переносится в базовую ветку."
                if (migrated_legacy_weights or migrated_legacy_stats or not prior_meta) else prior_meta.get("description", "Основная политика GravitrueX"))
        })


def list_model_records() -> list:
    ensure_model_library()
    records = []
    for folder in MODEL_ROOT.iterdir():
        if not folder.is_dir():
            continue
        model_id = folder.name
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", model_id):
            continue
        meta = read_model_meta(model_id)
        weight_path = model_weights_path(model_id)
        if not meta:
            if not weight_path.exists():
                continue
            width, depth = checkpoint_architecture(weight_path)
            meta = {"id": model_id, "name": model_id, "width": width, "depth": depth, "episodes": 0, "steps": 0}
            write_model_meta(model_id, meta)
        if weight_path.exists():
            width, depth = checkpoint_architecture(weight_path)
            meta["width"] = width
            meta["depth"] = depth
            # A checkpoint's embedded metadata is authoritative for its tensor layout.
            meta["architecture"] = checkpoint_network_type(weight_path)
        else:
            meta.setdefault("architecture", "shared_residual")
        meta["weights_path"] = str(weight_path)
        meta["has_weights"] = weight_path.exists()
        records.append(meta)
    records.sort(key=lambda x: (x.get("id") != DEFAULT_MODEL_ID, str(x.get("name", "")).casefold()))
    return records


def transliterate_model_slug(name: str) -> str:
    table = str.maketrans({
        "а":"a","б":"b","в":"v","г":"g","д":"d","е":"e","ё":"e","ж":"zh","з":"z","и":"i","й":"y","к":"k","л":"l","м":"m","н":"n","о":"o","п":"p","р":"r","с":"s","т":"t","у":"u","ф":"f","х":"kh","ц":"ts","ч":"ch","ш":"sh","щ":"sch","ъ":"","ы":"y","ь":"","э":"e","ю":"yu","я":"ya",
        "А":"a","Б":"b","В":"v","Г":"g","Д":"d","Е":"e","Ё":"e","Ж":"zh","З":"z","И":"i","Й":"y","К":"k","Л":"l","М":"m","Н":"n","О":"o","П":"p","Р":"r","С":"s","Т":"t","У":"u","Ф":"f","Х":"kh","Ц":"ts","Ч":"ch","Ш":"sh","Щ":"sch","Ъ":"","Ы":"y","Ь":"","Э":"e","Ю":"yu","Я":"ya"})
    value = name.translate(table).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return (value or "model")[:28]


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _rng_integer(rng, low: int, high: int) -> int:
    """Draw an integer from NumPy Generator or Python Random (high exclusive)."""
    numpy_integers = getattr(rng, "integers", None)
    if callable(numpy_integers):
        return int(numpy_integers(low, high))
    python_randrange = getattr(rng, "randrange", None)
    if callable(python_randrange):
        return int(python_randrange(low, high))
    raise TypeError(f"Unsupported random generator type: {type(rng).__name__}")


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrap_pos(x: float, size: float) -> float:
    return x % size


def wrap_delta(d: float, size: float) -> float:
    if d > size * 0.5:
        d -= size
    elif d < -size * 0.5:
        d += size
    return d


def torus_delta(ax: float, ay: float, bx: float, by: float) -> Tuple[float, float]:
    """Shortest vector from A to B in the wrap-around arena."""
    return wrap_delta(bx - ax, WORLD_W), wrap_delta(by - ay, WORLD_H)


def safe_norm(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


# ------------------------------- Ships / bullets ------------------------------
@dataclass
class Action:
    turn: int = 0       # -1 left, 0 coast, +1 right
    thrust: int = 0     # -1 reverse, 0 coast, +1 forward
    strafe: int = 0     # -1, 0, +1 in ship-local lateral direction
    fire: int = 0       # 0 / 1

    def as_array(self) -> np.ndarray:
        # Map -1,0,+1 to categorical indices 0,1,2.
        return np.asarray([self.turn + 1, self.thrust + 1, self.strafe + 1, self.fire], dtype=np.int64)

    @staticmethod
    def from_array(a: Sequence[int]) -> "Action":
        return Action(int(a[0]) - 1, int(a[1]) - 1, int(a[2]) - 1, int(a[3]))


@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    team: int
    owner_id: int
    ttl: float = BULLET_TTL
    alive: bool = True


class Ship:
    def __init__(self, sid: int, team: int, x: float, y: float, heading: float, rng: random.Random):
        self.sid = sid
        self.team = team  # 0 = blue/neural, 1 = red/heuristic opponent
        self.x, self.y = x, y
        self.vx = rng.uniform(-10.0, 10.0)
        self.vy = rng.uniform(-10.0, 10.0)
        self.heading = heading
        self.hp = MAX_HP
        self.shield = MAX_SHIELD
        self.cooldown = rng.uniform(0.05, 0.35)
        self.last_hit_time = -100.0
        self.alive = True
        self.damage_dealt = 0.0
        self.damage_taken = 0.0
        self.shots_fired = 0
        self.kills = 0
        self.deaths = 0
        self.last_action = Action()

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


class Arena:
    """Fixed-timestep 2D dogfight. Arena edges wrap around like a torus."""
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.ships: List[Ship] = []
        self.bullets: List[Bullet] = []
        self.sim_time = 0.0
        self.blue_n = 1
        self.red_n = 1
        self.round_over = False
        self.winner: Optional[int] = None
        self.step_count = 0
        self._next_sid = 0

    def reset(self, blue_n: int = 1, red_n: int = 1, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = random.Random(seed)
        self.blue_n = int(clamp(blue_n, 1, 5))
        self.red_n = int(clamp(red_n, 1, 5))
        self.ships = []
        self.bullets = []
        self.sim_time = 0.0
        self.round_over = False
        self.winner = None
        self.step_count = 0
        self._next_sid = 0
        # Spawn in the same engagement zone, but leave a large 5x arena around it.
        # This avoids wasting most early training episodes on long-distance travel.
        cx, cy = WORLD_W * 0.5, WORLD_H * 0.5
        team_offset = min(500.0, WORLD_W * 0.16)
        for team, count in ((0, self.blue_n), (1, self.red_n)):
            for i in range(count):
                x_base = cx + (-team_offset if team == 0 else team_offset)
                y_base = cy + (i - (count - 1) * 0.5) * 78.0
                x = x_base + self.rng.uniform(-55, 55)
                y = y_base + self.rng.uniform(-32, 32)
                heading = self.rng.uniform(-0.18, 0.18) if team == 0 else math.pi + self.rng.uniform(-0.18, 0.18)
                self.ships.append(Ship(self._next_sid, team, x, y, heading, self.rng))
                self._next_sid += 1

    def alive_ships(self, team: Optional[int] = None) -> List[Ship]:
        return [s for s in self.ships if s.alive and (team is None or s.team == team)]

    def ship_by_id(self, sid: int) -> Optional[Ship]:
        for s in self.ships:
            if s.sid == sid:
                return s
        return None

    def nearest_target(self, ship: Ship, candidates: Sequence[Ship]) -> Optional[Ship]:
        live = [e for e in candidates if e.alive and e.sid != ship.sid]
        if not live:
            return None
        # Distance is dominant; low-health enemies get a modest focus-fire bonus.
        return min(live, key=lambda e: self._target_score(ship, e))

    def _target_score(self, ship: Ship, target: Ship) -> float:
        dx, dy = torus_delta(ship.x, ship.y, target.x, target.y)
        return math.hypot(dx, dy) * (0.82 + 0.18 * target.hp / MAX_HP)

    def heuristic_action(self, ship: Ship) -> Action:
        """Simple non-learning bot: lead shots, maintain range, orbit and dodge."""
        enemies = self.alive_ships(1 - ship.team)
        target = self.nearest_target(ship, enemies)
        if target is None:
            return Action()
        dx, dy = torus_delta(ship.x, ship.y, target.x, target.y)
        dist = max(1.0, math.hypot(dx, dy))
        rvx, rvy = target.vx - ship.vx, target.vy - ship.vy
        lead_t = clamp(dist / BULLET_SPEED, 0.0, 0.72)
        aim_dx, aim_dy = dx + rvx * lead_t, dy + rvy * lead_t
        desired = math.atan2(aim_dy, aim_dx)
        err = wrap_angle(desired - ship.heading)
        turn = 0 if abs(err) < 0.045 else (1 if err > 0 else -1)
        if dist > 285:
            thrust = 1
        elif dist < 185:
            thrust = -1
        else:
            thrust = 0
        # Stable orbit direction; alternating sides prevent the whole team stacking.
        strafe = 1 if (ship.sid % 2 == 0) else -1
        # Prefer a lateral escape path if an incoming projectile is on collision course.
        dodge_score = 0.0
        dodge_side = 0
        nx, ny = -math.sin(ship.heading), math.cos(ship.heading)
        for b in self.bullets:
            if not b.alive or b.team == ship.team:
                continue
            rx, ry = torus_delta(ship.x, ship.y, b.x, b.y)  # ship -> bullet
            vx, vy = b.vx - ship.vx, b.vy - ship.vy
            vv = vx * vx + vy * vy
            if vv < 1e-5:
                continue
            t = -(rx * vx + ry * vy) / vv
            if not 0.0 < t < 0.72:
                continue
            cx, cy = rx + vx * t, ry + vy * t
            miss = math.hypot(cx, cy)
            if miss < 27.0:
                score = (27.0 - miss) / max(0.08, t)
                if score > dodge_score:
                    dodge_score = score
                    # Move toward the side which increases separation at predicted impact.
                    away_x, away_y = -cx, -cy
                    dot = nx * away_x + ny * away_y
                    dodge_side = 1 if dot >= 0 else -1
        if dodge_score > 0:
            strafe = dodge_side
            if dist < 230:
                thrust = 1
        fire = int(abs(err) < 0.105 and dist < 720 and ship.cooldown <= 0.06)
        return Action(turn, thrust, strafe, fire)

    def observation(self, ship: Ship) -> np.ndarray:
        """Fixed 96 features. Time-scale / camera zoom is intentionally absent; first 64 features preserve legacy meaning."""
        vals: List[float] = []
        c, s = math.cos(ship.heading), math.sin(ship.heading)
        local_vx = (ship.vx * c + ship.vy * s) / MAX_SPEED
        local_vy = (-ship.vx * s + ship.vy * c) / MAX_SPEED
        vals.extend([
            clamp(local_vx, -1, 1), clamp(local_vy, -1, 1),
            clamp(ship.speed / MAX_SPEED, 0, 1), ship.hp / MAX_HP,
            ship.shield / MAX_SHIELD, clamp(ship.cooldown / FIRE_COOLDOWN, 0, 1),
            ship.last_action.turn * 1.0, 1.0,
        ])

        enemies = self.alive_ships(1 - ship.team)
        target = self.nearest_target(ship, enemies)
        if target:
            dx, dy = torus_delta(ship.x, ship.y, target.x, target.y)
            rvx, rvy = target.vx - ship.vx, target.vy - ship.vy
            dist = math.hypot(dx, dy)
            err = wrap_angle(math.atan2(dy, dx) - ship.heading)
            vals.extend([
                clamp(dx / OBS_POS_X, -1, 1), clamp(dy / OBS_POS_Y, -1, 1),
                clamp(rvx / (MAX_SPEED * 2), -1, 1), clamp(rvy / (MAX_SPEED * 2), -1, 1),
                math.sin(err), math.cos(err), clamp(dist / 650.0, 0, 1),
                target.hp / MAX_HP, target.shield / MAX_SHIELD,
                clamp((rvx * dx + rvy * dy) / (max(1.0, dist) * MAX_SPEED * 2), -1, 1),
            ])
        else:
            vals.extend([0.0] * 10)

        # Three nearest enemies, six features each.
        enemy_rows = []
        for e in enemies:
            dx, dy = torus_delta(ship.x, ship.y, e.x, e.y)
            enemy_rows.append((math.hypot(dx, dy), [
                clamp(dx / OBS_POS_X, -1, 1), clamp(dy / OBS_POS_Y, -1, 1),
                clamp((e.vx - ship.vx) / (MAX_SPEED * 2), -1, 1),
                clamp((e.vy - ship.vy) / (MAX_SPEED * 2), -1, 1), e.hp / MAX_HP, e.shield / MAX_SHIELD,
            ]))
        enemy_rows.sort(key=lambda row: row[0])
        for i in range(3):
            vals.extend(enemy_rows[i][1] if i < len(enemy_rows) else [0.0] * 6)

        # Three nearest allies, four features each. Ignore self.
        allies = [a for a in self.alive_ships(ship.team) if a.sid != ship.sid]
        ally_rows = []
        for a in allies:
            dx, dy = torus_delta(ship.x, ship.y, a.x, a.y)
            ally_rows.append((math.hypot(dx, dy), [
                clamp(dx / OBS_POS_X, -1, 1), clamp(dy / OBS_POS_Y, -1, 1),
                a.hp / MAX_HP, clamp(a.speed / MAX_SPEED, 0, 1),
            ]))
        ally_rows.sort(key=lambda row: row[0])
        for i in range(3):
            vals.extend(ally_rows[i][1] if i < len(ally_rows) else [0.0] * 4)

        # Two nearest incoming projectiles: relative position and relative velocity.
        incoming = []
        for b in self.bullets:
            if b.alive and b.team != ship.team:
                dx, dy = torus_delta(ship.x, ship.y, b.x, b.y)
                incoming.append((math.hypot(dx, dy), [
                    clamp(dx / OBS_POS_X, -1, 1), clamp(dy / OBS_POS_Y, -1, 1),
                    clamp((b.vx - ship.vx) / (BULLET_SPEED * 1.5), -1, 1),
                    clamp((b.vy - ship.vy) / (BULLET_SPEED * 1.5), -1, 1),
                ]))
        incoming.sort(key=lambda row: row[0])
        for i in range(2):
            vals.extend(incoming[i][1] if i < len(incoming) else [0.0] * 4)

        # Team/tactical summary. No wall-clock time or time multiplier is included.
        living_allies = self.alive_ships(ship.team)
        living_enemies = self.alive_ships(1 - ship.team)
        mean_a = sum(a.hp / MAX_HP for a in living_allies) / max(1, len(living_allies))
        mean_e = sum(e.hp / MAX_HP for e in living_enemies) / max(1, len(living_enemies))
        closest_bullet_t = 1.0
        min_closing = 0.0
        if incoming:
            # Convert closest bullet distance to a small useful urgency feature.
            closest_bullet_t = clamp(incoming[0][0] / 240.0, 0, 1)
            rx, ry = torus_delta(ship.x, ship.y, next((b.x for b in self.bullets if b.alive and b.team != ship.team), ship.x),
                                  next((b.y for b in self.bullets if b.alive and b.team != ship.team), ship.y))
            min_closing = clamp(-((self._closest_incoming_velocity(ship)) / MAX_SPEED), -1, 1)
        target_angle = 0.0
        target_range = 1.0
        if target:
            dx, dy = torus_delta(ship.x, ship.y, target.x, target.y)
            target_angle = wrap_angle(math.atan2(dy, dx) - ship.heading) / math.pi
            target_range = clamp(math.hypot(dx, dy) / 650.0, 0, 1)
        vals.extend([
            len(living_allies) / 5.0, len(living_enemies) / 5.0,
            mean_a, mean_e, closest_bullet_t, min_closing,
            clamp(target_angle, -1, 1), target_range,
        ])

        # 32 added tactical features (the legacy 64 remain in the same order).
        # These give the policy more than raw proximity: interception, threat timing,
        # nearby formation geometry and local numerical superiority.
        if target:
            tdx, tdy = torus_delta(ship.x, ship.y, target.x, target.y)
            trvx, trvy = target.vx - ship.vx, target.vy - ship.vy
            tdist = max(1.0, math.hypot(tdx, tdy))
            closing = -(trvx * tdx + trvy * tdy) / tdist
            ttc = tdist / max(1.0, closing) if closing > 0 else 5.0
            lead_t = clamp(tdist / BULLET_SPEED, 0.0, 0.72)
            lead_err = wrap_angle(math.atan2(tdy + trvy * lead_t, tdx + trvx * lead_t) - ship.heading)
            velocity_angle = wrap_angle(math.atan2(trvy, trvx) - ship.heading) if abs(trvx) + abs(trvy) > 1e-5 else 0.0
            vals.extend([
                clamp(tdist / 1800.0, 0, 1), clamp(closing / (MAX_SPEED * 1.5), -1, 1),
                clamp(ttc / 5.0, 0, 1), math.sin(lead_err), math.cos(lead_err),
                clamp(math.hypot(trvx, trvy) / (MAX_SPEED * 2), 0, 1),
                math.sin(velocity_angle), math.cos(velocity_angle),
            ])
        else:
            vals.extend([0.0] * 8)

        extra_allies = []
        for ally in allies:
            adx, ady = torus_delta(ship.x, ship.y, ally.x, ally.y)
            extra_allies.append((math.hypot(adx, ady), [
                clamp(adx / 1500.0, -1, 1), clamp(ady / 1500.0, -1, 1),
                clamp((ally.vx - ship.vx) / (MAX_SPEED * 2), -1, 1),
                clamp((ally.vy - ship.vy) / (MAX_SPEED * 2), -1, 1),
                ally.hp / MAX_HP,
            ]))
        extra_allies.sort(key=lambda row: row[0])
        for i in range(2):
            vals.extend(extra_allies[i][1] if i < len(extra_allies) else [0.0] * 5)

        extra_enemies = []
        for enemy in enemies:
            edx, edy = torus_delta(ship.x, ship.y, enemy.x, enemy.y)
            evx, evy = enemy.vx - ship.vx, enemy.vy - ship.vy
            dist_e = max(1.0, math.hypot(edx, edy))
            vv = evx * evx + evy * evy
            tca = clamp(-(edx * evx + edy * evy) / vv, 0.0, 2.5) if vv > 1e-5 else 0.0
            miss = math.hypot(edx + evx * tca, edy + evy * tca)
            closing_e = -(edx * evx + edy * evy) / dist_e
            angle_e = wrap_angle(math.atan2(edy, edx) - ship.heading)
            extra_enemies.append((dist_e, [
                clamp(tca / 2.5, 0, 1), clamp(miss / 500.0, 0, 1),
                clamp(closing_e / (MAX_SPEED * 1.5), -1, 1), math.sin(angle_e),
            ]))
        extra_enemies.sort(key=lambda row: row[0])
        for i in range(2):
            vals.extend(extra_enemies[i][1] if i < len(extra_enemies) else [0.0] * 4)

        extra_bullets = []
        for bullet in self.bullets:
            if not bullet.alive or bullet.team == ship.team:
                continue
            bdx, bdy = torus_delta(ship.x, ship.y, bullet.x, bullet.y)
            bvx, bvy = bullet.vx - ship.vx, bullet.vy - ship.vy
            vv = bvx * bvx + bvy * bvy
            tca = clamp(-(bdx * bvx + bdy * bvy) / vv, 0.0, 2.5) if vv > 1e-5 else 0.0
            miss = math.hypot(bdx + bvx * tca, bdy + bvy * tca)
            extra_bullets.append((math.hypot(bdx, bdy), [clamp(tca / 2.5, 0, 1), clamp(miss / 60.0, 0, 1)]))
        extra_bullets.sort(key=lambda row: row[0])
        for i in range(2):
            vals.extend(extra_bullets[i][1] if i < len(extra_bullets) else [1.0, 1.0])

        nearby_allies = sum(1 for a in allies if math.hypot(*torus_delta(ship.x, ship.y, a.x, a.y)) < 600.0)
        nearby_enemies = sum(1 for e in enemies if math.hypot(*torus_delta(ship.x, ship.y, e.x, e.y)) < 600.0)
        vals.extend([nearby_allies / 4.0, nearby_enemies / 5.0])

        if len(vals) != OBS_SIZE:
            raise RuntimeError(f"Observation length {len(vals)} != {OBS_SIZE}")
        return np.asarray(vals, dtype=np.float32)

    def _closest_incoming_velocity(self, ship: Ship) -> float:
        best = None
        for b in self.bullets:
            if not b.alive or b.team == ship.team:
                continue
            dx, dy = torus_delta(ship.x, ship.y, b.x, b.y)
            dist = math.hypot(dx, dy)
            if best is None or dist < best[0]:
                best = (dist, b)
        if best is None:
            return 0.0
        b = best[1]
        dx, dy = torus_delta(ship.x, ship.y, b.x, b.y)
        dist = max(1.0, math.hypot(dx, dy))
        return ((b.vx - ship.vx) * dx + (b.vy - ship.vy) * dy) / dist

    def physics_tick(self, actions: Dict[int, Action]) -> None:
        self.sim_time += DT
        self.step_count += 1
        # Apply controls and translational inertia.
        for ship in self.ships:
            if not ship.alive:
                continue
            action = actions.get(ship.sid, Action())
            ship.last_action = action
            ship.heading = wrap_angle(ship.heading + action.turn * TURN_RATE * DT)
            fx, fy = math.cos(ship.heading), math.sin(ship.heading)
            lx, ly = -math.sin(ship.heading), math.cos(ship.heading)
            ship.vx += (fx * action.thrust * ACCEL_FORWARD + lx * action.strafe * ACCEL_STRAFE) * DT
            ship.vy += (fy * action.thrust * ACCEL_FORWARD + ly * action.strafe * ACCEL_STRAFE) * DT
            # Exponential drag is deliberately weak to preserve drift.
            drag = math.exp(-DRAG * DT)
            ship.vx *= drag
            ship.vy *= drag
            speed = ship.speed
            if speed > MAX_SPEED:
                factor = MAX_SPEED / speed
                ship.vx *= factor
                ship.vy *= factor
            ship.x = wrap_pos(ship.x + ship.vx * DT, WORLD_W)
            ship.y = wrap_pos(ship.y + ship.vy * DT, WORLD_H)
            ship.cooldown = max(0.0, ship.cooldown - DT)
            if ship.shield < MAX_SHIELD and self.sim_time - ship.last_hit_time > SHIELD_DELAY:
                ship.shield = min(MAX_SHIELD, ship.shield + SHIELD_REGEN * DT)
            if action.fire and ship.cooldown <= 1e-6:
                px = ship.x + fx * (SHIP_RADIUS + 4)
                py = ship.y + fy * (SHIP_RADIUS + 4)
                self.bullets.append(Bullet(
                    wrap_pos(px, WORLD_W), wrap_pos(py, WORLD_H),
                    ship.vx + fx * BULLET_SPEED, ship.vy + fy * BULLET_SPEED,
                    ship.team, ship.sid,
                ))
                ship.shots_fired += 1
                ship.cooldown = FIRE_COOLDOWN

        # Projectiles retain their velocity; collision checks happen every fixed tick.
        for b in self.bullets:
            if not b.alive:
                continue
            b.x = wrap_pos(b.x + b.vx * DT, WORLD_W)
            b.y = wrap_pos(b.y + b.vy * DT, WORLD_H)
            b.ttl -= DT
            if b.ttl <= 0:
                b.alive = False
                continue
            for ship in self.ships:
                if not ship.alive or ship.team == b.team or ship.sid == b.owner_id:
                    continue
                dx, dy = torus_delta(ship.x, ship.y, b.x, b.y)
                if dx * dx + dy * dy <= (SHIP_RADIUS + BULLET_RADIUS) ** 2:
                    damage = BULLET_DAMAGE
                    shield_damage = min(ship.shield, damage)
                    ship.shield -= shield_damage
                    remaining = damage - shield_damage
                    if remaining > 0:
                        ship.hp -= remaining
                    ship.last_hit_time = self.sim_time
                    ship.damage_taken += damage
                    owner = self.ship_by_id(b.owner_id)
                    if owner is not None:
                        owner.damage_dealt += damage
                    b.alive = False
                    if ship.hp <= 0 and ship.alive:
                        ship.hp = 0.0
                        ship.alive = False
                        ship.deaths += 1
                        if owner is not None and owner.team != ship.team:
                            owner.kills += 1
                    break
        if len(self.bullets) > 1500:
            # Safety cap; dead bullets are pruned immediately below.
            self.bullets = [b for b in self.bullets if b.alive]
        else:
            self.bullets = [b for b in self.bullets if b.alive]
        blue_alive = self.alive_ships(0)
        red_alive = self.alive_ships(1)
        if not blue_alive or not red_alive:
            self.round_over = True
            self.winner = 0 if blue_alive else 1 if red_alive else None

    def decision_step(self, actions: Dict[int, Action]) -> None:
        """Advance fixed physics ticks; supplied actions override the heuristic fallback."""
        all_actions = dict(actions)
        for ship in self.alive_ships(1):
            if ship.sid not in all_actions:
                all_actions[ship.sid] = self.heuristic_action(ship)
        for _ in range(ACTION_REPEAT):
            self.physics_tick(all_actions)
            if self.round_over:
                break


# ------------------------------- Neural policy -------------------------------
def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(np.clip(z, -40.0, 40.0))
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


def fire_prior_bias_numpy(obs: np.ndarray) -> np.ndarray:
    """Softly favor firing only with a live target, ready weapon, close range and lead alignment.

    Observation indices are the documented 96-feature layout: cooldown=5,
    enemy count=57, target distance/1800=64 and cosine(lead error)=68.
    The result is a logit bonus, not a hard-coded fire command; PPO can learn to override it. The larger bonus counters legacy weights that collapsed toward never-firing.
    """
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[-1] < OBS_SIZE:
        return np.zeros(x.shape[:-1], dtype=np.float32)
    ready = np.clip((0.24 - x[..., 5]) / 0.12, 0.0, 1.0)
    has_enemy = (x[..., 57] > 0.0).astype(np.float32)
    # A weapon has no ammunition cost: when the predicted intercept is good, firing
    # should be the default unless PPO has learned a genuinely strong reason not to.
    range_ok = np.clip((0.52 - x[..., 64]) / 0.14, 0.0, 1.0)
    aligned = np.clip((x[..., 68] - 0.94) / 0.06, 0.0, 1.0)
    return (8.0 * ready * has_enemy * range_ok * aligned).astype(np.float32)


def fire_prior_bias_torch(torch, obs):
    """Torch twin of fire_prior_bias_numpy; stays on the input device for CUDA graph capture."""
    if obs.shape[-1] < OBS_SIZE:
        return torch.zeros(obs.shape[:-1], dtype=obs.dtype, device=obs.device)
    ready = torch.clamp((0.24 - obs[..., 5]) / 0.12, 0.0, 1.0)
    has_enemy = (obs[..., 57] > 0.0).to(obs.dtype)
    range_ok = torch.clamp((0.52 - obs[..., 64]) / 0.14, 0.0, 1.0)
    aligned = torch.clamp((obs[..., 68] - 0.94) / 0.06, 0.0, 1.0)
    return 8.0 * ready * has_enemy * range_ok * aligned


def fire_window_mask_numpy(obs: np.ndarray) -> np.ndarray:
    """True only when a real enemy is in a good, legal firing window.

    This is a small action prior, not an extra observation: it keeps old policies
    that learned a strongly negative fire logit from becoming permanently pacifist.
    Conditions intentionally match the game's no-ammo-cost weapon and lead-aim features.
    """
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[-1] < OBS_SIZE:
        return np.zeros(x.shape[:-1], dtype=bool)
    return ((x[..., 57] > 0.0) & (x[..., 5] <= 0.28) &
            (x[..., 64] <= 0.50) & (x[..., 68] >= 0.94))


def apply_fire_prior_numpy(logits: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Apply a selectable firing prior: full (legacy), soft (no hard floor), or off."""
    values = np.asarray(logits, dtype=np.float32)
    mode = ACTION_PRIORS_MODE
    if mode == "off":
        return values
    bias = fire_prior_bias_numpy(obs)
    mask = fire_window_mask_numpy(obs)
    if values.ndim == bias.ndim + 1:
        bias = bias[..., None]
        mask = mask[..., None]
    if mode == "soft":
        # No forced 90% probability floor. The network may learn not to fire.
        return (values + 0.35 * bias).astype(np.float32)
    adjusted = values + bias
    # Legacy full prior retained exactly for controlled A/B tests.
    return np.where(mask, np.maximum(adjusted, np.float32(2.2)), adjusted).astype(np.float32)

def apply_fire_prior_torch(torch, logits, obs):
    """Torch twin of the selected firing prior; no host sync in the CUDA path."""
    mode = ACTION_PRIORS_MODE
    if mode == "off":
        return logits
    bias = fire_prior_bias_torch(torch, obs)
    adjusted = logits + (0.35 * bias if mode == "soft" else bias)
    if mode == "soft" or obs.shape[-1] < OBS_SIZE:
        return adjusted
    mask = ((obs[..., 57] > 0.0) & (obs[..., 5] <= 0.28) &
            (obs[..., 64] <= 0.50) & (obs[..., 68] >= 0.94))
    floor = torch.full_like(adjusted, 2.2)
    return torch.where(mask, torch.maximum(adjusted, floor), adjusted)


def apply_target_turn_prior_numpy(logits: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Weakly favor turning toward the lead intercept; shared by rollout and PPO.

    Observation 67/68 contains sin/cos of lead error. Positive angular error
    corresponds to a positive/right turn (action index 2), negative to left (0).
    """
    out = np.asarray(logits, dtype=np.float32).copy()
    if ACTION_PRIORS_MODE == "off":
        return out
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[-1] < OBS_SIZE or out.shape[-1] != 3:
        return out
    has_target = x[..., 57] > 0.0
    lead_sin = x[..., 67]
    lead_cos = x[..., 68]
    strength_scale = 0.35 if ACTION_PRIORS_MODE == "soft" else 1.0
    strength = strength_scale * 1.15 * np.clip(np.abs(lead_sin) / 0.40, 0.0, 1.0) * has_target
    left = has_target & (lead_sin < -0.02)
    right = has_target & (lead_sin > 0.02)
    out[left, 0] += strength[left]
    out[right, 2] += strength[right]
    # If the target is almost exactly behind, either direction is better than coasting.
    behind = has_target & (lead_cos < -0.92) & (np.abs(lead_sin) < 0.40)
    behind_bias = 0.2 if ACTION_PRIORS_MODE == "soft" else 0.55
    out[behind, 0] += behind_bias
    out[behind, 2] += behind_bias
    return out

def apply_target_turn_prior_torch(torch, logits, obs):
    """Torch twin of the selectable target-turn prior."""
    if ACTION_PRIORS_MODE == "off" or obs.shape[-1] < OBS_SIZE or logits.shape[-1] != 3:
        return logits
    has_target = obs[..., 57] > 0.0
    lead_sin = obs[..., 67]
    lead_cos = obs[..., 68]
    strength_scale = 0.35 if ACTION_PRIORS_MODE == "soft" else 1.0
    strength = strength_scale * 1.15 * torch.clamp(lead_sin.abs() / 0.40, 0.0, 1.0) * has_target.to(obs.dtype)
    left = has_target & (lead_sin < -0.02)
    right = has_target & (lead_sin > 0.02)
    behind = has_target & (lead_cos < -0.92) & (lead_sin.abs() < 0.40)
    bias = torch.zeros_like(logits)
    behind_bias = 0.2 if ACTION_PRIORS_MODE == "soft" else 0.55
    bias[..., 0] = left.to(obs.dtype) * strength + behind.to(obs.dtype) * behind_bias
    bias[..., 2] = right.to(obs.dtype) * strength + behind.to(obs.dtype) * behind_bias
    return logits + bias

def _dodge_choice_numpy(obs: np.ndarray):
    """Find a threatening incoming bullet and the better lateral dodge direction.

    Bullet slots 48..55 hold relative position/velocity; 90..93 hold normalized
    time-to-closest-approach and miss distance. Returns (valid, side) per observation.
    """
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    n = x.shape[0]
    if x.shape[-1] < OBS_SIZE:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=np.int64)
    tca0, miss0, tca1, miss1 = x[:, 90], x[:, 91], x[:, 92], x[:, 93]
    threat0 = (tca0 <= 0.55) & (miss0 <= 0.45)
    threat1 = (tca1 <= 0.55) & (miss1 <= 0.45)
    choose0 = threat0 & (~threat1 | ((tca0 + miss0) <= (tca1 + miss1)))
    choose1 = threat1 & ~choose0
    valid = (choose0 | choose1) & (x[:, 57] > 0.0)
    rx = np.where(choose0, x[:, 48] * OBS_POS_X, x[:, 52] * OBS_POS_X)
    ry = np.where(choose0, x[:, 49] * OBS_POS_Y, x[:, 53] * OBS_POS_Y)
    rvx = np.where(choose0, x[:, 50], x[:, 54])
    rvy = np.where(choose0, x[:, 51], x[:, 55])
    target_bearing = np.arctan2(x[:, 9] * OBS_POS_Y, x[:, 8] * OBS_POS_X)
    direct_err = np.arctan2(x[:, 12], x[:, 13])
    heading = target_bearing - direct_err
    v_forward = rvx * np.cos(heading) + rvy * np.sin(heading)
    cross = rx * rvy - ry * rvx
    signed = cross * v_forward
    valid &= np.abs(signed) > 1e-7
    side = np.where(signed > 0.0, 1, -1).astype(np.int64)
    return valid, side

def apply_dodge_prior_numpy(logits: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Prefer a safe strafe direction for imminent threats, controlled by A/B mode."""
    out = np.asarray(logits, dtype=np.float32).copy()
    if ACTION_PRIORS_MODE == "off":
        return out
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[-1] < OBS_SIZE or out.shape[-1] != 3:
        return out
    valid, side = _dodge_choice_numpy(x)
    if np.any(valid & (side > 0)):
        mask = valid & (side > 0)
        other_max = np.maximum(out[mask, 0], out[mask, 1])
        margin = 0.45 if ACTION_PRIORS_MODE == "soft" else 1.5
        out[mask, 2] = np.maximum(out[mask, 2], other_max + margin)
    if np.any(valid & (side < 0)):
        mask = valid & (side < 0)
        other_max = np.maximum(out[mask, 1], out[mask, 2])
        margin = 0.45 if ACTION_PRIORS_MODE == "soft" else 1.5
        out[mask, 0] = np.maximum(out[mask, 0], other_max + margin)
    return out

def apply_dodge_prior_torch(torch, logits, obs):
    """Torch / CUDA graph twin of apply_dodge_prior_numpy; no host sync or loops."""
    if ACTION_PRIORS_MODE == "off" or obs.shape[-1] < OBS_SIZE or logits.shape[-1] != 3:
        return logits
    tca0, miss0, tca1, miss1 = obs[..., 90], obs[..., 91], obs[..., 92], obs[..., 93]
    threat0 = (tca0 <= 0.55) & (miss0 <= 0.45)
    threat1 = (tca1 <= 0.55) & (miss1 <= 0.45)
    choose0 = threat0 & (~threat1 | ((tca0 + miss0) <= (tca1 + miss1)))
    choose1 = threat1 & ~choose0
    valid = (choose0 | choose1) & (obs[..., 57] > 0.0)
    rx = torch.where(choose0, obs[..., 48] * OBS_POS_X, obs[..., 52] * OBS_POS_X)
    ry = torch.where(choose0, obs[..., 49] * OBS_POS_Y, obs[..., 53] * OBS_POS_Y)
    rvx = torch.where(choose0, obs[..., 50], obs[..., 54])
    rvy = torch.where(choose0, obs[..., 51], obs[..., 55])
    target_bearing = torch.atan2(obs[..., 9] * OBS_POS_Y, obs[..., 8] * OBS_POS_X)
    direct_err = torch.atan2(obs[..., 12], obs[..., 13])
    heading = target_bearing - direct_err
    v_forward = rvx * torch.cos(heading) + rvy * torch.sin(heading)
    cross = rx * rvy - ry * rvx
    signed = cross * v_forward
    valid = valid & (signed.abs() > 1e-7)
    left = valid & (signed < 0.0)
    right = valid & (signed > 0.0)
    margin = 0.45 if ACTION_PRIORS_MODE == "soft" else 1.5
    left_logit = torch.maximum(logits[..., 0], torch.maximum(logits[..., 1], logits[..., 2]) + margin)
    right_logit = torch.maximum(logits[..., 2], torch.maximum(logits[..., 0], logits[..., 1]) + margin)
    out_left = torch.where(left, left_logit, logits[..., 0])
    out_right = torch.where(right, right_logit, logits[..., 2])
    return torch.stack((out_left, logits[..., 1], out_right), dim=-1)



class PackedExperience:
    """Compact rollout records stored as NumPy arrays instead of per-step dicts.

    GPU vector rollouts can contain hundreds of thousands of records. Keeping each
    record as a Python dictionary creates a large CPU and allocator bottleneck, so
    GPU batches return one packed array bundle per arena and PPO consumes it directly.
    """
    __slots__ = ("obs", "action", "logp", "value", "reward", "done", "adv", "returns",
                 "teacher_action", "teacher_mask")

    def __init__(self, obs, action, logp, value, reward, done, adv, returns,
                 teacher_action=None, teacher_mask=None):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.action = np.asarray(action, dtype=np.int64)
        self.logp = np.asarray(logp, dtype=np.float32).reshape(-1)
        self.value = np.asarray(value, dtype=np.float32).reshape(-1)
        self.reward = np.asarray(reward, dtype=np.float32).reshape(-1)
        self.done = np.asarray(done, dtype=np.bool_).reshape(-1)
        self.adv = np.asarray(adv, dtype=np.float32).reshape(-1)
        self.returns = np.asarray(returns, dtype=np.float32).reshape(-1)
        # Teacher labels are optional so older tests, checkpoints, and callers remain valid.
        self.teacher_action = (self.action.copy() if teacher_action is None else
                               np.asarray(teacher_action, dtype=np.int64))
        self.teacher_mask = (np.zeros(len(self.logp), dtype=np.bool_) if teacher_mask is None else
                             np.asarray(teacher_mask, dtype=np.bool_).reshape(-1))
        if self.teacher_action.shape != self.action.shape:
            raise ValueError("PackedExperience: teacher_action must match action shape")
        n = len(self.logp)
        if self.obs.shape[0] != n or self.action.shape[0] != n:
            raise ValueError("PackedExperience: размерности массивов не совпадают")
        for name in ("value", "reward", "done", "adv", "returns", "teacher_mask"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"PackedExperience: длина {name} не совпадает")

    def __len__(self):
        return int(self.logp.shape[0])

    def __bool__(self):
        return len(self) > 0

    @classmethod
    def from_records(cls, records):
        if not records:
            return cls(np.empty((0, OBS_SIZE), np.float32), np.empty((0, 4), np.int64),
                       np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.float32),
                       np.empty(0, np.bool_), np.empty(0, np.float32), np.empty(0, np.float32))
        return cls(
            np.asarray([r["obs"] for r in records], dtype=np.float32),
            np.asarray([r["action"] for r in records], dtype=np.int64),
            np.fromiter((r["logp"] for r in records), dtype=np.float32, count=len(records)),
            np.fromiter((r["value"] for r in records), dtype=np.float32, count=len(records)),
            np.fromiter((r["reward"] for r in records), dtype=np.float32, count=len(records)),
            np.fromiter((r["done"] for r in records), dtype=np.bool_, count=len(records)),
            np.fromiter((r["adv"] for r in records), dtype=np.float32, count=len(records)),
            np.fromiter((r["return"] for r in records), dtype=np.float32, count=len(records)),
            np.asarray([r.get("teacher_action", r["action"]) for r in records], dtype=np.int64),
            np.fromiter((bool(r.get("teacher_mask", False)) for r in records), dtype=np.bool_, count=len(records)),
        )

    @classmethod
    def concatenate(cls, batches):
        batches = [b for b in batches if b is not None and len(b) > 0]
        if not batches:
            return cls.from_records([])
        if len(batches) == 1:
            return batches[0]
        return cls(*(np.concatenate([getattr(b, field) for b in batches], axis=0)
                     for field in cls.__slots__))


class ActorCritic:
    """Residual actor-critic: compatible legacy policy + configurable deep residual MLP.

    Rollout inference is NumPy (so worker processes don't create CUDA contexts).
    PPO updates use PyTorch on CUDA when available / selected.
    """
    BASE_HIDDEN = 128

    def __init__(self, seed: Optional[int] = None, hidden: int = HIDDEN,
                 width: int = 1536, depth: int = 4, device_mode: str = "auto",
                 allocate: bool = True, architecture: str = "shared_residual"):
        self.rng = np.random.default_rng(seed)
        self.hidden = self.BASE_HIDDEN  # legacy branch width, retained for old APIs
        self.width = int(max(32, width))
        self.depth = int(max(2, depth))
        self.device_mode = str(device_mode or "auto").lower()
        self.architecture = str(architecture or "shared_residual").lower()
        if self.architecture not in {"shared_residual", "separate_critic"}:
            raise ValueError(f"Неизвестная архитектура: {self.architecture}")
        self.params: Dict[str, np.ndarray] = {}
        self.last_policy_grad_norm = 0.0
        self.last_value_grad_norm = 0.0
        self.last_grad_ratio = 0.0
        self.last_approx_kl = 0.0
        self.last_clip_fraction = 0.0
        self.last_entropy = 0.0
        self.last_value_loss = 0.0
        self.last_explained_variance = 0.0
        self.last_optimizer_steps = 0
        self.lr = 2.5e-4
        self._torch_params = None
        self._torch_optimizer = None
        self._torch_device = None
        self._inference_params = None
        self._inference_device = None
        self.allow_gpu_inference = False
        # Inference can be selected independently from the PPO optimizer device.
        # CPU is the low-latency default for small live battles; GPU is an explicit experiment.
        self.inference_mode = "cpu"  # cpu | gpu | auto
        self.last_inference_backend = "CPU / NumPy"
        self.last_device = "NumPy rollout / not training"
        if allocate:
            self._init_weights()
        else:
            self.params = {}

    @staticmethod
    def _xavier(rng, fan_in: int, fan_out: int) -> np.ndarray:
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        return rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)

    def _init_weights(self) -> None:
        h = self.BASE_HIDDEN
        w = self.width
        p: Dict[str, np.ndarray] = {
            # Legacy-compatible actor trunk and action heads.
            "W1": self._xavier(self.rng, OBS_SIZE, h), "b1": np.zeros(h, np.float32),
            "W2": self._xavier(self.rng, h, h), "b2": np.zeros(h, np.float32),
            "Wt": self._xavier(self.rng, h, 3), "bt": np.zeros(3, np.float32),
            "Wf": self._xavier(self.rng, h, 3), "bf": np.zeros(3, np.float32),
            "Ws": self._xavier(self.rng, h, 3), "bs": np.zeros(3, np.float32),
            "Wfire": self._xavier(self.rng, h, 1), "bfire": np.zeros(1, np.float32),
        }
        if self.architecture == "shared_residual":
            p["Wv"] = self._xavier(self.rng, h, 1)
            p["bv"] = np.zeros(1, np.float32)
        fan = OBS_SIZE
        for i in range(self.depth):
            p[f"G_W{i}"] = self._xavier(self.rng, fan, w)
            p[f"G_b{i}"] = np.zeros(w, np.float32)
            fan = w
        # Actor residual heads start at zero, so initial behavior remains the legacy base.
        for key, out in (("G_Wt", 3), ("G_Wf", 3), ("G_Ws", 3), ("G_Wfire", 1)):
            p[key] = np.zeros((w, out), np.float32)
            p[key.replace("_W", "_b")] = np.zeros(out, np.float32)
        if self.architecture == "shared_residual":
            p["G_Wv"] = np.zeros((w, 1), np.float32)
            p["G_bv"] = np.zeros(1, np.float32)
        else:
            # Experimental independent critic: its trunk cannot pull the actor by the
            # value-loss gradient. This is not centralized/MAPPO critic yet.
            cfan = OBS_SIZE
            for i in range(self.depth):
                p[f"C_W{i}"] = self._xavier(self.rng, cfan, w)
                p[f"C_b{i}"] = np.zeros(w, np.float32)
                cfan = w
            p["C_Wv"] = self._xavier(self.rng, w, 1)
            p["C_bv"] = np.zeros(1, np.float32)
        self.params = p

    def _large_residual_active(self) -> bool:
        # Check output heads only. Before first learning update these are all exactly zero,
        # so old checkpoints can play with legacy cost instead of multiplying a huge trunk.
        for key in ("G_Wt", "G_Wf", "G_Ws", "G_Wfire", "G_bt", "G_bf", "G_bs", "G_bfire"):
            if np.any(self.params[key]):
                return True
        return False

    def forward(self, obs: np.ndarray, cache: bool = False):
        x = np.asarray(obs, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        p = self.params
        # Base path with legacy names.
        h1 = np.tanh(x @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        logits = {
            "turn": h2 @ p["Wt"] + p["bt"],
            "thrust": h2 @ p["Wf"] + p["bf"],
            "strafe": h2 @ p["Ws"] + p["bs"],
            "fire": h2 @ p["Wfire"] + p["bfire"],
        }
        values = ((h2 @ p["Wv"] + p["bv"]).reshape(-1)
                  if self.architecture == "shared_residual" else None)
        # Large trainable residual path. If its output heads are still zero, skip all
        # large matrix multiplies until the first PPO update gives the branch a signal.
        if self._large_residual_active():
            g = x
            for i in range(self.depth):
                g = np.tanh(g @ p[f"G_W{i}"] + p[f"G_b{i}"])
            for key, wk, bk in (("turn", "G_Wt", "G_bt"), ("thrust", "G_Wf", "G_bf"),
                                ("strafe", "G_Ws", "G_bs"), ("fire", "G_Wfire", "G_bfire")):
                logits[key] = logits[key] + g @ p[wk] + p[bk]
            if self.architecture == "shared_residual":
                values = values + (g @ p["G_Wv"] + p["G_bv"]).reshape(-1)
        if self.architecture == "separate_critic":
            c = x
            for i in range(self.depth):
                c = np.tanh(c @ p[f"C_W{i}"] + p[f"C_b{i}"])
            values = (c @ p["C_Wv"] + p["C_bv"]).reshape(-1)
        # Same observation-derived combat priors are applied in sampling, live inference and PPO.
        logits["turn"] = apply_target_turn_prior_numpy(logits["turn"], x)
        logits["strafe"] = apply_dodge_prior_numpy(logits["strafe"], x)
        logits["fire"] = apply_fire_prior_numpy(logits["fire"], x)
        if cache:
            return logits, values, None
        return logits, values

    def _forward_gpu_inference(self, X: np.ndarray):
        import torch
        if not torch.cuda.is_available():
            return None
        if self._inference_params is None or self._inference_device != "cuda":
            # Cache weights on CUDA for play-time inference. This is inference-only;
            # workers deliberately stay on NumPy/CPU and the PPO optimizer is separate.
            self._inference_params = {k: torch.as_tensor(v.copy(), dtype=torch.float32, device="cuda")
                                      for k, v in self.params.items()}
            self._inference_device = "cuda"
        p = self._inference_params
        with torch.inference_mode():
            x = torch.as_tensor(X, dtype=torch.float32, device="cuda")
            h = torch.tanh(x @ p["W1"] + p["b1"])
            h = torch.tanh(h @ p["W2"] + p["b2"])
            logits = {
                "turn": h @ p["Wt"] + p["bt"],
                "thrust": h @ p["Wf"] + p["bf"],
                "strafe": h @ p["Ws"] + p["bs"],
                "fire": (h @ p["Wfire"] + p["bfire"]).squeeze(-1),
            }
            values = ((h @ p["Wv"] + p["bv"]).reshape(-1)
                      if self.architecture == "shared_residual" else None)
            if self._large_residual_active():
                g = x
                for i in range(self.depth):
                    g = torch.tanh(g @ p[f"G_W{i}"] + p[f"G_b{i}"])
                logits["turn"] = logits["turn"] + g @ p["G_Wt"] + p["G_bt"]
                logits["thrust"] = logits["thrust"] + g @ p["G_Wf"] + p["G_bf"]
                logits["strafe"] = logits["strafe"] + g @ p["G_Ws"] + p["G_bs"]
                logits["fire"] = logits["fire"] + (g @ p["G_Wfire"] + p["G_bfire"]).squeeze(-1)
                if self.architecture == "shared_residual":
                    values = values + (g @ p["G_Wv"] + p["G_bv"]).squeeze(-1)
            if self.architecture == "separate_critic":
                c = x
                for i in range(self.depth):
                    c = torch.tanh(c @ p[f"C_W{i}"] + p[f"C_b{i}"])
                values = (c @ p["C_Wv"] + p["C_bv"]).reshape(-1)
            logits["turn"] = apply_target_turn_prior_torch(torch, logits["turn"], x)
            logits["strafe"] = apply_dodge_prior_torch(torch, logits["strafe"], x)
            logits["fire"] = apply_fire_prior_torch(torch, logits["fire"], x)
            return ({k: v.detach().float().cpu().numpy() for k, v in logits.items()},
                    values.detach().float().cpu().numpy())

    def predict_batch(self, observations: Sequence[np.ndarray], deterministic: bool = True):
        X = np.asarray(observations, dtype=np.float32)
        gpu_result = None
        mode = str(getattr(self, "inference_mode", "cpu")).lower()
        strict_gpu = bool(getattr(self, "require_gpu_inference", False))
        use_gpu = strict_gpu or bool(self.allow_gpu_inference and mode in ("gpu", "auto"))
        # Auto avoids CUDA launch/copy overhead for small batches and small policies.
        # Visible battle policies set strict_gpu=True: they must not silently fall back to CPU.
        if mode == "auto" and not strict_gpu:
            use_gpu = use_gpu and (len(X) >= 4 or (self.width >= 1536 and self._large_residual_active()))
        if use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    if strict_gpu:
                        raise RuntimeError("CUDA недоступна, но для видимого боя выбран строгий GPU-инференс")
                else:
                    gpu_result = self._forward_gpu_inference(X)
                    if gpu_result is None and strict_gpu:
                        raise RuntimeError("CUDA inference backend вернул пустой результат")
            except Exception as exc:
                self.last_inference_backend = f"CUDA ERROR: {type(exc).__name__}"
                if strict_gpu:
                    raise RuntimeError(f"GPU-only inference failed: {type(exc).__name__}: {exc}") from exc
                gpu_result = None
        if gpu_result is not None:
            logits, values = gpu_result
            self.last_inference_backend = "CUDA / batched policy"
        elif strict_gpu:
            raise RuntimeError("GPU-only inference failed; CPU fallback is disabled for visible battle")
        else:
            logits, values = self.forward(X)
            self.last_inference_backend = "CPU / NumPy" if not use_gpu else "CPU / fallback (CUDA unavailable or error)"
        probs = {k: softmax(v) for k, v in logits.items() if k != "fire"}
        fire_p = 1.0 / (1.0 + np.exp(-np.clip(logits["fire"].reshape(-1), -30, 30)))
        actions, logps, outs = [], [], []
        for i in range(len(X)):
            a = []
            logp = 0.0
            for key in ("turn", "thrust", "strafe"):
                pr = probs[key][i]
                if deterministic:
                    idx = int(np.argmax(pr))
                else:
                    idx = int(self.rng.choice(3, p=pr / pr.sum()))
                a.append(idx)
                logp += math.log(max(1e-8, float(pr[idx])))
            fp = float(fire_p[i])
            if deterministic:
                fire = int(fp >= 0.52)
            else:
                fire = int(self.rng.random() < fp)
            a.append(fire)
            logp += math.log(max(1e-8, fp if fire else 1.0 - fp))
            actions.append(Action.from_array(a))
            logps.append(logp)
            outs.append(a)
        return actions, np.asarray(logps, np.float32), values.astype(np.float32), np.asarray(outs, np.int64)

    def count_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    def _reset_torch_state(self) -> None:
        self._torch_params = None
        self._torch_optimizer = None
        self._torch_device = None
        self._inference_params = None
        self._inference_device = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = {k: np.asarray(v, dtype=np.float32) for k, v in self.params.items()}
        payload["__meta_width"] = np.asarray([self.width], dtype=np.int32)
        payload["__meta_depth"] = np.asarray([self.depth], dtype=np.int32)
        payload["__meta_obs"] = np.asarray([OBS_SIZE], dtype=np.int32)
        payload["__meta_architecture"] = np.asarray([self.architecture], dtype="U32")
        # ZIP compression is disproportionately slow for large, nearly-random float weights.
        # Smaller checkpoints stay compact; large networks are saved uncompressed for speed.
        with open(tmp, "wb") as f:
            if self.count_params() <= 5_000_000:
                np.savez_compressed(f, **payload)
            else:
                np.savez(f, **payload)
        os.replace(tmp, path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            with np.load(path, allow_pickle=False) as data:
                saved_arch = (str(data["__meta_architecture"][0])
                              if "__meta_architecture" in data.files else "shared_residual")
                if saved_arch != self.architecture:
                    return False
                old = {k: np.asarray(data[k], dtype=np.float32).copy()
                       for k in data.files if not k.startswith("__meta_")}
            loaded_any = False
            # Copy every parameter whose shape is identical (allows width/depth changes
            # while preserving matching foundation or matching layers).
            for key, target in self.params.items():
                if key in old and old[key].shape == target.shape:
                    target[...] = old[key]
                    loaded_any = True
            # Support pre-v3 64x64 and v3 96x128 base policies. If only a legacy
            # checkpoint exists, map its old path into the base branch and leave the
            # new branch as a zero-residual extension.
            if "W1" in old and "W2" in old and not (old["W1"].shape == self.params["W1"].shape and old["W2"].shape == self.params["W2"].shape):
                old_w1, old_w2 = old["W1"], old["W2"]
                if old_w1.ndim == 2 and old_w2.ndim == 2 and old_w1.shape[0] in (64, OBS_SIZE) and old_w2.shape[0] == old_w2.shape[1]:
                    # Reset the base matrices first to avoid random weights on unseen features.
                    self.params["W1"].fill(0.0)
                    self.params["b1"].fill(0.0)
                    self.params["W2"].fill(0.0)
                    self.params["b2"].fill(0.0)
                    h1 = min(old_w1.shape[0], self.params["W1"].shape[0])
                    h2 = min(old_w1.shape[1], self.params["W1"].shape[1], old_w2.shape[0], self.params["W2"].shape[0])
                    self.params["W1"][:h1, :h2] = old_w1[:h1, :h2]
                    if "b1" in old:
                        self.params["b1"][:min(h2, old["b1"].size)] = old["b1"][:h2]
                    self.params["W2"][:h2, :h2] = old_w2[:h2, :h2]
                    if "b2" in old:
                        self.params["b2"][:min(h2, old["b2"].size)] = old["b2"][:h2]
                    for wk, bk in (("Wt", "bt"), ("Wf", "bf"), ("Ws", "bs"), ("Wfire", "bfire"), ("Wv", "bv")):
                        self.params[wk].fill(0.0)
                        if wk in old and old[wk].ndim == 2:
                            rows = min(h2, self.params[wk].shape[0], old[wk].shape[0])
                            cols = min(self.params[wk].shape[1], old[wk].shape[1])
                            self.params[wk][:rows, :cols] = old[wk][:rows, :cols]
                        if bk in old:
                            n = min(self.params[bk].size, old[bk].size)
                            self.params[bk][:n] = old[bk][:n]
                    loaded_any = True
            # Old checkpoints don't carry a large-branch state. Its output heads remain
            # zero-initialized, so behavior begins from the already-trained base policy.
            self._reset_torch_state()
            return loaded_any
        except Exception:
            traceback.print_exc()
            return False

    def _ensure_torch(self, device: str):
        try:
            import torch
        except Exception as exc:
            raise RuntimeError("Для обучения нужен PyTorch. Запусти run_gravitruex.bat, чтобы он установил нужные зависимости.") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Запрошена CUDA, но torch.cuda.is_available() вернул False. Проверь драйвер NVIDIA и CUDA-сборку PyTorch.")
        resolved = "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
        if self._torch_params is not None and self._torch_device == resolved:
            return torch, resolved
        self._reset_torch_state()
        torch.set_float32_matmul_precision("high")
        try:
            torch.set_num_threads(max(1, min(6, (os.cpu_count() or 4) // 2)))
        except Exception:
            pass
        if resolved == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        self._torch_device = resolved
        self._torch_params = {
            k: torch.nn.Parameter(torch.as_tensor(v.copy(), dtype=torch.float32, device=resolved))
            for k, v in self.params.items()
        }
        self._torch_optimizer = torch.optim.AdamW(self._torch_params.values(), lr=self.lr, eps=1e-5)
        self.last_device = f"PyTorch {torch.__version__} / {resolved.upper()}"
        return torch, resolved

    def _torch_forward(self, torch, x):
        p = self._torch_params
        h = torch.tanh(x @ p["W1"] + p["b1"])
        h = torch.tanh(h @ p["W2"] + p["b2"])
        logits = {
            "turn": h @ p["Wt"] + p["bt"],
            "thrust": h @ p["Wf"] + p["bf"],
            "strafe": h @ p["Ws"] + p["bs"],
            "fire": (h @ p["Wfire"] + p["bfire"]).squeeze(-1),
        }
        value = ((h @ p["Wv"] + p["bv"]).squeeze(-1)
                 if self.architecture == "shared_residual" else None)
        g = x
        for i in range(self.depth):
            g = torch.tanh(g @ p[f"G_W{i}"] + p[f"G_b{i}"])
        logits["turn"] = logits["turn"] + g @ p["G_Wt"] + p["G_bt"]
        logits["thrust"] = logits["thrust"] + g @ p["G_Wf"] + p["G_bf"]
        logits["strafe"] = logits["strafe"] + g @ p["G_Ws"] + p["G_bs"]
        logits["fire"] = logits["fire"] + (g @ p["G_Wfire"] + p["G_bfire"]).squeeze(-1)
        if self.architecture == "shared_residual":
            value = value + (g @ p["G_Wv"] + p["G_bv"]).squeeze(-1)
        else:
            c = x
            for i in range(self.depth):
                c = torch.tanh(c @ p[f"C_W{i}"] + p[f"C_b{i}"])
            value = (c @ p["C_Wv"] + p["C_bv"]).squeeze(-1)
        logits["turn"] = apply_target_turn_prior_torch(torch, logits["turn"], x)
        logits["strafe"] = apply_dodge_prior_torch(torch, logits["strafe"], x)
        logits["fire"] = apply_fire_prior_torch(torch, logits["fire"], x)
        return logits, value

    def update_ppo(self, records: List[dict], epochs: int = 3, batch_size: int = 2048,
                   clip_eps: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.012,
                   device: Optional[str] = None, teacher_coef: float = 0.0,
                   value_loss_mode: str = "mse", grad_diagnostics: bool = False) -> float:
        if not records:
            return 0.0
        requested = (device or self.device_mode or "auto").lower()
        torch, resolved = self._ensure_torch(requested)
        if isinstance(records, PackedExperience):
            X, A, old_lp = records.obs, records.action, records.logp
            returns, adv = records.returns, records.adv
            old_values = records.value
            teacher_actions, teacher_masks = records.teacher_action, records.teacher_mask
        else:
            X = np.asarray([r["obs"] for r in records], dtype=np.float32)
            A = np.asarray([r["action"] for r in records], dtype=np.int64)
            old_lp = np.asarray([r["logp"] for r in records], dtype=np.float32)
            returns = np.asarray([r["return"] for r in records], dtype=np.float32)
            adv = np.asarray([r["adv"] for r in records], dtype=np.float32)
            old_values = np.asarray([r.get("value", 0.0) for r in records], dtype=np.float32)
            teacher_actions = np.asarray([r.get("teacher_action", r["action"]) for r in records], dtype=np.int64)
            teacher_masks = np.asarray([bool(r.get("teacher_mask", False)) for r in records], dtype=np.bool_)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(records)
        minibatch = int(clamp(batch_size, 32, 16384))
        losses: List[float] = []
        imitation_losses: List[float] = []
        value_losses: List[float] = []
        approximate_kls: List[float] = []
        clip_fractions: List[float] = []
        entropy_values: List[float] = []
        self.last_policy_grad_norm = 0.0
        self.last_value_grad_norm = 0.0
        self.last_grad_ratio = 0.0
        self.last_approx_kl = 0.0
        self.last_clip_fraction = 0.0
        self.last_entropy = 0.0
        self.last_value_loss = 0.0
        self.last_optimizer_steps = 0
        self.last_value_loss_mode = str(value_loss_mode or "mse").lower()
        if self.last_value_loss_mode not in {"mse", "huber"}:
            self.last_value_loss_mode = "mse"
        variance_returns = float(np.var(returns)) if len(returns) else 0.0
        self.last_explained_variance = (float(1.0 - np.var(returns - old_values) / variance_returns)
                                        if variance_returns > 1e-8 else 0.0)
        diagnostic_done = False
        self.last_imitation_coef = float(max(0.0, teacher_coef)) if bool(np.any(teacher_masks)) else 0.0
        effective_teacher_coef = self.last_imitation_coef
        self.last_imitation_loss = 0.0
        rng = self.rng
        # Move the whole batch to the device once. Copying every minibatch over
        # PCIe separately costs five host-to-device transfers per minibatch and
        # was a measurable part of the PPO time on CUDA.
        on_device = resolved == "cuda"
        if on_device:
            Xd = torch.as_tensor(X, dtype=torch.float32, device=resolved)
            Ad = torch.as_tensor(A, dtype=torch.long, device=resolved)
            oldd = torch.as_tensor(old_lp, dtype=torch.float32, device=resolved)
            retd = torch.as_tensor(returns, dtype=torch.float32, device=resolved)
            advd = torch.as_tensor(adv, dtype=torch.float32, device=resolved)
            teacher_ad = torch.as_tensor(teacher_actions, dtype=torch.long, device=resolved)
            teacher_md = torch.as_tensor(teacher_masks, dtype=torch.bool, device=resolved)
            shuffle_gen = torch.Generator(device=resolved)
            shuffle_gen.manual_seed(_rng_integer(rng, 1, 2**31 - 1))
        for _ in range(max(1, int(epochs))):
            if on_device:
                order = torch.randperm(n, generator=shuffle_gen, device=resolved)
            else:
                order = rng.permutation(n)
            for start in range(0, n, minibatch):
                if on_device:
                    idx = order[start:start + minibatch]
                    if idx.numel() == 0:
                        continue
                    xb = Xd.index_select(0, idx)
                    ab = Ad.index_select(0, idx)
                    oldb = oldd.index_select(0, idx)
                    retb = retd.index_select(0, idx)
                    advb = advd.index_select(0, idx)
                    teacher_ab = teacher_ad.index_select(0, idx)
                    teacher_mb = teacher_md.index_select(0, idx)
                else:
                    idx = order[start:start + minibatch]
                    if len(idx) == 0:
                        continue
                    xb = torch.as_tensor(X[idx], dtype=torch.float32, device=resolved)
                    ab = torch.as_tensor(A[idx], dtype=torch.long, device=resolved)
                    oldb = torch.as_tensor(old_lp[idx], dtype=torch.float32, device=resolved)
                    retb = torch.as_tensor(returns[idx], dtype=torch.float32, device=resolved)
                    advb = torch.as_tensor(adv[idx], dtype=torch.float32, device=resolved)
                    teacher_ab = torch.as_tensor(teacher_actions[idx], dtype=torch.long, device=resolved)
                    teacher_mb = torch.as_tensor(teacher_masks[idx], dtype=torch.bool, device=resolved)
                logits, values = self._torch_forward(torch, xb)
                dist_turn = torch.distributions.Categorical(logits=logits["turn"])
                dist_thrust = torch.distributions.Categorical(logits=logits["thrust"])
                dist_strafe = torch.distributions.Categorical(logits=logits["strafe"])
                fire_logit = logits["fire"]
                new_lp = (dist_turn.log_prob(ab[:, 0]) + dist_thrust.log_prob(ab[:, 1]) +
                          dist_strafe.log_prob(ab[:, 2]) -
                          torch.nn.functional.binary_cross_entropy_with_logits(fire_logit, ab[:, 3].float(), reduction="none"))
                entropy = (dist_turn.entropy() + dist_thrust.entropy() + dist_strafe.entropy() +
                           torch.distributions.Bernoulli(logits=fire_logit).entropy())
                ratio = torch.exp(torch.clamp(new_lp - oldb, -15, 15))
                unclipped = ratio * advb
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advb
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                if self.last_value_loss_mode == "huber":
                    value_loss = torch.nn.functional.smooth_l1_loss(values, retb, beta=1.0)
                else:
                    value_loss = 0.5 * torch.mean((values - retb) ** 2)
                entropy_mean = entropy.mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean
                # Compact optimizer diagnostics are useful to distinguish no learning
                # signal from an unstable critic. Gradient norms are sampled once/update.
                approximate_kls.append(float((oldb - new_lp).mean().detach().cpu()))
                clip_fractions.append(float((torch.abs(ratio - 1.0) > clip_eps).float().mean().detach().cpu()))
                entropy_values.append(float(entropy_mean.detach().cpu()))
                value_losses.append(float(value_loss.detach().cpu()))
                if grad_diagnostics and not diagnostic_done:
                    params_for_diag = list(self._torch_params.values())
                    pg = torch.autograd.grad(policy_loss, params_for_diag, retain_graph=True, allow_unused=True)
                    vg = torch.autograd.grad(value_coef * value_loss, params_for_diag, retain_graph=True, allow_unused=True)
                    pg_sq = sum((g.detach().float().pow(2).sum() for g in pg if g is not None),
                                torch.zeros((), device=resolved))
                    vg_sq = sum((g.detach().float().pow(2).sum() for g in vg if g is not None),
                                torch.zeros((), device=resolved))
                    self.last_policy_grad_norm = float(torch.sqrt(pg_sq).detach().cpu())
                    self.last_value_grad_norm = float(torch.sqrt(vg_sq).detach().cpu())
                    self.last_grad_ratio = self.last_value_grad_norm / max(1e-8, self.last_policy_grad_norm)
                    diagnostic_done = True
                # Auxiliary behavior cloning: bootstrap combat basics from the built-in
                # lead-aim/shoot/dodge pilot while PPO learns outcomes. The coefficient
                # is annealed by the trainer, so the neural policy can exceed the teacher.
                if effective_teacher_coef > 0.0:
                    bc_turn = torch.nn.functional.cross_entropy(
                        logits["turn"], teacher_ab[:, 0], reduction="none")
                    bc_thrust = torch.nn.functional.cross_entropy(
                        logits["thrust"], teacher_ab[:, 1], reduction="none")
                    bc_strafe = torch.nn.functional.cross_entropy(
                        logits["strafe"], teacher_ab[:, 2], reduction="none")
                    # Fire labels are sparse: a fixed positive weight of 2 taught the
                    # policy that always-not-fire was often cheaper than copying the teacher.
                    # Balance positive/negative examples in this minibatch, bounded for stability.
                    fire_target = teacher_ab[:, 3].float()
                    bc_mask_bool = teacher_mb.bool()
                    pos_count = ((fire_target > 0.5) & bc_mask_bool).sum().float()
                    neg_count = ((fire_target <= 0.5) & bc_mask_bool).sum().float()
                    fire_weight_value = torch.clamp(neg_count / (pos_count + 1.0), min=2.0, max=24.0)
                    fire_pos_weight = fire_weight_value.detach()
                    bc_fire = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits["fire"].reshape(-1), fire_target,
                        reduction="none", pos_weight=fire_pos_weight)
                    bc_movement = (bc_turn + bc_thrust + bc_strafe) / 3.0
                    bc_per_record = 0.40 * bc_movement + 0.60 * bc_fire
                    bc_mask = teacher_mb.float()
                    bc_loss = (bc_per_record * bc_mask).sum() / bc_mask.sum().clamp_min(1.0)
                    self.last_teacher_fire_rate = float(((fire_target > 0.5) & bc_mask_bool).sum().detach().cpu()) / max(1, int(bc_mask_bool.sum().detach().cpu()))
                    self.last_teacher_fire_pos_weight = float(fire_pos_weight.detach().cpu())
                    loss = loss + effective_teacher_coef * bc_loss
                    imitation_losses.append(float(bc_loss.detach().cpu()))
                self._torch_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self._torch_params.values()), max_norm=1.0)
                self._torch_optimizer.step()
                losses.append(float(loss.detach().cpu()))
                self.last_optimizer_steps += 1
        self.last_approx_kl = float(np.mean(approximate_kls)) if approximate_kls else 0.0
        self.last_clip_fraction = float(np.mean(clip_fractions)) if clip_fractions else 0.0
        self.last_entropy = float(np.mean(entropy_values)) if entropy_values else 0.0
        self.last_value_loss = float(np.mean(value_losses)) if value_losses else 0.0
        # Keep a NumPy copy for rollout workers, game inference, and portable NPZ files.
        for key, value in self._torch_params.items():
            self.params[key][...] = value.detach().to("cpu").numpy()
        # The cached inference weights are a separate CUDA copy. Invalidate them after
        # PPO so GPU-action rollout cannot accidentally keep using a stale policy.
        self._inference_params = None
        self._inference_device = None
        if resolved == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        self.last_imitation_loss = float(np.mean(imitation_losses)) if imitation_losses else 0.0
        return float(np.mean(losses)) if losses else 0.0


def ensure_builtin_models() -> None:
    """Create first-run starter policies without touching any existing model files.

    The Team PPO baseline uses local 96-feature observations (including nearby
    allies) and the shared team-progress reward, with an independent critic trunk.
    It is not full MAPPO: no centralized global-state critic is claimed here.
    """
    presets = (
        {
            "id": "builtin_simple", "name": "Простая PPO",
            "seed": 61101, "width": 768, "depth": 3,
            "architecture": "shared_residual", "default_reward_mode": "baseline",
            "default_priors_mode": "soft",
            "description": "Компактная сеть PPO с общим стволом actor/critic. Контрольная стартовая модель; случайные начальные веса.",
        },
        {
            "id": "builtin_team", "name": "Командная PPO",
            "seed": 61102, "width": 1536, "depth": 4,
            "architecture": "separate_critic", "default_reward_mode": "team",
            "default_priors_mode": "soft",
            "description": "Командный эксперимент: локальная политика видит союзников через 96 признаков, независимый critic и общий командный прогресс. Это не централизованный MAPPO critic.",
        },
        {
            "id": "builtin_split_critic", "name": "Раздельный critic (контроль)",
            "seed": 61103, "width": 1536, "depth": 4,
            "architecture": "separate_critic", "default_reward_mode": "baseline",
            "default_priors_mode": "soft",
            "description": "Изолирует эффект независимого critic от командной награды: та же архитектура, но индивидуальный reward.",
        },
    )
    for item in presets:
        model_id = item["id"]
        weights = model_weights_path(model_id)
        stats_path = model_stats_path(model_id)
        folder = model_dir_for(model_id)
        folder.mkdir(parents=True, exist_ok=True)
        meta = read_model_meta(model_id)
        if not weights.exists():
            policy = ActorCritic(seed=item["seed"], width=item["width"], depth=item["depth"],
                                 device_mode="cpu", architecture=item["architecture"])
            policy.save(weights)
            if not stats_path.exists():
                _atomic_json(stats_path, {
                    "episodes": 0, "steps": 0, "wins": 0, "losses": 0, "draws": 0,
                    "win_rate_20": 0.0, "mean_reward_20": 0.0, "last_loss": 0.0,
                    "message": "Встроенная стартовая модель; обучение ещё не начато.",
                    "opponent_mode": "—", "width": item["width"], "depth": item["depth"],
                })
        # Preserve metadata the user may already have customized.
        meta.update({key: value for key, value in item.items() if key not in {"seed"}} if not meta else {})
        meta.setdefault("id", model_id)
        meta.setdefault("name", item["name"])
        meta.setdefault("width", item["width"])
        meta.setdefault("depth", item["depth"])
        meta.setdefault("architecture", item["architecture"])
        meta.setdefault("default_reward_mode", item["default_reward_mode"])
        meta.setdefault("default_priors_mode", item["default_priors_mode"])
        meta.setdefault("parameters", 0)
        meta.setdefault("episodes", 0)
        meta.setdefault("steps", 0)
        meta.setdefault("trained", False)
        meta.setdefault("training_history_known", True)
        meta.setdefault("built_in", True)
        meta.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
        meta.setdefault("description", item["description"])
        # Correct a legacy metadata omission from this build without overwriting a user's selection.
        if weights.exists():
            nw, nd = checkpoint_architecture(weights)
            meta["width"], meta["depth"] = nw, nd
            meta["architecture"] = checkpoint_network_type(weights)
            try:
                with np.load(weights, allow_pickle=False) as arrays:
                    meta["parameters"] = int(sum(arrays[k].size for k in arrays.files if not k.startswith("__meta_")))
            except Exception:
                pass
        _atomic_json(model_meta_path(model_id), meta)


def _gae_for_trajectory(trajectory: List[dict], gamma: float = 0.993, lam: float = 0.94) -> List[dict]:
    gae = 0.0
    for i in range(len(trajectory) - 1, -1, -1):
        r = trajectory[i]
        done = bool(r.get("done", False))
        next_v = 0.0 if done or i == len(trajectory) - 1 else float(trajectory[i + 1]["value"])
        delta = float(r["reward"]) + gamma * next_v * (0.0 if done else 1.0) - float(r["value"])
        gae = delta + gamma * lam * (0.0 if done else 1.0) * gae
        r["adv"] = float(gae)
        r["return"] = float(gae + r["value"])
    return trajectory


def _aim_context(env: Arena, ship: Ship) -> Tuple[float, float]:
    """Return lead-aim alignment and target distance for safe reward shaping."""
    target = env.nearest_target(ship, env.alive_ships(1 - ship.team))
    if target is None:
        return -1.0, float("inf")
    dx, dy = torus_delta(ship.x, ship.y, target.x, target.y)
    rvx, rvy = target.vx - ship.vx, target.vy - ship.vy
    dist = max(1.0, math.hypot(dx, dy))
    lead_t = clamp(dist / BULLET_SPEED, 0.0, 0.72)
    angle = math.atan2(dy + rvy * lead_t, dx + rvx * lead_t)
    return math.cos(wrap_angle(angle - ship.heading)), dist


def shot_reward_value(aim_alignment: float, range_factor: float) -> float:
    """Dense signal for firing: aligned shots are encouraged; blind shots are mildly discouraged."""
    aim = max(0.0, float(aim_alignment))
    rng = clamp(float(range_factor), 0.0, 1.0)
    return REWARD_SHOT_AIM_BONUS * (aim ** 2) * rng - REWARD_SHOT_BASE_COST


def teacher_coef_for_update(updates: int, opponent_mode: str) -> float:
    """Longer, annealed combat warm start; still reaches zero for unrestricted PPO/self-play."""
    if str(opponent_mode).lower() not in ("heuristic", "selfplay"):
        return 0.0
    progress = clamp(max(0, int(updates)) / 900.0, 0.0, 1.0)
    return 0.12 * (1.0 - progress)


def _aim_alignment(env: Arena, ship: Ship) -> float:
    """Cosine of lead-aim error; a dense but small shaping signal for PPO."""
    return _aim_context(env, ship)[0]


def _collect_episode_rollout(policy: ActorCritic, env: Arena, blue_n: int, red_n: int,
                             max_decisions: int = 480,
                             stop_event: Optional[threading.Event] = None,
                             opponent_mode: str = "heuristic",
                             opponent_policy: Optional[ActorCritic] = None):
    """Collect one complete rollout. Shared by serial and multiprocessing trainers."""
    trajectories: Dict[int, List[dict]] = {}
    total_reward = 0.0
    step_start = env.step_count
    for _decision_i in range(max_decisions):
        if stop_event is not None and stop_event.is_set():
            break
        blue_alive = env.alive_ships(0)
        red_alive = env.alive_ships(1)
        if not blue_alive or not red_alive:
            break
        # In self-play, both teams use the shared policy and both perspectives
        # contribute to PPO. In standard training, only blue is trainable.
        controlled = blue_alive + red_alive if opponent_mode == "selfplay" else blue_alive
        observations = [env.observation(ship) for ship in controlled]
        actions, logps, values, action_arrays = policy.predict_batch(observations, deterministic=False)
        teacher_arrays = {}
        if opponent_mode in ("heuristic", "selfplay"):
            # Bootstrap both self-play sides from the same baseline pilot at the start.
            # PPO's imitation coefficient decays to zero, so this is not a permanent
            # hard-coded action constraint and the policy can eventually diverge.
            for ship in controlled:
                if opponent_mode == "selfplay" or ship.team == 0:
                    teacher_arrays[ship.sid] = env.heuristic_action(ship).as_array()
        action_map: Dict[int, Action] = {}
        if opponent_mode == "model" and opponent_policy is not None:
            # The red team uses a frozen, separately selected checkpoint. Only the
            # blue/learning policy contributes PPO records in this mode.
            red_observations = [env.observation(ship) for ship in red_alive]
            red_actions, _, _, _ = opponent_policy.predict_batch(red_observations, deterministic=False)
            action_map.update({ship.sid: red_actions[i] for i, ship in enumerate(red_alive)})
        records_now: Dict[int, dict] = {}
        before = {
            ship.sid: (ship.damage_dealt, ship.damage_taken, ship.kills, ship.alive,
                       ship.shots_fired, *_aim_context(env, ship))
            for ship in env.ships
        }
        for i, ship in enumerate(controlled):
            record = {
                "obs": observations[i], "action": action_arrays[i].copy(),
                "logp": float(logps[i]), "value": float(values[i]),
                "reward": 0.0, "done": False,
                "teacher_action": teacher_arrays.get(ship.sid, action_arrays[i]).copy(),
                "teacher_mask": bool(ship.sid in teacher_arrays),
            }
            trajectories.setdefault(ship.sid, []).append(record)
            records_now[ship.sid] = record
            action_map[ship.sid] = actions[i]

        env.decision_step(action_map)
        blue_dead = sum(1 for ship in env.ships if ship.team == 0 and before[ship.sid][3] and not ship.alive)
        red_dead = sum(1 for ship in env.ships if ship.team == 1 and before[ship.sid][3] and not ship.alive)
        reward_cfg = training_reward_coefficients()
        team_progress_delta = {}
        for team_id in (0, 1):
            team_ships = [s for s in env.ships if s.team == team_id]
            team_dealt = sum(s.damage_dealt - before[s.sid][0] for s in team_ships)
            team_taken = sum(s.damage_taken - before[s.sid][1] for s in team_ships)
            team_progress_delta[team_id] = team_dealt - team_taken
        for sid, record in records_now.items():
            ship = env.ship_by_id(sid)
            if ship is None:
                continue
            d0, t0, k0, was_alive, shots0, aim0, distance0 = before[sid]
            dealt = ship.damage_dealt - d0
            taken = ship.damage_taken - t0
            kills = ship.kills - k0
            shots = ship.shots_fired - shots0
            died = was_alive and not ship.alive
            enemy_dead = red_dead if ship.team == 0 else blue_dead
            ally_dead = blue_dead if ship.team == 0 else red_dead
            # Baseline preserves Stage 6.14 rewards. Team mode shares progress and
            # makes terminal victory stronger than personal kill farming.
            reward = (reward_cfg["damage_dealt"] * dealt - reward_cfg["damage_taken"] * taken
                      + reward_cfg["kill"] * kills - (reward_cfg["death"] if died else 0.0))
            reward += reward_cfg["enemy_dead"] * enemy_dead - reward_cfg["ally_dead"] * ally_dead
            team_count = max(1, blue_n if ship.team == 0 else red_n)
            reward += reward_cfg["team_progress"] * team_progress_delta[ship.team] / team_count
            aim1, distance1 = _aim_context(env, ship) if ship.alive else (-1.0, float("inf"))
            range_factor = clamp((1000.0 - distance0) / 800.0, 0.0, 1.0)
            range_factor_after = clamp((1000.0 - distance1) / 800.0, 0.0, 1.0)
            reward += 0.065 * (aim1 - aim0)
            reward += 0.010 * (max(0.0, aim1) ** 2) * range_factor_after
            if TRAINING_DESIRED_RANGE > 0.0 and math.isfinite(distance0) and math.isfinite(distance1):
                desired_range = float(TRAINING_DESIRED_RANGE)
                potential_before = abs(distance0 - desired_range)
                potential_after = abs(distance1 - desired_range)
                reward += 0.016 * clamp((potential_before - potential_after) / 30.0, -1.0, 1.0)
            # Previous coefficients made almost every plausible shot negative unless
            # aim was nearly perfect, teaching the policy to suppress fire. Reward
            # moderately well-aimed shots while retaining a small blind-fire cost.
            reward += shots * shot_reward_value(aim0, range_factor)
            reward -= REWARD_STEP_COST
            record["reward"] += reward
            if ship.team == 0:
                total_reward += reward
            if died:
                record["done"] = True

        if env.round_over:
            break

    blue_alive = env.alive_ships(0)
    red_alive = env.alive_ships(1)
    blue_damage = sum(ship.damage_dealt for ship in env.ships if ship.team == 0)
    red_damage = sum(ship.damage_dealt for ship in env.ships if ship.team == 1)
    blue_kills = sum(ship.kills for ship in env.ships if ship.team == 0)
    red_kills = sum(ship.kills for ship in env.ships if ship.team == 1)
    if not blue_alive and red_alive:
        outcome = 1
    elif blue_alive and not red_alive:
        outcome = 0
    else:
        outcome = None

    blue_survivors = len(blue_alive)
    red_survivors = len(red_alive)
    def terminal_for(team: int) -> float:
        own_alive = blue_survivors if team == 0 else red_survivors
        opp_alive = red_survivors if team == 0 else blue_survivors
        own_kills = blue_kills if team == 0 else red_kills
        opp_kills = red_kills if team == 0 else blue_kills
        own_damage = blue_damage if team == 0 else red_damage
        opp_damage = red_damage if team == 0 else blue_damage
        reward_cfg = training_reward_coefficients()
        if outcome is not None:
            terminal = float(reward_cfg["terminal_win"])
            return terminal if outcome == team else -terminal
        return ((own_kills - opp_kills) * reward_cfg["terminal_kill"]
                + (own_alive - opp_alive) * reward_cfg["terminal_survivor"]
                + clamp((own_damage - opp_damage) / reward_cfg["terminal_damage_div"],
                        -reward_cfg["terminal_damage_cap"], reward_cfg["terminal_damage_cap"])
                + reward_cfg["terminal_bias"])

    records: List[dict] = []
    for sid, trajectory in trajectories.items():
        if trajectory:
            ship = env.ship_by_id(sid)
            team = ship.team if ship is not None else 0
            team_count = max(1, blue_n if team == 0 else red_n)
            trajectory[-1]["reward"] += terminal_for(team) / team_count
            trajectory[-1]["done"] = True
            _gae_for_trajectory(trajectory)
            records.extend(trajectory)
    blue_shots = sum(ship.shots_fired for ship in env.ships if ship.team == 0)
    red_shots = sum(ship.shots_fired for ship in env.ships if ship.team == 1)
    policy_fire_actions = [int(np.asarray(r.get("action", [0, 0, 0, 0]))[3]) for r in records]
    teacher_rows = [r for r in records if bool(r.get("teacher_mask", False))]
    teacher_fire_actions = [int(np.asarray(r.get("teacher_action", [0, 0, 0, 0]))[3]) for r in teacher_rows]
    metrics = {
        "blue_damage": blue_damage, "red_damage": red_damage,
        "blue_shots": blue_shots, "red_shots": red_shots,
        "blue_kills": blue_kills, "red_kills": red_kills,
        "blue_survivors": blue_survivors, "red_survivors": red_survivors,
        "blue_hit_rate": min(1.0, blue_damage / max(1.0, blue_shots * BULLET_DAMAGE)),
        "red_hit_rate": min(1.0, red_damage / max(1.0, red_shots * BULLET_DAMAGE)),
        "policy_fire_rate": float(np.mean(policy_fire_actions)) if policy_fire_actions else 0.0,
        "teacher_fire_rate": float(np.mean(teacher_fire_actions)) if teacher_fire_actions else 0.0,
    }
    return records, total_reward + terminal_for(0), outcome, blue_n, red_n, env.step_count - step_start, metrics


# Worker globals are initialized once per process. Large weights live in one shared
# memory segment instead of being pickled and retransmitted for every episode.
_WORKER_POLICY = None
_WORKER_OPPONENT_POLICY = None
_WORKER_SHM = None
_WORKER_FLAT = None
_WORKER_STOP_EVENT = None
_WORKER_CYCLE_EVENT = None


class _CombinedStopEvent:
    """Present trainer-stop and per-cycle cutoff as one event-like object."""
    __slots__ = ("events",)

    def __init__(self, *events):
        self.events = tuple(event for event in events if event is not None)

    def is_set(self):
        return any(event.is_set() for event in self.events)


def _init_rollout_worker(shm_name: str, total_floats: int, layout: list,
                         width: int, depth: int, stop_event, opponent_path: str = "",
                         low_priority: bool = False, cycle_event=None,
                         architecture: str = "shared_residual"):
    global _WORKER_POLICY, _WORKER_OPPONENT_POLICY, _WORKER_SHM, _WORKER_FLAT, _WORKER_STOP_EVENT, _WORKER_CYCLE_EVENT
    if low_priority:
        _lower_rollout_worker_priority()
    from multiprocessing import shared_memory as _shm_mod
    _WORKER_SHM = _shm_mod.SharedMemory(name=shm_name)
    _WORKER_FLAT = np.ndarray((int(total_floats),), dtype=np.float32, buffer=_WORKER_SHM.buf)
    _WORKER_POLICY = ActorCritic(seed=137, width=width, depth=depth, device_mode="cpu", allocate=False,
                                  architecture=architecture)
    _WORKER_POLICY.params = {
        name: _WORKER_FLAT[int(start):int(end)].reshape(tuple(shape))
        for name, shape, start, end in layout
    }
    _WORKER_STOP_EVENT = stop_event
    _WORKER_CYCLE_EVENT = cycle_event
    _WORKER_OPPONENT_POLICY = None
    if opponent_path:
        try:
            op_path = Path(opponent_path)
            if op_path.exists():
                ow, od = checkpoint_architecture(op_path)
                oa = checkpoint_network_type(op_path)
                _WORKER_OPPONENT_POLICY = ActorCritic(seed=451, width=ow, depth=od, device_mode="cpu", architecture=oa)
                _WORKER_OPPONENT_POLICY.load(op_path)
        except Exception:
            _WORKER_OPPONENT_POLICY = None


def _rollout_worker(task: dict):
    """Picklable worker; policy weights are shared and updated between rollout batches."""
    seed = int(task["seed"])
    policy = _WORKER_POLICY
    if policy is None:
        # Fallback path is useful for standalone diagnostics and tests.
        policy = ActorCritic(seed=seed ^ 0x5A17, width=int(task.get("width", 256)),
                             depth=int(task.get("depth", 2)), device_mode="cpu",
                             architecture=str(task.get("architecture", "shared_residual")))
        for key, value in task.get("params", {}).items():
            if key in policy.params and policy.params[key].shape == value.shape:
                policy.params[key][...] = value
    env = Arena(seed=seed)
    env.reset(int(task["blue_n"]), int(task["red_n"]))
    stop_event = (_CombinedStopEvent(_WORKER_STOP_EVENT, _WORKER_CYCLE_EVENT)
                  if _WORKER_CYCLE_EVENT is not None else _WORKER_STOP_EVENT)
    global ACTION_PRIORS_MODE, TRAINING_REWARD_MODE, TRAINING_DESIRED_RANGE
    old_modes = (ACTION_PRIORS_MODE, TRAINING_REWARD_MODE, TRAINING_DESIRED_RANGE)
    priors_mode = str(task.get("action_priors_mode", old_modes[0])).lower()
    reward_mode = str(task.get("reward_mode", old_modes[1])).lower()
    if priors_mode in {"full", "soft", "off"}:
        ACTION_PRIORS_MODE = priors_mode
    if reward_mode in {"baseline", "team"}:
        TRAINING_REWARD_MODE = reward_mode
    try:
        desired_range = float(task.get("desired_range", old_modes[2]))
        if math.isfinite(desired_range):
            TRAINING_DESIRED_RANGE = float(clamp(desired_range, 0.0, 1000.0))
    except (TypeError, ValueError):
        pass
    try:
        return _collect_episode_rollout(policy, env, int(task["blue_n"]), int(task["red_n"]),
                                        max_decisions=int(task.get("max_decisions", 480)),
                                        stop_event=stop_event,
                                        opponent_mode=str(task.get("opponent_mode", "heuristic")),
                                        opponent_policy=_WORKER_OPPONENT_POLICY)
    finally:
        ACTION_PRIORS_MODE, TRAINING_REWARD_MODE, TRAINING_DESIRED_RANGE = old_modes


class Trainer(threading.Thread):
    def __init__(self, base_policy: ActorCritic, fixed_blue: int, fixed_red: int,
                 random_compositions: bool, save_dir: Path, workers: int = 4,
                 opponent_mode: str = "heuristic", device_mode: str = "auto",
                 batch_size: int = 2048, epochs: int = 3,
                 rollout_gpu_inference: bool = False,
                 hybrid_gpu_rollout: bool = False, gpu_batch_envs: int = 16,
                 gpu_only_rollout: bool = False,
                 auto_rollout_balance: bool = False,
                 model_id: str = DEFAULT_MODEL_ID, model_name: str = "Основная модель",
                 opponent_path: Optional[Path] = None, opponent_name: Optional[str] = None,
                 max_decisions: int = 2700, action_priors_mode: str = "soft",
                 reward_mode: str = "baseline", desired_range: float = 480.0,
                 gpu_record_budget: int = GPU_ROLLOUT_RECORD_BUDGET):
        super().__init__(daemon=True, name="GravitrueX-PPO-Trainer")
        self.stop_event = threading.Event()
        self.mp_stop_event = None
        self.mp_cycle_event = None
        self._hybrid_low_yield_cycles = 0
        self.state_lock = threading.Lock()
        self.device_mode = str(device_mode).lower()
        self.rollout_gpu_inference = bool(rollout_gpu_inference)
        self.hybrid_gpu_rollout = bool(hybrid_gpu_rollout)
        self.gpu_only_rollout = bool(gpu_only_rollout)
        self.auto_rollout_balance = bool(auto_rollout_balance)
        self.auto_rollout_selection_note = ""
        self._last_cycle_hybrid_mode = False
        self._last_cycle_gpu_only_mode = False
        self._batch_gpu_device_seconds = 0.0
        self.gpu_batch_envs = int(clamp(gpu_batch_envs, 1, GPU_MAX_ARENAS))
        self.gpu_record_budget = int(clamp(int(gpu_record_budget), 60000, 2_880_000))
        self.gpu_rollout_fallback_reason = None
        self._last_gpu_device_seconds = 0.0
        self._last_gpu_host_overhead_seconds = 0.0
        self._last_gpu_graph_requested = False
        self._last_gpu_graph_used = False
        self._last_gpu_graph_error = ""
        self._last_gpu_max_decisions = 0
        self._last_gpu_record_budget = self.gpu_record_budget
        self._gpu_rollout_executor = None
        self._last_hybrid_cpu_episodes = 0
        self._last_hybrid_cpu_records = 0
        self._last_gpu_batch_seconds = 0.0
        self._last_gpu_batch_wall_seconds = 0.0
        self._last_gpu_episodes_per_sec = 0.0
        self._last_rollout_episodes_per_sec = 0.0
        self._last_hybrid_buffer_cap = 0
        self._hybrid_pending_worker_futures = []
        self._last_hybrid_worker_drain_seconds = 0.0
        self._gpu_batch_requested_envs = self.gpu_batch_envs
        self._last_gpu_oom_fallback = ""
        self.batch_size = int(clamp(batch_size, 128, 8192))
        self.epochs = int(clamp(epochs, 1, 6))
        self.policy = ActorCritic(seed=int(time.time()) % 1_000_000, width=base_policy.width,
                                  depth=base_policy.depth, device_mode=self.device_mode,
                                  architecture=base_policy.architecture)
        self.max_decisions = int(clamp(max_decisions, 120, 6000))
        self.action_priors_mode = str(action_priors_mode if action_priors_mode in {"full", "soft", "off"} else "soft")
        self.reward_mode = str(reward_mode if reward_mode in {"baseline", "team"} else "baseline")
        self.desired_range = float(clamp(float(desired_range), 0.0, 1000.0))
        self.policy.allow_gpu_inference = self.rollout_gpu_inference or self.gpu_only_rollout
        self.policy.inference_mode = "gpu" if (self.rollout_gpu_inference or self.gpu_only_rollout) else "cpu"
        for k in self.policy.params:
            if k in base_policy.params and self.policy.params[k].shape == base_policy.params[k].shape:
                self.policy.params[k][...] = base_policy.params[k]
        self._shared_memory = None
        self._shared_array = None
        self._shared_layout = None
        self._save_batches = 0
        self._last_checkpoint_time = time.monotonic()
        self.fixed_blue = fixed_blue
        self.fixed_red = fixed_red
        self.random_compositions = random_compositions
        self.opponent_mode = opponent_mode
        self.model_id = str(model_id)
        self.model_name = str(model_name)
        self.opponent_path = Path(opponent_path) if opponent_path else None
        self.opponent_name = str(opponent_name or opponent_mode)
        self.opponent_policy = None
        if self.opponent_mode == "model" and self.opponent_path and self.opponent_path.exists():
            ow, od = checkpoint_architecture(self.opponent_path)
            oa = checkpoint_network_type(self.opponent_path)
            self.opponent_policy = ActorCritic(seed=int(time.time()) % 100003, width=ow, depth=od,
                                               device_mode="cpu", architecture=oa)
            self.opponent_policy.load(self.opponent_path)
        # The legacy single-arena GPU-action mode is serial. Hybrid mode keeps the
        # requested CPU worker pool AND adds a tensor-batched GPU physics rollout.
        self.workers = (1 if (self.rollout_gpu_inference and not self.hybrid_gpu_rollout) or self.gpu_only_rollout
                        else int(clamp(workers, 1, 32)))
        self.cpu_workers_active = self.workers
        self.hybrid_limits = None
        self._hybrid_priority_applied = False
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = save_dir / f"weights_{self.model_id}_latest.npz"
        self.stats_path = save_dir / "training_stats.json"
        self.metrics_jsonl_path = save_dir / "training_metrics.jsonl"
        self.cpu_rollouts_jsonl_path = save_dir / "cpu_rollouts.jsonl"
        self.gpu_rollouts_jsonl_path = save_dir / "gpu_rollouts.jsonl"
        self.ppo_updates_jsonl_path = save_dir / "ppo_updates.jsonl"
        self.report_path = save_dir / "training_report.txt"
        self.runtime_log_path = save_dir / "training_runtime.log"
        self.crash_report_path = save_dir / "training_crash_report.txt"
        self.fatal_error: Optional[str] = None
        self.fatal_traceback = ""
        self.failed_phase = ""
        self.ppo_fallback_reason = ""
        self._stage_history = []
        self._last_cpu_rollout_seconds = 0.0
        self._last_cpu_rollout_episodes = 0
        self._last_cpu_rollout_records = 0
        self._last_cpu_rollout_steps = 0
        self._last_gpu_rollout_seconds = 0.0
        self._last_gpu_rollout_episodes = 0
        self._last_gpu_rollout_records = 0
        self._last_gpu_rollout_steps = 0
        self._last_cpu_wait_gpu_seconds = 0.0
        self._last_rollout_wall_seconds = 0.0
        self.stats = {
            "episodes": 0, "wins": 0, "losses": 0, "draws": 0,
            "win_rate_20": 0.0, "mean_reward_20": 0.0,
            "cycle_reward_total": 0.0, "cycle_mean_reward": 0.0, "last_loss": 0.0,
            "blue_kills_20": 0.0, "red_kills_20": 0.0,
            "blue_survivors_20": 0.0, "red_survivors_20": 0.0,
            "opponent_mode": self.opponent_mode, "opponent_name": self.opponent_name,
            "steps": 0, "steps_per_sec": 0.0, "message": "Подготовка обучения...",
            "blue_n": fixed_blue, "red_n": fixed_red, "workers": self.workers,
            "cpu_workers_requested": self.workers, "cpu_workers_active": self.cpu_workers_active,
            "hybrid_available_ram_gb": None, "hybrid_cpu_record_cap": 0,
            "cuda_memory_snapshot": {}, "cuda_memory_snapshot_after_stop": {},
            "gpu_record_slots": 0, "gpu_obs_history_estimate_gib": 0.0,
            "hybrid_cpu_priority_requested": bool(self.hybrid_gpu_rollout),
            "device": self.device_mode.upper(), "width": self.policy.width, "depth": self.policy.depth,
            "architecture": self.policy.architecture, "max_decisions": self.max_decisions,
            "action_priors_mode": self.action_priors_mode, "reward_mode": self.reward_mode,
            "desired_range": self.desired_range,
            "batch_size": self.batch_size,
            "rollout_gpu_inference": self.rollout_gpu_inference,
            "hybrid_gpu_rollout": self.hybrid_gpu_rollout,
            "gpu_only_rollout": self.gpu_only_rollout,
            "rollout_physics_device": "CUDA" if (self.hybrid_gpu_rollout or self.gpu_only_rollout) else "CPU",
            "rollout_policy_device": "CUDA" if (self.hybrid_gpu_rollout or self.gpu_only_rollout or self.rollout_gpu_inference) else "CPU/NumPy",
            "gpu_batch_envs": self.gpu_batch_envs,
            "rollout_sec_last": 0.0, "ppo_sec_last": 0.0,
            "cpu_episodes_during_gpu": 0, "cpu_records_during_gpu": 0,
            "gpu_batch_seconds": 0.0, "gpu_batch_wall_seconds": 0.0, "gpu_episodes_per_sec": 0.0,
            "gpu_device_seconds": 0.0, "gpu_host_overhead_seconds": 0.0,
            "gpu_cuda_graph_requested": False, "gpu_cuda_graph_used": False,
            "gpu_cuda_graph_status": "не проверялся",
            "gpu_max_decisions_last": 0, "gpu_record_budget": self.gpu_record_budget,
            "gpu_rollout_envs_requested": self.gpu_batch_envs,
            "gpu_rollout_envs": 0,
            "gpu_rollout_envs_budget_reduced": 0,
            "auto_rollout_balance": self.auto_rollout_balance, "auto_rollout_selection_note": "",
            "rollout_episodes_per_sec": 0.0, "cpu_record_buffer_cap": 0,
            "cpu_rollout_sec_last": 0.0, "cpu_rollout_episodes_last": 0,
            "cpu_rollout_records_last": 0, "cpu_rollout_steps_last": 0,
            "cpu_rollout_steps_per_sec": 0.0,
            "gpu_rollout_sec_last": 0.0, "gpu_rollout_episodes_last": 0,
            "gpu_rollout_records_last": 0, "gpu_rollout_steps_last": 0,
            "gpu_rollout_steps_per_sec": 0.0, "cpu_wait_gpu_sec_last": 0.0,
            "ppo_records_last": 0, "ppo_updates": 0, "epochs_last": self.epochs,
            "last_two_cycles": [], "resource_snapshot": {},
            "phase": "Подготовка обучения",
        }
        # Continue displayed cumulative counters from the previous session.
        try:
            if self.stats_path.exists():
                with open(self.stats_path, "r", encoding="utf-8") as f:
                    prior = json.load(f)
                for key in ("episodes", "steps", "ppo_updates"):
                    if isinstance(prior.get(key), (int, float)):
                        self.stats[key] = int(prior[key])
                self.stats["message"] = "Продолжение предыдущего обучения."
        except Exception:
            pass
        self.initial_steps = int(self.stats.get("steps", 0))
        self.session_steps = 0
        self.recent_results: List[float] = []
        self.recent_rewards: List[float] = []
        self.started_at = time.time()
        self.rng = random.Random(int(time.time() * 1000) % 2**31)
        self.episode_seed = int(time.time())

    def stop(self) -> None:
        self.stop_event.set()
        if self.mp_stop_event is not None:
            try:
                self.mp_stop_event.set()
            except Exception:
                pass

    def get_stats(self) -> dict:
        with self.state_lock:
            return dict(self.stats)

    def _set_phase(self, phase: str) -> None:
        """Expose whether the trainer is generating experience or updating PPO."""
        with self.state_lock:
            self.stats["phase"] = str(phase)

    def _runtime_log(self, message: str, details: str = "") -> None:
        """Persist runtime decisions/failures; a vanished UI must not hide the cause."""
        try:
            stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            with open(self.runtime_log_path, "a", encoding="utf-8") as handle:
                handle.write(f"[{stamp}] {message}\n")
                if details:
                    handle.write(details.rstrip() + "\n")
        except Exception:
            pass

    def _switch_rollout_to_cpu(self, exc: BaseException, phase: str) -> None:
        """Disable the failing accelerated collector without discarding training state."""
        detail = f"{type(exc).__name__}: {exc}"
        note = f"GPU rollout failed during {phase}; switched to CPU rollout: {detail}"[:600]
        self._runtime_log(note, traceback.format_exc())
        self.gpu_rollout_fallback_reason = note
        self.auto_rollout_selection_note = note
        self.gpu_only_rollout = False
        self.hybrid_gpu_rollout = False
        self.rollout_gpu_inference = False
        self.auto_rollout_balance = False
        self.policy.allow_gpu_inference = True
        self.policy.inference_mode = "gpu"
        if self._gpu_rollout_executor is not None:
            try:
                self._gpu_rollout_executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._gpu_rollout_executor = None
        with self.state_lock:
            self.stats.update({
                "gpu_only_rollout": False,
                "hybrid_gpu_rollout": False,
                "rollout_gpu_inference": False,
                "auto_rollout_balance": False,
                "auto_rollout_selection_note": note,
                "gpu_rollout_fallback_reason": note,
                "message": note,
                "phase": "GPU rollout недоступен; продолжаем на CPU",
            })

    def _collect_cpu_rollouts(self, executor=None, batch_n: int = 1):
        """Collect a CPU batch; shared by normal CPU mode and GPU-failure recovery."""
        tasks = [self._make_task() for _ in range(max(1, int(batch_n)))]
        if executor is not None:
            return list(executor.map(_rollout_worker, tasks))
        results = []
        for task in tasks:
            env = Arena(seed=task["seed"])
            env.reset(task["blue_n"], task["red_n"])
            results.append(_collect_episode_rollout(
                self.policy, env, task["blue_n"], task["red_n"],
                task["max_decisions"], self.stop_event,
                opponent_mode=self.opponent_mode, opponent_policy=self.opponent_policy))
            if self.stop_event.is_set():
                break
        return results

    def _update_ppo_with_fallback(self, records_for_update):
        """Retry a failed non-CPU PPO update once on CPU, preserving NumPy weights."""
        started = time.perf_counter()
        try:
            updates = int(self.stats.get("ppo_updates", 0))
            teacher_coef = teacher_coef_for_update(updates, self.opponent_mode)
            loss = self.policy.update_ppo(records_for_update, epochs=self.epochs,
                                          batch_size=self.batch_size, device=self.device_mode,
                                          teacher_coef=teacher_coef)
            return loss, max(0.0, time.perf_counter() - started)
        except Exception as exc:
            requested_device = self.device_mode
            if requested_device == "cpu":
                raise
            if requested_device == "auto":
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise exc
                except ImportError:
                    raise exc
            detail = f"{type(exc).__name__}: {exc}"
            note = f"PPO on {self.device_mode.upper()} failed; retrying the same batch on CPU: {detail}"[:600]
            self.ppo_fallback_reason = note
            self._runtime_log(note, traceback.format_exc())
            # update_ppo only copies optimized tensors into the NumPy checkpoint
            # parameters after all minibatches finish. Resetting torch state discards
            # a partially updated optimizer/model and retries from the last safe weights.
            reset = getattr(self.policy, "_reset_torch_state", None)
            if callable(reset):
                reset()
            self.device_mode = "cpu"
            self.policy.device_mode = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            with self.state_lock:
                self.stats["device"] = "CPU (аварийный fallback после ошибки PPO)"
                self.stats["message"] = note
                self.stats["phase"] = "PPO CUDA недоступен; повторяем обновление на CPU"
                self.stats["ppo_fallback_reason"] = note
            retry_started = time.perf_counter()
            updates = int(self.stats.get("ppo_updates", 0))
            teacher_coef = teacher_coef_for_update(updates, self.opponent_mode)
            loss = self.policy.update_ppo(records_for_update, epochs=self.epochs,
                                          batch_size=self.batch_size, device="cpu",
                                          teacher_coef=teacher_coef)
            return loss, max(0.0, time.perf_counter() - retry_started)

    def set_resource_snapshot(self, perf_snapshot: dict) -> None:
        """Accept a low-frequency hardware snapshot from the UI monitor."""
        keys = ("cpu", "app_cpu", "ram_percent", "process_ram", "gpu_util", "gpu_mem_util",
                "vram_used", "vram_total", "gpu_temp", "gpu_power_w")
        data = {k: perf_snapshot.get(k) for k in keys if perf_snapshot.get(k) is not None}
        with self.state_lock:
            self.stats["resource_snapshot"] = data

    def _episode(self, episode_no: int):
        if self.random_compositions:
            blue_n = self.rng.randint(1, 5)
            red_n = self.rng.randint(1, 5)
        else:
            blue_n, red_n = self.fixed_blue, self.fixed_red
        seed = self.rng.randint(1, 2**31 - 1)
        env = Arena(seed=seed)
        env.reset(blue_n, red_n)
        # Apply this trainer's selected behavior profile for the entire rollout,
        # then restore process defaults so another trainer/test cannot inherit it.
        global ACTION_PRIORS_MODE, TRAINING_REWARD_MODE, TRAINING_DESIRED_RANGE
        old_modes = (ACTION_PRIORS_MODE, TRAINING_REWARD_MODE, TRAINING_DESIRED_RANGE)
        ACTION_PRIORS_MODE = self.action_priors_mode
        TRAINING_REWARD_MODE = self.reward_mode
        TRAINING_DESIRED_RANGE = self.desired_range
        try:
            return _collect_episode_rollout(self.policy, env, blue_n, red_n, self.max_decisions, self.stop_event,
                                            opponent_mode=self.opponent_mode,
                                            opponent_policy=self.opponent_policy)
        finally:
            ACTION_PRIORS_MODE, TRAINING_REWARD_MODE, TRAINING_DESIRED_RANGE = old_modes

    def _make_task(self) -> dict:
        if self.random_compositions:
            blue_n = self.rng.randint(1, 5)
            red_n = self.rng.randint(1, 5)
        else:
            blue_n, red_n = self.fixed_blue, self.fixed_red
        return {
            "seed": self.rng.randint(1, 2**31 - 1), "blue_n": blue_n, "red_n": red_n,
            "max_decisions": self.max_decisions, "opponent_mode": self.opponent_mode,
            "width": self.policy.width, "depth": self.policy.depth,
            "architecture": self.policy.architecture,
            "action_priors_mode": self.action_priors_mode, "reward_mode": self.reward_mode,
            "desired_range": self.desired_range,
        }

    def _run_batches(self, executor=None) -> None:
        win_window: List[float] = []
        reward_window: List[float] = []
        blue_damage_window: List[float] = []
        red_damage_window: List[float] = []
        blue_shots_window: List[float] = []
        red_shots_window: List[float] = []
        hit_rate_window: List[float] = []
        red_hit_rate_window: List[float] = []
        blue_kills_window: List[float] = []
        red_kills_window: List[float] = []
        blue_survivors_window: List[float] = []
        red_survivors_window: List[float] = []
        policy_fire_window: List[float] = []
        teacher_fire_window: List[float] = []
        session_start = time.time()
        while not self.stop_event.is_set():
            gpu_only_this_cycle = bool(self.gpu_only_rollout)
            hybrid_this_cycle = bool(self.hybrid_gpu_rollout)
            batch_n = (self.cpu_workers_active if hybrid_this_cycle and executor is not None
                       else 0 if (gpu_only_this_cycle or hybrid_this_cycle) else 1)
            self._last_cycle_hybrid_mode = hybrid_this_cycle
            self._last_cycle_gpu_only_mode = gpu_only_this_cycle
            if gpu_only_this_cycle:
                phase_text = (f"Сбор опыта: CUDA-физика и GPU-инференс · запрошено {self.gpu_batch_envs} арен; "
                              f"фактический размер определяется record budget {self.gpu_record_budget:,}")
            elif hybrid_this_cycle:
                phase_text = (f"Сбор опыта параллельно: {batch_n} CPU-арен + "
                              f"{self.gpu_batch_envs} CUDA-арен запрошено; фактическое число ограничено "
                              f"record budget {self.gpu_record_budget:,}")
            elif self.rollout_gpu_inference:
                phase_text = "Сбор опыта: одна арена, физика CPU + пакет действий CUDA"
            else:
                phase_text = f"Сбор опыта: {batch_n} CPU-сред(ы), физика + NumPy-инференс"
            self._set_phase(phase_text)
            rollout_started = time.perf_counter()
            try:
                if gpu_only_this_cycle:
                    # True GPU-only rollout: no CPU Arena episodes are launched.
                    results = self._collect_gpu_batch(self.rng.randrange(1, 2**31 - 1))
                elif hybrid_this_cycle:
                    results = self._collect_hybrid_async(executor)
                else:
                    results = self._collect_cpu_rollouts(executor, batch_n)
            except Exception as exc:
                # CUDA Graph capture itself can eagerly fall back inside the backend;
                # this catches later failures too (batch allocation, CUDA kernels,
                # GPU inference, or a broken hybrid GPU future). Do not let a
                # recoverable GPU rollout error silently terminate the Trainer thread.
                gpu_rollout_was_active = (gpu_only_this_cycle or hybrid_this_cycle
                                          or self.rollout_gpu_inference)
                if self.stop_event.is_set() or not gpu_rollout_was_active:
                    raise
                failed_phase = phase_text
                self._switch_rollout_to_cpu(exc, failed_phase)
                gpu_only_this_cycle = False
                hybrid_this_cycle = False
                self._last_cycle_hybrid_mode = False
                self._last_cycle_gpu_only_mode = False
                batch_n = self.cpu_workers_active if executor is not None else 1
                self._set_phase("Аварийный переход: сбор опыта на CPU")
                rollout_started = time.perf_counter()
                results = self._collect_cpu_rollouts(executor, batch_n)
            if not results:
                if self.stop_event.is_set():
                    break
                continue
            # A stop request can interrupt worker rollouts. Do not learn from truncated
            # episodes; leave the last complete checkpoint intact and exit cleanly.
            if self.stop_event.is_set():
                break
            rollout_elapsed = max(0.0, time.perf_counter() - rollout_started)
            self._last_rollout_wall_seconds = rollout_elapsed
            # In non-hybrid modes the scalar Arena physics runs on CPU even when the
            # policy's forward pass or PPO optimizer is CUDA-backed.
            if not hybrid_this_cycle and not gpu_only_this_cycle:
                # CPU physics rollout (PPO may still use CUDA; these are separate devices).
                self._last_cpu_rollout_seconds = rollout_elapsed
                self._last_cpu_rollout_episodes = len(results)
                self._last_cpu_rollout_records = sum(len(row[0]) for row in results)
                self._last_cpu_rollout_steps = sum(int(row[5]) for row in results)
                self._last_gpu_rollout_seconds = 0.0
                self._last_gpu_rollout_episodes = 0
                self._last_gpu_rollout_records = 0
                self._last_gpu_rollout_steps = 0
                self._last_gpu_batch_seconds = 0.0
                self._batch_gpu_device_seconds = 0.0
                self._last_gpu_device_seconds = 0.0
                self._last_gpu_host_overhead_seconds = 0.0
                self._last_cpu_wait_gpu_seconds = 0.0
                self._last_gpu_graph_requested = False
                self._last_gpu_graph_used = False
                self._last_gpu_graph_error = ""
                self._last_gpu_max_decisions = 0
                with self.state_lock:
                    # These are active-cycle metrics, not saved GUI preferences.
                    # A CPU-only cycle must not display requested GPU arenas as if they ran.
                    self.stats["gpu_rollout_envs"] = 0
                    self.stats["gpu_rollout_envs_attempted"] = 0
                    self.stats["gpu_rollout_envs_budget_reduced"] = 0
                    self.stats["gpu_cuda_graph_used"] = False
                    self.stats["gpu_cuda_graph_status"] = "не использовался (CPU rollout)"
            elif gpu_only_this_cycle:
                # Explicitly mark CPU physics metrics as zero; all these episodes were
                # produced by the tensor-vectorized CUDA arena batch.
                self._last_cpu_rollout_seconds = 0.0
                self._last_cpu_rollout_episodes = 0
                self._last_cpu_rollout_records = 0
                self._last_cpu_rollout_steps = 0
                self._last_hybrid_cpu_episodes = 0
                self._last_hybrid_cpu_records = 0
                self._last_cpu_wait_gpu_seconds = 0.0
                self._last_gpu_rollout_seconds = max(0.001, self._last_gpu_batch_seconds or rollout_elapsed)
                self._last_gpu_rollout_episodes = len(results)
                self._last_gpu_rollout_records = sum(len(row[0]) for row in results)
                self._last_gpu_rollout_steps = sum(int(row[5]) for row in results)
                self._last_gpu_device_seconds = float(getattr(self, "_batch_gpu_device_seconds", 0.0) or 0.0)
                self._last_gpu_host_overhead_seconds = max(0.0, self._last_gpu_rollout_seconds - self._last_gpu_device_seconds)
                self._last_gpu_episodes_per_sec = len(results) / max(0.001, self._last_gpu_rollout_seconds)
                self._last_gpu_batch_wall_seconds = self._last_gpu_rollout_seconds
            records = []
            packed_batches = []
            batch_reward = 0.0
            batch_steps = 0
            batch_compositions = (self.fixed_blue, self.fixed_red)
            for episode_records, ep_reward, outcome, bn, rn, phys_steps, metrics in results:
                if isinstance(episode_records, PackedExperience):
                    if len(episode_records):
                        packed_batches.append(episode_records)
                else:
                    records.extend(episode_records)
                batch_reward += ep_reward
                batch_steps += phys_steps
                batch_compositions = (bn, rn)
                reward_window.append(ep_reward)
                win_window.append(1.0 if outcome == 0 else 0.0 if outcome == 1 else 0.5)
                blue_damage_window.append(float(metrics.get("blue_damage", 0.0)))
                red_damage_window.append(float(metrics.get("red_damage", 0.0)))
                blue_shots_window.append(float(metrics.get("blue_shots", 0.0)))
                red_shots_window.append(float(metrics.get("red_shots", 0.0)))
                hit_rate_window.append(float(metrics.get("blue_hit_rate", 0.0)))
                red_hit_rate_window.append(float(metrics.get("red_hit_rate", 0.0)))
                blue_kills_window.append(float(metrics.get("blue_kills", 0.0)))
                red_kills_window.append(float(metrics.get("red_kills", 0.0)))
                blue_survivors_window.append(float(metrics.get("blue_survivors", 0.0)))
                red_survivors_window.append(float(metrics.get("red_survivors", 0.0)))
                policy_fire_window.append(float(metrics.get("policy_fire_rate", 0.0)))
                teacher_fire_window.append(float(metrics.get("teacher_fire_rate", 0.0)))
            policy_fire_window = policy_fire_window[-20:]
            teacher_fire_window = teacher_fire_window[-20:]
            reward_window = reward_window[-20:]
            win_window = win_window[-20:]
            blue_damage_window = blue_damage_window[-20:]
            red_damage_window = red_damage_window[-20:]
            blue_shots_window = blue_shots_window[-20:]
            red_shots_window = red_shots_window[-20:]
            hit_rate_window = hit_rate_window[-20:]
            red_hit_rate_window = red_hit_rate_window[-20:]
            blue_kills_window = blue_kills_window[-20:]
            red_kills_window = red_kills_window[-20:]
            blue_survivors_window = blue_survivors_window[-20:]
            red_survivors_window = red_survivors_window[-20:]
            # Rollouts intentionally stay CPU/NumPy to avoid creating CUDA contexts
            # inside each spawned worker. CUDA is only used for the batched PPO update.
            if self.device_mode == "cuda":
                update_device = "CUDA"
            elif self.device_mode == "cpu":
                update_device = "CPU"
            else:
                try:
                    import torch
                    update_device = "CUDA" if torch.cuda.is_available() else "CPU"
                except Exception:
                    update_device = "CPU"
            if packed_batches:
                if records:
                    packed_batches.insert(0, PackedExperience.from_records(records))
                records_for_update = PackedExperience.concatenate(packed_batches)
            else:
                records_for_update = records
            record_count = len(records_for_update)
            self._set_phase(
                f"Обновление PPO на {update_device}: {record_count:,} записей, batch={self.batch_size}"
            )
            ppo_started = time.perf_counter()
            if record_count:
                loss, ppo_elapsed = self._update_ppo_with_fallback(records_for_update)
                if self.device_mode == "cpu":
                    update_device = "CPU"
            else:
                loss = 0.0
                ppo_elapsed = max(0.0, time.perf_counter() - ppo_started)
            self._sync_shared_weights()
            self.session_steps += batch_steps
            elapsed = max(0.001, time.time() - session_start)
            with self.state_lock:
                self.stats.update({
                    "episodes": int(self.stats.get("episodes", 0)) + len(results),
                    "wins": sum(1 for x in win_window if x == 1.0),
                    "losses": sum(1 for x in win_window if x == 0.0),
                    "draws": sum(1 for x in win_window if x == 0.5),
                    "win_rate_20": sum(win_window) / len(win_window) if win_window else 0.0,
                    "mean_reward_20": sum(reward_window) / max(1, len(reward_window)),
                    "cycle_reward_total": float(batch_reward),
                    "cycle_mean_reward": float(batch_reward) / max(1, len(results)),
                    "blue_damage_20": sum(blue_damage_window) / max(1, len(blue_damage_window)),
                    "red_damage_20": sum(red_damage_window) / max(1, len(red_damage_window)),
                    "blue_shots_20": sum(blue_shots_window) / max(1, len(blue_shots_window)),
                    "red_shots_20": sum(red_shots_window) / max(1, len(red_shots_window)),
                    "blue_hit_rate_20": sum(hit_rate_window) / max(1, len(hit_rate_window)),
                    "red_hit_rate_20": sum(red_hit_rate_window) / max(1, len(red_hit_rate_window)),
                    "blue_kills_20": sum(blue_kills_window) / max(1, len(blue_kills_window)),
                    "red_kills_20": sum(red_kills_window) / max(1, len(red_kills_window)),
                    "blue_survivors_20": sum(blue_survivors_window) / max(1, len(blue_survivors_window)),
                    "red_survivors_20": sum(red_survivors_window) / max(1, len(red_survivors_window)),
                    "policy_fire_rate_20": sum(policy_fire_window) / max(1, len(policy_fire_window)),
                    "teacher_fire_rate_20": sum(teacher_fire_window) / max(1, len(teacher_fire_window)),
                    "opponent_mode": self.opponent_mode, "opponent_name": self.opponent_name,
                    "last_loss": loss,
                    "steps": int(self.stats.get("steps", 0)) + batch_steps,
                    "steps_per_sec": self.session_steps / elapsed,
                    "message": (f"Обучается: {self.model_name} [{self.model_id}]; PPO {self.device_mode.upper()}; "
                                + (f"GPU-физика: фактически {int(self.stats.get('gpu_rollout_envs', 0) or 0)}/{int(self.stats.get('gpu_rollout_envs_requested', self.gpu_batch_envs) or self.gpu_batch_envs)} арен (effective/requested)" if gpu_only_this_cycle else
                                   f"{self.cpu_workers_active if executor is not None else 1} CPU-арен + GPU фактически {int(self.stats.get('gpu_rollout_envs', 0) or 0)}/{int(self.stats.get('gpu_rollout_envs_requested', self.gpu_batch_envs) or self.gpu_batch_envs)} арен" if hybrid_this_cycle else
                                   "одна CPU-физика + GPU policy" if self.rollout_gpu_inference else
                                   f"{self.cpu_workers_active if executor is not None else 1} CPU-арен")
                                + f"; противник: {self.opponent_name}; MLP {self.policy.width}×{self.policy.depth}."
                                + (f" {self.auto_rollout_selection_note}" if self.auto_rollout_selection_note else "")
                                + (f" GPU fallback: {self.gpu_rollout_fallback_reason}" if self.gpu_rollout_fallback_reason else "")),
                    "training_model_id": self.model_id, "training_model_name": self.model_name,
                    "device": self.policy.last_device if self.policy.last_device != "NumPy rollout / not training" else self.device_mode.upper(),
                    "phase": "Подготовка следующего пакета опыта",
                    "rollout_sec_last": rollout_elapsed, "ppo_sec_last": ppo_elapsed,
                    "rollout_episodes_last": len(results),
                    "cpu_episodes_during_gpu": self._last_hybrid_cpu_episodes if hybrid_this_cycle else 0,
                    "cpu_records_during_gpu": self._last_hybrid_cpu_records if hybrid_this_cycle else 0,
                    "gpu_batch_seconds": self._last_gpu_batch_seconds,
                    "gpu_device_seconds": self._last_gpu_device_seconds,
                    "gpu_host_overhead_seconds": self._last_gpu_host_overhead_seconds,
                    "gpu_episodes_per_sec": self._last_gpu_episodes_per_sec,
                    "rollout_episodes_per_sec": len(results) / max(0.001, rollout_elapsed),
                    "cpu_record_buffer_cap": self._last_hybrid_buffer_cap,
                    "cpu_rollout_sec_last": self._last_cpu_rollout_seconds,
                    "cpu_rollout_episodes_last": self._last_cpu_rollout_episodes,
                    "cpu_rollout_records_last": self._last_cpu_rollout_records,
                    "cpu_rollout_steps_last": self._last_cpu_rollout_steps,
                    "cpu_rollout_steps_per_sec": self._last_cpu_rollout_steps / max(0.001, self._last_cpu_rollout_seconds),
                    "gpu_rollout_sec_last": self._last_gpu_rollout_seconds,
                    "gpu_rollout_episodes_last": self._last_gpu_rollout_episodes,
                    "gpu_rollout_records_last": self._last_gpu_rollout_records,
                    "gpu_rollout_steps_last": self._last_gpu_rollout_steps,
                    "gpu_rollout_steps_per_sec": self._last_gpu_rollout_steps / max(0.001, self._last_gpu_rollout_seconds),
                    "cpu_wait_gpu_sec_last": self._last_cpu_wait_gpu_seconds,
                    "ppo_records_last": record_count,
                    "imitation_loss_last": float(getattr(self.policy, "last_imitation_loss", 0.0)),
                    "imitation_coef_last": float(getattr(self.policy, "last_imitation_coef", 0.0)),
                    "ppo_updates": int(self.stats.get("ppo_updates", 0)) + 1,
                    "epochs_last": self.epochs,
                    "cpu_rollout_episodes_derived": sum(1 for r in results if len(r) > 6 and isinstance(r[6], dict) and not r[6].get("rollout_backend")),
                    "gpu_rollout_episodes_derived": sum(1 for r in results if len(r) > 6 and isinstance(r[6], dict) and r[6].get("rollout_backend") == "GPU tensor physics"),
                    "rollout_gpu_inference": self.rollout_gpu_inference,
                    "hybrid_gpu_rollout": self.hybrid_gpu_rollout,
                    "gpu_only_rollout": self.gpu_only_rollout,
                    "rollout_physics_device": "CUDA" if (gpu_only_this_cycle or hybrid_this_cycle) else "CPU",
                    "rollout_policy_device": "CUDA" if (gpu_only_this_cycle or hybrid_this_cycle or self.rollout_gpu_inference) else "CPU/NumPy",
                    "gpu_batch_envs": self.gpu_batch_envs,
                    "blue_n": batch_compositions[0], "red_n": batch_compositions[1],
                    "workers": 0 if gpu_only_this_cycle else (self.cpu_workers_active if executor is not None else 1),
                })
            self._write_training_cycle_log(
                record_count=record_count, loss=loss,
                rollout_elapsed=rollout_elapsed, ppo_elapsed=ppo_elapsed,
                batch_steps=batch_steps, results=results,
                cpu_seconds=self._last_cpu_rollout_seconds,
                gpu_seconds=self._last_gpu_rollout_seconds,
                cpu_wait_seconds=self._last_cpu_wait_gpu_seconds,
            )
            # Large GPU policies can be tens/hundreds of MB. Avoid compressing the full
            # checkpoint after every rollout; save every 5 updates, immediately on stop,
            # and always once more in the finalizer.
            self._save_batches += 1
            if self._save_batches >= 5 or self.stop_event.is_set() or time.monotonic() - self._last_checkpoint_time >= 30.0:
                self.policy.save(self.save_path)
                self._save_batches = 0
                self._last_checkpoint_time = time.monotonic()
            self._save_stats()

    def _collect_hybrid_async(self, executor=None):
        """Collect on CPU only while CPU futures can finish before the GPU batch.

        Every rollout belongs to one frozen policy version. CPU work is submitted as
        independent futures and polled together with the GPU future; a slow episode
        is cancelled at the cycle boundary rather than holding PPO hostage. Results
        from unfinished/truncated CPU episodes are discarded. This matters greatly
        for large policies whose NumPy CPU inference can be orders of magnitude slower
        than the vectorized CUDA backend.
        """
        # Usually the previous cycle's CPU tasks were joined at the post-PPO weight
        # publish. This guard covers early exits where no PPO update was performed.
        self._drain_hybrid_worker_futures()
        if self._gpu_rollout_executor is None:
            self._gpu_rollout_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="GravitrueX-GPU-rollout")
        gpu_seed = self.rng.randrange(1, 2**31 - 1)
        self._last_gpu_batch_seconds = 0.0
        self._batch_gpu_device_seconds = 0.0
        gpu_started = time.perf_counter()
        gpu_future = self._gpu_rollout_executor.submit(self._collect_gpu_batch, gpu_seed)

        cpu_results = []
        cpu_records = 0
        cpu_episodes = 0
        cpu_steps = 0
        cpu_collect_started = time.perf_counter()
        available_ram = _available_ram_bytes()
        limits = resolve_hybrid_limits(self.workers, self.gpu_batch_envs,
                                       logical_cpus=os.cpu_count(),
                                       available_ram_bytes=available_ram)
        pool_capacity = (int(getattr(executor, "_max_workers", self.cpu_workers_active))
                         if executor is not None else 0)
        # RAM can fall after startup (especially when the CUDA rollout stages
        # pinned host buffers). Re-evaluate each cycle and submit only the number
        # of CPU tasks supported by current headroom, up to the existing pool size.
        batch_n = min(max(0, pool_capacity), int(limits["active_workers"]))
        max_cpu_records = int(limits["cpu_record_cap"])
        max_cpu_episodes = 64
        self.hybrid_limits = dict(limits)
        cycle_event = self.mp_cycle_event
        if cycle_event is not None:
            # The preceding cutoff tasks are drained before publishing new weights.
            cycle_event.clear()
        with self.state_lock:
            self.stats.update({
                "cpu_workers_requested": int(self.workers),
                "cpu_workers_active": int(batch_n),
                "hybrid_available_ram_gb": limits["available_ram_gb"],
                "hybrid_cpu_record_cap": max_cpu_records,
                "cpu_record_cap_hit": False,
                "cpu_rejected_over_cap_records": 0,
                "hybrid_worker_drain_seconds": 0.0,
                "hybrid_cpu_sidecar_state": ("active" if batch_n else
                                             "disabled (RAM safety floor or no worker pool)"),
            })
        self._last_hybrid_buffer_cap = max_cpu_records
        self._set_phase(
            f"Гибрид async: GPU запрос {self.gpu_batch_envs} арен (фактический размер определяется лимитом {self.gpu_record_budget:,}); CPU буфер до {max_cpu_records:,} записей")

        pending = set()
        cpu_collection_open = bool(executor is not None and batch_n > 0)
        stop_reason = "gpu-complete"
        cpu_cap_hit = False
        cpu_rejected_records = 0

        def submit_cpu_task():
            if not cpu_collection_open or self.stop_event.is_set():
                return False
            if cpu_episodes + len(pending) >= max_cpu_episodes:
                return False
            if cpu_records >= max_cpu_records:
                return False
            pending.add(executor.submit(_rollout_worker, self._make_task()))
            return True

        def harvest_cpu_future(future):
            nonlocal cpu_records, cpu_episodes, cpu_steps, cpu_cap_hit, cpu_rejected_records
            try:
                row = future.result()
            except Exception as exc:
                self._runtime_log(f"Hybrid CPU rollout worker failed: {type(exc).__name__}: {exc}",
                                  traceback.format_exc())
                return
            # Only futures harvested before the cycle cutoff can contribute. Any
            # future still pending at the cutoff may return a truncated trajectory.
            if not row or len(row) < 6:
                return
            count = len(row[0])
            if count <= 0:
                return
            # Episodes are atomic: do not overshoot the cap by appending a whole
            # episode that would take the record buffer above the RAM safety limit.
            if not episode_fits_record_budget(cpu_records, count, max_cpu_records):
                cpu_cap_hit = True
                cpu_rejected_records += count
                with self.state_lock:
                    self.stats.update({
                        "cpu_record_cap_hit": True,
                        "cpu_rejected_over_cap_records": cpu_rejected_records,
                    })
                return
            cpu_results.append(row)
            cpu_episodes += 1
            cpu_records += count
            cpu_steps += int(row[5])
            with self.state_lock:
                self.stats.update({
                    "cpu_episodes_during_gpu": cpu_episodes,
                    "cpu_records_during_gpu": cpu_records,
                    "cpu_record_buffer_cap": max_cpu_records,
                    "cpu_record_cap_hit": cpu_cap_hit,
                    "cpu_rejected_over_cap_records": cpu_rejected_records,
                    "phase": (f"GPU собирает (запрошено {self.gpu_batch_envs}, фактический батч ограничен лимитом {self.gpu_record_budget:,}); CPU накопил "
                              f"{cpu_episodes} эп. / {cpu_records:,} записей"),
                })

        # Prime no more than one task per CPU process. Refill only when a full CPU
        # episode has actually completed, while polling GPU completion concurrently.
        for _ in range(min(batch_n, max_cpu_episodes)):
            submit_cpu_task()

        while not self.stop_event.is_set():
            if gpu_future.done():
                stop_reason = "gpu-complete"
                break
            if cpu_cap_hit or cpu_records >= max_cpu_records:
                stop_reason = "cpu-record-cap"
                break
            if cpu_episodes + len(pending) >= max_cpu_episodes:
                stop_reason = "cpu-episode-cap"
                break
            if not pending:
                # With no eligible CPU workers left, wait only on the GPU future.
                wait([gpu_future], timeout=0.025, return_when=FIRST_COMPLETED)
                continue
            done, _ = wait([gpu_future, *pending], timeout=0.025,
                           return_when=FIRST_COMPLETED)
            # GPU takes precedence at the boundary. Futures that are already done
            # are salvaged below; any running tasks are told to stop immediately.
            if gpu_future in done or gpu_future.done():
                stop_reason = "gpu-complete"
                break
            completed_cpu = [future for future in done if future is not gpu_future]
            for future in completed_cpu:
                pending.discard(future)
                harvest_cpu_future(future)
                if (not self.stop_event.is_set() and not cpu_cap_hit
                        and cpu_records < max_cpu_records
                        and cpu_episodes + len(pending) < max_cpu_episodes):
                    submit_cpu_task()
                if cpu_cap_hit or cpu_records >= max_cpu_records:
                    stop_reason = "cpu-record-cap"
                    break
            if stop_reason == "cpu-record-cap":
                break
        if self.stop_event.is_set():
            stop_reason = "user-stop"

        # Salvage fully finished episodes before setting the cycle cutoff event.
        # Do not harvest anything that completes after cutoff, as it may be partial.
        if stop_reason != "user-stop":
            for future in list(pending):
                if future.done() and not future.cancelled():
                    pending.discard(future)
                    harvest_cpu_future(future)

        cpu_producer_elapsed = max(0.001, time.perf_counter() - cpu_collect_started)
        cpu_collection_open = False
        if cycle_event is not None:
            cycle_event.set()
        active_after_cutoff = []
        for future in list(pending):
            cancelled = future.cancel()  # queued tasks cancel; active tasks see cycle_event
            if not cancelled and not future.done() and not future.cancelled():
                active_after_cutoff.append(future)
        # Keep the shared policy snapshot immutable while CPU tasks wind down, but do
        # not block GPU completion/PPO on them. _sync_shared_weights joins them after
        # the PPO update, overlapping the small shutdown tail with accelerator work.
        self._hybrid_pending_worker_futures.extend(active_after_cutoff)

        if self.stop_event.is_set():
            try:
                gpu_future.result(timeout=30.0)
            except Exception:
                pass
            self._last_hybrid_cpu_episodes = cpu_episodes
            self._last_hybrid_cpu_records = cpu_records
            self._last_cpu_rollout_seconds = cpu_producer_elapsed
            self._last_cpu_rollout_episodes = cpu_episodes
            self._last_cpu_rollout_records = cpu_records
            self._last_cpu_rollout_steps = cpu_steps
            self._last_gpu_rollout_seconds = 0.0
            self._last_gpu_rollout_episodes = 0
            self._last_gpu_rollout_records = 0
            self._last_gpu_rollout_steps = 0
            self._last_cpu_wait_gpu_seconds = 0.0
            return cpu_results

        # If the RAM/record cap stopped CPU first, this is the only intentional wait:
        # CPU has already quiesced, and we still need the in-flight GPU batch.
        wait_started = time.perf_counter()
        gpu_was_done_before_wait = gpu_future.done()
        gpu_results = []
        try:
            gpu_results = gpu_future.result()
        except Exception as gpu_exc:
            # Keep any completed CPU experience; disable only the failing GPU collector.
            self.hybrid_gpu_rollout = False
            self.gpu_rollout_fallback_reason = str(gpu_exc)[:240]
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            with self.state_lock:
                self.stats["message"] = (
                    f"GPU пакет отключён ({self.gpu_rollout_fallback_reason}); продолжение на CPU-аренах.")
                self.stats["hybrid_gpu_rollout"] = False
                self.stats["gpu_rollout_backend"] = "fallback to CPU"
            traceback.print_exc()
        gpu_wall_elapsed = max(0.001, time.perf_counter() - gpu_started)
        cpu_wait_gpu = (0.0 if gpu_was_done_before_wait
                        else max(0.0, time.perf_counter() - wait_started))
        gpu_records = sum(len(row[0]) for row in gpu_results)
        gpu_steps = sum(int(row[5]) for row in gpu_results)
        gpu_elapsed = max(0.001, self._last_gpu_batch_seconds or gpu_wall_elapsed)
        self._last_gpu_batch_wall_seconds = gpu_wall_elapsed
        self._last_hybrid_cpu_episodes = cpu_episodes
        self._last_hybrid_cpu_records = cpu_records
        self._last_gpu_batch_seconds = gpu_elapsed
        self._last_gpu_episodes_per_sec = len(gpu_results) / gpu_elapsed if gpu_results else 0.0
        self._last_cpu_rollout_seconds = cpu_producer_elapsed
        self._last_cpu_rollout_episodes = cpu_episodes
        self._last_cpu_rollout_records = cpu_records
        self._last_cpu_rollout_steps = cpu_steps
        self._last_gpu_rollout_seconds = gpu_elapsed
        self._last_gpu_rollout_episodes = len(gpu_results)
        self._last_gpu_rollout_records = gpu_records
        self._last_gpu_rollout_steps = gpu_steps
        self._last_cpu_wait_gpu_seconds = cpu_wait_gpu
        self._last_gpu_device_seconds = float(getattr(self, "_batch_gpu_device_seconds", 0.0) or 0.0)
        self._last_gpu_host_overhead_seconds = max(0.0, gpu_elapsed - self._last_gpu_device_seconds)
        cpu_rate = cpu_steps / max(0.001, cpu_producer_elapsed)
        gpu_rate = gpu_steps / max(0.001, gpu_elapsed)

        # Adaptive sidecar backoff: for very large policies a CPU episode may take
        # many times longer than the whole CUDA batch. After two cycles with no
        # completed CPU episode, stop spending CPU on trajectories that cannot join
        # the current on-policy batch. GPU rollout and CUDA PPO continue unchanged.
        if gpu_results and cpu_episodes == 0:
            self._hybrid_low_yield_cycles += 1
        else:
            self._hybrid_low_yield_cycles = 0
        if (gpu_results and self._hybrid_low_yield_cycles >= 2
                and self.hybrid_gpu_rollout and not self.stop_event.is_set()):
            self.hybrid_gpu_rollout = False
            self.gpu_only_rollout = True
            self.policy.allow_gpu_inference = True
            self.policy.inference_mode = "gpu"
            note = ("Гибрид отключён автоограничителем: CPU-воркеры не завершили ни одного эпизода "
                    "за два GPU-цикла. Продолжаем GPU-only, чтобы не тратить CPU и не задерживать PPO.")
            self.auto_rollout_selection_note = note
            self._runtime_log(note)
            with self.state_lock:
                self.stats.update({
                    "hybrid_gpu_rollout": False,
                    "gpu_only_rollout": True,
                    "hybrid_cpu_sidecar_state": "auto-disabled: zero completed episodes in two GPU cycles",
                    "auto_rollout_selection_note": note,
                    "message": note,
                    "phase": "Автоограничитель: продолжаем только на GPU",
                })
            # All CPU futures are joined above, so the process pool is now idle.
            # Shutting it down releases memory/handles; the context manager's later
            # shutdown is idempotent.
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass

        if self.auto_rollout_balance and cpu_steps > 0 and gpu_steps > 0:
            # If scalar CPU physics benchmarks faster, retain CUDA for PPO but stop
            # adding the slower tensor rollout. Otherwise keep the hybrid, unless its
            # zero-yield guard above has already selected GPU-only.
            if gpu_rate < cpu_rate * 0.85 and not self.gpu_only_rollout:
                self.hybrid_gpu_rollout = False
                self.auto_rollout_selection_note = (
                    f"Автовыбор: CPU rollout быстрее ({cpu_rate:,.0f} шаг/с против GPU {gpu_rate:,.0f}); "
                    "GPU-rollout отключён, PPO продолжает использовать выбранное устройство.")
            elif not self.auto_rollout_selection_note:
                self.auto_rollout_selection_note = (
                    f"Автовыбор оставил гибрид: CPU {cpu_rate:,.0f} шаг/с, GPU {gpu_rate:,.0f} шаг/с.")
            with self.state_lock:
                self.stats["auto_rollout_balance"] = True
                self.stats["auto_rollout_selection_note"] = self.auto_rollout_selection_note
                self.stats["hybrid_gpu_rollout"] = self.hybrid_gpu_rollout
                self.stats["gpu_only_rollout"] = self.gpu_only_rollout
                self.stats["message"] = (self.auto_rollout_selection_note
                                          if self.gpu_only_rollout else self.stats.get("message", ""))

        with self.state_lock:
            self.stats.update({
                "gpu_device_seconds": self._last_gpu_device_seconds,
                "gpu_host_overhead_seconds": self._last_gpu_host_overhead_seconds,
                "auto_rollout_selection_note": self.auto_rollout_selection_note,
                "cpu_episodes_during_gpu": cpu_episodes,
                "cpu_records_during_gpu": cpu_records,
                "gpu_batch_seconds": gpu_elapsed,
                "gpu_batch_wall_seconds": gpu_wall_elapsed,
                "gpu_episodes_per_sec": self._last_gpu_episodes_per_sec,
                "cpu_record_buffer_cap": max_cpu_records,
                "cpu_record_cap_hit": cpu_cap_hit,
                "cpu_rejected_over_cap_records": cpu_rejected_records,
                "hybrid_worker_drain_seconds": self._last_hybrid_worker_drain_seconds,
                "hybrid_low_yield_cycles": self._hybrid_low_yield_cycles,
            })
        return cpu_results + gpu_results

    def _collect_gpu_batch(self, seed: Optional[int] = None):
        """Collect independent tensor-physics battles on CUDA, lowering batch on OOM."""
        import torch
        gpu_started = time.perf_counter()
        self._last_gpu_graph_requested = False
        self._last_gpu_graph_used = False
        self._last_gpu_graph_error = ""
        self._last_gpu_max_decisions = 0
        self._last_gpu_record_budget = self.gpu_record_budget
        self._gpu_batch_requested_envs = int(self.gpu_batch_envs)
        self._last_gpu_oom_fallback = ""
        if not torch.cuda.is_available():
            raise RuntimeError("GPU-пакет боёв выбран, но CUDA недоступна. Запусти BAT в режиме NVIDIA CUDA.")
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        if self.opponent_mode == "model":
            raise RuntimeError("GPU-пакет боёв на этом этапе поддерживает эвристического противника и self-play. Для отдельной модели-противника выбери CPU: параллельные арены.")
        import sys
        GPUVectorArenaBatch = _load_gpu_batch_class()
        engine = sys.modules[__name__]
        requested_count = int(self.gpu_batch_envs)
        candidate_count = requested_count
        batch_seed = int(seed if seed is not None else self.rng.randrange(1, 2**31 - 1))
        batch = None
        results = None
        memory_snapshot = {}
        record_slots = 0
        obs_history_gib = 0.0
        while True:
            try:
                batch_kwargs = dict(
                    engine=engine, policy=self.policy, count=candidate_count,
                    blue_n=self.fixed_blue, red_n=self.fixed_red,
                    random_compositions=self.random_compositions,
                    opponent_mode=self.opponent_mode, max_decisions=self.max_decisions,
                    device="cuda", seed=batch_seed,
                )
                if "max_records" in inspect.signature(GPUVectorArenaBatch.__init__).parameters:
                    batch_kwargs["max_records"] = self.gpu_record_budget
                batch = GPUVectorArenaBatch(**batch_kwargs)
                self._last_gpu_max_decisions = int(getattr(batch, "max_decisions", 0) or 0)
                with self.state_lock:
                    self.stats["gpu_max_decisions_last"] = self._last_gpu_max_decisions
                    self.stats["gpu_record_budget"] = self._last_gpu_record_budget
                    self.stats["gpu_rollout_envs_requested"] = requested_count
                    self.stats["gpu_rollout_envs_attempted"] = candidate_count
                results = batch.collect(self.stop_event)
                memory_snapshot = _cuda_memory_snapshot()
                record_slots = int(getattr(batch, "record_slots", 10) or 10)
                obs_history_gib = (
                    int(getattr(batch, "max_decisions", self.max_decisions))
                    * int(getattr(batch, "E", candidate_count))
                    * record_slots * int(getattr(engine, "OBS_SIZE", OBS_SIZE)) * 4
                    / float(1024 ** 3)
                )
                break
            except Exception as exc:
                oom = is_cuda_oom_error(exc)
                # If the record budget already reduced the request to e.g. 400 actual
                # arenas, retrying 4096 -> 3840 recreates the same tensors. On OOM in
                # collect(), reduce from the effective allocated batch instead.
                oom_basis = int(getattr(batch, "E", 0) or candidate_count)
                next_count = next_gpu_arena_fallback(oom_basis) if oom else None
                if next_count is None and oom_basis != candidate_count:
                    next_count = next_gpu_arena_fallback(candidate_count)
                if next_count is None:
                    raise
                old_count = oom_basis
                candidate_count = min(next_count, max(1, candidate_count - 1))
                self.gpu_batch_envs = candidate_count
                self._last_gpu_oom_fallback = (
                    f"CUDA OOM: {old_count} арен не поместились; пакет автоматически снижен до {candidate_count}.")
                self._runtime_log(self._last_gpu_oom_fallback,
                                  f"{type(exc).__name__}: {exc}")
                with self.state_lock:
                    self.stats.update({
                        "message": self._last_gpu_oom_fallback,
                        "phase": self._last_gpu_oom_fallback,
                        "gpu_rollout_oom_fallback": self._last_gpu_oom_fallback,
                        "gpu_rollout_envs_requested": requested_count,
                        "gpu_rollout_envs": candidate_count,
                    })
                try:
                    batch = None
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                # A fixed batch_seed keeps the randomized setup comparable across retries.
        self._batch_gpu_device_seconds = float(getattr(batch, "last_cuda_device_seconds", 0.0) or 0.0)
        self._last_gpu_graph_requested = bool(getattr(batch, "use_cuda_graph", False))
        self._last_gpu_graph_used = bool(getattr(batch, "last_cuda_graph_used", False))
        self._last_gpu_graph_error = str(getattr(batch, "last_cuda_graph_error", "") or "")
        graph_status = ("использован" if self._last_gpu_graph_used else
                        ("eager fallback: " + self._last_gpu_graph_error[:180]
                         if self._last_gpu_graph_requested and self._last_gpu_graph_error else
                         "eager (CUDA Graph отключён)" if not self._last_gpu_graph_requested else
                         "eager fallback без подробной причины"))
        with self.state_lock:
            self.stats["gpu_cuda_graph_requested"] = self._last_gpu_graph_requested
            self.stats["gpu_cuda_graph_used"] = self._last_gpu_graph_used
            self.stats["gpu_cuda_graph_status"] = graph_status
        try:
            if str(batch.dev) == "cuda":
                torch.cuda.synchronize()
        except Exception:
            pass
        # Include allocation/retry time: this is the true wall cost of GPU rollout.
        self._last_gpu_batch_seconds = max(0.001, time.perf_counter() - gpu_started)
        with self.state_lock:
            effective_count = int(getattr(batch, "E", candidate_count) or candidate_count)
            budget_reduced = int(getattr(batch, "record_budget_reduced_arenas", 0) or 0)
            self.stats.update({
                "gpu_rollout_backend": "CUDA tensor physics",
                "gpu_rollout_envs_requested": requested_count,
                "gpu_rollout_envs_attempted": candidate_count,
                "gpu_rollout_envs": effective_count,
                "gpu_rollout_envs_budget_reduced": budget_reduced,
                "gpu_max_decisions_last": int(getattr(batch, "max_decisions", self.max_decisions)),
                "gpu_rollout_oom_fallback": self._last_gpu_oom_fallback,
                "cuda_memory_snapshot": memory_snapshot,
                "gpu_record_slots": record_slots,
                "gpu_obs_history_estimate_gib": round(obs_history_gib, 3),
            })
        return results

    def _make_shared_layout(self):
        layout = []
        offset = 0
        for name, array in self.policy.params.items():
            arr = np.asarray(array, dtype=np.float32)
            size = int(arr.size)
            layout.append((name, tuple(arr.shape), offset, offset + size))
            offset += size
        return layout, offset

    def _drain_hybrid_worker_futures(self) -> float:
        """Join cutoff CPU tasks only when safe; this usually overlaps the PPO update."""
        pending = [f for f in self._hybrid_pending_worker_futures
                   if f is not None and not f.done() and not f.cancelled()]
        self._hybrid_pending_worker_futures = []
        if not pending:
            self._last_hybrid_worker_drain_seconds = 0.0
            return 0.0
        started = time.perf_counter()
        wait(pending)
        elapsed = max(0.0, time.perf_counter() - started)
        self._last_hybrid_worker_drain_seconds = elapsed
        with self.state_lock:
            self.stats["hybrid_worker_drain_seconds"] = elapsed
        return elapsed

    def _sync_shared_weights(self):
        if self._shared_array is None or self._shared_layout is None:
            return
        # CPU workers read an immutable shared snapshot while PPO updates its private
        # policy. Join them here, after PPO has run, before publishing the next version.
        self._drain_hybrid_worker_futures()
        for name, _shape, start, end in self._shared_layout:
            self._shared_array[start:end] = self.policy.params[name].reshape(-1)

    def run(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Keep the hybrid CPU sidecar conservative: leave scheduling capacity for
            # CUDA launch/dispatch and the UI; lower worker priority in the pool init.
            if self.hybrid_gpu_rollout:
                limits = resolve_hybrid_limits(
                    self.workers, self.gpu_batch_envs, logical_cpus=os.cpu_count(),
                    available_ram_bytes=_available_ram_bytes())
                self.cpu_workers_active = int(limits["active_workers"])
                self.hybrid_limits = dict(limits)
                with self.state_lock:
                    self.stats.update({
                        "workers": self.cpu_workers_active,
                        "cpu_workers_requested": self.workers,
                        "cpu_workers_active": self.cpu_workers_active,
                        "hybrid_available_ram_gb": limits["available_ram_gb"],
                        "hybrid_cpu_record_cap": int(limits["cpu_record_cap"]),
                        "hybrid_cpu_priority_requested": True,
                    })
                self._runtime_log(
                    f"Hybrid resource plan: CPU workers {self.cpu_workers_active}/{self.workers}; "
                    f"GPU arenas {self.gpu_batch_envs}; CPU record cap {limits['cpu_record_cap']}; "
                    f"available RAM {limits['available_ram_gb'] if limits['available_ram_gb'] is not None else 'unknown'} GB; "
                    "CPU workers run at below-normal priority when supported.")
            else:
                self.cpu_workers_active = self.workers
            # Load/migrate the latest disk checkpoint before creating shared views.
            self.policy.load(self.save_path)
            self.policy.save(self.save_path)
            if (not self.gpu_only_rollout and (self.workers > 1 or self.hybrid_gpu_rollout)
                and not (self.hybrid_gpu_rollout and self.cpu_workers_active <= 0)):
                try:
                    from multiprocessing import shared_memory
                    ctx = mp.get_context("spawn")
                    layout, total = self._make_shared_layout()
                    self._shared_memory = shared_memory.SharedMemory(create=True, size=total * np.dtype(np.float32).itemsize)
                    self._shared_array = np.ndarray((total,), dtype=np.float32, buffer=self._shared_memory.buf)
                    self._shared_layout = layout
                    self._sync_shared_weights()
                    self.mp_stop_event = ctx.Event()
                    self.mp_cycle_event = ctx.Event()
                    with ProcessPoolExecutor(
                        max_workers=self.cpu_workers_active, mp_context=ctx,
                        initializer=_init_rollout_worker,
                        initargs=(self._shared_memory.name, total, layout, self.policy.width,
                                  self.policy.depth, self.mp_stop_event,
                                  str(self.opponent_path) if self.opponent_mode == "model" and self.opponent_path else "",
                                  bool(self.hybrid_gpu_rollout), self.mp_cycle_event,
                                  self.policy.architecture),
                    ) as executor:
                        self._run_batches(executor)
                except Exception as exc:
                    worker_tb = traceback.format_exc()
                    self._runtime_log(
                        f"Parallel rollout executor failed; attempting safe fallback: {type(exc).__name__}: {exc}",
                        worker_tb,
                    )
                    traceback.print_exc()
                    if self.hybrid_gpu_rollout:
                        self.gpu_rollout_fallback_reason = f"CPU worker pool недоступен: {exc}"[:240]
                        self.hybrid_gpu_rollout = False
                    with self.state_lock:
                        self.stats["message"] = f"Параллельный режим недоступен ({exc}); переход на CPU-арены без гибридного распараллеливания."
                        self.stats["hybrid_gpu_rollout"] = False
                    if not self.stop_event.is_set():
                        self._run_batches(None)
                finally:
                    if self._shared_memory is not None:
                        try:
                            self._shared_memory.close()
                        except Exception:
                            pass
                        try:
                            self._shared_memory.unlink()
                        except Exception:
                            pass
                        self._shared_memory = None
                        self._shared_array = None
                        self._shared_layout = None
            else:
                self._run_batches(None)
        except Exception as exc:
            tb_text = traceback.format_exc()
            try:
                with self.state_lock:
                    self.failed_phase = str(self.stats.get("phase", "этап неизвестен"))
            except Exception:
                self.failed_phase = "этап неизвестен"
            self.fatal_error = f"{type(exc).__name__}: {exc}"
            self.fatal_traceback = tb_text
            with self.state_lock:
                self.stats["message"] = f"Ошибка обучения: {self.fatal_error}"
                self.stats["phase"] = f"СБОЙ: {self.failed_phase}"
                self.stats["fatal_error"] = self.fatal_error
                self.stats["failed_phase"] = self.failed_phase
                self.stats["crash_report"] = str(self.crash_report_path)
            report = [
                "GRAVITRUEX — TRAINING CRASH REPORT",
                f"Время: {datetime.now().astimezone().isoformat(timespec='seconds')}",
                f"Модель: {self.model_name} [{self.model_id}]",
                f"Архитектура: {self.policy.width}x{self.policy.depth} ({self.policy.count_params():,} параметров)",
                f"Этап: {self.failed_phase}",
                f"PPO device mode: {self.device_mode}",
                f"GPU-only rollout: {self.gpu_only_rollout}; hybrid: {self.hybrid_gpu_rollout}; GPU inference: {self.rollout_gpu_inference}",
                f"Последнее сообщение: {self.fatal_error}",
                "",
                "TRACEBACK:",
                tb_text,
            ]
            try:
                tmp = self.crash_report_path.with_suffix(".txt.tmp")
                tmp.write_text("\n".join(report), encoding="utf-8")
                os.replace(tmp, self.crash_report_path)
            except Exception:
                pass
            self._runtime_log(f"FATAL training error at phase '{self.failed_phase}': {self.fatal_error}", tb_text)
            self._save_stats()
            traceback.print_exc()
        finally:
            if self._gpu_rollout_executor is not None:
                try:
                    self._gpu_rollout_executor.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    pass
                self._gpu_rollout_executor = None
            try:
                self.policy.save(self.save_path)
                # NumPy weights are the durable source of truth. Drop the trainer's
                # CUDA parameters/optimizer after checkpointing so a stopped session
                # doesn't keep those live allocations until the window is closed.
                reset_torch = getattr(self.policy, "_reset_torch_state", None)
                if callable(reset_torch):
                    reset_torch()
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        snapshot_after_stop = _cuda_memory_snapshot()
                    else:
                        snapshot_after_stop = {}
                except Exception:
                    snapshot_after_stop = {}
                with self.state_lock:
                    self.stats["cuda_memory_snapshot_after_stop"] = snapshot_after_stop
                self._save_stats()
            except Exception:
                pass
            with self.state_lock:
                if not self.stats.get("message", "").startswith("Ошибка"):
                    self.stats["message"] = "Обучение остановлено. Веса сохранены."

    def _write_training_cycle_log(self, record_count: int, loss: float,
                                  rollout_elapsed: float, ppo_elapsed: float,
                                  batch_steps: int, results: list,
                                  cpu_seconds: float, gpu_seconds: float,
                                  cpu_wait_seconds: float) -> None:
        """Write per-source rollouts and a compact, shareable report after each update."""
        stamp = datetime.now().isoformat(timespec="seconds")
        def rec_count(row):
            try:
                return len(row[0])
            except Exception:
                return 0
        cpu_eps = int(self._last_cpu_rollout_episodes)
        gpu_eps = int(self._last_gpu_rollout_episodes)
        cpu_records = int(self._last_cpu_rollout_records)
        gpu_records = int(self._last_gpu_rollout_records)
        cpu_steps = int(self._last_cpu_rollout_steps)
        gpu_steps = int(self._last_gpu_rollout_steps)
        cpu_rate = cpu_steps / max(0.001, cpu_seconds)
        gpu_rate = gpu_steps / max(0.001, gpu_seconds)
        update_no = int(self.stats.get("ppo_updates", 0))
        row = {
            "time": stamp, "model_id": self.model_id, "model_name": self.model_name,
            "gpu_batch_requested_envs": int(getattr(self, "_gpu_batch_requested_envs", self.gpu_batch_envs))
                if (self._last_cycle_hybrid_mode or self._last_cycle_gpu_only_mode) else 0,
            "gpu_batch_actual_envs": int(self.stats.get("gpu_rollout_envs", 0) or 0)
                if (self._last_cycle_hybrid_mode or self._last_cycle_gpu_only_mode) else 0,
            "gpu_batch_oom_fallback": str(getattr(self, "_last_gpu_oom_fallback", "")),
            "gpu_record_budget": int(self._last_gpu_record_budget),
            "gpu_batch_budget_reduced_arenas": int(self.stats.get("gpu_rollout_envs_budget_reduced", 0) or 0),
            "gpu_episode_results": gpu_eps,
            "hybrid_worker_drain_seconds": float(getattr(self, "_last_hybrid_worker_drain_seconds", 0.0)),
            "cpu_record_cap_hit": bool(self.stats.get("cpu_record_cap_hit", False)),
            "cpu_rejected_over_cap_records": int(self.stats.get("cpu_rejected_over_cap_records", 0) or 0),
            "profile": f"{self.policy.width}x{self.policy.depth}",
            "opponent": self.opponent_name,
            "rollout_mode": ("gpu_physics_only" if self._last_cycle_gpu_only_mode else
                             "hybrid_cpu_gpu" if self._last_cycle_hybrid_mode else
                             "cpu_physics_gpu_policy" if self.rollout_gpu_inference else "cpu_physics"),
            "physics_device": "CUDA" if (self._last_cycle_gpu_only_mode or self._last_cycle_hybrid_mode) else "CPU",
            "policy_inference_device": "CUDA" if (self._last_cycle_gpu_only_mode or self._last_cycle_hybrid_mode or self.rollout_gpu_inference) else "CPU/NumPy",
            "ppo_update_device": str(self.policy.last_device),
            "cpu_workers": 0 if self._last_cycle_gpu_only_mode else (
                int(self.stats.get("cpu_workers_active", self.cpu_workers_active))
                if self._last_cycle_hybrid_mode else self.workers),
            "gpu_arenas": int(self.stats.get("gpu_rollout_envs", 0) or 0)
                if (self._last_cycle_hybrid_mode or self._last_cycle_gpu_only_mode) else 0,
            "cpu_seconds": round(float(cpu_seconds), 4), "cpu_episodes": cpu_eps,
            "cpu_records": cpu_records, "cpu_physics_steps": cpu_steps,
            "cpu_steps_per_sec": round(cpu_rate, 2),
            "gpu_seconds": round(float(gpu_seconds), 4), "gpu_episodes": gpu_eps,
            "gpu_records": gpu_records, "gpu_physics_steps": gpu_steps,
            "gpu_steps_per_sec": round(gpu_rate, 2),
            "gpu_device_seconds": round(float(self._last_gpu_device_seconds), 4),
            "gpu_host_overhead_seconds": round(float(self._last_gpu_host_overhead_seconds), 4),
            "gpu_cuda_graph_requested": bool(self._last_gpu_graph_requested),
            "gpu_cuda_graph_used": bool(self._last_gpu_graph_used),
            "gpu_cuda_graph_error": str(self._last_gpu_graph_error)[:240],
            "gpu_max_decisions_per_arena": int(self._last_gpu_max_decisions),
            "requested_max_decisions_per_arena": int(self.max_decisions),
            "action_priors_mode": self.action_priors_mode,
            "reward_mode": self.reward_mode,
            "desired_range": self.desired_range,
            "auto_rollout_balance": bool(self.auto_rollout_balance),
            "auto_rollout_selection_note": str(self.auto_rollout_selection_note),
            "cpu_wait_gpu_seconds": round(float(cpu_wait_seconds), 4),
            "rollout_wall_seconds": round(float(rollout_elapsed), 4),
            "ppo_seconds": round(float(ppo_elapsed), 4), "ppo_device": str(self.policy.last_device),
            "ppo_records": int(record_count), "minibatch": self.batch_size,
            "epochs": self.epochs, "ppo_loss": round(float(loss), 6),
            "imitation_loss": round(float(getattr(self.policy, "last_imitation_loss", 0.0)), 6),
            "imitation_coef": round(float(getattr(self.policy, "last_imitation_coef", 0.0)), 5),
            "total_physics_steps": int(batch_steps),
            "rollout_steps_per_sec": round(cycle_throughput(batch_steps, rollout_elapsed, ppo_elapsed)["rollout_steps_per_sec"], 2),
            "end_to_end_steps_per_sec": round(cycle_throughput(batch_steps, rollout_elapsed, ppo_elapsed)["end_to_end_steps_per_sec"], 2),
            # Backward-compatible field: "combined" now correctly means rollout + PPO, not rollout alone.
            "combined_steps_per_sec": round(cycle_throughput(batch_steps, rollout_elapsed, ppo_elapsed)["end_to_end_steps_per_sec"], 2),
            "session_average_steps_per_sec": round(float(self.stats.get("steps_per_sec", 0.0) or 0.0), 2),
            "cpu_buffer_records": int(self._last_hybrid_cpu_records),
            "cpu_buffer_cap": int(self._last_hybrid_buffer_cap),
            "cuda_memory_snapshot": dict(self.stats.get("cuda_memory_snapshot", {}) or {}),
            "gpu_record_slots": int(self.stats.get("gpu_record_slots", 0) or 0),
            "gpu_obs_history_estimate_gib": float(self.stats.get("gpu_obs_history_estimate_gib", 0.0) or 0.0),
            "episodes_total": int(self.stats.get("episodes", 0)),
            "steps_total": int(self.stats.get("steps", 0)),
            "win_rate_20": round(float(self.stats.get("win_rate_20", 0.0)), 4),
            "cycle_reward_total": round(float(self.stats.get("cycle_reward_total", 0.0) or 0.0), 4),
            "cycle_mean_reward": round(float(self.stats.get("cycle_mean_reward", 0.0) or 0.0), 4),
            "mean_reward_20": round(float(self.stats.get("mean_reward_20", 0.0)), 4),
            "blue_damage_20": round(float(self.stats.get("blue_damage_20", 0.0)), 3),
            "red_damage_20": round(float(self.stats.get("red_damage_20", 0.0)), 3),
            "blue_shots_20": round(float(self.stats.get("blue_shots_20", 0.0)), 3),
            "red_shots_20": round(float(self.stats.get("red_shots_20", 0.0)), 3),
            "blue_hit_rate_20": round(float(self.stats.get("blue_hit_rate_20", 0.0)), 4),
            "red_hit_rate_20": round(float(self.stats.get("red_hit_rate_20", 0.0)), 4),
            "blue_kills_20": round(float(self.stats.get("blue_kills_20", 0.0)), 3),
            "red_kills_20": round(float(self.stats.get("red_kills_20", 0.0)), 3),
            "blue_survivors_20": round(float(self.stats.get("blue_survivors_20", 0.0)), 3),
            "red_survivors_20": round(float(self.stats.get("red_survivors_20", 0.0)), 3),
            "policy_fire_rate_20": round(float(self.stats.get("policy_fire_rate_20", 0.0)), 4),
            "teacher_fire_rate_20": round(float(self.stats.get("teacher_fire_rate_20", 0.0)), 4),
            "teacher_fire_pos_weight": round(float(getattr(self.policy, "last_teacher_fire_pos_weight", 1.0)), 3),
        }
        self._stage_history.append(row)
        self._stage_history = self._stage_history[-2:]
        with self.state_lock:
            self.stats["last_two_cycles"] = [dict(x) for x in self._stage_history]
        def append_jsonl(path, payload):
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception:
                pass
        append_jsonl(self.metrics_jsonl_path, row)
        if cpu_eps or cpu_records:
            append_jsonl(self.cpu_rollouts_jsonl_path, {k: v for k, v in row.items()
                if k in ("time", "model_id", "profile", "cpu_workers", "cpu_seconds", "cpu_episodes", "cpu_records", "cpu_physics_steps", "cpu_steps_per_sec", "cpu_wait_gpu_seconds")})
        if gpu_eps or gpu_records:
            append_jsonl(self.gpu_rollouts_jsonl_path, {k: v for k, v in row.items()
                if k in ("time", "model_id", "profile", "gpu_arenas", "gpu_seconds", "gpu_episodes", "gpu_records", "gpu_physics_steps", "gpu_steps_per_sec")})
        append_jsonl(self.ppo_updates_jsonl_path, {k: v for k, v in row.items()
            if k in ("time", "model_id", "profile", "ppo_seconds", "ppo_device", "ppo_records", "minibatch", "epochs", "ppo_loss", "imitation_loss", "imitation_coef")})
        cuda_mem = dict(row.get("cuda_memory_snapshot", {}) or {})
        lines = [
            "GRAVITRUEX — КРАТКИЙ ОТЧЁТ ОБУЧЕНИЯ",
            f"Время: {stamp}",
            f"Модель: {self.model_name} [{self.model_id}] · {self.policy.width}×{self.policy.depth} · {self.policy.count_params():,} параметров",
            f"Противник: {self.opponent_name} · режим цикла: {'только GPU-физика' if self._last_cycle_gpu_only_mode else 'гибрид CPU+GPU' if self._last_cycle_hybrid_mode else 'CPU-физика'}",
            f"Устройства отдельно: физика={row['physics_device']} · действия policy={row['policy_inference_device']} · PPO={row['ppo_update_device']}",
            f"Всего: {row['episodes_total']:,} эпизодов · {row['steps_total']:,} физических шагов · обновлений PPO: {update_no}",
            f"Скорость GPU-rollout: {row['gpu_steps_per_sec']:,.0f} шаг/с только во время сбора; скорость полного цикла (сбор {row['rollout_wall_seconds']:.2f}с + PPO {row['ppo_seconds']:.2f}с) {row['end_to_end_steps_per_sec']:,.0f} шаг/с.",
            f"CPU: {cpu_eps} эп. / {cpu_records:,} записей / {cpu_steps:,} шагов за {cpu_seconds:.2f} с = {cpu_rate:,.0f} шаг/с; процессов {0 if self._last_cycle_gpu_only_mode else (self.cpu_workers_active if self._last_cycle_hybrid_mode else self.workers)}",
            f"GPU: {gpu_eps} эп. / {gpu_records:,} записей / {gpu_steps:,} шагов за {gpu_seconds:.2f} с = {gpu_rate:,.0f} шаг/с; арен фактически {int(self.stats.get('gpu_rollout_envs', 0) or 0) if (self._last_cycle_hybrid_mode or self._last_cycle_gpu_only_mode) else 0}",
            (f"GPU batch: requested {getattr(self, '_gpu_batch_requested_envs', self.gpu_batch_envs)} → actual {int(self.stats.get('gpu_rollout_envs', 0) or 0)}; record budget {self._last_gpu_record_budget:,}; arenas reduced by record cap {int(self.stats.get('gpu_rollout_envs_budget_reduced', 0) or 0)}; {self._last_gpu_oom_fallback}" if self._last_cycle_hybrid_mode or self._last_cycle_gpu_only_mode else ""),
            (f"Hybrid limits: CPU workers {row['cpu_workers']}/{self.workers}; "
             f"RAM available at start {self.hybrid_limits.get('available_ram_gb') if self.hybrid_limits else 'unknown'} GB; "
             f"CPU record cap {self._last_hybrid_buffer_cap:,}; workers low-priority when supported" if self._last_cycle_hybrid_mode else ""),
            (f"Автоограничитель гибрида: {self.auto_rollout_selection_note}" if self.auto_rollout_selection_note else ""),
            f"GPU device timeline: {self._last_gpu_device_seconds:.2f} с; rest (Python dispatch, init, copy/pack): {self._last_gpu_host_overhead_seconds:.2f} с",
            f"CUDA Graph: {'used' if self._last_gpu_graph_used else 'eager'}; requested={self._last_gpu_graph_requested}; {self._last_gpu_graph_error[:200] if self._last_gpu_graph_error else 'no fallback error'}",
            (f"PyTorch CUDA memory: allocated={cuda_mem.get('allocated_gib', 'n/a')} GiB · reserved={cuda_mem.get('reserved_gib', 'n/a')} GiB · peak allocated={cuda_mem.get('peak_allocated_gib', 'n/a')} GiB · device free/total={cuda_mem.get('free_gib', 'n/a')}/{cuda_mem.get('total_gib', 'n/a')} GiB · history slots/arena={row.get('gpu_record_slots', 0)} · observation history estimate={row.get('gpu_obs_history_estimate_gib', 0):.3f} GiB"),
            f"GPU rollout: арен запрошено {int(self.stats.get('gpu_rollout_envs_requested', 0) or 0)} · реально в батче {int(self.stats.get('gpu_rollout_envs', 0) or 0)} · сокращено лимитом записей {int(self.stats.get('gpu_rollout_envs_budget_reduced', 0) or 0)}; горизонт {self._last_gpu_max_decisions if self._last_gpu_max_decisions else 'не использовался'} решений/арену · record budget {self._last_gpu_record_budget:,}",
            f"Ожидание GPU после завершения CPU-сбора: {cpu_wait_seconds:.2f} с; CPU-буфер {self._last_hybrid_cpu_records:,}/{self._last_hybrid_buffer_cap:,} (строгий лимит; отброшено сверх лимита записей: {int(self.stats.get('cpu_rejected_over_cap_records', 0) or 0)})",
            f"Дренаж CPU-воркеров, не перекрытый PPO: {float(getattr(self, '_last_hybrid_worker_drain_seconds', 0.0)):.3f} с",
            f"PPO: записей {record_count:,} · mini-batch {self.batch_size:,} · эпох {self.epochs} · loss {loss:.5f}",
            f"Combat bootstrap: aux-loss {float(getattr(self.policy, 'last_imitation_loss', 0.0)):.4f} · коэффициент {float(getattr(self.policy, 'last_imitation_coef', 0.0)):.4f} (0 к 900 обновлениям) · teacher fire weight {row['teacher_fire_pos_weight']:.1f}" if self.opponent_mode in ("heuristic", "selfplay") else "Imitation loss: отключён для противника-модели",
            f"Действия огня, последние 20: policy {row['policy_fire_rate_20']:.1%} · teacher {row['teacher_fire_rate_20']:.1%}",
            f"Качество, последние 20: победы {row['win_rate_20']:.0%} · reward цикла Σ={row['cycle_reward_total']:+.2f} · средняя/цикл={row['cycle_mean_reward']:+.3f} · средняя 20 эп.={row['mean_reward_20']:+.3f} · убийства B/R {row['blue_kills_20']:.2f}/{row['red_kills_20']:.2f} · выжившие B/R {row['blue_survivors_20']:.2f}/{row['red_survivors_20']:.2f} · выстрелы B/R {row.get('blue_shots_20', 0.0):.1f}/{row.get('red_shots_20', 0.0):.1f} · попадания B/R {row.get('blue_hit_rate_20', 0.0):.0%}/{row.get('red_hit_rate_20', 0.0):.0%} · урон B/R {row.get('blue_damage_20', 0.0):.1f}/{row.get('red_damage_20', 0.0):.1f}",
            "",
            "ДВА ПОСЛЕДНИХ ЦИКЛА:",
        ]
        if self.auto_rollout_selection_note:
            lines.append("Автовыбор rollout: " + self.auto_rollout_selection_note)
        resources = dict(self.stats.get("resource_snapshot", {}) or {})
        if resources:
            lines.append("Ресурсы UI snapshot: " + " · ".join(
                f"{key}={value}" for key, value in resources.items()))
        for i, item in enumerate(reversed(self._stage_history), 1):
            lines.append(f"{i}. {item['time']} | reward Σ {item.get('cycle_reward_total', 0.0):+.2f} · средняя/цикл {item.get('cycle_mean_reward', 0.0):+.3f} · средняя 20 эп. {item.get('mean_reward_20', 0.0):+.3f} | CPU-сбор {item['cpu_seconds']:.2f}с, {item['cpu_episodes']} эп., {item['cpu_steps_per_sec']:,.0f} phys.step/с | GPU-сбор {item['gpu_seconds']:.2f}с, {item['gpu_episodes']} эп., {item['gpu_steps_per_sec']:,.0f} phys.step/с | GPU arenas {item.get('gpu_batch_actual_envs', 0)}/{item.get('gpu_batch_requested_envs', 0)} | PPO {item['ppo_seconds']:.2f}с | wall {item['rollout_wall_seconds'] + item['ppo_seconds']:.2f}с | full {item.get('end_to_end_steps_per_sec', item['combined_steps_per_sec']):,.0f} phys.step/с")
        lines += ["", f"Компактный диагностический TXT на рабочем столе: {resolve_desktop_diagnostic_path()}",
                  "Подробные JSONL остаются рядом с моделью; для первичного разбора достаточно компактного TXT."]
        try:
            tmp = self.report_path.with_suffix(".txt.tmp")
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(tmp, self.report_path)
        except Exception:
            pass

    def _save_stats(self) -> None:
        try:
            with self.state_lock:
                data = dict(self.stats)
            tmp = self.stats_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.stats_path)
            write_model_meta(self.model_id, {
                "name": self.model_name, "width": self.policy.width, "depth": self.policy.depth,
                "parameters": self.policy.count_params(),
                "episodes": int(data.get("episodes", 0)), "steps": int(data.get("steps", 0)),
                "trained": bool(read_model_meta(self.model_id).get("trained", False) or int(data.get("episodes", 0)) > 0),
                "last_opponent": str(data.get("opponent_name", self.opponent_name)), "last_opponent_mode": self.opponent_mode,
                "last_device": str(data.get("device", self.device_mode)),
                "last_message": str(data.get("message", "")),
            })
        except Exception:
            pass


# ----------------------------- Performance monitor -----------------------------
class PerformanceMonitor:
    """Low-overhead background sampler. GPU telemetry is optional via nvidia-smi."""
    def __init__(self):
        self._lock = threading.Lock()
        self._psutil = None
        self._process = None
        try:
            import psutil  # optional, listed in requirements.txt
            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
        except Exception:
            pass
        self._values = {
            "cpu": None, "app_cpu": None, "ram_used": None, "ram_total": None,
            "ram_percent": None, "process_ram": None, "gpu_util": None, "gpu_mem_util": None,
            "vram_used": None, "vram_total": None, "gpu_temp": None,
            "gpu_power_w": None, "gpu_clock_mhz": None, "sampled_at": 0.0,
        }
        self._stop = threading.Event()
        self._gpu_exe = shutil.which("nvidia-smi")
        if not self._gpu_exe:
            candidates = [
                Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
                Path(r"C:\Windows\System32\nvidia-smi.exe"),
            ]
            self._gpu_exe = next((str(c) for c in candidates if c.exists()), None)
        self._thread = threading.Thread(target=self._run, name="GravitrueX-PerfMonitor", daemon=True)
        self._thread.start()

    def _sample_gpu(self):
        if not self._gpu_exe:
            return None
        try:
            result = subprocess.run(
                [self._gpu_exe, "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.2, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None
            raw = [x.strip() for x in result.stdout.strip().splitlines()[0].split(",")]
            if len(raw) < 5:
                return None
            def optional_num(i):
                try:
                    v = raw[i].strip()
                    return None if not v or v.upper() == "N/A" else float(v)
                except Exception:
                    return None
            return {
                "gpu_util": optional_num(0), "gpu_mem_util": optional_num(1),
                "vram_used": optional_num(2), "vram_total": optional_num(3),
                "gpu_temp": optional_num(4), "gpu_power_w": optional_num(5),
                "gpu_clock_mhz": optional_num(6),
            }
        except Exception:
            return None

    def _run(self):
        # Sample the NVIDIA hardware counters twice per second. These are SM/GPU
        # and memory-controller metrics from nvidia-smi; Task Manager can show a
        # different engine graph (e.g. Copy/3D/Compute_0), so the values are labelled.
        next_gpu = 0.0
        while not self._stop.is_set():
            update = {"sampled_at": time.time()}
            if self._psutil is not None:
                try:
                    vm = self._psutil.virtual_memory()
                    app_procs = [self._process] if self._process else []
                    if self._process:
                        try:
                            app_procs.extend(self._process.children(recursive=True))
                        except Exception:
                            pass
                    app_cpu_raw = 0.0
                    app_ram = 0.0
                    for proc in app_procs:
                        try:
                            app_cpu_raw += proc.cpu_percent(interval=None)
                            app_ram += proc.memory_info().rss
                        except Exception:
                            continue
                    logical_cpus = max(1, self._psutil.cpu_count(logical=True) or 1)
                    update.update({
                        "cpu": float(self._psutil.cpu_percent(interval=None)),
                        "app_cpu": float(app_cpu_raw / logical_cpus),
                        "ram_used": float(vm.used), "ram_total": float(vm.total),
                        "ram_percent": float(vm.percent),
                        "process_ram": float(app_ram) if app_procs else None,
                    })
                except Exception:
                    pass
            now = time.monotonic()
            if now >= next_gpu:
                gpu = self._sample_gpu()
                if gpu:
                    update.update(gpu)
                next_gpu = now + 0.5
            with self._lock:
                # Preserve last GPU values if nvidia-smi did not respond on this sample.
                for key, value in update.items():
                    if value is not None:
                        self._values[key] = value
            self._stop.wait(0.5)

    def snapshot(self):
        with self._lock:
            return dict(self._values)

    def close(self):
        self._stop.set()


def _fmt_bytes(value):
    if value is None:
        return "н/д"
    value = float(value)
    if value >= 1024 ** 3:
        return f"{value / (1024 ** 3):.1f} ГБ"
    return f"{value / (1024 ** 2):.0f} МБ"


class ToolTip:
    """Small dark hover tooltip for Tk widgets; no external dependency."""
    def __init__(self, widget, text: str, delay_ms: int = 450):
        self.widget = widget
        self.text = str(text or "").strip()
        self.delay_ms = int(delay_ms)
        self._after_id = None
        self._window = None
        if self.text:
            widget.bind("<Enter>", self._schedule, add="+")
            widget.bind("<Leave>", self._hide, add="+")
            widget.bind("<ButtonPress>", self._hide, add="+")
            widget.bind("<Destroy>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel_timer()
        try:
            self._after_id = self.widget.after(self.delay_ms, self._show)
        except Exception:
            self._after_id = None

    def _cancel_timer(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        self._after_id = None
        if self._window is not None or not self.text:
            return
        try:
            import tkinter as tk
            if not self.widget.winfo_exists() or not self.widget.winfo_ismapped():
                return
            window = tk.Toplevel(self.widget)
            window.wm_overrideredirect(True)
            window.attributes("-topmost", True)
            label = tk.Label(
                window, text=self.text, justify="left", anchor="w",
                bg="#172b3e", fg="#e5eef8", relief="solid", borderwidth=1,
                font=("Segoe UI", 9), padx=9, pady=7, wraplength=340,
            )
            label.pack()
            window.update_idletasks()
            x = self.widget.winfo_rootx() + min(18, max(0, self.widget.winfo_width() // 4))
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
            screen_w = window.winfo_screenwidth()
            screen_h = window.winfo_screenheight()
            w, h = window.winfo_reqwidth(), window.winfo_reqheight()
            x = max(4, min(x, screen_w - w - 4))
            if y + h > screen_h - 4:
                y = max(4, self.widget.winfo_rooty() - h - 5)
            window.geometry(f"+{x}+{y}")
            self._window = window
        except Exception:
            self._window = None

    def _hide(self, _event=None):
        self._cancel_timer()
        if self._window is not None:
            try:
                self._window.destroy()
            except Exception:
                pass
            self._window = None


# ---------------------------------- Tk UI -------------------------------------
class GravitrueXApp:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk, messagebox
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.root = root
        root.title("GravitrueX — Neural Arena")
        root.configure(bg=BG)
        try:
            screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
            default_w = max(1000, min(1660, screen_w - 70))
            default_h = max(700, min(920, screen_h - 130))
            root.geometry(f"{default_w}x{default_h}+24+24")
        except Exception:
            root.geometry("1260x820")
        root.minsize(1000, 650)
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        ensure_model_library()
        ensure_builtin_models()
        self.model_records = list_model_records()
        self.model_display_to_id: Dict[str, str] = {}
        self.model_id_to_display: Dict[str, str] = {}
        for record in self.model_records:
            mid = str(record.get("id", DEFAULT_MODEL_ID))
            display = f"{record.get('name', mid)} [{mid}]"
            self.model_display_to_id[display] = mid
            self.model_id_to_display[mid] = display
        main_weights = model_weights_path(DEFAULT_MODEL_ID)
        saved_weights_path = main_weights if main_weights.exists() else WEIGHTS_PATH
        saved_width, saved_depth = checkpoint_architecture(saved_weights_path)
        saved_network_type = checkpoint_network_type(saved_weights_path)
        self.policy = ActorCritic(seed=42, width=saved_width, depth=saved_depth,
                                  device_mode="cpu" if FORCE_CPU else "auto",
                                  architecture=saved_network_type)
        self.policy.allow_gpu_inference = True
        self.policy.require_gpu_inference = True
        self.policy.inference_mode = "gpu"
        loaded = self.policy.load(main_weights if main_weights.exists() else WEIGHTS_PATH)
        if not main_weights.exists():
            self.policy.save(main_weights)
        self.policy_loaded = loaded
        self.active_model_id = DEFAULT_MODEL_ID
        self.policy_cache: Dict[str, ActorCritic] = {DEFAULT_MODEL_ID: self.policy}
        if not read_model_meta(DEFAULT_MODEL_ID):
            write_model_meta(DEFAULT_MODEL_ID, {"name": "Основная модель", "width": saved_width, "depth": saved_depth,
                                                 "architecture": self.policy.architecture,
                                                 "default_reward_mode": "baseline", "default_priors_mode": "full",
                                                 "parameters": self.policy.count_params(), "episodes": 0, "steps": 0,
                                                 "trained": bool(loaded), "description": "Основная политика GravitrueX"})
        main_meta = read_model_meta(DEFAULT_MODEL_ID)
        if "architecture" not in main_meta or "default_priors_mode" not in main_meta:
            write_model_meta(DEFAULT_MODEL_ID, {
                "architecture": self.policy.architecture,
                "default_reward_mode": main_meta.get("default_reward_mode", "baseline"),
                "default_priors_mode": main_meta.get("default_priors_mode", "full"),
            })
            main_meta = read_model_meta(DEFAULT_MODEL_ID)
        self.policy_loaded = bool(loaded and main_meta.get("trained", int(main_meta.get("episodes", 0) or 0) > 0))
        self.arena = Arena(seed=90210)
        self.camera_x = WORLD_W * 0.5
        self.camera_y = WORLD_H * 0.5
        self.view_zoom = 1.0
        self.blue_n = tk.IntVar(value=1)
        self.red_n = tk.IntVar(value=1)
        self.random_compositions = tk.BooleanVar(value=False)
        main_display = self.model_id_to_display.get(DEFAULT_MODEL_ID, "Основная модель [main]")
        self.active_model_var = tk.StringVar(value=main_display)
        self.blue_ai = tk.StringVar(value=main_display)
        self.red_ai = tk.StringVar(value=HEURISTIC_LABEL)
        self.training_opponent = tk.StringVar(value=HEURISTIC_LABEL)
        self.new_model_name_var = tk.StringVar(value="Эксперимент")
        self.model_detail_var = tk.StringVar(value="Модель: загружаю сведения…")
        self.training_target_var = tk.StringVar(value=f"Обучается: {main_display}")
        self.match_blue_ai = main_display
        self.match_red_ai = HEURISTIC_LABEL
        self.train_workers = tk.IntVar(value=max(1, min(6, (os.cpu_count() or 4) // 2)))
        self.train_workers_text = tk.StringVar(value=str(self.train_workers.get()))
        # GPU arenas are vectorized environments in one tensor batch, not CUDA streams.
        self.train_gpu_envs = tk.IntVar(value=4096)
        self.train_gpu_envs_text = tk.StringVar(value=str(self.train_gpu_envs.get()))
        # Training composition is intentionally separate from the visible battle roster.
        self.train_blue_n = tk.IntVar(value=1)
        self.train_red_n = tk.IntVar(value=1)
        self.train_record_budget = tk.StringVar(value="720 000")
        profile_name = next((name for name, dims in ARCH_PROFILES.items() if dims == (saved_width, saved_depth)), "Тактическая 1536×4 (~7,3 млн)")
        self.model_profile = tk.StringVar(value=profile_name)
        self.training_reward_mode = tk.StringVar(value="По профилю модели")
        self.training_priors_mode = tk.StringVar(value="По профилю модели")
        self.training_desired_range = tk.StringVar(value="480")
        self.train_horizon = tk.StringVar(value="2700 (180 с)")
        self._visible_test_horizon = None
        self._test_battle_active = False
        self._test_battle_start_step = 0
        self.training_device = tk.StringVar(value="CPU только" if FORCE_CPU else "Авто: CUDA если доступна")
        self.train_batch = tk.StringVar(value="4096")
        self.train_epochs = tk.StringVar(value="3")
        self.game_inference_mode = tk.StringVar(value="GPU/CUDA (единственный режим в бою)")
        self.training_rollout_mode = tk.StringVar(value=("CPU: параллельные арены" if FORCE_CPU else "GPU: параллельные арены (CUDA-физика)"))
        self.mode = "play"
        self.speed = 1.0
        self.speed_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        self.paused = False
        self.accumulator = 0.0
        self.last_wall = time.perf_counter()
        self.perf = PerformanceMonitor()
        self._fps_frames = 0
        self._fps_window_start = time.perf_counter()
        self.display_fps = 0.0
        self._last_stats_update = 0.0
        self._last_resource_log_time = 0.0
        self._last_paused_hud_refresh = 0.0
        self._scene_dirty = True
        self._background_dirty = True
        self._background_size = (0, 0)
        self.game_steps_per_sec = 0.0
        self.ticks_last_frame = 0
        self._physics_window_start = time.perf_counter()
        self._physics_steps_window_start = 0
        self.trainer: Optional[Trainer] = None
        self._pending_battle_test = False
        self.training_notice = ""
        self.episode_label = tk.StringVar(value="Матч не начат")
        self.speed_label = tk.StringVar(value="Скорость симуляции: 1×")
        self.model_label = tk.StringVar(value=f"Модель: {'веса загружены' if loaded else 'новая, ещё не обучена'}")
        self.mode_label = tk.StringVar(value="РЕЖИМ: ИГРА")
        self.stats_labels: Dict[str, object] = {}
        self._tooltips: List[ToolTip] = []
        self.desktop_diagnostic_path = resolve_desktop_diagnostic_path()
        self._last_desktop_diagnostic_write = 0.0
        self._desktop_diagnostic_write_error = ""
        self._make_ui()
        config_loaded = self._load_settings(quiet=True, startup=True)
        # Resolve Tk's deferred pack/layout calculations before the first canvas
        # frame. Avoid redrawing the scene against a transient 1×1 canvas during startup.
        try:
            self.root.update_idletasks()
        except Exception:
            pass
        self._new_match()
        if not config_loaded and not APP_CONFIG_PATH.exists():
            # Produce a starter config only on first launch. Never silently overwrite
            # an existing config that is malformed or from a newer application version.
            self._save_settings(quiet=True)
        self._update_controls()
        self._draw()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(16, self._loop)

    def _display_for_model_id(self, model_id: str) -> str:
        if model_id not in self.model_id_to_display:
            meta = read_model_meta(model_id)
            name = meta.get("name", model_id)
            display = f"{name} [{model_id}]"
            self.model_id_to_display[model_id] = display
            self.model_display_to_id[display] = model_id
        return self.model_id_to_display[model_id]

    def _model_id_for_display(self, display: str) -> Optional[str]:
        if display == HEURISTIC_LABEL or display == SELFPLAY_LABEL:
            return None
        return self.model_display_to_id.get(display)

    def _active_weights_path(self) -> Path:
        return model_weights_path(self.active_model_id)

    def _active_stats_path(self) -> Path:
        return model_stats_path(self.active_model_id)

    def _refresh_model_catalog(self) -> None:
        self.model_records = list_model_records()
        self.model_display_to_id = {}
        self.model_id_to_display = {}
        for record in self.model_records:
            mid = str(record.get("id", DEFAULT_MODEL_ID))
            display = f"{record.get('name', mid)} [{mid}]"
            self.model_display_to_id[display] = mid
            self.model_id_to_display[mid] = display
        displays = [self._display_for_model_id(str(r.get("id", DEFAULT_MODEL_ID))) for r in self.model_records]
        for attr, values in (("active_model_combo", displays),
                              ("blue_ai_combo", [HEURISTIC_LABEL] + displays),
                              ("red_ai_combo", [HEURISTIC_LABEL] + displays),
                              ("training_opponent_combo", [HEURISTIC_LABEL, SELFPLAY_LABEL] + displays)):
            combo = getattr(self, attr, None)
            if combo is not None:
                combo.configure(values=values)
        for var_name, fallback in (("active_model_var", self._display_for_model_id(DEFAULT_MODEL_ID)),
                                   ("blue_ai", self._display_for_model_id(DEFAULT_MODEL_ID)),
                                   ("red_ai", HEURISTIC_LABEL),
                                   ("training_opponent", HEURISTIC_LABEL)):
            var = getattr(self, var_name, None)
            if var is not None and var.get() not in (displays if var_name in ("active_model_var", "blue_ai") else ([HEURISTIC_LABEL] + displays if var_name == "red_ai" else [HEURISTIC_LABEL, SELFPLAY_LABEL] + displays)):
                var.set(fallback)

    def _policy_for_model_id(self, model_id: str) -> ActorCritic:
        """Load/cached policy for a named saved model. Active policy stays shared with trainer UI."""
        if model_id in self.policy_cache:
            return self.policy_cache[model_id]
        path = model_weights_path(model_id)
        width, depth = checkpoint_architecture(path)
        meta = read_model_meta(model_id)
        if not path.exists():
            width = int(meta.get("width", width) or width)
            depth = int(meta.get("depth", depth) or depth)
        architecture = (checkpoint_network_type(path) if path.exists()
                        else str(meta.get("architecture") or "shared_residual"))
        if architecture not in {"shared_residual", "separate_critic"}:
            architecture = "shared_residual"
        policy = ActorCritic(seed=abs(hash(model_id)) % 2_000_000_000, width=width, depth=depth,
                             device_mode="cpu" if FORCE_CPU else "auto", architecture=architecture)
        loaded = policy.load(path) if path.exists() else False
        policy.allow_gpu_inference = True
        policy.require_gpu_inference = True
        policy.inference_mode = "gpu"
        if not path.exists():
            policy.save(path)
        self.policy_cache[model_id] = policy
        return policy

    def _active_model_changed(self, _event=None):
        global ACTION_PRIORS_MODE
        if self.trainer and self.trainer.is_alive():
            self.messagebox.showwarning("Обучение запущено", "Нельзя переключать обучаемую модель во время PPO. Останови обучение и выбери модель после сохранения чекпоинта.")
            self.active_model_var.set(self._display_for_model_id(self.active_model_id))
            return
        new_id = self._model_id_for_display(self.active_model_var.get()) or DEFAULT_MODEL_ID
        self.active_model_id = new_id
        self.policy = self._policy_for_model_id(new_id)
        active_meta = read_model_meta(new_id)
        if hasattr(self, "training_priors_mode") and self.training_priors_mode.get() == "По профилю модели":
            profiled_priors = str(active_meta.get("default_priors_mode", "full" if new_id == DEFAULT_MODEL_ID else "soft"))
            ACTION_PRIORS_MODE = profiled_priors if profiled_priors in {"full", "soft", "off"} else "soft"
        self.policy_loaded = bool(model_weights_path(new_id).exists() and active_meta.get("trained", int(active_meta.get("episodes", 0) or 0) > 0))
        self.model_profile.set(next((name for name, dims in ARCH_PROFILES.items()
                                     if dims == (self.policy.width, self.policy.depth)), self.model_profile.get()))
        self._apply_game_inference_mode()
        meta = read_model_meta(new_id)
        self.model_label.set(f"Модель: {meta.get('name', new_id)} · {self.policy.width}×{self.policy.depth} · {network_architecture_label(self.policy.architecture)}")
        self._update_model_details()
        self._update_composition_note()
        self._update_controls()

    def _update_model_details(self):
        if not hasattr(self, "model_detail_var"):
            return
        mid = self.active_model_id
        meta = read_model_meta(mid)
        stats = {}
        try:
            with open(model_stats_path(mid), "r", encoding="utf-8") as f:
                stats = json.load(f)
        except Exception:
            pass
        history_known = bool(meta.get("training_history_known", True))
        episodes = int(stats.get("episodes", meta.get("episodes", 0)) or 0)
        steps = int(stats.get("steps", meta.get("steps", 0)) or 0)
        opp = stats.get("opponent_name", meta.get("last_opponent", "ещё не обучалась"))
        training_now = bool(self.trainer and self.trainer.is_alive() and getattr(self.trainer, "model_id", None) == mid)
        if training_now:
            live = self.trainer.get_stats()
            episodes = int(live.get("episodes", episodes) or 0)
            steps = int(live.get("steps", steps) or 0)
            opp = live.get("opponent_name", opp)
            phase = live.get("phase", "обучение")
            status = f"СЕЙЧАС ОБУЧАЕТСЯ · {phase}"
        elif not model_weights_path(mid).exists():
            status = "нет чекпоинта"
        elif not history_known:
            status = "веса импортированы · история обучения неизвестна"
        elif meta.get("trained", episodes > 0):
            status = "обученная/импортированная политика готова к использованию"
        else:
            status = "чекпоинт есть · обучение ещё не подтверждено"
        parent = meta.get("parent_model")
        if parent:
            inherited_known = bool(meta.get("parent_training_history_known", True))
            if inherited_known:
                origin = f"Копия: {parent} · унаследовано {int(meta.get('inherited_episodes', 0) or 0):,} эпизодов"
            else:
                origin = f"Копия: {parent} · история родительских весов неизвестна"
        elif meta.get("source_archive"):
            origin = f"Импорт из {meta.get('source_archive')}"
        else:
            origin = "Происхождение: самостоятельная модель"
        if history_known:
            train_line = f"Эпизоды этой версии: {episodes:,} · физ. шаги: {steps:,}"
            inherited = int(meta.get("inherited_episodes", 0) or 0)
            if inherited:
                train_line += f" · унаследовано: {inherited:,}"
            result_line = (f"Последние 20: W/L/D {int(stats.get('wins', 0) or 0)}/"
                           f"{int(stats.get('losses', 0) or 0)}/{int(stats.get('draws', 0) or 0)} · "
                           f"очки/матч (победа=1, ничья=0,5): {float(stats.get('win_rate_20', 0.0) or 0.0):.0%} · "
                           f"средняя награда {float(stats.get('mean_reward_20', 0.0) or 0.0):+.3f}")
        else:
            train_line = "История обучения: неизвестна (не была в исходном архиве)"
            result_line = "Статистика побед/награды: отсутствует"
        if training_now:
            live = self.trainer.get_stats()
            result_line = (f"Награда {float(live.get('mean_reward_20', 0.0) or 0.0):+.3f} · "
                           f"убийства синих/красных {float(live.get('blue_kills_20', 0.0) or 0.0):.2f}/"
                           f"{float(live.get('red_kills_20', 0.0) or 0.0):.2f} · "
                           f"выжившие {float(live.get('blue_survivors_20', 0.0) or 0.0):.2f}/"
                           f"{float(live.get('red_survivors_20', 0.0) or 0.0):.2f}")
        self.model_detail_var.set(
            f"Активная: {meta.get('name', mid)} [{mid}]\n"
            f"Профиль: {self.policy.width}×{self.policy.depth} · {network_architecture_label(self.policy.architecture)} · {self.policy.count_params():,} параметров\n"
            f"{train_line}\n{result_line}\n"
            f"{origin}\n"
            f"Последний/текущий противник: {opp}\n{status}")
        episodes_label = f"{episodes:,}" if history_known else "история неизвестна"
        self.training_target_var.set(f"Цель: {meta.get('name', mid)} [{mid}] · {episodes_label} эпизодов" + (" · В РАБОТЕ" if training_now else ""))

    def _create_model_variant(self, fresh: bool = False):
        if self.trainer and self.trainer.is_alive():
            self.messagebox.showwarning("Обучение запущено", "Останови обучение перед созданием варианта.")
            return
        name = self.new_model_name_var.get().strip()
        if not name:
            name = "Эксперимент"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"model_{transliterate_model_slug(name)}_{stamp}"
        folder = model_dir_for(model_id)
        suffix = 1
        while folder.exists():
            model_id = f"model_{transliterate_model_slug(name)}_{stamp}_{suffix}"
            folder = model_dir_for(model_id)
            suffix += 1
        folder.mkdir(parents=True, exist_ok=False)
        width, depth = ARCH_PROFILES.get(self.model_profile.get(), (self.policy.width, self.policy.depth))
        variant = ActorCritic(seed=random.randint(1, 2**31 - 1), width=width, depth=depth,
                              device_mode="cpu" if FORCE_CPU else "auto", architecture=self.policy.architecture)
        if not fresh:
            # Keep every shape-compatible parameter so a variant starts from the selected model.
            for key, target in variant.params.items():
                source = self.policy.params.get(key)
                if source is not None and source.shape == target.shape:
                    target[...] = source
        path = model_weights_path(model_id)
        variant.save(path)
        _atomic_json(model_stats_path(model_id), {
            "episodes": 0, "steps": 0, "wins": 0, "losses": 0, "draws": 0,
            "win_rate_20": 0.0, "mean_reward_20": 0.0, "last_loss": 0.0,
            "message": "Создан новый вариант; обучение ещё не начато.",
            "opponent_mode": "—", "width": width, "depth": depth,
        })
        parent_meta = read_model_meta(self.active_model_id)
        inherited_trained = bool(not fresh and (self.policy_loaded or parent_meta.get("trained", False)))
        write_model_meta(model_id, {
            "name": name, "width": width, "depth": depth, "architecture": variant.architecture,
            "parameters": variant.count_params(),
            "episodes": 0, "steps": 0, "trained": inherited_trained,
            "parent_model": None if fresh else self.active_model_id,
            "parent_training_history_known": bool(parent_meta.get("training_history_known", True)) if not fresh else True,
            "training_history_known": True,
            "inherited_episodes": int(parent_meta.get("episodes", 0) or 0) if not fresh and bool(parent_meta.get("training_history_known", True)) else 0,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "description": "Новая случайная инициализация" if fresh else f"Копия модели {self.active_model_id}",
        })
        self.policy_cache[model_id] = variant
        self._refresh_model_catalog()
        display = self._display_for_model_id(model_id)
        self.active_model_var.set(display)
        self._active_model_changed()
        # Offer the new variant as default blue policy but do not mutate an ongoing match.
        self.blue_ai.set(display)
        if hasattr(self, "blue_ai_combo"):
            self.blue_ai_combo.set(display)
        self._update_model_details()
        self.messagebox.showinfo("Модель создана",
            f"Создан вариант «{name}».\nФайл весов: {path}\n" +
            ("Скопированы совпадающие веса активной модели." if not fresh else "Веса инициализированы случайно; это новая модель без обучения."))

    def _switch_ui_tab(self, name: str) -> None:
        """Switch the global application context (battle or training)."""
        name = "training" if name == "training" else "battle"
        self.ui_tab = name
        if name == "training" and self.mode == "play" and not self.paused and not self.arena.round_over:
            # Switching away from the live battlefield pauses the visible match.
            # The training dashboard is not an invisible source of background CPU load.
            self.paused = True
            self.accumulator = 0.0
            self.mode_label.set("РЕЖИМ: ИГРА / ПАУЗА")
        for page in (getattr(self, "_battle_page", None), getattr(self, "_training_page", None)):
            if page is not None:
                page.pack_forget()
        # Training becomes a full-window workspace: the battlefield is hidden and
        # the former sidebar expands into three readable columns. Battle retains
        # the compact controls next to its live arena.
        left = getattr(self, "_left_host", None)
        right_host = getattr(self, "_right_host", None)
        if name == "training":
            if left is not None:
                left.pack_forget()
            if right_host is not None:
                right_host.pack_forget()
                right_host.configure(width=0)
                right_host.pack_propagate(True)
                right_host.pack(side="top", fill="both", expand=True, padx=0, pady=0)
        else:
            if right_host is not None:
                right_host.pack_forget()
                right_host.configure(width=285)
                right_host.pack_propagate(False)
            if left is not None:
                left.pack_forget()
                left.pack(side="left", fill="both", expand=True)
            if right_host is not None:
                right_host.pack(side="right", fill="y", padx=(12, 0))
        page = self._training_page if name == "training" else self._battle_page
        page.pack(fill="both", expand=True)
        if hasattr(self, "main_tab_battle") and hasattr(self, "main_tab_training"):
            active_bg, active_fg = "#214b58", "#8be0ca"
            idle_bg, idle_fg = "#13263a", "#9fb8cc"
            self.main_tab_battle.configure(bg=active_bg if name == "battle" else idle_bg,
                                           fg=active_fg if name == "battle" else idle_fg)
            self.main_tab_training.configure(bg=active_bg if name == "training" else idle_bg,
                                             fg=active_fg if name == "training" else idle_fg)
        self._scene_dirty = True
        self._background_dirty = True
        if hasattr(self, "play_button"):
            self._update_controls()
        try:
            self.root.after_idle(self._draw)
        except Exception:
            pass

    def _make_ui(self) -> None:
        tk, ttk = self.tk, self.ttk
        root = self.root
        try:
            style = ttk.Style(root)
            style.theme_use("clam")
            style.configure("Arena.TCombobox", fieldbackground="#14283a", background="#1b3246",
                            foreground=TEXT, arrowcolor=ACCENT, bordercolor="#36516a",
                            lightcolor="#36516a", darkcolor="#0c1929", padding=4)
            style.map("Arena.TCombobox", fieldbackground=[("readonly", "#14283a"), ("disabled", "#0c1929")],
                      foreground=[("readonly", TEXT), ("disabled", MUTED)],
                      selectbackground=[("readonly", "#214b58")],
                      selectforeground=[("readonly", TEXT)])
        except Exception:
            pass
        header = tk.Frame(root, bg=BG)
        header.pack(fill="x", padx=12, pady=(9, 5))
        tk.Label(header, text="GRAVITRUEX", bg=BG, fg=TEXT,
                 font=("Segoe UI", 17, "bold"), anchor="w").pack(side="left", padx=(2, 16))
        header_tabs = tk.Frame(header, bg=BG)
        header_tabs.pack(side="right")
        self.main_tab_battle = tk.Button(header_tabs, text="⚔  БОЙ",
            command=lambda: self._switch_ui_tab("battle"), relief="flat", bd=0,
            font=("Segoe UI", 10, "bold"), padx=18, pady=8, cursor="hand2",
            bg="#214b58", fg="#8be0ca", activebackground="#2b5d6b", activeforeground="#ffffff")
        self.main_tab_battle.pack(side="left", padx=(0, 4))
        self.main_tab_training = tk.Button(header_tabs, text="🧠  ОБУЧЕНИЕ",
            command=lambda: self._switch_ui_tab("training"), relief="flat", bd=0,
            font=("Segoe UI", 10, "bold"), padx=18, pady=8, cursor="hand2",
            bg="#13263a", fg="#9fb8cc", activebackground="#2b5d6b", activeforeground="#ffffff")
        self.main_tab_training.pack(side="left")
        outer = tk.Frame(root, bg=BG)
        outer.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        left = tk.Frame(outer, bg=BG)
        # Two task-focused tabs. Each tab owns its own scroll area, so battle controls
        # and training diagnostics never compete for space in one universal panel.
        right_host = tk.Frame(outer, bg=BG, width=285)
        right_host.pack(side="right", fill="y", padx=(12, 0))
        left.pack(side="left", fill="both", expand=True)
        self.control_tab_container = tk.Frame(right_host, bg="#0c1929")
        self.control_tab_container.pack(fill="both", expand=True)

        def make_scroll_tab():
            page = tk.Frame(self.control_tab_container, bg="#0c1929")
            holder = tk.Frame(page, bg="#0c1929")
            holder.pack(fill="both", expand=True)
            tab_canvas = tk.Canvas(holder, bg="#0c1929", highlightthickness=0, bd=0, width=265)
            tab_scroll = tk.Scrollbar(holder, orient="vertical", command=tab_canvas.yview,
                                      bg="#21384d", troughcolor="#0c1929",
                                      activebackground="#3c617c", relief="flat", bd=0,
                                      highlightthickness=0)
            tab_canvas.pack(side="left", fill="both", expand=True)
            tab_scroll.pack(side="right", fill="y")
            tab_frame = tk.Frame(tab_canvas, bg="#0c1929", width=258,
                                 highlightbackground="#1d344b", highlightthickness=1)
            tab_window = tab_canvas.create_window((0, 0), window=tab_frame, anchor="nw")
            tab_canvas.configure(yscrollcommand=tab_scroll.set)
            tab_frame.bind("<Configure>", lambda _e, cv=tab_canvas: cv.configure(scrollregion=cv.bbox("all")))
            tab_canvas.bind("<Configure>", lambda e, cv=tab_canvas, win=tab_window: cv.itemconfigure(win, width=e.width))
            tab_canvas.bind("<MouseWheel>", lambda e, cv=tab_canvas: cv.yview_scroll(-1 if e.delta > 0 else 1, "units"))
            return page, tab_frame

        battle_page, battle_right = make_scroll_tab()
        training_page, training_right = make_scroll_tab()
        self._battle_page = battle_page
        self._training_page = training_page
        self.battle_right = battle_right
        self.training_right = training_right
        # Full-width training workspace. Controls use two columns; the rightmost
        # column is a pinned-at-top dashboard with large KPIs and the last two cycles.
        self.training_main_row = tk.Frame(training_right, bg="#0c1929")
        self.training_main_row.pack(fill="both", expand=True, anchor="n")
        for _col_i, _weight in enumerate((10, 10, 12)):
            self.training_main_row.grid_columnconfigure(_col_i, weight=_weight, uniform="training_main")
        self.training_main_row.grid_rowconfigure(0, weight=1)
        self.training_columns = []
        for _column_index in range(3):
            _column = tk.Frame(self.training_main_row, bg="#0c1929")
            _column.grid(row=0, column=_column_index, sticky="nsew", padx=(0, 10 if _column_index < 2 else 0))
            self.training_columns.append(_column)
        # Existing detailed performance/results sections append below this headline dashboard.
        self.training_summary = self.training_columns[2]
        self.training_summary_columns = [self.training_columns[2], self.training_columns[2]]
        self.training_kpi_vars = {}
        self._section(self.training_summary, "Сводка обучения · последние результаты")
        kpi_specs = [
            ("updates", "PPO ОБНОВЛЕНИЯ / ЭПИЗОДЫ", "0 обновлений · 0 эп."),
            ("steps", "СКОРОСТЬ / ШАГИ", "— шаг/с · 0 шагов"),
            ("reward", "НАГРАДА / СРЕДНЯЯ", "Σ reward — · avg/cycle — · avg20 — · loss —"),
            ("win", "ПОБЕДЫ · ПОСЛЕДНИЕ 20", "—% · W/D/L —/—/—"),
            ("fire", "СТРЕЛЬБА / ПОПАДАНИЯ", "— выстр./бой · попадания —"),
            ("kills", "УБИЙСТВА / УРОН", "— убийств/бой · урон —"),
        ]
        kpi_grid = tk.Frame(self.training_summary, bg="#0c1929")
        kpi_grid.pack(fill="x", padx=7, pady=(2, 8))
        kpi_grid.grid_columnconfigure(0, weight=1, uniform="kpi")
        kpi_grid.grid_columnconfigure(1, weight=1, uniform="kpi")
        for idx, (key, title, default) in enumerate(kpi_specs):
            card = tk.Frame(kpi_grid, bg="#14283a", highlightbackground="#27435a", highlightthickness=1)
            card.grid(row=idx // 2, column=idx % 2, sticky="nsew", padx=3, pady=3)
            tk.Label(card, text=title, bg="#14283a", fg="#90aeca",
                     font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x", padx=9, pady=(7, 1))
            value_var = tk.StringVar(value=default)
            self.training_kpi_vars[key] = value_var
            tk.Label(card, textvariable=value_var, bg="#14283a", fg="#8be0ca",
                     font=("Consolas", 11, "bold"), anchor="w", justify="left",
                     wraplength=330).pack(fill="x", padx=9, pady=(1, 8))
        self._section(self.training_summary, "Последние два цикла")
        self.training_cycle_vars = [tk.StringVar(value="Ожидание первого PPO-обновления") for _ in range(2)]
        self.training_cycle_labels = []
        for i, var in enumerate(self.training_cycle_vars):
            label = tk.Label(self.training_summary, textvariable=var, bg="#102235",
                             fg="#a9c6dc" if i == 0 else "#8be0ca",
                             font=("Consolas", 9, "bold"), anchor="w", justify="left",
                             wraplength=520, padx=9, pady=7, relief="flat")
            label.pack(fill="x", padx=10, pady=2)
            self.training_cycle_labels.append(label)
        self._section(self.training_summary, "Статус")
        self.training_dashboard_status = tk.StringVar(value="Обучение ещё не запущено")
        tk.Label(self.training_summary, textvariable=self.training_dashboard_status, bg="#0c1929",
                 fg="#ffc777", font=("Segoe UI", 9, "bold"), justify="left", anchor="w",
                 wraplength=520).pack(fill="x", padx=12, pady=(0, 8))
        self._outer = outer
        self._left_host = left
        self._right_host = right_host
        right_host.pack_propagate(False)
        self.ui_tab = "battle"

        # Keep the canvas self-explanatory: no redundant tooltips about what an arena is.
        self.canvas = tk.Canvas(left, width=900, height=680, bg=BG, highlightthickness=1,
                                highlightbackground="#20364c")
        self.canvas.pack(fill="both", expand=True)
        foot = tk.Label(left, text="СИНИЕ — команда 1     •     КРАСНЫЕ — команда 2",
                        bg=BG, fg=MUTED, font=("Segoe UI", 9), anchor="w")
        foot.pack(fill="x", pady=(7, 0))

        # Battle tab: commands for the visible fight and roster.
        right = battle_right
        self._section(right, "Бой")
        self.play_button = self._make_button(right, "▶  ПРОДОЛЖИТЬ БОЙ", self._enter_play, accent=True)
        self.play_button.pack(fill="x", padx=12, pady=3)
        self.new_match_button = self._make_button(right, "↻  НОВЫЙ БОЙ", self._new_match, accent=True)
        self.new_match_button.pack(fill="x", padx=12, pady=3)
        self._section(right, "Настройки и модели")
        settings_row = tk.Frame(right, bg="#0c1929")
        settings_row.pack(fill="x", padx=10, pady=(2, 5))
        self.save_settings_button = self._make_button(
            settings_row, "💾 Настройки", self._save_settings,
            tooltip=f"Сохраняет модели, составы команд, параметры обучения/боя и интерфейса в {APP_CONFIG_PATH.name}. Веса хранятся отдельно.")
        self.save_settings_button.pack(side="left", fill="x", expand=True, padx=2)
        self.load_settings_button = self._make_button(
            settings_row, "↩ Загрузить", self._load_settings,
            tooltip=f"Загружает настройки из {APP_CONFIG_PATH.name}. Не удаляет веса и не сбрасывает текущий бой.")
        self.load_settings_button.pack(side="left", fill="x", expand=True, padx=2)

        # Training tab: optimizer/session actions and checkpoint management.
        right = self.training_columns[0]
        self._section(right, "Обучение")
        self.train_button = self._make_button(right, "🧠  НАЧАТЬ / ОСТАНОВИТЬ ОБУЧЕНИЕ", self._toggle_training, accent=True)
        self.train_button.pack(fill="x", padx=12, pady=3)
        self.stop_all_button = tk.Button(right, text="■  СТОП ВСЁ", command=self._stop_everything,
                                         bg="#a52d3b", fg="white", activebackground="#c63c4a",
                                         activeforeground="white", relief="flat", font=("Segoe UI", 11, "bold"),
                                         padx=8, pady=8, cursor="hand2")
        self.stop_all_button.pack(fill="x", padx=12, pady=3)
        self._set_tooltip(self.stop_all_button, "Безопасно останавливает обучение и замораживает обычный бой.")
        self.test_battle_button = self._make_button(
            right, "⚔  ПРОВЕРИТЬ В БОЮ", self._check_in_battle, accent=True,
            tooltip="Безопасно останавливает обучение, загружает последний сохранённый чекпоинт активной модели и начинает видимый бой. Если в обучении выбран self-play, модель играет против копии самой себя; иначе — против выбранного эвристического/модельного противника.")
        self.test_battle_button.pack(fill="x", padx=12, pady=(3, 7))
        weight_row = tk.Frame(right, bg="#0c1929")
        weight_row.pack(fill="x", padx=10, pady=3)
        self._make_button(weight_row, "💾 Сохранить веса", self._save_weights).pack(side="left", fill="x", expand=True, padx=2)
        self._make_button(weight_row, "↩ Загрузить веса", self._load_weights).pack(side="left", fill="x", expand=True, padx=2)
        training_settings_row = tk.Frame(right, bg="#0c1929")
        training_settings_row.pack(fill="x", padx=10, pady=3)
        self.save_settings_button_training = self._make_button(training_settings_row, "💾 Сохранить конфиг", self._save_settings)
        self.save_settings_button_training.pack(side="left", fill="x", expand=True, padx=2)
        self.load_settings_button_training = self._make_button(training_settings_row, "↩ Загрузить конфиг", self._load_settings)
        self.load_settings_button_training.pack(side="left", fill="x", expand=True, padx=2)
        reset_row = tk.Frame(right, bg="#0c1929")
        reset_row.pack(fill="x", padx=12, pady=3)
        self._make_button(reset_row, "♻  Сброс весов…", self._request_reset_weights).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self.reset_confirmation_frame = tk.Frame(reset_row, bg="#0c1929")
        self._make_button(self.reset_confirmation_frame, "ПОДТВЕРДИТЬ", self._confirm_reset_weights).pack(side="left", fill="x", expand=True, padx=2)
        self._make_button(self.reset_confirmation_frame, "×", self._cancel_reset_weights).pack(side="left", fill="x", padx=2)
        self.reset_confirmation_label = tk.Label(right, text="Старые веса и статистика будут сохранены в резервную копию.",
                                                  bg="#0c1929", fg="#ffc777", font=("Segoe UI", 8), wraplength=280)
        self._set_tooltip(self.reset_confirmation_label, "При подтверждении веса и статистика архивируются с временной отметкой перед созданием новой политики.")

        right = self.training_columns[0]
        self._section(right, "Библиотека моделей и весов")
        displays = [self._display_for_model_id(str(r.get("id", DEFAULT_MODEL_ID))) for r in self.model_records]
        self.active_model_combo = self._make_combo_row(
            right, "Активная модель / цель обучения", self.active_model_var, displays,
            tooltip="Выбирает конкретную модель, чьи веса будут загружены в интерфейс и продолжены при старте обучения. Встроенные модели: «Простая PPO» — компактный общий actor/critic; «Командная PPO» — локальная политика с независимым critic и командной наградой; «Раздельный critic (контроль)» изолирует пользу независимого critic при индивидуальной награде. Они создаются со случайной инициализацией, а не являются заранее обученными сильными пилотами. Каждая модель хранится отдельно в GravitrueX/models/ID/weights_ID_latest.npz. Переключение не изменяет уже начатый бой; чтобы применить модель к бою, выбери её в команде и нажми «Новый бой».")
        self.active_model_combo.bind("<<ComboboxSelected>>", self._active_model_changed, add="+")
        name_row = tk.Frame(right, bg="#0c1929")
        name_row.pack(fill="x", padx=12, pady=(3, 1))
        name_entry = tk.Entry(name_row, textvariable=self.new_model_name_var, bg="#14283a", fg=TEXT,
                              insertbackground=TEXT, relief="flat", font=("Segoe UI", 9))
        name_entry.pack(fill="x", pady=2)
        self._set_tooltip(name_entry, "Имя для нового варианта модели. Копия переносит все параметры, совпадающие по форме, из активной модели; новая с нуля получает случайные веса. У каждой модели будет собственная папка, статистика и чекпоинт.")
        model_buttons = tk.Frame(right, bg="#0c1929")
        model_buttons.pack(fill="x", padx=10, pady=2)
        self.create_copy_button = self._make_button(model_buttons, "＋ Копия активной", lambda: self._create_model_variant(False),
            tooltip="Создать отдельный вариант текущей модели. Подходит для экспериментов: оригинальные веса не затрагиваются.")
        self.create_copy_button.pack(side="left", fill="x", expand=True, padx=2)
        self.create_fresh_button = self._make_button(model_buttons, "＋ С нуля", lambda: self._create_model_variant(True),
            tooltip="Создать новую модель с тем же выбранным профилем, но со случайно инициализированными весами. Используй для чистого эксперимента без переноса навыков.")
        self.create_fresh_button.pack(side="left", fill="x", expand=True, padx=2)
        self.model_detail_label = tk.Label(right, textvariable=self.model_detail_var, bg="#0c1929", fg="#a9c4d9",
                                           font=("Consolas", 8), wraplength=280, justify="left", anchor="w")
        self.model_detail_label.pack(fill="x", padx=14, pady=(3, 5))
        self._set_tooltip(self.model_detail_label, "Сведения относятся к выбранной активной модели: архитектура, размер, накопленные тренировочные эпизоды и физические шаги, последний противник и наличие сохранённых весов. При обучении сюда добавится активный статус.")

        right = battle_right
        self._section(right, "Состав команд и ИИ")
        self._counter_row(right, "Синие корабли", self.blue_n, lambda d: self._adjust_count(self.blue_n, d))
        self._counter_row(right, "Красные корабли", self.red_n, lambda d: self._adjust_count(self.red_n, d))
        self.composition_note = tk.StringVar(value="Состав применяется при запуске нового боя.")
        composition_note_label = tk.Label(right, textvariable=self.composition_note, bg="#0c1929", fg="#ffc777",
                                          font=("Segoe UI", 8), wraplength=280, justify="left")
        composition_note_label.pack(fill="x", padx=14, pady=(3, 5))
        self._set_tooltip(composition_note_label, "Показывает состав активного боя и выбранный состав для следующего боя. Изменения счётчиков не трогают текущую арену — нажми «Новый бой», чтобы применить их.")
        team_choices = [HEURISTIC_LABEL] + displays
        self.blue_ai_combo = self._make_combo_row(
            right, "Модель синей команды", self.blue_ai, team_choices,
            tooltip="Выбери эвристический ИИ или конкретную сохранённую модель. Выбранный участник фиксируется в момент нажатия «Новый бой», поэтому изменение списка не перезапускает текущий матч.")
        self.red_ai_combo = self._make_combo_row(
            right, "Модель красной команды", self.red_ai, team_choices,
            tooltip="Можно сражать разные сохранённые веса друг с другом. Выбор применяется только к следующему бою; если выбрать ту же модель с обеих сторон, это будет зеркальный матч.")
        self.game_inference_combo = self._make_combo_row(
            right, "Вычисление ИИ в бою", self.game_inference_mode,
            ("GPU/CUDA (единственный режим в бою)",),
            tooltip=("Видимый бой использует только CUDA для нейросетевого inference. CPU/NumPy fallback запрещён: если CUDA недоступна или операция падает, бой ставится на паузу с ошибкой вместо тихого замедления на CPU. Скалярная физика отображаемой арены остаётся CPU; пакетная CUDA-физика применяется в обучении."))
        right = self.training_columns[1]
        self._section(right, "Противник в обучении")
        self.training_opponent_combo = self._make_combo_row(
            right, "Противник обучаемой модели", self.training_opponent,
            [HEURISTIC_LABEL, SELFPLAY_LABEL] + displays,
            tooltip="Выбери эвристического противника, self-play (обучаемая политика используется за обе стороны) или сохранённую модель. При выборе сохранённой модели её веса замораживаются на время сессии; обновляется только активная модель. Если выбрана сама активная модель, противник будет использовать стартовую копию её весов, то есть получится обучение против предыдущего чекпоинта.")
        self.training_target_label = tk.Label(right, textvariable=self.training_target_var, bg="#0c1929", fg=ACCENT,
                                              font=("Segoe UI", 8, "bold"), wraplength=280, justify="left")
        self.training_target_label.pack(fill="x", padx=14, pady=(2, 3))
        self._set_tooltip(self.training_target_label, "Это конкретная модель, параметры которой будут изменяться PPO в следующей тренировочной сессии. Смотри ID в квадратных скобках, чтобы не перепутать похожие названия.")
        selfplay_note = tk.Label(right, text="Можно обучать против бота, прошлой версии модели или через self-play. У каждой именованной модели свой файл весов и статистика.",
                                 bg="#0c1929", fg=MUTED, font=("Segoe UI", 8), wraplength=280,
                                 justify="left")
        selfplay_note.pack(fill="x", padx=14, pady=(2, 4))
        self._set_tooltip(selfplay_note, "Self-play: обе стороны используют обучаемую политику, и PPO получает записи обеих команд. Именованный противник: его чекпоинт не обновляется во время этой сессии. Для объективного сравнения сохраняй отдельные варианты моделей и тестируй их на одинаковых составах/seed.")
        self.training_blue_counter_row = self._counter_row(
            right, "Кораблей в обучаемой команде", self.train_blue_n, self._adjust_training_blue_n,
            tooltip="Фиксированный размер синей команды в тренировочных аренах. Используется, когда выключены случайные составы. Не меняет состав видимого боя и не влияет на сохранённые модели.")
        self.training_red_counter_row = self._counter_row(
            right, "Кораблей в команде противника", self.train_red_n, self._adjust_training_red_n,
            tooltip="Фиксированный размер красной команды в тренировочных аренах (1–5). Для эвристического ИИ или сохранённой модели это количество её кораблей; в self-play — количество кораблей второй команды. Игнорируется, если включены случайные составы.")
        self.training_record_budget_combo = self._make_combo_row(
            right, "Лимит записей GPU", self.train_record_budget,
            GPU_RECORD_BUDGET_CHOICES,
            tooltip="Максимальный объём опыта на один GPU-rollout. Доступны 720k, 1,44 млн (×2), 2,16 млн (×3), 2,88 млн (×4). Больший бюджет может увеличить фактический батч, но расходует больше VRAM/RAM. На RTX 5060 8 ГБ повышай ступенчато; следи за памятью и OOM fallback. Горизонт сохраняется, уменьшается фактическое число арен.")
        cb = tk.Checkbutton(right, text="Случайные составы 1–5 (вместо выбранных ниже)",
                            variable=self.random_compositions, bg="#0c1929", fg=TEXT,
                            selectcolor="#14283a", activebackground="#0c1929", activeforeground=TEXT,
                            font=("Segoe UI", 9), wraplength=280, justify="left")
        cb.pack(anchor="w", padx=14, pady=(5, 3))
        self._set_tooltip(cb, "Если включено, каждый тренировочный эпизод случайно выбирает от 1 до 5 синих и от 1 до 5 красных кораблей. При фиксированных составах используй два счётчика выше — так проще сравнивать обучение. Случайные составы разнообразят опыт, но средняя скорость может быть ниже: лимит записей оставляет меньше параллельных арен для крупных команд.")
        self._section(right, "Поведение и длина боёв")
        self.training_reward_combo = self._make_combo_row(
            right, "Награда поведения", self.training_reward_mode,
            ("По профилю модели", "Индивидуальная", "Командная"),
            tooltip=("По профилю модели — берёт рекомендуемую награду из карточки модели. Индивидуальная — преимущественно поощряет личный урон и убийства; это контроль. Командная — сильнее учитывает общий прогресс команды, победу и выживших, чтобы уменьшать награду за личное убийство в ущерб союзникам. Изменяет только сигнал PPO, а не физику боя. Для встроенной «Командной PPO» оставь «По профилю модели»; это экспериментальная командная награда, а не гарантия координации."))
        self.training_priors_combo = self._make_combo_row(
            right, "Подсказки действий", self.training_priors_mode,
            ("По профилю модели", "Мягкие", "Жёсткие (старые)", "Выключены"),
            tooltip=("Выбирает насколько программа вручную подталкивает политику. Мягкие — слабый бонус к наведению/уклонению/стрельбе, рекомендуемый старт. Жёсткие (старые) — сильнее вмешиваются в выбор, включая нижний порог вероятности выстрела в удобном окне; полезно лишь для сравнения. Выключены — решение полностью остаётся сети. Подсказки применяются при обучении и при выборе действий в бою, поэтому тестируй варианты на одинаковых настройках."))
        self.training_range_combo = self._make_combo_row(
            right, "Целевая дистанция, ед.", self.training_desired_range,
            ("Без целевой дистанции", "250", "350", "480", "600", "750"),
            tooltip=("«Без целевой дистанции» отключает только явный shaping-бонус за приближение к заданному диапазону. Расстояние остаётся входным признаком, а награда за наведение, стрельбу, урон и победу сохраняется. Это экспериментальный режим: сравни его с 480 при одинаковых seed и настройках. Числовые значения — мягкая цель, не ограничение движения."))
        self.training_horizon_combo = self._make_combo_row(
            right, "Длина тренировочного / тестового боя", self.train_horizon,
            HORIZON_CHOICES,
            tooltip=("Одно решение удерживается 4 физических тика. При 60 тиках/с: 450=30 с, 900=60 с, 1350=90 с, 1800=120 с, 2700=180 с, 3600=240 с, 5400=360 с. Старые короткие горизонты сохранены. Если бой не завершился за лимит, это unresolved/timeout, не победа и не поражение. Длинный горизонт может снизить число параллельных арен при том же бюджете записей."))
        self._section(right, "Параллелизм обучения")
        rollout_modes = (("CPU: параллельные арены",) if FORCE_CPU else
                         ("CPU: параллельные арены",
                          "GPU: параллельные арены (CUDA-физика)",
                          "GPU: действия + PPO (CPU-физика, 1 бой)",
                          "Гибрид: CPU + GPU-параллельные бои",
                          "Авто: CPU/GPU по скорости"))
        self.training_rollout_combo = self._make_combo_row(
            right, "Сбор опыта", self.training_rollout_mode, rollout_modes,
            tooltip=("CPU: параллельные арены — физика и действия на CPU-процессах; PPO обновляется на выбранном устройстве. "
                     "GPU: параллельные арены (CUDA-физика) — только векторный GPU-rollout; CPU-арены не запускаются. Число в поле GPU-параллельные арены задаёт размер батча независимых боёв, а не число реальных CUDA-потоков. PPO можно выбрать на CUDA или CPU. "
                     "GPU: действия + PPO (CPU-физика, 1 бой) — устаревший диагностический режим: игровая физика остаётся CPU, на GPU идут действия/PPO. "
                     "Гибрид — CPU-процессы и GPU-пакет работают одновременно, затем опыт объединяется. Авто сравнивает скорости и отключает более медленный GPU-rollout, но может оставить CUDA для PPO. "
                     "GPU-режимы требуют CUDA и поддерживают эвристического противника либо self-play, но не отдельную именованную модель."))
        self.cpu_worker_counter_row = self._counter_row(right, "Процессы CPU", self.train_workers, self._adjust_workers)
        self.gpu_env_counter_row = None
        if not FORCE_CPU:
            self.gpu_env_counter_row = self._counter_row(right, "GPU-параллельные арены", self.train_gpu_envs, self._adjust_gpu_envs)
            gpu_env_note = tk.Label(right, text="Диапазон 1–4096 арен; 4096 — экспериментальный верхний предел с OOM fallback. Для RTX 5060 сравни 2048, 3072 и 4096 на одинаковом профиле.",
                                    bg="#0c1929", fg=MUTED, font=("Segoe UI", 8), wraplength=280, justify="left")
            gpu_env_note.pack(fill="x", padx=14, pady=(1, 3))
            self._set_tooltip(gpu_env_note, "Число независимых боёв в одном CUDA-тензорном батче (1–4096), НЕ число буквальных CUDA-потоков. Для RTX 5060 с 8 ГБ начни с 3072; 4096 — экспериментальная ступень. Бюджет записей 360 000 ограничивает число одновременно собираемых арен: при большом составе или длинном горизонте программа уменьшает размер батча, сохраняя выбранную длину каждого боя. Поэтому сравни не только throughput и VRAM, но и фактическое число арен; при CUDA OOM размер пакета автоматически уменьшается.")
        worker_note = tk.Label(right, text="Можно ввести вручную; Shift+−/+ меняет на 5. В гибриде число автоматически ограничится по запасу CPU и RAM.",
                               bg="#0c1929", fg=MUTED, font=("Segoe UI", 8), wraplength=280,
                               justify="left")
        worker_note.pack(fill="x", padx=14, pady=(2, 4))
        self._set_tooltip(worker_note, "В обычном CPU-режиме можешь сравнивать 4–8 процессов и увеличивать только если шагов/с становится больше. В гибриде Stage 6.7 число активных CPU-воркеров автоматически ограничивается максимумом 4, резервирует около четверти логических потоков для CUDA-dispatch/UI и снижается ещё сильнее при дефиците RAM. В низкоприоритетном пуле CPU собирает эпизоды, пока GPU считает свой батч. Значение поля — желаемое число, отчёт показывает активное. Только для старого режима «GPU: действия + PPO» значение принудительно становится 1.")
        self._section(right, "Большая нейросеть / GPU")
        self.model_profile_combo = self._make_combo_row(
            right, "Размер модели", self.model_profile, tuple(ARCH_PROFILES.keys()),
            tooltip="Меняет размер активной архитектуры при следующем старте/подготовке обучения. Перед сменой размера программа сохраняет checkpoint в резервную папку; параметры переносятся только если форма совпадает. Во время PPO это поле блокируется, чтобы не было путаницы с реально обучаемой сетью.")
        self._make_combo_row(right, "Устройство PPO", self.training_device,
                             (("CPU только",) if FORCE_CPU else ("Авто: CUDA если доступна", "CUDA обязательно", "CPU только")),
                             tooltip="Где обновляются веса после сбора опытa. CUDA обычно быстрее для крупных сетей и больших mini-batch. CPU полезен как контрольный тест и может быть разумным, если CPU-rollout завершился, а GPU-физика уже остановилась; однако CPU PPO не выполняется одновременно с GPU-rollout в текущей архитектуре. Если выбрать CPU в гибридном режиме, физика GPU всё равно остаётся на CUDA, а только обратное распространение/optimizer переходят на CPU.")
        self._make_combo_row(right, "Мини-пакет PPO", self.train_batch,
                             ("512", "1024", "2048", "4096", "8192"))
        self._make_combo_row(right, "Эпохи PPO за обновление", self.train_epochs,
                             ("1", "2", "3", "4", "6"),
                             tooltip="Эпоха = один полный проход по всем записям текущего сбора опыта (внутри данные делятся на mini-batch и перемешиваются). Например, 3 эпохи означают до 3 проходов по одному и тому же rollout, а не 3 боя. Больше эпох — дольше PPO и сильнее обучение на этих данных, но не больше новых боёв; слишком много может переобучить свежую партию/сделать обновление слишком резким. Начни с 3; если PPO занимает много времени, попробуй 1–2; если PPO короткое и политика улучшается медленно, сравни 4–6, но оценивай победы на тестовых боях.")
        gpu_note = tk.Label(right, text="Сбалансированный гибрид: GPU считает выбранный пакет, CPU-процессы параллельно собирают дополнительные бои на пониженном приоритете. Число CPU-процессов и буфер автоматически ограничиваются по доступным ядрам и RAM.",
                            bg="#0c1929", fg="#b9c9d9", font=("Segoe UI", 8), wraplength=280,
                            justify="left")
        gpu_note.pack(fill="x", padx=14, pady=(2, 4))
        self._set_tooltip(gpu_note, "CPU и GPU собирают независимые эпизоды одновременно, пока политика заморожена; затем записи объединяются и выполняется одно PPO-обновление на выбранном устройстве. В гибриде CPU-процессы имеют пониженный приоритет, оставляя ресурсы для GPU-dispatch и интерфейса. При дефиците RAM число процессов и буфер автоматически уменьшаются. PPO не запускается параллельно со сбором, чтобы не менять policy посреди on-policy rollout. Если PPO — узкое место, тестируй mini-batch 2048→4096; 8192 только если VRAM свободна. Сравни шаги/с и качество на одинаковом числе эпизодов.")

        right = battle_right
        self._section(right, "Время симуляции")
        speed_frame = tk.Frame(right, bg="#0c1929")
        speed_frame.pack(fill="x", padx=9, pady=2)
        for i, speed_value in enumerate(self.speed_values):
            label = f"{speed_value:g}×"
            b = tk.Button(speed_frame, text=label, command=lambda v=speed_value: self._set_speed(v),
                          bg="#1b3246" if speed_value != 1.0 else "#24504e", fg=TEXT,
                          activebackground="#2d526a", activeforeground=TEXT, relief="flat",
                          font=("Segoe UI", 9, "bold"), padx=4, pady=5)
            b.grid(row=i // 4, column=i % 4, sticky="ew", padx=2, pady=2)
            self._set_tooltip(b, f"Скорость воспроизведения игрового боя: {speed_value:g}×. Это запускает больше фиксированных физических шагов за кадр, но не меняет физику и не ускоряет фоновое обучение. Высокие множители могут уменьшить FPS интерфейса.")
        for col in range(4):
            speed_frame.grid_columnconfigure(col, weight=1)
        self.pause_button = self._make_button(
            right, "Ⅱ  ПАУЗА", self._toggle_pause,
            tooltip="Переключает паузу текущего обычного боя. Не сбрасывает таймер, корабли или выбранный состав. Во время фонового обучения эта кнопка отключена; используйте «СТОП ВСЁ» для остановки обучения."
        )
        self.pause_button.pack(fill="x", padx=12, pady=(5, 2))
        speed_status = tk.Label(right, textvariable=self.speed_label, bg="#0c1929", fg=ACCENT,
                                font=("Segoe UI", 9))
        speed_status.pack(anchor="w", padx=14, pady=(3, 0))
        self._set_tooltip(speed_status, "Множитель относится только к отображаемому бою. Обучение запускает отдельные безоконные арены и управляется параметрами числа процессов, размера мини-пакета и эпох PPO.")

        self._section(right, "Масштаб камеры")
        zoom_frame = tk.Frame(right, bg="#0c1929")
        zoom_frame.pack(fill="x", padx=10, pady=2)
        self._make_button(zoom_frame, "−", lambda: self._change_zoom(0.8), tooltip="Отдалить камеру примерно на 20%. На физику и решения ИИ не влияет.").pack(side="left", fill="x", expand=True, padx=2)
        self._make_button(zoom_frame, "1×", self._reset_zoom, tooltip="Вернуть стандартный масштаб камеры.").pack(side="left", fill="x", expand=True, padx=2)
        self._make_button(zoom_frame, "+", lambda: self._change_zoom(1.25), tooltip="Приблизить камеру примерно на 25%. На физику и решения ИИ не влияет.").pack(side="left", fill="x", expand=True, padx=2)
        self._make_button(right, "Вся карта", self._zoom_all, tooltip="Установить масштаб так, чтобы обзор охватывал почти весь сектор. Корабли будут выглядеть мельче.").pack(fill="x", padx=12, pady=2)
        zoom_note = tk.Label(right, text="Камера следует за боем. Колесо мыши меняет масштаб.",
                             bg="#0c1929", fg=MUTED, font=("Segoe UI", 8), wraplength=280,
                             justify="left")
        zoom_note.pack(fill="x", padx=14, pady=(2, 1))
        self._set_tooltip(zoom_note, "Масштаб меняет только визуальное представление. Координаты, дистанции, снаряды и наблюдения нейросети остаются неизменными.")

        right = self.training_summary_columns[0]
        self._section(right, "Производительность")
        for key, caption in [
            ("perf_fps", "FPS интерфейса: измерение…"),
            ("perf_cpu", "CPU системы: измерение…"),
            ("perf_ram", "RAM системы / процесс: измерение…"),
            ("perf_gpu", "GPU / VRAM: поиск nvidia-smi…"),
            ("perf_infer", "ИИ в бою: CUDA/GPU (CPU fallback запрещён)"),
            ("perf_steps", "Скорость симуляции: —"),
            ("perf_stage", "Этап вычислений: ожидание"),
        ]:
            v = tk.StringVar(value=caption)
            self.stats_labels[key] = v
            perf_label = tk.Label(right, textvariable=v, bg="#0c1929", fg=ACCENT if key == "perf_fps" else MUTED,
                                  font=("Consolas", 8), anchor="w", justify="left", wraplength=520)
            perf_label.pack(fill="x", padx=14, pady=2)
            self._set_tooltip(perf_label, self._performance_tooltip(key))
        perf_note = tk.Label(right, text="GPU/VRAM через nvidia-smi (опрос ~1 раз/с). Во время сбора опыта GPU может простаивать — см. «Этап вычислений».",
                             bg="#0c1929", fg=MUTED, font=("Segoe UI", 8), wraplength=500,
                             justify="left")
        perf_note.pack(fill="x", padx=14, pady=(2, 4))
        self._set_tooltip(perf_note, "nvidia-smi сообщает общую загрузку GPU, занятую VRAM и температуру. Это не процент выполнения именно нейросетевых операций GravitrueX; другие приложения также могут загружать видеокарту.")
        copy_report_button = self._make_button(right, "▤  Копировать краткий отчёт", self._copy_training_report,
            tooltip="Копирует в буфер обмена текстовый отчёт обучения: модель, CPU/GPU-физика отдельно, время ожидания, время PPO, параметры mini-batch/эпох и две последние итерации. Этот текст удобно вставить в чат для диагностики.")
        copy_report_button.pack(fill="x", padx=12, pady=(2, 6))

        # Battle stats are intentionally compact; training metrics live on the training tab.
        right = battle_right
        self._section(right, "Статистика боя")
        for key, default in [("mode", "РЕЖИМ: ИГРА"), ("model", "Модель"),
                             ("episode", "Матч"), ("params", "Параметры")]:
            v = tk.StringVar(value=default)
            self.stats_labels[key] = v
            status_label = tk.Label(right, textvariable=v, bg="#0c1929",
                                    fg=ACCENT if key == "mode" else TEXT, font=("Consolas", 9),
                                    anchor="w", justify="left", wraplength=285)
            status_label.pack(fill="x", padx=14, pady=3)
            self._set_tooltip(status_label, self._status_tooltip(key))
        keys_note = tk.Label(right, text="SPACE — пауза · R — новый бой · 1–7 — скорость · +/- — масштаб · 0 — вся карта",
                             bg="#0c1929", fg=MUTED, font=("Segoe UI", 8), wraplength=285, justify="left")
        keys_note.pack(fill="x", padx=14, pady=(12, 8), anchor="w")
        right = self.training_summary_columns[1]
        self._section(right, "Результаты обучения")
        for key, default in [("train", "Обучение"), ("win", "Доля побед"), ("reward", "Награда"),
                             ("loss", "Loss"), ("accuracy", "Попадания"), ("damage", "Урон"),
                             ("kills", "Убийства"), ("survivors", "Выжившие"), ("speed", "Шагов/с")]:
            v = tk.StringVar(value=default)
            self.stats_labels[key] = v
            color = "#8be0ca" if key in ("train", "win", "kills") else TEXT
            status_label = tk.Label(right, textvariable=v, bg="#0c1929", fg=color,
                                    font=("Consolas", 9), anchor="w", justify="left", wraplength=520)
            status_label.pack(fill="x", padx=14, pady=3)
            self._set_tooltip(status_label, self._status_tooltip(key))
        weights_path_label = tk.Label(right, text=f"Веса: {MODEL_ROOT}", bg="#0c1929", fg="#668098",
                                      font=("Segoe UI", 7), wraplength=500, justify="left")
        weights_path_label.pack(fill="x", padx=14, pady=(10, 10))
        root.bind("<space>", lambda e: self._toggle_pause())
        root.bind("r", lambda e: self._new_match())
        root.bind("1", lambda e: self._set_speed(0.5))
        root.bind("2", lambda e: self._set_speed(1.0))
        root.bind("3", lambda e: self._set_speed(2.0))
        root.bind("4", lambda e: self._set_speed(4.0))
        root.bind("5", lambda e: self._set_speed(8.0))
        root.bind("6", lambda e: self._set_speed(16.0))
        root.bind("7", lambda e: self._set_speed(32.0))
        root.bind("+", lambda e: self._change_zoom(1.25))
        root.bind("=", lambda e: self._change_zoom(1.25))
        root.bind("-", lambda e: self._change_zoom(0.8))
        root.bind("0", lambda e: self._zoom_all())
        self.canvas.bind("<MouseWheel>", self._mousewheel_zoom)
        self._switch_ui_tab("battle")
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self._refresh_model_catalog()
        self._update_model_details()

    def _set_tooltip(self, widget, text: str):
        if not text:
            return widget
        tooltip = ToolTip(widget, text)
        self._tooltips.append(tooltip)
        return widget

    def _make_button(self, parent, text, command, accent=False, tooltip=None):
        tk = self.tk
        button = tk.Button(parent, text=text, command=command,
                           bg="#24504e" if accent else "#172b3e", fg=TEXT,
                           activebackground="#2f625e" if accent else "#29465f", activeforeground="white",
                           relief="flat", borderwidth=0, padx=10, pady=7,
                           font=("Segoe UI", 9, "bold"))
        if tooltip is None:
            tooltip = self._button_tooltip(text)
        self._set_tooltip(button, tooltip)
        return button

    @staticmethod
    def _button_tooltip(text: str) -> str:
        low = str(text).lower()
        if "игра" in low or "бой идёт" in low or "продолжить бой" in low or "вернуться в игру" in low:
            return "Перейти в режим обычного боя или продолжить текущий матч. Не создаёт новый бой, не применяет новый состав и не сбрасывает время. Во время активного обучения переход недоступен."
        if "обучен" in low or "остановить обучение" in low or "остановка" in low:
            return "Запускает PPO-обучение или запрашивает безопасную остановку текущей тренировки. При запуске применится выбранный профиль модели. При остановке сохраняется последний завершённый пакет весов."
        if "стоп" in low or "остановить" in low:
            return "Останавливает активный процесс: обычный бой ставится на паузу, обучение получает запрос на завершение текущей партии и сохранение весов."
        if "новый бой" in low:
            return "Создаёт новую арену с выбранным числом кораблей и сбрасывает таймер боя. Это единственная кнопка, которая применяет изменённый состав команд."
        if "сохранить веса" in low:
            return "Сохраняет веса выбранной активной модели в GravitrueX/models/ID/weights_ID_latest.npz. У каждой модели своя папка и training_stats.json; другие модели не перезаписываются. Во время обучения чекпоинт сохраняется автоматически."
        if "загрузить веса" in low:
            return "Загружает совместимые параметры из файла весов. Делай это после остановки обучения; загрузка заменит политику в памяти."
        if "сброс весов" in low or "сброс" in low:
            return "Создаёт новую случайно инициализированную модель выбранного размера. Перед сбросом программа сохраняет резервную копию весов и статистики. Изменение необратимо только для активного экземпляра, но копии остаются на диске."
        if "подтвердить" in low:
            return "Подтверждает сброс весов: будут созданы резервные копии старой модели и статистики, затем начнётся новая инициализация."
        if text.strip() in {"×", "x", "X"}:
            return "Отменяет подтверждение сброса и ничего не меняет."
        if text.strip() in {"−", "+"}:
            return "Изменяет соответствующий параметр на единицу. Для числа кораблей новый состав применится только после нажатия «Новый бой»."
        if text.strip() == "Вся карта":
            return "Отдаляет камеру так, чтобы было видно большую часть всего сектора. На симуляцию и нейросеть не влияет."
        return "Нажми, чтобы изменить настройку. Наведи курсор на связанные параметры, чтобы увидеть пояснения."

    def _combo_tooltip(self, label: str) -> str:
        tips = {
            "ИИ синей команды": "Выбирает управление синими кораблями в обычном бою. «Нейросеть» использует сохранённую обучаемую политику; «Эвристический ИИ» использует вручную заданную логику наведения, манёвра и уклонения. Настройка влияет на игровой режим, а не напрямую на уже запущенные эпизоды обучения.",
            "ИИ красной команды": "Выбирает управление красными кораблями в обычном бою. Нейросеть позволяет играть против себя или против эвристики. Применяется к текущей игре на следующих решениях, без перезапуска матча.",
            "Режим обучения": "«Эвристический ИИ» обучает синюю команду против фиксированного бота. Self-play использует одну и ту же обучаемую политику для обеих сторон и собирает опыт обоих лагерей. Self-play может порождать ничьи или циклы, поэтому сравнивай прогресс на фиксированном контрольном противнике.",
            "Размер модели": "Задаёт ширину и глубину крупной MLP-ветки. Больше параметров может выразить более сложные зависимости, но увеличивает время прямого прохода на CPU, память и стоимость PPO-обновления. Выбранный профиль применяется при запуске обучения; совпадающие веса переносятся, несовпадающие слои инициализируются заново.",
            "Устройство PPO": "Выбирает устройство только для обновления весов PPO. «CPU только» не использует CUDA; «Авто» выбирает CUDA, если PyTorch и драйвер её видят; «CUDA обязательно» выдаёт ошибку, если CUDA недоступна. Это не переносит физику боя на GPU. Когда менять: выбирай CUDA, если этап PPO заметно долгий и VRAM свободна; если тормозит сбор опыта, смена устройства PPO почти не поможет.",
            "Мини-пакет PPO": "Сколько записей опыта обрабатывается за один шаг оптимизатора. Что делать: если GPU загружается короткими всплесками именно на этапе PPO и VRAM свободна — перейди с 2048 на 4096, затем на 8192. Если память почти заполнена, обучение выдаёт OOM или PPO стал слишком долгим — снизь пакет. Большой batch не ускоряет CPU-физику.",
            "Эпохи PPO за обновление": "Сколько проходов по собранному опыту делается за обновление. Что делать: начни с 2–3. Если PPO занимает мало времени, но улучшения нестабильны, можно проверить 4; если обучение медленное или показатели ухудшаются — верни 2–3. 6 эпох дают больше вычислений, но не гарантируют лучший результат и могут переобучить политику на один пакет.",
            "Вычисление ИИ в бою": "Устройство прямого прохода при обычной игре. CPU — предпочтительно для 1–5 управляемых нейросетью кораблей: маленький пакет часто быстрее без копирования на GPU. GPU: пакет действий отправляет решения всех нейрокораблей одним батчем на CUDA; тестируй с крупной моделью/многими кораблями, но физика остаётся на CPU. Авто включает GPU только если батч/модель достаточно велики. Если GPU не поддерживается, будет CPU fallback.",
            "Сбор опыта": "CPU: параллельные арены — обычная физика в нескольких CPU-процессах. GPU: действия + PPO — один бой, физика CPU и политика/PPO на CUDA; это диагностический режим. Гибрид: CPU + GPU-параллельные бои — несколько независимых боёв одновременно на CPU-процессах плюс ещё отдельный пакет GPU-арен с векторной физикой в PyTorch. Опыт обоих источников собирается вместе и используется одним PPO-обновлением. Гибрид требует CUDA и пока поддерживает эвристического противника и self-play, но не отдельную именованную модель.",
        }
        return tips.get(label, f"Настройка «{label}». Наведи курсор на неё и на соседние элементы, чтобы узнать, как они влияют на бой или обучение.")

    @staticmethod
    def _counter_tooltip(label: str) -> str:
        if label == "Синие корабли":
            return ("Число синих кораблей в следующем обычном бою и в фиксированном составе обучения. "
                    "Можно выставить от 1 до 5. Текущая арена не меняется; для применения в игре нажми «Новый бой».")
        if label == "Красные корабли":
            return ("Число красных кораблей в следующем обычном бою и в фиксированном составе обучения. "
                    "Можно выставить от 1 до 5. Текущая арена не меняется; для применения в игре нажми «Новый бой».")
        if label == "Процессы CPU":
            return "Число независимых боёв в отдельных CPU-процессах (1–32). Это процессы, не потоки. Ryzen 7 5700X: 8 физических ядер и 16 логических потоков; сравни 8/10/12/16. Увеличивай, пока физ. шаг/с растёт. Если CPU 95–100%, а скорость стоит или падает, уменьши. Shift+клик −/+ меняет значение на 5; число можно вводить вручную."
        if label == "GPU-параллельные арены":
            return "Число независимых боёв в одном GPU-тензорном батче (1–4096), это НЕ отдельные CUDA-потоки. На RTX 5060 8 ГБ проверь 3072 и 4096: оставляй не менее 1.5–2 ГБ запаса для Windows, дисплея и других приложений. Бюджет rollout 240 000 записей. При self-play это примерно 64 решения/арену на 3072 или 48 на 4096; больше арен не всегда лучше, если из-за бюджета слишком короткая траектория. Оцени шаги/с, решения/арену, выстрелы, попадания и победы. При CUDA OOM пакет автоматически снижается."
        return f"Параметр «{label}»: используй −/+ для изменения значения."

    @staticmethod
    def _performance_tooltip(key: str) -> str:
        tips = {
            "perf_fps": "Число отрисованных кадров интерфейса в секунду, не скорость физики. Во время обучения UI намеренно обновляется примерно 10 Гц. На паузе сцена кэшируется и цикл опускается до 10 Гц; цифра показывает последнюю скорость активной игры, а не бесполезные циклы ожидания. Если активная игра ниже 55 FPS, выключи GPU-инференс для малого боя, уменьши число отображаемых объектов или проверь нагрузку Canvas.",
            "perf_cpu": "CPU системы — общая загрузка процессора. Arena — суммарная доля процессорной мощности, занятой программой и дочерними процессами, пересчитанная относительно всех логических ядер. Во время 12 параллельных CPU-rollout значение около 100% может быть ожидаемым.",
            "perf_ram": "Показывает занятую оперативную память всей системы и суммарную память процесса Arena с дочерними рабочими процессами. Для крупных моделей и множества параллельных арен следи за доступным запасом RAM.",
            "perf_gpu": "Общая загрузка NVIDIA GPU, VRAM и температура из nvidia-smi. В гибридном обучении CUDA одновременно считает физику нескольких боёв, а CPU-процессы генерируют дополнительные эпизоды. Если GPU загружена слабо и VRAM свободна — повышай GPU-арены; если память близка к пределу/появляется OOM — снижай. Процент загрузки не показывает скорость сам по себе: сравнивай эпизоды/с и rollout_sec_last.",
            "perf_infer": "В обычном бою действия каждой выбранной нейросети рассчитываются через CUDA. Для нескольких кораблей пакет небольшой, поэтому задержка может быть выше CPU; при больших сетях и ускорении времени GPU-инференс снижает CPU-нагрузку. Если написано CPU fallback, CUDA недоступна или произошла ошибка. Физика отображаемой Arena остаётся на CPU, а CUDA здесь отвечает за forward policy; в обучении GPU также выполняет батч-физику и PPO.",
            "perf_steps": "Физические шаги в секунду. Во время обучения это скорость генерации боевого опыта, а не количество шагов оптимизатора. Если CPU 100%, а шагов/с почти не растёт при добавлении workers — убавь процессы. Если CPU недогружен — прибавляй по одному и сравнивай.",
            "perf_stage": "Текущий этап обучения. Отдельные таймеры CPU, GPU, ожидания и PPO доступны в сводке. CPU PPO в гибриде возможен: CPU-арены закончены, и процессор обучает веса, пока GPU простаивает; это может быть полезно для сравнения, но обычно крупная модель обновляется быстрее на CUDA. Если GPU сбор быстрее CPU, оставь CPU PPO; если PPO — узкое место и VRAM хватает, используй CUDA.",
        }
        return tips.get(key, "Мониторинг производительности GravitrueX.")

    @staticmethod
    def _status_tooltip(key: str) -> str:
        tips = {
            "mode": "Текущий режим программы: игра, пауза или обучение.",
            "model": "Активная архитектура политики и состояние весов. Смена профиля может быть запланирована на следующее обучение, пока текущая политика остаётся активной.",
            "episode": "Состав и время текущего игрового матча.",
            "train": "Суммарное число завершённых тренировочных эпизодов и учтённых физических шагов. Сохраняется между запусками в статистике.",
            "win": "Исходы последних 20 тренировочных эпизодов. Ничья учитывается отдельно и не равна победе.",
            "reward": "Средняя суммарная награда за последние 20 эпизодов. Это внутренний сигнал обучения; для оценки силы ИИ важнее также реальные победы, убийства и выжившие корабли.",
            "loss": "Значение функции потерь PPO на последнем обновлении. Само по себе уменьшение loss не гарантирует улучшение тактики.",
            "accuracy": "Приблизительная доля нанесённого урона относительно теоретического урона всех выстрелов, плюс среднее число выстрелов за бой. Не является идеальной статистикой точности при разных условиях попадания.",
            "damage": "Средний суммарный урон каждой команды на последних тренировочных эпизодах.",
            "kills": "Среднее число уничтоженных кораблей каждой командой за последние эпизоды — один из ключевых критериев боевой эффективности.",
            "survivors": "Среднее число оставшихся кораблей каждой команды после эпизода. Важный показатель эффективности вместе с убийствами.",
            "speed": "В зависимости от режима показывает скорость сбора опыта. В гибридном режиме одновременно работают CPU-процессы со скалярной физикой и GPU с пакетной тензорной физикой; смотри этап и счётчик CPU/GPU арен в статусе обучения.",
            "params": "Число всех обучаемых параметров модели и число входных признаков. Больше параметров повышает выразительность, но увеличивает вычислительную цену.",
        }
        return tips.get(key, "Статистика текущего матча или обучения.")

    def _section(self, parent, title):
        header = self.tk.Label(parent, text=title.upper(), bg="#0c1929", fg="#7f9bb4",
                               font=("Segoe UI", 8, "bold"), anchor="w")
        header.pack(fill="x", padx=14, pady=(13, 4))
        self._set_tooltip(header, {
            "Библиотека моделей и весов": "Каждая модель имеет независимые веса и статистику в GravitrueX/models/ID. Активная модель — цель следующего обучения; копия создаёт экспериментальную ветку, «С нуля» — случайную политику.",
            "Состав команд и ИИ": "Выбирай разные именованные модели для синей и красной сторон. На поле показываются реальные участники матча. Изменения применяются только после «Новый бой».",
            "Противник в обучении": "PPO изменяет только активную модель. Противник может быть эвристикой, self-play или отдельным сохранённым checkpoint, который замораживается на время сессии.",
            "Параллелизм обучения": "CPU-процессы и GPU-параллельные арены задаются отдельно. В гибридном режиме обычные арены считаются на CPU одновременно с векторной физикой целого пакета боёв на CUDA; весь собранный опыт объединяется перед PPO. Число CPU-процессов можно вводить вручную; Shift+клик по +/− меняет шаг на 5.",
            "Большая нейросеть / GPU": "Размер модели и параметры PPO. В гибридном режиме CUDA одновременно считает физику нескольких боёв, а затем выполняет PPO-обновление на том же устройстве; CPU-процессы продолжают генерировать отдельные арены.",
            "Время симуляции": "Меняет воспроизведение текущего боя, а не темп обучения или физические константы.",
            "Масштаб камеры": "Меняет масштаб визуального обзора, не затрагивая координаты и тактику.",
            "Производительность": "Показывает FPS, общую нагрузку CPU/RAM/GPU и скорость симуляции. GPU-пики могут быть краткими между стадиями сбора опыта.",
            "Состояние": "Сводные метрики. Победы, убийства и выжившие лучше характеризуют тактику, чем одна только средняя награда или loss.",
        }.get(title, f"Раздел настроек: {title}. Наведи курсор на элементы раздела для подробностей."))

    def _copy_training_report(self):
        try:
            use_trainer_report = bool(self.trainer and self.trainer.model_id == self.active_model_id)
            if use_trainer_report and getattr(self.trainer, "fatal_error", None) and self.trainer.crash_report_path.exists():
                path = self.trainer.crash_report_path
            else:
                path = (self.trainer.report_path if use_trainer_report else model_dir_for(self.active_model_id) / "training_report.txt")
            if path.exists():
                report = path.read_text(encoding="utf-8")
            else:
                meta = read_model_meta(self.active_model_id)
                report = (f"GravitrueX summary\nModel: {meta.get('name', self.active_model_id)} [{self.active_model_id}]\n"
                          f"Profile: {self.policy.width}x{self.policy.depth}; arch={self.policy.architecture}; params={self.policy.count_params():,}\n"
                          f"Weights: {model_weights_path(self.active_model_id)}\nReport is not available yet; start at least one training update.")
            self.root.clipboard_clear()
            self.root.clipboard_append(report)
            self.root.update_idletasks()
            self.training_notice = "Отчёт диагностики скопирован. Его можно вставить в чат."
            self.messagebox.showinfo("Отчёт скопирован", "Отчёт обучения или полный traceback аварии скопирован в буфер обмена.")
        except Exception as exc:
            self.messagebox.showerror("Не удалось скопировать отчёт", str(exc))

    def _counter_row(self, parent, label, var, adjust, tooltip=None):
        tk = self.tk
        row = tk.Frame(parent, bg="#0c1929")
        row.pack(fill="x", padx=12, pady=2)
        tip = tooltip or self._counter_tooltip(label)
        title = tk.Label(row, text=label, bg="#0c1929", fg=TEXT, font=("Segoe UI", 9))
        title.pack(side="left", fill="x", expand=True)

        is_workers = var is self.train_workers
        is_gpu_envs = var is self.train_gpu_envs
        entry_var = (self.train_workers_text if is_workers else
                     self.train_gpu_envs_text if is_gpu_envs else None)
        minimum, maximum = (1, 32) if is_workers else ((1, GPU_MAX_ARENAS) if is_gpu_envs else (1, 5))

        def adjust_and_sync(delta):
            adjust(delta)
            if entry_var is not None:
                entry_var.set(str(var.get()))

        minus = tk.Button(row, text="−", width=2, command=lambda: adjust_and_sync(-1), bg="#21374b", fg=TEXT,
                          relief="flat", activebackground="#314e66")
        minus.pack(side="left", padx=2)
        if entry_var is not None:
            value = tk.Entry(row, textvariable=entry_var, width=5, justify="center", bg="#14283a",
                             fg=ACCENT, insertbackground=TEXT, relief="flat", font=("Consolas", 11, "bold"))
            value.pack(side="left", padx=2)
            value.bind("<Return>", lambda _e, v=var, ev=entry_var, lo=minimum, hi=maximum: self._commit_counter_entry(v, ev, lo, hi))
            value.bind("<FocusOut>", lambda _e, v=var, ev=entry_var, lo=minimum, hi=maximum: self._commit_counter_entry(v, ev, lo, hi))
            value.bind("<KeyRelease>", lambda _e, v=var, ev=entry_var, lo=minimum, hi=maximum: self._preview_counter_entry(v, ev, lo, hi))
        else:
            value = tk.Label(row, textvariable=var, width=2, bg="#0c1929", fg=ACCENT,
                             font=("Consolas", 11, "bold"))
            value.pack(side="left")
        plus = tk.Button(row, text="+", width=2, command=lambda: adjust_and_sync(1), bg="#21374b", fg=TEXT,
                         relief="flat", activebackground="#314e66")
        plus.pack(side="left", padx=2)

        if entry_var is not None:
            # Shift-click is bound ahead of Tk's class-level button command; returning
            # "break" prevents a second +/-1 action on the same click.
            minus.bind("<Shift-ButtonRelease-1>", lambda _e: (adjust_and_sync(-5), "break")[1])
            plus.bind("<Shift-ButtonRelease-1>", lambda _e: (adjust_and_sync(5), "break")[1])
        for widget in (row, title, value):
            self._set_tooltip(widget, tip)
        self._set_tooltip(minus, tip + " Обычное нажатие: −1; Shift+нажатие: −5.")
        self._set_tooltip(plus, tip + " Обычное нажатие: +1; Shift+нажатие: +5.")
        row._counter_controls = (minus, value, plus)
        return row

    def _preview_counter_entry(self, int_var, text_var, low, high):
        try:
            raw = str(text_var.get()).strip()
            if raw and raw.lstrip("+-").isdigit():
                int_var.set(int(clamp(int(raw), low, high)))
        except Exception:
            pass

    def _commit_counter_entry(self, int_var, text_var, low, high):
        try:
            raw = str(text_var.get()).strip()
            parsed = int(raw) if raw else int(int_var.get())
        except Exception:
            parsed = int(int_var.get())
        parsed = int(clamp(parsed, low, high))
        int_var.set(parsed)
        text_var.set(str(parsed))
        if hasattr(self, "training_notice"):
            if int_var is self.train_workers and parsed > 16:
                self.training_notice = "CPU-процессов больше 16: разрешено для эксперимента, но обычно не быстрее на CPU с 16 логическими потоками. Сравни физ. шаги/с."
        return "break"

    def _make_combo_row(self, parent, label, variable, values, tooltip=None):
        tk, ttk = self.tk, self.ttk
        row = tk.Frame(parent, bg="#0c1929")
        row.pack(fill="x", padx=12, pady=3)
        title = tk.Label(row, text=label, bg="#0c1929", fg=TEXT,
                         font=("Segoe UI", 8), anchor="w")
        title.pack(fill="x", anchor="w")
        combo = ttk.Combobox(row, textvariable=variable, values=values, state="readonly", width=27,
                             style="Arena.TCombobox")
        combo.pack(fill="x", pady=(2, 0))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._selection_changed())
        tip = tooltip or self._combo_tooltip(label)
        self._set_tooltip(row, tip)
        self._set_tooltip(title, tip)
        self._set_tooltip(combo, tip)
        return combo

    def _selection_changed(self):
        """Settings changes never restart a running battle or clear its pause state."""
        self._apply_game_inference_mode()
        self._scene_dirty = True
        self._update_composition_note()
        if hasattr(self, "active_model_id"):
            self._update_model_details()
        try:
            target = ARCH_PROFILES.get(self.model_profile.get())
            if target and target != (self.policy.width, self.policy.depth):
                self.model_label.set(
                    f"Модель: активна {self.policy.width}×{self.policy.depth}; "
                    f"профиль {target[0]}×{target[1]} применится при старте обучения"
                )
            elif target:
                self.model_label.set(
                    f"Модель: {self.policy.width}×{self.policy.depth} · "
                    f"{'обученные веса' if self.policy_loaded else 'новая инициализация'}"
                )
        except Exception:
            pass
        self._update_controls()

    def _apply_game_inference_mode(self):
        """Keep visible policy inference on CUDA; strict GPU mode pauses on CUDA failure."""
        if not hasattr(self, "policy"):
            return
        if hasattr(self, "game_inference_mode"):
            self.game_inference_mode.set("GPU/CUDA (единственный режим в бою)")
        self.policy.inference_mode = "gpu"
        self.policy.allow_gpu_inference = True
        self.policy.require_gpu_inference = True
        self.policy._inference_params = None
        self.policy._inference_device = None
        self.policy.last_inference_backend = "CUDA / ожидает первого батча"

    def _on_canvas_resize(self, _event=None):
        self._scene_dirty = True
        self._background_dirty = True

    def _refresh_paused_hud(self):
        """Refresh only changing performance text while the paused scene stays cached."""
        try:
            perf = self.perf.snapshot()
            cpu_txt = f"CPU {perf['cpu']:.0f}%" if perf.get("cpu") is not None else "CPU n/a"
            ram_txt = f"RAM {_fmt_bytes(perf['ram_used'])}" if perf.get("ram_used") is not None else "RAM n/a"
            if perf.get("gpu_util") is not None:
                gpu_txt = f"GPU/SM {perf['gpu_util']:.0f}%"
                if perf.get("gpu_mem_util") is not None:
                    gpu_txt += f" MEM {perf['gpu_mem_util']:.0f}%"
            else:
                gpu_txt = "GPU n/a"
            self.canvas.itemconfigure(
                "hud_perf",
                text=(f"FPS {self.display_fps:.1f} (пауза) · {cpu_txt} · {ram_txt} · {gpu_txt}"),
                fill=ACCENT if self.display_fps >= 55 else "#ffc777",
            )
            self._last_paused_hud_refresh = time.perf_counter()
        except Exception:
            self._scene_dirty = True

    def _update_composition_note(self):
        if not hasattr(self, "composition_note") or not hasattr(self, "arena"):
            return
        requested = (int(self.blue_n.get()), int(self.red_n.get()))
        current = (int(self.arena.blue_n), int(self.arena.red_n))
        requested_models = (self.blue_ai.get(), self.red_ai.get()) if hasattr(self, "blue_ai") else ("?", "?")
        current_models = (getattr(self, "match_blue_ai", HEURISTIC_LABEL), getattr(self, "match_red_ai", HEURISTIC_LABEL))
        model_note = (f"ИИ сейчас: СИНИЕ — {current_models[0]} / КРАСНЫЕ — {current_models[1]}. "
                      f"Для следующего боя выбрано: {requested_models[0]} / {requested_models[1]}.")
        if requested != current or requested_models != current_models:
            self.composition_note.set(
                f"Текущий бой: {current[0]}×{current[1]}. На следующий: {requested[0]}×{requested[1]}.\n"
                + model_note + "\nНовые настройки применятся только после «НОВЫЙ БОЙ»."
            )
        else:
            self.composition_note.set(
                f"Текущий состав: {current[0]}×{current[1]}. " + model_note + "\nИзменения не перезапускают бой."
            )

    def _update_controls(self):
        """Keep prominent controls consistent with play/pause/training state."""
        if not hasattr(self, "play_button"):
            return
        training_alive = bool(self.trainer and self.trainer.is_alive() and self.mode == "training")
        stop_requested = bool(training_alive and self.trainer.stop_event.is_set())
        round_over = bool(getattr(self.arena, "round_over", False))

        if training_alive:
            self.play_button.configure(text="▶  ИГРА ПОСЛЕ ОБУЧЕНИЯ", state="disabled")
            self.train_button.configure(
                text="ОСТАНОВКА…" if stop_requested else "■  ОСТАНОВИТЬ ОБУЧЕНИЕ",
                state="disabled" if stop_requested else "normal"
            )
            self.new_match_button.configure(state="disabled")
            self.pause_button.configure(text="Ⅱ  ПАУЗА (ИГРА)", state="disabled")
            self.stop_all_button.configure(
                text="■  ОСТАНОВКА…" if stop_requested else "■  СТОП ОБУЧЕНИЯ И БОЯ",
                state="disabled" if stop_requested else "normal"
            )
            if hasattr(self, "test_battle_button"):
                self.test_battle_button.configure(
                    text="ОСТАНОВКА · ЗАПУСК БОЯ…" if self._pending_battle_test else "⚔  ПРОВЕРИТЬ В БОЮ",
                    state="disabled" if (stop_requested or self._pending_battle_test) else "normal")
            for attr in ("load_settings_button", "load_settings_button_training"):
                widget = getattr(self, attr, None)
                if widget is not None:
                    widget.configure(state="disabled")
            if hasattr(self, "training_rollout_combo"):
                self.training_rollout_combo.configure(state="disabled")
            for attr in ("training_reward_combo", "training_priors_combo", "training_range_combo",
                         "training_horizon_combo", "model_profile_combo"):
                combo = getattr(self, attr, None)
                if combo is not None:
                    combo.configure(state="disabled")
            if hasattr(self, "gpu_env_counter_row"):
                for ctl in getattr(self.gpu_env_counter_row, "_counter_controls", ()):
                    ctl.configure(state="disabled")
            if hasattr(self, "cpu_worker_counter_row"):
                for ctl in getattr(self.cpu_worker_counter_row, "_counter_controls", ()):
                    ctl.configure(state="disabled")
            if hasattr(self, "game_inference_combo"):
                self.game_inference_combo.configure(state="disabled")
            for attr in ("active_model_combo", "create_copy_button", "create_fresh_button", "blue_ai_combo", "red_ai_combo", "training_opponent_combo"):
                widget = getattr(self, attr, None)
                if widget is not None:
                    try:
                        widget.configure(state="disabled")
                    except Exception:
                        pass
            return

        self.train_button.configure(text="⚙  НАЧАТЬ ОБУЧЕНИЕ" if self.mode == "play" else "⚙  ПРОДОЛЖИТЬ ОБУЧЕНИЕ",
                                    state="normal")
        if hasattr(self, "test_battle_button"):
            self.test_battle_button.configure(text="⚔  ПРОВЕРИТЬ В БОЮ", state="disabled" if self._pending_battle_test else "normal")
        for attr in ("load_settings_button", "load_settings_button_training"):
            widget = getattr(self, attr, None)
            if widget is not None:
                widget.configure(state="normal")
        self.new_match_button.configure(state="normal")
        if hasattr(self, "training_rollout_combo"):
            self.training_rollout_combo.configure(state="readonly")
        for attr in ("training_reward_combo", "training_priors_combo", "training_range_combo",
                     "training_horizon_combo", "model_profile_combo"):
            combo = getattr(self, attr, None)
            if combo is not None:
                combo.configure(state="readonly")
        if hasattr(self, "cpu_worker_counter_row"):
            cpu_controls_state = "disabled" if (self.training_rollout_mode.get().startswith("GPU: действия") or self.training_rollout_mode.get().startswith("GPU: параллельные арены")) else "normal"
            for ctl in getattr(self.cpu_worker_counter_row, "_counter_controls", ()):
                ctl.configure(state=cpu_controls_state)
        if hasattr(self, "gpu_env_counter_row"):
            gpu_controls_state = "normal" if (self.training_rollout_mode.get().startswith("Гибрид") or self.training_rollout_mode.get().startswith("GPU: параллельные арены")) else "disabled"
            for ctl in getattr(self.gpu_env_counter_row, "_counter_controls", ()):
                ctl.configure(state=gpu_controls_state)
        if hasattr(self, "game_inference_combo"):
            self.game_inference_combo.configure(state="readonly")
        for attr in ("active_model_combo", "blue_ai_combo", "red_ai_combo", "training_opponent_combo"):
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    widget.configure(state="readonly")
                except Exception:
                    pass
        for attr in ("create_copy_button", "create_fresh_button"):
            widget = getattr(self, attr, None)
            if widget is not None:
                try:
                    widget.configure(state="normal")
                except Exception:
                    pass
        if self.mode == "play":
            if round_over:
                self.play_button.configure(text="✓  БОЙ ЗАВЕРШЁН", state="disabled")
                self.pause_button.configure(text="Ⅱ  ПАУЗА", state="disabled")
                self.stop_all_button.configure(text="■  БОЙ ЗАВЕРШЁН", state="disabled")
            elif self.paused:
                self.play_button.configure(text="▶  ПРОДОЛЖИТЬ БОЙ", state="normal")
                self.pause_button.configure(text="▶  ПРОДОЛЖИТЬ БОЙ", state="normal")
                self.stop_all_button.configure(text="■  БОЙ ОСТАНОВЛЕН", state="disabled")
            else:
                self.play_button.configure(text="●  БОЙ ИДЁТ", state="disabled")
                self.pause_button.configure(text="Ⅱ  ПАУЗА", state="normal")
                self.stop_all_button.configure(text="■  СТОП БОЙ", state="normal")
        else:
            self.play_button.configure(text="▶  ВЕРНУТЬСЯ В ИГРУ", state="normal")
            self.pause_button.configure(text="Ⅱ  ПАУЗА (СНАЧАЛА ИГРА)", state="disabled")
            self.stop_all_button.configure(text="■  БОЙ НА ПАУЗЕ", state="disabled")

    def _request_reset_weights(self):
        if self.trainer and self.trainer.is_alive():
            self.messagebox.showwarning("Обучение запущено", "Сначала останови обучение. Сброс во время PPO запрещён, чтобы не повредить веса.")
            return
        self.reset_confirmation_frame.pack(side="right", fill="x", padx=(3, 0))
        self.reset_confirmation_label.pack(fill="x", padx=14, pady=(2, 3))

    def _cancel_reset_weights(self):
        self.reset_confirmation_frame.pack_forget()
        self.reset_confirmation_label.pack_forget()

    def _confirm_reset_weights(self):
        if self.trainer and self.trainer.is_alive():
            self.messagebox.showwarning("Обучение запущено", "Сначала останови обучение.")
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        weights_path = self._active_weights_path()
        stats_path = self._active_stats_path()
        try:
            if weights_path.exists():
                shutil.copy2(weights_path, weights_path.with_name(f"weights_{self.active_model_id}_before_reset_{stamp}.npz"))
            if stats_path.exists():
                shutil.copy2(stats_path, stats_path.with_name(f"training_stats_before_reset_{stamp}.json"))
            model_width, model_depth = ARCH_PROFILES.get(self.model_profile.get(), (self.policy.width, self.policy.depth))
            self.policy = ActorCritic(seed=random.randint(0, 2**31 - 1), width=model_width, depth=model_depth,
                                      device_mode="cpu" if FORCE_CPU else "auto", architecture=self.policy.architecture)
            self.policy.allow_gpu_inference = True
            self.policy.inference_mode = "gpu"
            self.policy.allow_gpu_inference = False
            self.policy.inference_mode = "cpu"
            self._apply_game_inference_mode()
            self.policy.save(weights_path)
            self.policy_cache[self.active_model_id] = self.policy
            write_model_meta(self.active_model_id, {"width": model_width, "depth": model_depth,
                "architecture": self.policy.architecture, "parameters": self.policy.count_params(), "episodes": 0, "steps": 0, "trained": False,
                "last_opponent": "reset", "description": "Веса сброшены к случайной инициализации"})
            fresh_stats = {
                "episodes": 0, "wins": 0, "losses": 0, "draws": 0,
                "win_rate_20": 0.0, "mean_reward_20": 0.0, "last_loss": 0.0,
                "steps": 0, "steps_per_sec": 0.0,
                "message": "Веса сброшены. Готова новая сессия обучения.",
                "blue_n": self.blue_n.get(), "red_n": self.red_n.get(),
                "blue_kills_20": 0.0, "red_kills_20": 0.0,
                "blue_survivors_20": 0.0, "red_survivors_20": 0.0,
            }
            _atomic_json(stats_path, fresh_stats)
            self.policy_loaded = False
            self.model_label.set("Модель: веса сброшены, новая инициализация")
            self._update_model_details()
            self._cancel_reset_weights()
            self._update_controls()
            self.messagebox.showinfo("Веса сброшены", "Создана новая случайная модель.\nТекущий бой и пауза не изменены.\nСтарые веса и статистика сохранены в резервных копиях.")
        except Exception as exc:
            self.messagebox.showerror("Ошибка сброса", str(exc))

    def _adjust_count(self, variable, delta):
        # Team composition is a setup value for the NEXT match; never mutate the
        # active Arena, pause flag, simulation clock, or accumulated physics time.
        variable.set(int(clamp(variable.get() + delta, 1, 5)))
        self._update_composition_note()
        self._update_controls()

    def _adjust_workers(self, delta):
        self.train_workers.set(int(clamp(self.train_workers.get() + delta, 1, 32)))
        if hasattr(self, "train_workers_text"):
            self.train_workers_text.set(str(self.train_workers.get()))

    def _adjust_gpu_envs(self, delta):
        self.train_gpu_envs.set(int(clamp(self.train_gpu_envs.get() + delta, 1, GPU_MAX_ARENAS)))
        if hasattr(self, "train_gpu_envs_text"):
            self.train_gpu_envs_text.set(str(self.train_gpu_envs.get()))

    def _adjust_training_blue_n(self, delta):
        self.train_blue_n.set(int(clamp(self.train_blue_n.get() + delta, 1, 5)))

    def _adjust_training_red_n(self, delta):
        self.train_red_n.set(int(clamp(self.train_red_n.get() + delta, 1, 5)))

    def _change_zoom(self, factor):
        self.view_zoom = float(clamp(self.view_zoom * factor, 0.25, 2.6))
        self._scene_dirty = True

    def _reset_zoom(self):
        self.view_zoom = 1.0
        self._scene_dirty = True

    def _zoom_all(self):
        self.view_zoom = CAMERA_BASE_WIDTH / WORLD_W
        self._scene_dirty = True

    def _mousewheel_zoom(self, event):
        self._change_zoom(1.15 if event.delta > 0 else 1 / 1.15)
        return "break"

    def _new_match(self):
        """Explicitly create a fresh match using the selected next-match composition."""
        if not hasattr(self, "arena"):
            return
        if self.trainer and self.trainer.is_alive():
            self.training_notice = "Новый бой в окне игры недоступен во время обучения. Останови обучение первым."
            self._update_controls()
            return
        self.match_blue_ai = self.blue_ai.get()
        self.match_red_ai = self.red_ai.get()
        self.arena.reset(self.blue_n.get(), self.red_n.get(), seed=random.randint(1, 10**9))
        self._test_battle_active = False
        self._visible_test_horizon = None
        self._scene_dirty = True
        self.camera_x, self.camera_y = WORLD_W * 0.5, WORLD_H * 0.5
        self.accumulator = 0.0
        self.held_actions = {}
        self.last_wall = time.perf_counter()
        self.episode_label.set("Матч начат")
        self.mode = "play"
        self.mode_label.set("РЕЖИМ: ИГРА")
        self.paused = False
        self._update_composition_note()
        self._update_controls()

    def _check_in_battle(self):
        """Stop training safely, then launch the selected checkpoint in the visible arena."""
        try:
            import torch
            if not torch.cuda.is_available():
                self.messagebox.showerror("CUDA недоступна", "По требованию GPU-only режима бой не будет запускать нейросеть на CPU. Проверь PyTorch CUDA и драйвер NVIDIA.")
                return
        except Exception as exc:
            self.messagebox.showerror("Не удалось проверить CUDA", str(exc))
            return
        if self.trainer and self.trainer.is_alive():
            self._pending_battle_test = True
            self.training_notice = "Останавливаю обучение и сохраняю checkpoint. Проверочный бой запустится после безопасного завершения PPO."
            try:
                self.trainer.stop()
            except Exception:
                self._pending_battle_test = False
                raise
            if hasattr(self, "test_battle_button"):
                self.test_battle_button.configure(text="ОСТАНОВКА · ЗАПУСК БОЯ…", state="disabled")
            self.paused = True
            self.accumulator = 0.0
            self._update_controls()
            return
        self._launch_battle_test()

    def _launch_battle_test(self):
        global ACTION_PRIORS_MODE
        if not self._pending_battle_test and self.trainer and self.trainer.is_alive():
            return
        if self.trainer and self.trainer.is_alive():
            self._pending_battle_test = True
            return
        self._pending_battle_test = False
        model_id = str(self.active_model_id)
        checkpoint = model_weights_path(model_id)
        try:
            if hasattr(self, "training_priors_mode"):
                priors_choice = self.training_priors_mode.get()
                if priors_choice == "По профилю модели":
                    active_meta = read_model_meta(model_id)
                    profiled = str(active_meta.get("default_priors_mode", "full" if model_id == DEFAULT_MODEL_ID else "soft"))
                    ACTION_PRIORS_MODE = profiled if profiled in {"full", "soft", "off"} else "soft"
                else:
                    ACTION_PRIORS_MODE = {"Мягкие": "soft", "Жёсткие (старые)": "full", "Выключены": "off"}.get(priors_choice, ACTION_PRIORS_MODE)
            policy = self.policy_cache.get(model_id, self.policy)
            if checkpoint.exists():
                policy.load(checkpoint)
            policy.allow_gpu_inference = True
            policy.require_gpu_inference = True
            policy.inference_mode = "gpu"
            policy._inference_params = None
            policy._inference_device = None
            self.policy = policy
            self.policy_cache[model_id] = policy
            blue_display = self._display_for_model_id(model_id)
            opponent_choice = str(self.training_opponent.get())
            if opponent_choice == SELFPLAY_LABEL or opponent_choice.startswith("Self-play"):
                red_display = blue_display
            elif opponent_choice == HEURISTIC_LABEL:
                red_display = HEURISTIC_LABEL
            else:
                red_id = self._model_id_for_display(opponent_choice)
                red_display = self._display_for_model_id(red_id) if red_id else HEURISTIC_LABEL
                if red_id and not model_weights_path(red_id).exists():
                    red_display = HEURISTIC_LABEL
            self.blue_ai.set(blue_display)
            self.red_ai.set(red_display)
            if hasattr(self, "blue_ai_combo"):
                self.blue_ai_combo.set(blue_display)
            if hasattr(self, "red_ai_combo"):
                self.red_ai_combo.set(red_display)
            self._apply_game_inference_mode()
            self._switch_ui_tab("battle")
            self._new_match()
            self._visible_test_horizon = parse_decision_horizon(self.train_horizon.get(), 2700)
            self._test_battle_start_step = int(self.arena.step_count)
            self._test_battle_active = True
            self.episode_label.set(f"Проверочный бой: {blue_display} против {red_display} · лимит {self._visible_test_horizon} решений")
            self.training_notice = (f"Проверочный бой запущен на GPU inference с лимитом {self._visible_test_horizon} решений; "
                                    "по достижении лимита бой остановится без объявления победителя. Физика видимой арены CPU.")
        except Exception as exc:
            self.messagebox.showerror("Не удалось запустить проверочный бой", str(exc))
        finally:
            if hasattr(self, "test_battle_button"):
                self.test_battle_button.configure(text="⚔  ПРОВЕРИТЬ В БОЮ", state="normal")

    def _stop_everything(self):
        """One prominent stop: freeze the match and request a safe training stop."""
        self.paused = True
        self._scene_dirty = True
        self.accumulator = 0.0
        if self.trainer and self.trainer.is_alive():
            self.trainer.stop()
            self.train_button.configure(text="ОСТАНОВКА…", state="disabled")
            self.training_notice = "Останавливаем сбор опыта, завершаем безопасный чекпоинт и сохраняем веса."
            self.stop_all_button.configure(text="■  ОСТАНОВКА…")
        else:
            self.mode = "play"
            self.mode_label.set("РЕЖИМ: ИГРА / ПАУЗА")
            self.training_notice = "Бой на паузе. «ИГРА» / «Продолжить бой» продолжает матч; «Новый бой» сбрасывает его."
        self._update_controls()
        self._draw()
        self._update_stats()

    def _configure_policy_profile(self):
        width, depth = ARCH_PROFILES.get(self.model_profile.get(), (1536, 4))
        if self.policy.width == width and self.policy.depth == depth:
            return
        # Never destroy a successful checkpoint merely to experiment with scale.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = model_dir_for(self.active_model_id) / "checkpoints"
        backup_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.policy.save(backup_dir / f"weights_{self.active_model_id}_before_profile_switch_{self.policy.width}x{self.policy.depth}_{stamp}.npz")
        except Exception:
            pass
        old = self.policy
        new = ActorCritic(seed=random.randint(0, 2**31 - 1), width=width, depth=depth,
                          device_mode="cpu" if FORCE_CPU else "auto", architecture=self.policy.architecture)
        new.allow_gpu_inference = True
        new.require_gpu_inference = True
        new.inference_mode = "gpu"
        # Preserve the legacy/base policy exactly; only carry over large layers whose
        # dimensions match. New residual heads remain zero where shape changed.
        for key, arr in new.params.items():
            if key in old.params and old.params[key].shape == arr.shape:
                arr[...] = old.params[key]
        self.policy = new
        self.policy_cache[self.active_model_id] = self.policy
        self._apply_game_inference_mode()
        self.policy_loaded = True
        self.model_label.set(f"Модель: перенесена база, профиль {width}×{depth}")
        self.policy.save(self._active_weights_path())
        write_model_meta(self.active_model_id, {"width": width, "depth": depth, "architecture": self.policy.architecture,
                                                  "parameters": self.policy.count_params()})
        self._update_model_details()

    def _enter_play(self):
        if self.trainer and self.trainer.is_alive():
            self.messagebox.showinfo("Обучение продолжается", "Сначала останови обучение, затем переходи в игру.")
            return
        # Return to the existing arena; the explicit NEW MATCH button is the only
        # action that resets ships, match time, or applies a changed composition.
        self.mode = "play"
        self._switch_ui_tab("battle")
        self._scene_dirty = True
        self.mode_label.set("РЕЖИМ: ИГРА" if not self.arena.round_over else "РЕЖИМ: ИГРА / БОЙ ЗАВЕРШЁН")
        self.paused = False
        self.accumulator = 0.0
        self.last_wall = time.perf_counter()
        self._update_controls()

    def _toggle_training(self):
        global ACTION_PRIORS_MODE, TRAINING_REWARD_MODE, TRAINING_DESIRED_RANGE
        self._switch_ui_tab("training")
        if self.trainer and self.trainer.is_alive():
            self.trainer.stop()
            self.train_button.configure(text="ОСТАНОВКА…", state="disabled")
            self.training_notice = "Останавливаем сбор опыта и сохраняем последний полный пакет весов..."
            self.paused = True
            self.accumulator = 0.0
            self._update_controls()
            return
        try:
            self._configure_policy_profile()
            active_meta = read_model_meta(self.active_model_id)
            reward_choice = self.training_reward_mode.get()
            priors_choice = self.training_priors_mode.get()
            if reward_choice == "По профилю модели":
                resolved_reward_mode = str(active_meta.get("default_reward_mode", "baseline"))
            else:
                resolved_reward_mode = "team" if reward_choice == "Командная" else "baseline"
            if priors_choice == "По профилю модели":
                resolved_priors_mode = str(active_meta.get("default_priors_mode", "full" if self.active_model_id == DEFAULT_MODEL_ID else "soft"))
            else:
                resolved_priors_mode = {"Мягкие": "soft", "Жёсткие (старые)": "full", "Выключены": "off"}.get(priors_choice, "soft")
            if resolved_reward_mode not in {"baseline", "team"}:
                resolved_reward_mode = "baseline"
            if resolved_priors_mode not in {"full", "soft", "off"}:
                resolved_priors_mode = "soft"
            ACTION_PRIORS_MODE = resolved_priors_mode
            TRAINING_REWARD_MODE = resolved_reward_mode
            try:
                range_choice = str(self.training_desired_range.get()).strip()
                TRAINING_DESIRED_RANGE = (0.0 if range_choice == "Без целевой дистанции"
                                          else float(clamp(float(range_choice), 150.0, 1000.0)))
            except (TypeError, ValueError):
                TRAINING_DESIRED_RANGE = 480.0
            max_decisions = parse_decision_horizon(self.train_horizon.get(), 2700)
            opponent_choice = self.training_opponent.get()
            opponent_path = None
            if opponent_choice == SELFPLAY_LABEL or opponent_choice.startswith("Self-play"):
                opponent_mode = "selfplay"
            elif opponent_choice == HEURISTIC_LABEL:
                opponent_mode = "heuristic"
            else:
                opponent_id = self._model_id_for_display(opponent_choice)
                if opponent_id is None:
                    opponent_mode = "heuristic"
                else:
                    opponent_mode = "model"
                    opponent_path = model_weights_path(opponent_id)
                    if not opponent_path.exists():
                        self.messagebox.showerror("Нет весов противника", f"Не найден файл весов для {opponent_choice}:\n{opponent_path}")
                        return
            device_mode = "cpu" if FORCE_CPU else ("cuda" if self.training_device.get().startswith("CUDA обязательно") else ("cpu" if self.training_device.get().startswith("CPU") else "auto"))
            rollout_choice = self.training_rollout_mode.get()
            auto_rollout_balance = (not FORCE_CPU and rollout_choice.startswith("Авто: CPU/GPU"))
            hybrid_gpu = (not FORCE_CPU and (rollout_choice.startswith("Гибрид") or auto_rollout_balance))
            gpu_only = (not FORCE_CPU and rollout_choice.startswith("GPU: параллельные арены"))
            rollout_gpu = (not FORCE_CPU and rollout_choice.startswith("GPU: действия"))
            auto_fallback_note = ""
            if hybrid_gpu or gpu_only:
                import torch
                if not torch.cuda.is_available():
                    if auto_rollout_balance:
                        hybrid_gpu = False
                        auto_rollout_balance = False
                        if device_mode == "auto":
                            device_mode = "cpu"
                        auto_fallback_note = "CUDA недоступна; авто-режим начал с CPU-rollout."
                    else:
                        self.messagebox.showerror("CUDA недоступна", "Этот режим требует CUDA-видеокарту и CUDA-сборку PyTorch. Выбери «CPU: параллельные арены» либо перезапусти программу через BAT в режиме NVIDIA CUDA.")
                        return
                # PPO may intentionally run on CPU while tensor physics runs on CUDA.
                # This is slower in many cases, but useful as a benchmark and can free
                # VRAM for large rollout batches. CPU workers are idle during the PPO phase.
                if opponent_mode == "model":
                    self.messagebox.showerror("Режим противника не поддерживается", "Тензорная GPU-арена на этом этапе поддерживает эвристического противника и self-play. Для отдельной сохранённой модели выбери «CPU: параллельные арены».")
                    return
            effective_workers = 1 if (rollout_gpu or gpu_only) and not hybrid_gpu else self.train_workers.get()
            active_meta = read_model_meta(self.active_model_id)
            active_name = str(active_meta.get("name", self.active_model_id))
            fixed_train_blue = int(clamp(self.train_blue_n.get(), 1, 5))
            fixed_train_red = int(clamp(self.train_red_n.get(), 1, 5))
            gpu_budget_value = int(self.train_record_budget.get().replace(" ", ""))
            self.trainer = Trainer(self.policy, fixed_train_blue, fixed_train_red,
                                   bool(self.random_compositions.get()), model_dir_for(self.active_model_id),
                                   workers=effective_workers, opponent_mode=opponent_mode,
                                   device_mode=device_mode, batch_size=int(self.train_batch.get()),
                                   epochs=int(self.train_epochs.get()), rollout_gpu_inference=rollout_gpu,
                                   hybrid_gpu_rollout=hybrid_gpu, gpu_batch_envs=int(self.train_gpu_envs.get()),
                                   gpu_only_rollout=gpu_only, auto_rollout_balance=auto_rollout_balance,
                                   model_id=self.active_model_id, model_name=active_name, opponent_path=opponent_path,
                                   opponent_name=opponent_choice, max_decisions=max_decisions,
                                   action_priors_mode=resolved_priors_mode, reward_mode=resolved_reward_mode,
                                   desired_range=TRAINING_DESIRED_RANGE,
                                   gpu_record_budget=gpu_budget_value)
        except Exception as exc:
            self.messagebox.showerror("Не удалось подготовить обучение", str(exc))
            return
        write_model_meta(self.active_model_id, {
            "architecture": self.policy.architecture,
            "last_training_reward_mode": resolved_reward_mode,
            "last_training_priors_mode": resolved_priors_mode,
            "last_training_desired_range": TRAINING_DESIRED_RANGE,
            "last_training_horizon": max_decisions,
            "last_training_blue_n": fixed_train_blue, "last_training_red_n": fixed_train_red,
            "last_gpu_record_budget": gpu_budget_value,
        })
        # Snapshot the exact selected starting model before training; the Trainer works on its own policy copy.
        self.policy.save(self._active_weights_path())
        self.policy_cache[self.active_model_id] = self.policy
        self.trainer.start()
        self.mode = "training"
        self.paused = True
        self.accumulator = 0.0
        self.mode_label.set("РЕЖИМ: ОБУЧЕНИЕ (без отрисовки боёв)")
        opponent_label = opponent_choice
        rollout_desc = (f"авто-выбор CPU/GPU: стартовый тест {effective_workers} CPU-арен + {int(self.train_gpu_envs.get())} GPU-арен"
                        if auto_rollout_balance else f"GPU-only: {int(self.train_gpu_envs.get())} GPU-арен, CPU-физика выключена"
                        if gpu_only else f"гибрид: {effective_workers} CPU-арен + {int(self.train_gpu_envs.get())} GPU-арен с тензорной физикой"
                        if hybrid_gpu else "GPU-действия + CPU-физика, 1 арена" if rollout_gpu else "параллельный CPU rollout")
        cpu_env_description = 0 if gpu_only else effective_workers
        self.training_notice = (f"Обучается {active_name} [{self.active_model_id}]: {self.policy.count_params():,} параметров; "
                                f"сеть: {network_architecture_label(self.policy.architecture)}; PPO {device_mode}; "
                                f"состав обучения: {'случайный 1–5' if bool(self.random_compositions.get()) else f'{fixed_train_blue}v{fixed_train_red}'}; GPU record budget: {gpu_budget_value:,}; "
                                f"награда: {resolved_reward_mode}; priors: {resolved_priors_mode}; горизонт: {max_decisions}; "                                f"противник: {opponent_label}; сбор {rollout_desc}; CPU-сред: {cpu_env_description}."
                                + (f" {auto_fallback_note}" if auto_fallback_note else ""))
        self._update_model_details()
        self._update_controls()

    def _settings_token(self, display: str) -> str:
        if display == HEURISTIC_LABEL:
            return "heuristic"
        if display == SELFPLAY_LABEL:
            return "selfplay"
        return self._model_id_for_display(display) or DEFAULT_MODEL_ID

    def _settings_display(self, token: str, allow_selfplay: bool = False) -> str:
        token = str(token or "")
        if token == "heuristic":
            return HEURISTIC_LABEL
        if token == "selfplay" and allow_selfplay:
            return SELFPLAY_LABEL
        if token in self.model_id_to_display:
            return self.model_id_to_display[token]
        return HEURISTIC_LABEL if allow_selfplay is False else HEURISTIC_LABEL

    def _collect_settings_payload(self) -> dict:
        """Capture every selectable combat/training setting; weights remain separate."""
        return {
            "active_model_id": self.active_model_id,
            "models": {
                "active": self.active_model_id,
                "blue": self._settings_token(self.blue_ai.get()),
                "red": self._settings_token(self.red_ai.get()),
                "training_opponent": self._settings_token(self.training_opponent.get()),
            },
            "battle": {
                "blue_n": int(clamp(self.blue_n.get(), 1, 5)),
                "red_n": int(clamp(self.red_n.get(), 1, 5)),
                "inference_mode": self.game_inference_mode.get(),
                "speed": float(clamp(self.speed, min(self.speed_values), max(self.speed_values))),
                "view_zoom": float(clamp(self.view_zoom, 0.25, 2.6)),
            },
            "training": {
                "model_profile": self.model_profile.get(),
                "device": self.training_device.get(),
                "rollout_mode": self.training_rollout_mode.get(),
                "cpu_workers": int(clamp(self.train_workers.get(), 1, 32)),
                "gpu_arenas": int(clamp(self.train_gpu_envs.get(), 1, GPU_MAX_ARENAS)),
                "blue_n": int(clamp(self.train_blue_n.get(), 1, 5)),
                "red_n": int(clamp(self.train_red_n.get(), 1, 5)),
                "gpu_record_budget": self.train_record_budget.get(),
                "ppo_batch": self.train_batch.get(),
                "ppo_epochs": self.train_epochs.get(),
                "random_compositions": bool(self.random_compositions.get()),
                "reward_mode": self.training_reward_mode.get(),
                "action_priors_mode": self.training_priors_mode.get(),
                "desired_range": self.training_desired_range.get(),
                "max_decisions": self.train_horizon.get(),
            },
            "variant": {"new_model_name": self.new_model_name_var.get()},
            "window": {"geometry": str(self.root.geometry())},
        }

    def _save_settings(self, quiet: bool = False):
        try:
            payload = self._collect_settings_payload()
            save_app_config(APP_CONFIG_PATH, payload)
            self.training_notice = f"Настройки сохранены: {APP_CONFIG_PATH.name}"
            if not quiet:
                self.messagebox.showinfo(
                    "Настройки сохранены",
                    f"Сохранены выборы моделей/вариантов, параметры боя и обучения.\n\nФайл: {APP_CONFIG_PATH}\n\nВеса и статистика моделей не копируются — они остаются в GravitrueX/models/.")
            return True
        except Exception as exc:
            if not quiet:
                self.messagebox.showerror("Не удалось сохранить настройки", str(exc))
            return False

    def _load_settings(self, quiet: bool = False, startup: bool = False):
        if not startup and self.trainer and self.trainer.is_alive():
            if not quiet:
                self.messagebox.showwarning("Обучение запущено", "Останови обучение перед загрузкой настроек. Это не позволит случайно менять активный профиль во время PPO.")
            return False
        settings = load_app_config(APP_CONFIG_PATH)
        if settings is None:
            if not quiet:
                self.messagebox.showwarning(
                    "Конфиг не найден или повреждён",
                    f"Не удалось прочитать совместимый конфиг: {APP_CONFIG_PATH}\n\nПроверь JSON. Текущие параметры не изменены.")
            return False
        try:
            self._refresh_model_catalog()
            model_settings = settings.get("models", {}) if isinstance(settings.get("models", {}), dict) else {}
            active_id = str(model_settings.get("active", settings.get("active_model_id", DEFAULT_MODEL_ID)))
            if active_id not in self.model_id_to_display:
                active_id = DEFAULT_MODEL_ID
            self.active_model_var.set(self._display_for_model_id(active_id))
            # This loads the selected variant's own weights/metadata from models/<id>.
            self._active_model_changed()

            battle = settings.get("battle", {}) if isinstance(settings.get("battle", {}), dict) else {}
            training = settings.get("training", {}) if isinstance(settings.get("training", {}), dict) else {}
            variant = settings.get("variant", {}) if isinstance(settings.get("variant", {}), dict) else {}
            window = settings.get("window", {}) if isinstance(settings.get("window", {}), dict) else {}

            def safe_int(value, fallback, low, high):
                try:
                    return int(clamp(int(value), low, high))
                except (TypeError, ValueError, OverflowError):
                    return fallback

            def set_choice(var, value, choices, fallback):
                candidate = str(value or "")
                var.set(candidate if candidate in choices else fallback)

            self.blue_n.set(safe_int(battle.get("blue_n", 1), 1, 1, 5))
            self.red_n.set(safe_int(battle.get("red_n", 1), 1, 1, 5))
            self.train_workers.set(safe_int(training.get("cpu_workers", self.train_workers.get()), self.train_workers.get(), 1, 32))
            self.train_workers_text.set(str(self.train_workers.get()))
            self.train_gpu_envs.set(safe_int(training.get("gpu_arenas", GPU_MAX_ARENAS), GPU_MAX_ARENAS, 1, GPU_MAX_ARENAS))
            self.train_gpu_envs_text.set(str(self.train_gpu_envs.get()))
            self.train_blue_n.set(safe_int(training.get("blue_n", 1), 1, 1, 5))
            self.train_red_n.set(safe_int(training.get("red_n", 1), 1, 1, 5))
            set_choice(self.train_record_budget, training.get("gpu_record_budget", "720 000"),
                       GPU_RECORD_BUDGET_CHOICES, "720 000")
            profile = str(training.get("model_profile", self.model_profile.get()))
            self.model_profile.set(profile if profile in ARCH_PROFILES else self.model_profile.get())
            set_choice(self.train_batch, training.get("ppo_batch", "4096"), ("512", "1024", "2048", "4096", "8192"), "4096")
            set_choice(self.train_epochs, training.get("ppo_epochs", "3"), ("1", "2", "3", "4", "6"), "3")
            self.random_compositions.set(bool(training.get("random_compositions", False)))
            set_choice(self.training_reward_mode, training.get("reward_mode", "По профилю модели"),
                       ("По профилю модели", "Индивидуальная", "Командная"), "По профилю модели")
            set_choice(self.training_priors_mode, training.get("action_priors_mode", "По профилю модели"),
                       ("По профилю модели", "Мягкие", "Жёсткие (старые)", "Выключены"), "По профилю модели")
            set_choice(self.training_desired_range, training.get("desired_range", "480"),
                       ("Без целевой дистанции", "250", "350", "480", "600", "750"), "480")
            saved_horizon = str(training.get("max_decisions", "2700 (180 с)"))
            saved_horizon_decisions = parse_decision_horizon(saved_horizon, 2700)
            self.train_horizon.set(min(
                HORIZON_CHOICES,
                key=lambda option: abs(parse_decision_horizon(option, 2700) - saved_horizon_decisions)))

            if FORCE_CPU:
                valid_rollouts = ("CPU: параллельные арены",)
                valid_devices = ("CPU только",)
                valid_inference = ("GPU/CUDA (единственный режим в бою)",)
                self.training_rollout_mode.set("CPU: параллельные арены")
                self.training_device.set("CPU только")
                self.game_inference_mode.set("GPU/CUDA (единственный режим в бою)")
            else:
                valid_rollouts = ("CPU: параллельные арены", "GPU: параллельные арены (CUDA-физика)",
                                  "GPU: действия + PPO (CPU-физика, 1 бой)", "Гибрид: CPU + GPU-параллельные бои",
                                  "Авто: CPU/GPU по скорости")
                valid_devices = ("Авто: CUDA если доступна", "CUDA обязательно", "CPU только")
                valid_inference = ("GPU/CUDA (единственный режим в бою)",)
                set_choice(self.training_rollout_mode, training.get("rollout_mode", self.training_rollout_mode.get()), valid_rollouts, "GPU: параллельные арены (CUDA-физика)")
                set_choice(self.training_device, training.get("device", self.training_device.get()), valid_devices, "Авто: CUDA если доступна")
                set_choice(self.game_inference_mode, "GPU/CUDA (единственный режим в бою)", valid_inference, "GPU/CUDA (единственный режим в бою)")

            self.blue_ai.set(self._settings_display(model_settings.get("blue", active_id), allow_selfplay=False))
            self.red_ai.set(self._settings_display(model_settings.get("red", "heuristic"), allow_selfplay=False))
            self.training_opponent.set(self._settings_display(model_settings.get("training_opponent", "heuristic"), allow_selfplay=True))
            self.new_model_name_var.set(str(variant.get("new_model_name", "Эксперимент"))[:100] or "Эксперимент")
            try:
                speed = float(battle.get("speed", 1.0))
                if not math.isfinite(speed):
                    speed = 1.0
                speed = min(self.speed_values, key=lambda x: abs(x - speed))
                self._set_speed(speed)
            except (TypeError, ValueError):
                self._set_speed(1.0)
            try:
                zoom = float(battle.get("view_zoom", 1.0))
                self.view_zoom = float(clamp(zoom if math.isfinite(zoom) else 1.0, 0.25, 2.6))
            except (TypeError, ValueError):
                self.view_zoom = 1.0
            geometry = str(window.get("geometry", ""))
            if re.fullmatch(r"\d{3,5}x\d{3,5}(?:[+-]\d{1,5}){0,2}", geometry):
                # Keep the user's saved window size where possible, but clamp it to
                # the current screen so old 1550px layouts do not clip the global tabs.
                match = re.match(r"(\d{3,5})x(\d{3,5})", geometry)
                if match:
                    try:
                        sw = int(self.root.winfo_screenwidth())
                        sh = int(self.root.winfo_screenheight())
                        # Old saved configs may carry a small Stage 6.11 window.
                        # Expand it toward a comfortable 1920x1080 desktop layout,
                        # while respecting the actual screen bounds.
                        preferred_w = max(1000, min(1660, sw - 70))
                        preferred_h = max(700, min(920, sh - 130))
                        width = min(max(preferred_w, int(match.group(1))), max(1000, sw - 24))
                        height = min(max(preferred_h, int(match.group(2))), max(650, sh - 72))
                        self.root.geometry(f"{width}x{height}+12+12")
                    except Exception:
                        self.root.geometry("1260x820")
            self._apply_game_inference_mode()
            self._update_composition_note()
            self._update_model_details()
            self._update_controls()
            self._scene_dirty = True
            self.training_notice = f"Настройки загружены из {APP_CONFIG_PATH.name}. Состав бойцов применяется к следующему бою."
            if not quiet:
                self.messagebox.showinfo(
                    "Настройки загружены",
                    f"Применены настройки боя, моделей/вариантов и обучения из:\n{APP_CONFIG_PATH}\n\nТекущий бой не сброшен; выбранные состав и модели применятся после «Новый бой».")
            return True
        except Exception as exc:
            if not quiet:
                self.messagebox.showerror("Ошибка загрузки настроек", f"Конфиг прочитан, но не все параметры удалось применить:\n{exc}")
            return False

    def _save_weights(self):
        if self.trainer and self.trainer.is_alive():
            self.messagebox.showinfo("Обучение", "Во время обучения веса сохраняются после каждого пакета боёв. Останови обучение для ручного сохранения.")
            return
        try:
            path = self._active_weights_path()
            self.policy.save(path)
            write_model_meta(self.active_model_id, {"width": self.policy.width, "depth": self.policy.depth,
                              "parameters": self.policy.count_params()})
            self._update_model_details()
            self.messagebox.showinfo("Сохранено", f"Веса записаны в:\n{path}")
        except Exception as e:
            self.messagebox.showerror("Ошибка", str(e))

    def _load_weights(self):
        if self.trainer and self.trainer.is_alive():
            self.messagebox.showinfo("Обучение", "Сначала останови обучение, чтобы не читать веса во время их обновления.")
            return
        path = self._active_weights_path()
        if self.policy.load(path):
            meta = read_model_meta(self.active_model_id)
            self.policy_loaded = bool(meta.get("trained", int(meta.get("episodes", 0) or 0) > 0))
            self.policy_cache[self.active_model_id] = self.policy
            self.model_label.set(f"Модель: загружены веса {self.active_model_id}")
            self._update_model_details()
            self.messagebox.showinfo("Загружено", f"Загружены веса:\n{path}")
        else:
            self.messagebox.showwarning("Нет весов", "Файл весов не найден или не совместим. Начни обучение.")

    def _set_speed(self, value: float):
        self.speed = float(value)
        self.speed_label.set(f"Скорость симуляции: {self.speed:g}×; обучение отдельно")

    def _toggle_pause(self):
        if self.mode == "play" and not self.arena.round_over:
            was_running = not self.paused
            transition_time = time.perf_counter()
            if was_running and self._fps_frames > 0:
                sample_elapsed = transition_time - self._fps_window_start
                if sample_elapsed > 0.10:
                    self.display_fps = self._fps_frames / sample_elapsed
            self.paused = not self.paused
            self._scene_dirty = True
            self.accumulator = 0.0 if self.paused else self.accumulator
            # FPS is measured only while the scene is actively being rendered.
            # Reset the sample window at the transition so idle time is not counted.
            self._fps_frames = 0
            self._fps_window_start = transition_time
            self.mode_label.set("РЕЖИМ: ИГРА / ПАУЗА" if self.paused else "РЕЖИМ: ИГРА")
            self._update_controls()

    def _loop(self):
        now = time.perf_counter()
        wall_dt = min(0.12, now - self.last_wall)
        self.last_wall = now
        # While paused, don't count idle/UI polling ticks as rendered FPS.
        # During training the UI intentionally redraws at ~10 Hz.
        measure_fps = (self.ui_tab == "training") or (self.mode in ("training", "training_error")) or (self.mode == "play" and not self.paused)
        if measure_fps:
            self._fps_frames += 1
            fps_elapsed = now - self._fps_window_start
            if fps_elapsed >= 0.75:
                self.display_fps = self._fps_frames / max(0.001, fps_elapsed)
                self._fps_frames = 0
                self._fps_window_start = now
        else:
            self._fps_frames = 0
            self._fps_window_start = now
        ticks_run = 0
        round_over_before = self.arena.round_over
        if self.mode == "play" and not self.paused and not self.arena.round_over:
            # Time scaling controls how many FIXED physics ticks run per display frame.
            # The neural policy never receives this multiplier as an observation.
            self.accumulator += wall_dt * self.speed
            max_ticks = 300
            tick_count = 0
            while self.accumulator >= DT and tick_count < max_ticks and not self.arena.round_over:
                blue_alive = self.arena.alive_ships(0)
                if self.arena.step_count % ACTION_REPEAT == 0 or not hasattr(self, "held_actions"):
                    held: Dict[int, Action] = {}
                    neural_groups: Dict[str, List[Ship]] = {}
                    team_selectors = ((0, getattr(self, "match_blue_ai", self.blue_ai.get())),
                                      (1, getattr(self, "match_red_ai", self.red_ai.get())))
                    for team, selector in team_selectors:
                        team_ships = self.arena.alive_ships(team)
                        if selector == HEURISTIC_LABEL:
                            for ship in team_ships:
                                held[ship.sid] = self.arena.heuristic_action(ship)
                        else:
                            model_id = self._model_id_for_display(selector)
                            if model_id is None:
                                for ship in team_ships:
                                    held[ship.sid] = self.arena.heuristic_action(ship)
                            else:
                                neural_groups.setdefault(model_id, []).extend(team_ships)
                    for model_id, ships_for_model in neural_groups.items():
                        model_policy = self._policy_for_model_id(model_id)
                        model_policy.inference_mode = "gpu"
                        model_policy.allow_gpu_inference = True
                        model_policy.require_gpu_inference = True
                        observations = [self.arena.observation(ship) for ship in ships_for_model]
                        try:
                            actions, _, _, _ = model_policy.predict_batch(observations, deterministic=True)
                        except Exception as exc:
                            self.paused = True
                            self.accumulator = 0.0
                            self.held_actions = {}
                            message = f"GPU-only ИИ не смог выполнить inference: {type(exc).__name__}: {exc}"
                            self.mode_label.set("РЕЖИМ: GPU ИИ ОШИБКА / ПАУЗА")
                            self.training_notice = message
                            self._scene_dirty = True
                            self._update_controls()
                            try:
                                self.messagebox.showerror("Ошибка GPU inference", message + "\n\nCPU fallback отключён по настройке.")
                            except Exception:
                                pass
                            break
                        held.update({ship.sid: actions[i] for i, ship in enumerate(ships_for_model)})
                    if self.paused:
                        break
                    self.held_actions = held
                    for ship in self.arena.alive_ships():
                        ship.last_action = held.get(ship.sid, Action())
                self.arena.physics_tick(self.held_actions)
                self.accumulator -= DT
                tick_count += 1
                if self._test_battle_active and self._visible_test_horizon is not None:
                    if visible_test_horizon_reached(self._test_battle_start_step,
                                                    int(self.arena.step_count),
                                                    int(self._visible_test_horizon)) and not self.arena.round_over:
                        self._test_battle_active = False
                        self.paused = True
                        self.accumulator = 0.0
                        self.training_notice = (f"Проверочный бой остановлен по лимиту {self._visible_test_horizon} решений. "
                                                "Это не победа и не поражение: команды ещё были живы.")
                        self.episode_label.set(f"Тест остановлен по лимиту: {self._visible_test_horizon} решений")
                        self._update_controls()
                        self._scene_dirty = True
                        break
            ticks_run = tick_count
            if tick_count:
                self._scene_dirty = True
        self.ticks_last_frame = ticks_run
        physics_elapsed = now - self._physics_window_start
        if physics_elapsed >= 0.75:
            self.game_steps_per_sec = (self.arena.step_count - self._physics_steps_window_start) / max(0.001, physics_elapsed)
            self._physics_window_start = now
            self._physics_steps_window_start = self.arena.step_count
        if self.arena.round_over != round_over_before:
            self._update_controls()
        if self.trainer and not self.trainer.is_alive() and self.mode == "training":
            failed = bool(getattr(self.trainer, "fatal_error", None))
            self.mode = "training_error" if failed else "training_done"
            self.paused = True
            self._scene_dirty = True
            self._background_dirty = True
            self.accumulator = 0.0
            if failed:
                self.mode_label.set("РЕЖИМ: ОБУЧЕНИЕ ПРЕРВАНО ОШИБКОЙ")
                self.training_notice = (
                    f"Обучение аварийно завершилось на этапе: {self.trainer.failed_phase}. "
                    f"Причина: {self.trainer.fatal_error}. Полный traceback записан в "
                    f"{self.trainer.crash_report_path}"
                )
            else:
                self.mode_label.set("РЕЖИМ: ОБУЧЕНИЕ ОСТАНОВЛЕНО / БОЙ НА ПАУЗЕ")
            if self.trainer and self.policy.load(self.trainer.save_path):
                self.policy_loaded = True
                self.policy_cache[self.active_model_id] = self.policy
                self.model_label.set(f"Модель: последние обученные веса загружены [{self.active_model_id}]")
            self._update_model_details()
            self._update_controls()
            if self._pending_battle_test:
                self.root.after(0, self._launch_battle_test)
        # While paused, Tk only polls state at 10 Hz; the static scene remains cached.
        # Input events still dispatch immediately through Tk's event loop.
        if self.ui_tab == "training" or self.mode in ("training", "training_error"):
            self._draw()
        elif self.paused:
            if self._scene_dirty:
                self._draw()
            elif now - self._last_paused_hud_refresh >= 0.75:
                self._refresh_paused_hud()
                self._update_stats()
                self._last_stats_update = now
        else:
            self._draw()
        # Target 60 Hz while actively playing; use 10 Hz for training UI and paused idle.
        elapsed_ms = (time.perf_counter() - now) * 1000.0
        if self.ui_tab == "training" or self.mode in ("training", "training_error") or self.paused:
            target_ms = 1000.0 / TRAINING_UI_FPS_TARGET
        else:
            target_ms = DISPLAY_FRAME_MS
        delay_ms = max(1, int(target_ms - elapsed_ms))
        self.root.after(delay_ms, self._loop)

    def _draw_static_background(self, width: int, height: int):
        """Cache only the flat background; the coordinate grid moves with the world camera."""
        c = self.canvas
        c.delete("background")
        c.create_rectangle(0, 0, width, height, fill=BG, outline=BG, tags=("background",))
        c.tag_lower("background")
        self._background_size = (width, height)
        self._background_dirty = False

    def _draw(self):
        c = self.canvas
        width = max(1, c.winfo_width())
        height = max(1, c.winfo_height())
        if width < 100 or height < 100:
            self._scene_dirty = True
            self._background_dirty = True
            return
        if self.ui_tab == "training":
            c.delete("all")
            c.create_rectangle(0, 0, width, height, fill=BG, outline=BG)
            self._background_dirty = True
            self._draw_training_screen(width, height)
        else:
            if self._background_dirty or self._background_size != (width, height):
                c.delete("all")
                self._draw_static_background(width, height)
            c.delete("dynamic")
            self._draw_arena(width, height)
            c.addtag_withtag("dynamic", "all")
            c.dtag("background", "dynamic")
        now = time.perf_counter()
        if now - self._last_stats_update >= 0.20:
            self._update_stats()
            self._last_stats_update = now
        self._scene_dirty = False

    def _update_camera(self):
        if self.paused:
            return
        live = self.arena.alive_ships()
        if not live:
            return
        anchor = live[0]
        dxs = [wrap_delta(ship.x - anchor.x, WORLD_W) for ship in live]
        dys = [wrap_delta(ship.y - anchor.y, WORLD_H) for ship in live]
        target_x = wrap_pos(anchor.x + sum(dxs) / len(dxs), WORLD_W)
        target_y = wrap_pos(anchor.y + sum(dys) / len(dys), WORLD_H)
        # Smoothly track the group centre without depending on time-scale.
        self.camera_x = wrap_pos(self.camera_x + 0.16 * wrap_delta(target_x - self.camera_x, WORLD_W), WORLD_W)
        self.camera_y = wrap_pos(self.camera_y + 0.16 * wrap_delta(target_y - self.camera_y, WORLD_H), WORLD_H)

    def _screen_points(self, x, y, scale, width, height, margin=30):
        dx = wrap_delta(x - self.camera_x, WORLD_W)
        dy = wrap_delta(y - self.camera_y, WORLD_H)
        out = []
        for ox in (-WORLD_W, 0.0, WORLD_W):
            for oy in (-WORLD_H, 0.0, WORLD_H):
                sx = width * 0.5 + (dx + ox) * scale
                sy = height * 0.5 + (dy + oy) * scale
                if -margin <= sx <= width + margin and -margin <= sy <= height + margin:
                    out.append((sx, sy))
        return out

    def _draw_world_grid(self, width, height, scale):
        """Draw grid lines at fixed world coordinates (not screen pixels)."""
        c = self.canvas
        steps = (50.0, 100.0, 250.0, 500.0, 1000.0)
        step = next((candidate for candidate in steps if candidate * scale >= 46.0), steps[-1])
        cx, cy = width * 0.5, height * 0.5
        for world_x in range(0, int(WORLD_W), int(step)):
            dx = wrap_delta(float(world_x) - self.camera_x, WORLD_W)
            screen_x = cx + dx * scale
            if -1 <= screen_x <= width + 1:
                major = (world_x % 250 == 0)
                c.create_line(screen_x, 0, screen_x, height,
                              fill="#1b3145" if major else GRID, width=1)
        for world_y in range(0, int(WORLD_H), int(step)):
            dy = wrap_delta(float(world_y) - self.camera_y, WORLD_H)
            screen_y = cy + dy * scale
            if -1 <= screen_y <= height + 1:
                major = (world_y % 250 == 0)
                c.create_line(0, screen_y, width, screen_y,
                              fill="#1b3145" if major else GRID, width=1)

    def _draw_training_screen(self, width, height):
        c = self.canvas
        stats = self.trainer.get_stats() if self.trainer else {}
        perf = self.perf.snapshot()
        margin = 24
        c.create_rectangle(margin, margin, width - margin, height - margin,
                           fill="#091726", outline="#27425a", width=1)
        cx = width / 2
        c.create_text(cx, 48, text="ОБУЧЕНИЕ PPO · ТАКТИЧЕСКАЯ ПОЛИТИКА", fill=ACCENT,
                      font=("Segoe UI", 20, "bold"))
        active_name = stats.get("training_model_name", getattr(self.trainer, "model_name", self.active_model_var.get().split(" [", 1)[0]))
        active_id = stats.get("training_model_id", getattr(self.trainer, "model_id", self.active_model_id))
        c.create_text(cx, 82, text=f"Модель: {active_name} [{active_id}] · {stats.get('width', self.policy.width)}×{stats.get('depth', self.policy.depth)} · {self.policy.count_params():,} параметров",
                      fill=TEXT, font=("Segoe UI", 10, "bold"))
        gpu_requested = int(stats.get("gpu_rollout_envs_requested", stats.get("gpu_batch_envs", self.train_gpu_envs.get())) or 0)
        gpu_effective = int(stats.get("gpu_rollout_envs", 0) or 0)
        gpu_budget = int(stats.get("gpu_record_budget", getattr(self.trainer, "gpu_record_budget", 0)) or 0)
        gpu_batch_label = (f"GPU-арены {gpu_effective}/{gpu_requested} фактически/запрошено · budget {gpu_budget:,}"
                           if gpu_effective else f"GPU-батч ещё не завершён · запрос {gpu_requested} арен · budget {gpu_budget:,}")
        rollout_desc = (
            f"Готов к запуску · сбор: {self.training_rollout_mode.get()}"
            if not self.trainer and not stats else
            f"GPU-only физика: {gpu_batch_label}"
            if stats.get("gpu_only_rollout") else
            f"Гибрид: {stats.get('workers', self.train_workers.get())} CPU-процессов + {gpu_batch_label}"
            if stats.get("hybrid_gpu_rollout") else
            "GPU-действия + CPU-физика (1 арена)" if stats.get("rollout_gpu_inference") else
            f"CPU-параллельные арены: {stats.get('workers', self.train_workers.get())} процесс(ов)"
        )
        mode_note = stats.get("auto_rollout_selection_note", "")
        c.create_text(cx, 108, text=rollout_desc + f" · PPO: {stats.get('device', self.training_device.get())} · противник: {stats.get('opponent_name', '—')}",
                      fill=MUTED, font=("Segoe UI", 9), width=max(300, width * 0.9))
        if mode_note:
            c.create_text(cx, 128, text=str(mode_note)[:170], fill="#f1c27d", font=("Segoe UI", 8), width=max(300, width * 0.9))
        elif stats.get("ppo_fallback_reason"):
            c.create_text(cx, 130, text="PPO fallback: " + str(stats["ppo_fallback_reason"])[:140],
                          fill="#f1c27d", font=("Segoe UI", 8), width=max(300, width * 0.9))
        c.create_text(cx, 159, text=f"Эпизоды {stats.get('episodes', 0):,}   ·   физические шаги {stats.get('steps', 0):,}   ·   обновления PPO {stats.get('ppo_updates', 0):,}",
                      fill=TEXT, font=("Consolas", 12))
        c.create_text(cx, 192, text=f"W/L/D последние 20: {stats.get('wins', 0)}/{stats.get('losses', 0)}/{stats.get('draws', 0)}   ·   средняя награда {stats.get('mean_reward_20', 0):+.3f}   ·   loss {stats.get('last_loss', 0):.4f}",
                      fill=ACCENT, font=("Consolas", 10))
        c.create_text(cx, 222, text=f"Убийства B/R: {stats.get('blue_kills_20', 0):.2f}/{stats.get('red_kills_20', 0):.2f}   ·   выжившие B/R: {stats.get('blue_survivors_20', 0):.2f}/{stats.get('red_survivors_20', 0):.2f}",
                      fill=TEXT, font=("Consolas", 10))
        c.create_text(cx, 256, text=f"Производительность: CPU {perf.get('cpu', 0) if perf.get('cpu') is not None else 'н/д'}% · RAM {_fmt_bytes(perf.get('process_ram'))}/{_fmt_bytes(perf.get('ram_total'))} · GPU {str(perf.get('gpu_util')) + '%' if perf.get('gpu_util') is not None else 'н/д'} · VRAM {_fmt_bytes(perf.get('vram_used'))}/{_fmt_bytes(perf.get('vram_total'))}",
                      fill="#8be0ca", font=("Consolas", 9))
        c.create_text(cx, 282, text=f"Сбор: wall {stats.get('rollout_sec_last', 0):.2f} с · PPO {stats.get('ppo_sec_last', 0):.2f} с · mini-batch {stats.get('batch_size', self.train_batch.get())} · эпох {stats.get('epochs_last', self.train_epochs.get())}",
                      fill="#a9c6dc", font=("Consolas", 9))
        # Dedicated, non-overlapping performance breakdown.
        col_l = max(40, width * 0.05)
        col_r = width * 0.52
        c.create_text(col_l, 330, anchor="w", text="ПОСЛЕДНИЙ ЦИКЛ: РАЗДЕЛЬНЫЕ ЭТАПЫ", fill=ACCENT,
                      font=("Segoe UI", 10, "bold"))
        cpu_txt = (f"CPU rollout: {stats.get('cpu_rollout_episodes_last', 0)} эп. · {stats.get('cpu_rollout_records_last', 0):,} записей · "
                   f"{stats.get('cpu_rollout_steps_per_sec', 0):,.0f} шаг/с · {stats.get('cpu_rollout_sec_last', 0):.2f} с")
        gpu_txt = (f"GPU rollout: {stats.get('gpu_rollout_episodes_last', 0)} эп. · {stats.get('gpu_rollout_records_last', 0):,} записей · "
                   f"{stats.get('gpu_rollout_steps_per_sec', 0):,.0f} шаг/с · {stats.get('gpu_rollout_sec_last', 0):.2f} с")
        wait_txt = (f"Ожидание GPU после CPU: {stats.get('cpu_wait_gpu_sec_last', 0):.2f} с · "
                    f"CUDA device: {stats.get('gpu_device_seconds', 0):.2f} с · host/launch/pack: {stats.get('gpu_host_overhead_seconds', 0):.2f} с")
        ppo_txt = f"PPO: {stats.get('ppo_records_last', 0):,} записей · {stats.get('ppo_sec_last', 0):.2f} с · {stats.get('device', 'CPU')}"
        for i, line in enumerate((cpu_txt, gpu_txt, wait_txt, ppo_txt)):
            c.create_text(col_l + 8, 358 + i * 24, anchor="w", text=line, fill=TEXT if i in (0, 1) else MUTED,
                          font=("Consolas", 9), width=max(200, width * 0.88))
        # The last two cycles make stalls and source imbalance immediately visible.
        c.create_text(col_l, 470, anchor="w", text="ДВА ПОСЛЕДНИХ ЦИКЛА · CPU / GPU / ОЖИДАНИЕ / PPO", fill=ACCENT,
                      font=("Segoe UI", 10, "bold"))
        history = list(stats.get("last_two_cycles", []))
        if not history:
            history = self.trainer._stage_history[-2:] if self.trainer and hasattr(self.trainer, "_stage_history") else []
        if not history:
            c.create_text(col_l + 8, 498, anchor="w", text="Первый цикл ещё не завершён.", fill=MUTED, font=("Consolas", 9))
        for i, item in enumerate(reversed(history[-2:])):
            line = (f"{item.get('time', '')[-8:]}  CPU {item.get('cpu_steps_per_sec', 0):,.0f}/с ({item.get('cpu_seconds', 0):.1f}с)  ·  "
                    f"GPU {item.get('gpu_steps_per_sec', 0):,.0f}/с ({item.get('gpu_seconds', 0):.1f}с)  ·  "
                    f"wait {item.get('cpu_wait_gpu_seconds', 0):.1f}с  ·  PPO {item.get('ppo_seconds', 0):.1f}с")
            c.create_text(col_l + 8, 498 + i * 24, anchor="w", text=line, fill=TEXT, font=("Consolas", 9),
                          width=max(200, width * 0.88))
        is_error = self.mode == "training_error" and bool(getattr(self.trainer, "fatal_error", None))
        status_y = min(height - (92 if is_error else 54), 556)
        c.create_text(col_l, status_y, anchor="w", text=f"Фаза: {stats.get('phase', 'подготовка')}", fill=ACCENT,
                      font=("Segoe UI", 9), width=max(200, width * 0.88))
        c.create_text(col_l, status_y + 22, anchor="w", text=f"Отчёт: GravitrueX/models/{self.active_model_id}/training_report.txt · кнопка «Копировать краткий отчёт» на вкладке обучения",
                      fill=MUTED, font=("Segoe UI", 8), width=max(200, width * 0.88))
        if is_error:
            error_text = str(self.trainer.fatal_error).replace("\n", " ")
            c.create_text(col_l, status_y + 43, anchor="w",
                          text=f"ОБУЧЕНИЕ ПРЕРВАНО: {error_text[:260]}",
                          fill="#ff8585", font=("Segoe UI", 9, "bold"),
                          width=max(200, width * 0.88))
            c.create_text(col_l, status_y + 66, anchor="w",
                          text=f"Полный traceback: GravitrueX/models/{self.active_model_id}/training_crash_report.txt · журнал: training_runtime.log",
                          fill=TEXT, font=("Segoe UI", 8), width=max(200, width * 0.88))

    def _draw_arena(self, width, height):
        c = self.canvas
        self._update_camera()
        target_view_w = min(WORLD_W, CAMERA_BASE_WIDTH / max(0.25, self.view_zoom))
        target_view_h = target_view_w * WORLD_H / WORLD_W
        scale = min(width / max(1.0, target_view_w), height / max(1.0, target_view_h))
        self._draw_world_grid(width, height, scale)

        ship_scale = max(0.58, min(1.20, scale / 0.66))
        bullet_r = max(1.6, min(3.0, 2.2 * ship_scale))
        for bullet in self.arena.bullets:
            col = BLUE if bullet.team == 0 else RED
            for x, y in self._screen_points(bullet.x, bullet.y, scale, width, height, margin=8):
                c.create_oval(x - bullet_r, y - bullet_r, x + bullet_r, y + bullet_r,
                              fill=col, outline="")

        for ship in self.arena.ships:
            if not ship.alive:
                continue
            col = BLUE if ship.team == 0 else RED
            a = ship.heading
            for x, y in self._screen_points(ship.x, ship.y, scale, width, height, margin=30):
                points = []
                for da, radius in ((0.0, 17 * ship_scale), (2.48, 11 * ship_scale), (-2.48, 11 * ship_scale)):
                    points.extend((x + math.cos(a + da) * radius, y + math.sin(a + da) * radius))
                c.create_polygon(*points, fill=col, outline="")
                # Compact two-line in-world HUD: shield (cyan) above hull HP.
                # Keep indicators minimal so they remain legible without restoring old outlines/trails.
                bar_w = max(22.0, 34.0 * ship_scale)
                bar_h = max(2.0, 2.4 * ship_scale)
                bar_left = x - bar_w * 0.5
                shield_top = y - 24.0 * ship_scale
                hp_top = shield_top + bar_h + max(1.0, 1.2 * ship_scale)
                shield_ratio = clamp(ship.shield / max(1.0, MAX_SHIELD), 0.0, 1.0)
                hp_ratio = clamp(ship.hp / max(1.0, MAX_HP), 0.0, 1.0)
                c.create_rectangle(bar_left, shield_top, bar_left + bar_w, shield_top + bar_h,
                                   fill="#172b3d", outline="")
                if shield_ratio > 0.0:
                    c.create_rectangle(bar_left, shield_top, bar_left + bar_w * shield_ratio,
                                       shield_top + bar_h, fill="#63d9ff", outline="")
                c.create_rectangle(bar_left, hp_top, bar_left + bar_w, hp_top + bar_h,
                                   fill="#172b3d", outline="")
                if hp_ratio > 0.0:
                    c.create_rectangle(bar_left, hp_top, bar_left + bar_w * hp_ratio,
                                       hp_top + bar_h, fill=health_bar_color(hp_ratio), outline="")

        blue_alive = len(self.arena.alive_ships(0))
        red_alive = len(self.arena.alive_ships(1))
        c.create_text(14, 12, anchor="nw", text=f"СИНИЕ  {blue_alive}/{self.arena.blue_n}",
                      fill=BLUE, font=("Consolas", 11, "bold"))
        c.create_text(width - 14, 12, anchor="ne", text=f"КРАСНЫЕ  {red_alive}/{self.arena.red_n}",
                      fill=RED, font=("Consolas", 11, "bold"))
        c.create_text(14, height - 12, anchor="sw",
                      text=f"{self.arena.sim_time:.1f} с  ·  масштаб {self.view_zoom:.2f}×",
                      fill=MUTED, font=("Consolas", 9))
        fps_color = "#8be0ca" if self.display_fps >= 50.0 else ("#ffc777" if self.display_fps >= 30.0 else "#ff8794")
        c.create_text(width - 14, height - 12, anchor="se",
                      text=f"FPS {self.display_fps:.1f}  ·  {self.game_steps_per_sec:,.0f} физ. шаг/с",
                      fill=fps_color, font=("Consolas", 10, "bold"))
        if self.paused:
            c.create_rectangle(width / 2 - 72, height / 2 - 27, width / 2 + 72, height / 2 + 27,
                               fill="#0c1929", outline="#3b5c78")
            c.create_text(width / 2, height / 2, text="ПАУЗА", fill=TEXT, font=("Segoe UI", 16, "bold"))
        if self.arena.round_over:
            if self.arena.winner == 0:
                title, color = "ПОБЕДА СИНИХ", BLUE
            elif self.arena.winner == 1:
                title, color = "ПОБЕДА КРАСНЫХ", RED
            else:
                title, color = "НИЧЬЯ", TEXT
            c.create_rectangle(width / 2 - 150, height / 2 - 40, width / 2 + 150, height / 2 + 40,
                               fill="#0b1725", outline=color, width=1)
            c.create_text(width / 2, height / 2, text=title, fill=color, font=("Segoe UI", 17, "bold"))

    def _update_stats(self):
        labels = self.stats_labels
        labels["mode"].set(self.mode_label.get())
        labels["model"].set(self.model_label.get())
        labels["episode"].set(f"Матч: {self.arena.blue_n} синих / {self.arena.red_n} красных · {self.arena.sim_time:.1f} с")
        labels["params"].set(f"Параметров: {self.policy.count_params():,} · вход: {OBS_SIZE} признака")
        if self.trainer:
            stats = self.trainer.get_stats()
            labels["train"].set(f"Обучение {stats.get('training_model_name', self.trainer.model_name)}: {stats.get('episodes', 0)} эпизодов · {stats.get('steps', 0):,} физ. шагов · противник: {stats.get('opponent_name', self.trainer.opponent_name)}")
            labels["win"].set(f"Последние 20: {stats.get('wins', 0)} побед / {stats.get('draws', 0)} ничьих / {stats.get('losses', 0)} поражений")
            labels["reward"].set(f"Reward цикла Σ {stats.get('cycle_reward_total', 0):+.1f} · средняя/цикл {stats.get('cycle_mean_reward', 0):+.3f} · средняя 20 эп. {stats.get('mean_reward_20', 0):+.3f}")
            labels["loss"].set(f"PPO loss: {stats.get('last_loss', 0):.4f}")
            labels["accuracy"].set(f"Попадания: {stats.get('blue_hit_rate_20', 0):.0%} · {stats.get('blue_shots_20', 0):.1f} выстр./бой")
            labels["damage"].set(f"Урон за бой: синие {stats.get('blue_damage_20', 0):.1f} / красные {stats.get('red_damage_20', 0):.1f}")
            labels["kills"].set(f"Убийства: синие {stats.get('blue_kills_20', 0):.2f} / красные {stats.get('red_kills_20', 0):.2f}")
            labels["survivors"].set(f"Выжившие: синие {stats.get('blue_survivors_20', 0):.2f} / красные {stats.get('red_survivors_20', 0):.2f}")
            labels["speed"].set(f"Обучение: {stats.get('steps_per_sec', 0):.0f} физ. шагов/с · {stats.get('device', 'CPU')}")
            labels["perf_stage"].set(
                f"Этап: {stats.get('phase', 'подготовка')} · сбор {stats.get('rollout_sec_last', 0):.1f}с / PPO {stats.get('ppo_sec_last', 0):.1f}с")
        else:
            labels["train"].set("Обучение: не запущено")
            labels["win"].set("Результат последних боёв: —")
            labels["reward"].set("Reward цикла / средняя награда: —")
            labels["loss"].set("PPO loss: —")
            labels["accuracy"].set("Попадания: —")
            labels["damage"].set("Урон за бой: —")
            labels["kills"].set("Убийства: —")
            labels["survivors"].set("Выжившие: —")
            labels["speed"].set(f"Скорость времени: {self.speed:g}×")
            labels["perf_stage"].set("Этап вычислений: обучение не запущено")
        if self.trainer and self.mode != "training":
            # Keep historical training metrics visible, but do not claim that a finished
            # trainer is still collecting rollouts or updating PPO.
            labels["perf_stage"].set("Этап вычислений: обучение не запущено")
        # Large at-a-glance dashboard: no hunting through a paragraph of status text.
        stats = self.trainer.get_stats() if self.trainer else {}
        if hasattr(self, "training_kpi_vars"):
            values = self.training_kpi_vars
            values["updates"].set(f"{stats.get('ppo_updates', 0):,} PPO · {stats.get('episodes', 0):,} эп.")
            latest_cycle = (list(stats.get("last_two_cycles", []))[-1:] or [{}])[0]
            gpu_rollout_rate = float(stats.get("gpu_rollout_steps_per_sec", 0) or 0)
            full_cycle_rate = float(latest_cycle.get("end_to_end_steps_per_sec", latest_cycle.get("combined_steps_per_sec", 0)) or 0)
            cpu_rollout_rate = float(stats.get("cpu_rollout_steps_per_sec", 0) or 0)
            latest_cpu_sec = float(stats.get("cpu_rollout_sec_last", 0) or 0)
            latest_gpu_sec = float(stats.get("gpu_rollout_sec_last", 0) or 0)
            def compact_rate(value: float) -> str:
                return f"{value / 1000.0:.1f}k" if abs(value) >= 1000.0 else f"{value:.0f}"
            values["steps"].set(
                f"CPU {compact_rate(cpu_rollout_rate)}/с · {latest_cpu_sec:.1f}с\n"
                f"GPU {compact_rate(gpu_rollout_rate)}/с · {latest_gpu_sec:.1f}с\n"
                f"Полный цикл {compact_rate(full_cycle_rate)}/с")
            values["reward"].set(f"Σ {stats.get('cycle_reward_total', 0):+.1f} · avg/cycle {stats.get('cycle_mean_reward', 0):+.3f} · avg20 {stats.get('mean_reward_20', 0):+.3f}\nloss {stats.get('last_loss', 0):.4f}")
            win_rate = float(stats.get("win_rate_20", 0.0)) * 100.0
            values["win"].set(f"{win_rate:.0f}% · W/D/L {stats.get('wins', 0)}/{stats.get('draws', 0)}/{stats.get('losses', 0)}")
            values["fire"].set(f"{stats.get('blue_shots_20', 0):.1f} выстр./бой · hit {stats.get('blue_hit_rate_20', 0):.0%} · policy {stats.get('policy_fire_rate_20', 0):.0%}")
            values["kills"].set(f"{stats.get('blue_kills_20', 0):.2f} убийств/бой · {stats.get('blue_damage_20', 0):.1f} урона/бой")
            history = list(stats.get("last_two_cycles", []))[-2:]
            for idx, var in enumerate(getattr(self, "training_cycle_vars", [])):
                if idx >= len(history):
                    var.set("Ожидание второго цикла" if idx == 1 else "Циклы появятся после PPO-update")
                    continue
                item = history[idx]
                stamp = str(item.get("time", ""))[-8:]
                gpu_rate = float(item.get("gpu_steps_per_sec", 0) or 0)
                gpu_sec = float(item.get("gpu_seconds", 0) or 0)
                cpu_rate = float(item.get("cpu_steps_per_sec", 0) or 0)
                cpu_sec = float(item.get("cpu_seconds", 0) or 0)
                cpu_ep = int(item.get("cpu_episodes", 0) or 0)
                gpu_ep = int(item.get("gpu_episodes", 0) or 0)
                requested_envs = int(item.get("gpu_batch_requested_envs", 0) or 0)
                ppo_sec = float(item.get("ppo_seconds", 0) or 0)
                envs = int(item.get("gpu_batch_actual_envs", item.get("gpu_arenas", 0)) or 0)
                records = int(item.get("ppo_records", item.get("gpu_records", 0)) or 0)
                full_rate = float(item.get("end_to_end_steps_per_sec", item.get("combined_steps_per_sec", 0)) or 0)
                var.set(
                    f"{stamp or '—'} · CPU {compact_rate(cpu_rate)}/с {cpu_sec:.1f}с/{cpu_ep} эп. · "
                    f"GPU {compact_rate(gpu_rate)}/с {gpu_sec:.1f}с/{gpu_ep} эп.\n"
                    f"GPU-арены {envs}/{requested_envs} · PPO {ppo_sec:.2f}с · {records:,} записей · цикл {compact_rate(full_rate)}/с\n"
                    f"Reward Σ {float(item.get('cycle_reward_total', 0) or 0):+.1f} · avg/cycle {float(item.get('cycle_mean_reward', 0) or 0):+.3f} · avg20 {float(item.get('mean_reward_20', 0) or 0):+.3f}")
            if hasattr(self, "training_dashboard_status"):
                phase = str(stats.get("phase", "Обучение не запущено"))
                note = str(stats.get("auto_rollout_selection_note", "") or "")
                message = str(stats.get("message", "") or "")
                log_hint = f"TXT лог: {getattr(self, 'desktop_diagnostic_path', Path('Desktop') / DESKTOP_DIAGNOSTIC_FILENAME)}"
                combined_status = phase + (" · " + (note or message) if (note or message) else "")
                log_error = str(getattr(self, "_desktop_diagnostic_write_error", "") or "")
                if log_error:
                    log_hint += f"\nОШИБКА записи TXT: {log_error[:120]}"
                self.training_dashboard_status.set((combined_status + "\n" + log_hint)[:520])
        perf = self.perf.snapshot()
        if self.trainer and self.trainer.is_alive():
            try:
                self.trainer.set_resource_snapshot(perf)
                now_log = time.monotonic()
                if now_log - self._last_resource_log_time >= 1.0:
                    row = {"time": datetime.now().isoformat(timespec="seconds"),
                           "model_id": self.trainer.model_id,
                           "phase": self.trainer.get_stats().get("phase", ""),
                           "steps_per_sec": self.trainer.get_stats().get("steps_per_sec", 0),
                           "cpu": perf.get("cpu"), "app_cpu": perf.get("app_cpu"),
                           "ram_percent": perf.get("ram_percent"), "process_ram": perf.get("process_ram"),
                           "gpu_util": perf.get("gpu_util"), "gpu_mem_util": perf.get("gpu_mem_util"),
                           "vram_used": perf.get("vram_used"), "vram_total": perf.get("vram_total"),
                           "gpu_temp": perf.get("gpu_temp"), "gpu_power_w": perf.get("gpu_power_w")}
                    try:
                        with open(self.trainer.save_dir / "resource_monitor.jsonl", "a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    except Exception:
                        pass
                    self._last_resource_log_time = now_log
            except Exception:
                pass
        cpu = perf.get("cpu")
        ram_used, ram_total = perf.get("ram_used"), perf.get("ram_total")
        proc_ram = perf.get("process_ram")
        gpu = perf.get("gpu_util")
        vram_used, vram_total = perf.get("vram_used"), perf.get("vram_total")
        if self.paused and self.mode == "play":
            labels["perf_fps"].set(f"FPS последней активной игры: {self.display_fps:.1f} · пауза: сцена кэширована")
        elif self.paused:
            labels["perf_fps"].set("Интерфейс в ожидании · сцена кэширована")
        else:
            is_training_view = self.mode in ("training", "training_error")
            labels["perf_fps"].set(f"FPS интерфейса: {self.display_fps:.1f} / {TRAINING_UI_FPS_TARGET if is_training_view else DISPLAY_FPS_TARGET}" + (" (обучение)" if is_training_view else ""))
        app_cpu = perf.get("app_cpu")
        labels["perf_cpu"].set((f"CPU системы {cpu:.0f}% · Arena {app_cpu:.0f}%" if app_cpu is not None else f"CPU системы: {cpu:.0f}%") if cpu is not None else "CPU системы: н/д (нужен psutil)")
        if ram_used is not None and ram_total is not None:
            labels["perf_ram"].set(f"RAM: {_fmt_bytes(ram_used)}/{_fmt_bytes(ram_total)} ({perf.get('ram_percent', 0):.0f}%) · Arena {_fmt_bytes(proc_ram)}")
        else:
            labels["perf_ram"].set("RAM: н/д (установи psutil)")
        if gpu is not None and vram_used is not None and vram_total is not None:
            extra_temp = f" · {perf['gpu_temp']}°C" if perf.get("gpu_temp") is not None else ""
            mem_util = f" · mem engine {perf['gpu_mem_util']}%" if perf.get("gpu_mem_util") is not None else ""
            power = f" · {perf['gpu_power_w']:.0f}W" if perf.get("gpu_power_w") is not None else ""
            sm_clock = f" · SM {perf['gpu_clock_mhz']}MHz" if perf.get("gpu_clock_mhz") is not None else ""
            labels["perf_gpu"].set(f"NVIDIA SM {gpu}%{mem_util} · VRAM {_fmt_bytes(vram_used * 1024**2)}/{_fmt_bytes(vram_total * 1024**2)}{power}{sm_clock}{extra_temp}")
        else:
            labels["perf_gpu"].set("GPU: недоступно через nvidia-smi")
        if "perf_infer" in labels:
            if self.mode in ("training", "training_error") and self.trainer:
                tstats = self.trainer.get_stats()
                if tstats.get("gpu_only_rollout"):
                    labels["perf_infer"].set(f"ИИ в rollout: CUDA physics · арен {int(tstats.get('gpu_rollout_envs', 0) or 0)}/{int(tstats.get('gpu_rollout_envs_requested', tstats.get('gpu_batch_envs', 0)) or 0)} фактически/запрошено; CPU arenas=0")
                elif tstats.get("hybrid_gpu_rollout"):
                    labels["perf_infer"].set("ИИ в rollout: гибрид async · CPU-арены + CUDA tensor physics")
                elif tstats.get("rollout_gpu_inference"):
                    labels["perf_infer"].set("ИИ в rollout: GPU policy + PPO; физика CPU, 1 бой")
                else:
                    labels["perf_infer"].set("ИИ в rollout: CPU-физика / NumPy-параллельно")
            else:
                labels["perf_infer"].set(f"ИИ в бою: {self.policy.last_inference_backend} · режим {self.policy.inference_mode.upper()}")
        if self.mode in ("training", "training_error") and self.trainer:
            train_rate = self.trainer.get_stats().get("steps_per_sec", 0)
            labels["perf_steps"].set(f"Обучение: {train_rate:,.0f} физ. шаг/с")
        else:
            labels["perf_steps"].set(f"Игра: {self.game_steps_per_sec:,.0f} физ. шаг/с · {self.ticks_last_frame} за кадр")
        detail_now = time.perf_counter()
        if detail_now - getattr(self, "_last_model_detail_update", 0.0) >= 1.0:
            self._last_model_detail_update = detail_now
            self._update_model_details()
        if detail_now - getattr(self, "_last_desktop_diagnostic_write", 0.0) >= 5.0:
            self._write_desktop_diagnostics()

    def _write_desktop_diagnostics(self, force: bool = False) -> None:
        """Overwrite one compact desktop snapshot with current app/training/resource state.

        The file is deliberately a rolling snapshot, not an ever-growing log. It also
        contains the last recent metric rows so a single text attachment is useful.
        """
        now = time.monotonic()
        if not force and now - getattr(self, "_last_desktop_diagnostic_write", 0.0) < 5.0:
            return
        self._last_desktop_diagnostic_write = now
        try:
            perf = self.perf.snapshot() if hasattr(self, "perf") else {}
            trainer = getattr(self, "trainer", None)
            ts = trainer.get_stats() if trainer is not None else {}
            arena = getattr(self, "arena", None)
            active_meta = read_model_meta(getattr(self, "active_model_id", DEFAULT_MODEL_ID))
            if arena is not None:
                blue = [ship for ship in arena.ships if ship.team == 0]
                red = [ship for ship in arena.ships if ship.team == 1]
                alive_blue = sum(1 for ship in blue if ship.alive)
                alive_red = sum(1 for ship in red if ship.alive)
                total_shots_blue = sum(int(getattr(ship, "shots_fired", 0)) for ship in blue)
                total_shots_red = sum(int(getattr(ship, "shots_fired", 0)) for ship in red)
                damage_blue = sum(float(getattr(ship, "damage_dealt", 0.0)) for ship in blue)
                damage_red = sum(float(getattr(ship, "damage_dealt", 0.0)) for ship in red)
                kills_blue = sum(int(getattr(ship, "kills", 0)) for ship in blue)
                kills_red = sum(int(getattr(ship, "kills", 0)) for ship in red)
                hp_blue = sum(max(0.0, float(getattr(ship, "hp", 0.0))) for ship in blue if ship.alive)
                hp_red = sum(max(0.0, float(getattr(ship, "hp", 0.0))) for ship in red if ship.alive)
                shield_blue = sum(max(0.0, float(getattr(ship, "shield", 0.0))) for ship in blue if ship.alive)
                shield_red = sum(max(0.0, float(getattr(ship, "shield", 0.0))) for ship in red if ship.alive)
            else:
                alive_blue = alive_red = total_shots_blue = total_shots_red = 0
                damage_blue = damage_red = hp_blue = hp_red = shield_blue = shield_red = 0.0
                kills_blue = kills_red = 0
            lines = [
                "GRAVITRUEX — КОМПАКТНЫЙ ДИАГНОСТИЧЕСКИЙ СНИМОК",
                f"Обновлено: {datetime.now().astimezone().isoformat(timespec='seconds')}",
                f"Сборка: {APP_BUILD_LABEL}",
                f"Файл перезаписывается каждые ~5 секунд; это актуальный снимок, а не бесконечный журнал.",
                f"Путь к проекту: {SCRIPT_DIR}",
                f"Путь к текущему логу: {self.desktop_diagnostic_path}",
                "",
                "[ПРИЛОЖЕНИЕ / БОЙ]",
                f"Режим={getattr(self, 'mode', 'unknown')} · вкладка={getattr(self, 'ui_tab', 'unknown')} · пауза={bool(getattr(self, 'paused', False))} · скорость времени={getattr(self, 'speed', 1.0)}x",
                f"FPS={getattr(self, 'display_fps', 0.0):.1f} · физ. шаг/с={getattr(self, 'game_steps_per_sec', 0.0):,.0f} · модель UI={getattr(self, 'active_model_id', DEFAULT_MODEL_ID)} ({getattr(getattr(self, 'policy', None), 'width', '?')}x{getattr(getattr(self, 'policy', None), 'depth', '?')}; {getattr(getattr(self, 'policy', None), 'count_params', lambda: 0)():,} params)",
                f"Стороны: {getattr(self, 'match_blue_ai', '—')} VS {getattr(self, 'match_red_ai', '—')} · живы B/R={alive_blue}/{alive_red} · пули={len(getattr(arena, 'bullets', [])) if arena else 0} · sim_time={getattr(arena, 'sim_time', 0.0):.2f}s · шаг={getattr(arena, 'step_count', 0)}",
                f"Видимый бой, всего: shots B/R={total_shots_blue}/{total_shots_red} · damage B/R={damage_blue:.1f}/{damage_red:.1f} · kills B/R={kills_blue}/{kills_red} · HP B/R={hp_blue:.1f}/{hp_red:.1f} · shield B/R={shield_blue:.1f}/{shield_red:.1f}",
                f"Inference: {getattr(getattr(self, 'policy', None), 'last_inference_backend', 'n/a')} · mode={getattr(getattr(self, 'policy', None), 'inference_mode', 'n/a')} · require_gpu={getattr(getattr(self, 'policy', None), 'require_gpu_inference', False)}",
                "",
                "[РЕСУРСЫ]",
                " · ".join(f"{k}={v}" for k, v in perf.items() if k != "sampled_at"),
                "",
            ]
            if trainer is not None:
                model_id = getattr(trainer, "model_id", "?")
                width = getattr(getattr(trainer, "policy", None), "width", 0)
                depth = getattr(getattr(trainer, "policy", None), "depth", 0)
                params = getattr(getattr(trainer, "policy", None), "count_params", lambda: 0)()
                history = list(ts.get("last_two_cycles", []))[-2:]
                recent_rate_rows = _read_jsonl_tail(getattr(trainer, "metrics_jsonl_path", Path("/nonexistent")), 8)
                cuda_mem = dict(ts.get("cuda_memory_snapshot", {}) or {})
                cuda_mem_after_stop = dict(ts.get("cuda_memory_snapshot_after_stop", {}) or {})
                lines.extend([
                    "[ОБУЧЕНИЕ]",
                    f"Модель={ts.get('training_model_name', getattr(trainer, 'model_name', '?'))} [{model_id}] · архитектура={width}x{depth} · параметры={params:,}",
                    f"Чекпоинт: {_compact_file_status(getattr(trainer, 'save_path', None))} · сессия trainer_alive={bool(getattr(trainer, 'is_alive', lambda: False)())}",
                    f"Противник={ts.get('opponent_name', getattr(trainer, 'opponent_name', '?'))} · состав={'random 1–5' if getattr(trainer, 'random_compositions', False) else f'{getattr(trainer, 'fixed_blue', '?')}v{getattr(trainer, 'fixed_red', '?')}'} · PPO mini-batch={getattr(trainer, 'batch_size', '?')} · эпох={getattr(trainer, 'epochs', '?')}",
                    f"Устройства: rollout physics={ts.get('rollout_physics_device', 'n/a')} · policy={ts.get('rollout_policy_device', 'n/a')} · PPO={ts.get('device', getattr(trainer, 'device_mode', '?'))} · gpu_only={ts.get('gpu_only_rollout', False)} · hybrid={ts.get('hybrid_gpu_rollout', False)}",
                    f"Суммарно: episodes={ts.get('episodes', 0):,} · physics_steps={ts.get('steps', 0):,} · PPO updates={ts.get('ppo_updates', 0):,} · records/cycle={ts.get('ppo_records_last', 0):,} · loss={float(ts.get('last_loss', 0) or 0):+.6f}",
                    f"Качество last20: reward={float(ts.get('mean_reward_20', 0) or 0):+.4f} · win={float(ts.get('win_rate_20', 0) or 0):.1%} · W/D/L={ts.get('wins', 0)}/{ts.get('draws', 0)}/{ts.get('losses', 0)} · shots B/R={float(ts.get('blue_shots_20', 0) or 0):.2f}/{float(ts.get('red_shots_20', 0) or 0):.2f} · hit B/R={float(ts.get('blue_hit_rate_20', 0) or 0):.1%}/{float(ts.get('red_hit_rate_20', 0) or 0):.1%}",
                    f"Damage B/R={float(ts.get('blue_damage_20', 0) or 0):.2f}/{float(ts.get('red_damage_20', 0) or 0):.2f} · kills B/R={float(ts.get('blue_kills_20', 0) or 0):.2f}/{float(ts.get('red_kills_20', 0) or 0):.2f} · policy fire={float(ts.get('policy_fire_rate_20', 0) or 0):.2%} · teacher fire={float(ts.get('teacher_fire_rate_20', 0) or 0):.2%} · imitation={float(ts.get('imitation_loss_last', 0) or 0):.5f}/{float(ts.get('imitation_coef_last', 0) or 0):.4f}",
                    f"Throughput: session={float(ts.get('steps_per_sec', 0) or 0):,.0f} phys.step/s · CPU last={float(ts.get('cpu_rollout_steps_per_sec', 0) or 0):,.0f}/s за {float(ts.get('cpu_rollout_sec_last', 0) or 0):.2f}s · GPU last={float(ts.get('gpu_rollout_steps_per_sec', 0) or 0):,.0f}/s за {float(ts.get('gpu_rollout_sec_last', 0) or 0):.2f}s · arenas actual/requested={int(ts.get('gpu_rollout_envs', 0) or 0)}/{int(ts.get('gpu_rollout_envs_requested', 0) or 0)} · horizon={getattr(trainer, '_last_gpu_max_decisions', 0)} · budget={getattr(trainer, '_last_gpu_record_budget', 0):,}",
                    f"Timing: total rollout wall={float(ts.get('rollout_sec_last', 0) or 0):.3f}s · CPU collect={float(ts.get('cpu_rollout_sec_last', 0) or 0):.3f}s · GPU collect={float(ts.get('gpu_rollout_sec_last', 0) or 0):.3f}s (device={float(ts.get('gpu_device_seconds', 0) or 0):.3f}s, host={float(ts.get('gpu_host_overhead_seconds', 0) or 0):.3f}s) · PPO={float(ts.get('ppo_sec_last', 0) or 0):.3f}s · phase={ts.get('phase', 'n/a')}",
                    f"CUDA Graph: {ts.get('gpu_cuda_graph_status', 'not reported')} · OOM fallback: {ts.get('gpu_rollout_oom_fallback', 'none') or 'none'} · budget-reduced arenas={int(ts.get('gpu_rollout_envs_budget_reduced', 0) or 0):,}",
                    (f"PyTorch CUDA allocator: rollout allocated/reserved/peak={cuda_mem.get('allocated_gib', 'n/a')}/{cuda_mem.get('reserved_gib', 'n/a')}/{cuda_mem.get('peak_allocated_gib', 'n/a')} GiB · free/total={cuda_mem.get('free_gib', 'n/a')}/{cuda_mem.get('total_gib', 'n/a')} GiB · after stop allocated/reserved={cuda_mem_after_stop.get('allocated_gib', 'n/a')}/{cuda_mem_after_stop.get('reserved_gib', 'n/a')} GiB · history slots={int(ts.get('gpu_record_slots', 0) or 0)} · obs buffer estimate={float(ts.get('gpu_obs_history_estimate_gib', 0) or 0):.3f} GiB"),
                    f"Fallback/error: {str(ts.get('message', '') or getattr(trainer, 'fatal_error', '') or 'none')[:500]}",
                    f"Checkpoint={getattr(trainer, 'save_path', 'n/a')} · crash report={getattr(trainer, 'crash_report_path', 'n/a')}",
                    "Последние два цикла (скорость сбора отдельно от полного цикла с PPO):",
                ])
                for row in reversed(history):
                    lines.append(
                        f"  {row.get('time', '')} | CPU-сбор={float(row.get('cpu_seconds', 0) or 0):.2f}s ({int(row.get('cpu_episodes', 0) or 0)} эп., {float(row.get('cpu_steps_per_sec', 0) or 0):,.0f}/s) "
                        f"· GPU-сбор={float(row.get('gpu_seconds', 0) or 0):.2f}s ({int(row.get('gpu_episodes', 0) or 0)} эп., {float(row.get('gpu_steps_per_sec', 0) or 0):,.0f}/s) "
                        f"· arenas={int(row.get('gpu_batch_actual_envs', 0) or 0)}/{int(row.get('gpu_batch_requested_envs', 0) or 0)} "
                        f"· PPO={float(row.get('ppo_seconds', 0) or 0):.2f}s · wall={float(row.get('rollout_wall_seconds', 0) or 0) + float(row.get('ppo_seconds', 0) or 0):.2f}s "
                        f"· records={int(row.get('ppo_records', 0) or 0):,}")
                lines.append("Последние 8 сводных записей метрик (краткие):")
                for row in recent_rate_rows:
                    lines.append(
                        f"  {str(row.get('time', ''))[-8:]} ep={row.get('episodes_total', 0)} upd={row.get('ppo_updates', 0)} "
                        f"reward={row.get('mean_reward_20', 0)} win={row.get('win_rate_20', 0)} "
                        f"shot={row.get('blue_shots_20', 0)} hit={row.get('blue_hit_rate_20', 0)} "
                        f"rollout={row.get('gpu_steps_per_sec', 0)} full={row.get('end_to_end_steps_per_sec', row.get('combined_steps_per_sec', 0))}")
                runtime_path = getattr(trainer, "runtime_log_path", None)
                if runtime_path and Path(runtime_path).exists():
                    try:
                        runtime_lines = Path(runtime_path).read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
                        if runtime_lines:
                            lines.extend(["Последние события training_runtime.log:", *["  " + x[:300] for x in runtime_lines]])
                    except Exception:
                        pass
                crash_path = getattr(trainer, "crash_report_path", None)
                if crash_path and Path(crash_path).exists():
                    try:
                        crash_lines = Path(crash_path).read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
                        lines.extend(["Последние строки последнего crash report:", *["  " + x[:300] for x in crash_lines]])
                    except Exception:
                        pass
            else:
                lines.extend([
                    "[ОБУЧЕНИЕ]",
                    "Тренер сейчас не запущен.",
                    f"Выбранная модель={self.active_model_id} · name={active_meta.get('name', self.active_model_id)} · profile={active_meta.get('width', getattr(self.policy, 'width', '?'))}x{active_meta.get('depth', getattr(self.policy, 'depth', '?'))} · trained={active_meta.get('trained', False)} · saved episodes={active_meta.get('episodes', 0)} · saved steps={active_meta.get('steps', 0)}",
                ])
            # Compact model catalogue helps identify which checkpoint was active without sending weights.
            try:
                records = list(getattr(self, "model_records", []))
                # Refresh only the currently selected/training entry; don't reopen every NPZ every 5 seconds.
                for record in records:
                    if record.get("id") in (getattr(self, "active_model_id", None), getattr(trainer, "model_id", None)):
                        fresh = read_model_meta(str(record.get("id")))
                        if fresh:
                            record.update({k: v for k, v in fresh.items() if k in ("episodes", "steps", "trained", "width", "depth", "name")})
                lines.extend(["", f"[БИБЛИОТЕКА МОДЕЛЕЙ] количество={len(records)} (без весов):"])
                for record in records[:16]:
                    lines.append(
                        f"  {record.get('id', '?')} | {record.get('name', '?')} | "
                        f"{record.get('width', '?')}x{record.get('depth', '?')} | "
                        f"episodes={record.get('episodes', 0)} steps={record.get('steps', 0)} "
                        f"trained={record.get('trained', False)} weights={record.get('has_weights', False)}")
            except Exception as exc:
                lines.append(f"Модельная библиотека: не удалось прочитать ({type(exc).__name__}: {exc})")
            lines.extend([
                "",
                "[ПРИМЕЧАНИЕ] Этот снимок перезаписывается и содержит текущее состояние + последние циклы. "
                "Для первичного разбора достаточно прислать только этот TXT; чекпоинты/конфиг не включаются.",
            ])
            write_compact_diagnostic_snapshot(self.desktop_diagnostic_path, "\n".join(lines))
            self._desktop_diagnostic_write_error = ""
        except Exception as exc:
            self._desktop_diagnostic_write_error = f"{type(exc).__name__}: {exc}"

    def _on_close(self):
        # Persist a final compact diagnostic snapshot and UI selections; don't touch checkpoints/data.
        self._write_desktop_diagnostics(force=True)
        self._save_settings(quiet=True)
        self.perf.close()
        if self.trainer and self.trainer.is_alive():
            self.trainer.stop()
            # Saving occurs at the end of the current episode; the thread is daemonized.
            self.root.after(300, self._finish_close)
        else:
            self.root.destroy()

    def _finish_close(self):
        if self.trainer and self.trainer.is_alive():
            self.root.after(300, self._finish_close)
        else:
            self._write_desktop_diagnostics(force=True)
            self.root.destroy()


def run_self_test() -> int:
    print("[1/6] Инициализация политики")
    policy = ActorCritic(seed=1, width=32, depth=2, device_mode="cpu")
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    actions, logps, values, arrs = policy.predict_batch([obs, obs], deterministic=False)
    assert len(actions) == 2 and arrs.shape == (2, 4)
    assert all(np.isfinite(logps)) and all(np.isfinite(values))
    print(f"  OK: {policy.count_params()} параметров, вход {OBS_SIZE}, выходы {arrs.shape}")

    print("[2/6] Проверка арены и наблюдений")
    env = Arena(seed=2)
    env.reset(3, 4)
    assert len(env.ships) == 7
    for ship in env.alive_ships(0):
        o = env.observation(ship)
        assert o.shape == (OBS_SIZE,) and np.all(np.isfinite(o))
    print(f"  OK: 3 на 4, {OBS_SIZE} признаков на корабль")

    print("[3/6] Фиксированная физика")
    blue = env.alive_ships(0)
    batch = [env.observation(s) for s in blue]
    acts, _, _, _ = policy.predict_batch(batch, deterministic=True)
    for _ in range(20):
        env.decision_step({s.sid: acts[i] for i, s in enumerate(blue) if s.alive})
        if env.round_over:
            break
    assert all(math.isfinite(s.x) and math.isfinite(s.y) for s in env.ships)
    print(f"  OK: {env.step_count} fixed physics ticks, {len(env.bullets)} projectiles alive")

    print("[4/6] Проверка PPO backprop на синтетическом пакете")
    records = []
    for i in range(32):
        o = np.random.default_rng(i).normal(0, 0.3, OBS_SIZE).astype(np.float32)
        a, lp, v, arr = policy.predict_batch([o], deterministic=False)
        records.append({"obs": o, "action": arr[0], "logp": float(lp[0]), "value": float(v[0]),
                        "reward": float((i % 3) - 1), "done": True})
    _gae_for_trajectory(records)
    loss = policy.update_ppo(records, epochs=1, batch_size=16)
    assert math.isfinite(loss)
    print(f"  OK: PPO loss {loss:.4f}; веса обновлены")

    print("[5/6] Проверка self-play и наград за обе команды")
    env2 = Arena(seed=33)
    env2.reset(1, 1, seed=33)
    records2, reward2, outcome2, _, _, steps2, metrics2 = _collect_episode_rollout(
        policy, env2, 1, 1, max_decisions=12, opponent_mode="selfplay")
    assert records2 and steps2 > 0
    assert "blue_survivors" in metrics2 and "red_survivors" in metrics2
    probe = Arena(seed=34)
    probe.reset(1, 1, seed=34)
    red_ship = probe.alive_ships(1)[0]
    given = Action(turn=1, thrust=-1, strafe=1, fire=1)
    probe.decision_step({probe.alive_ships(0)[0].sid: Action(), red_ship.sid: given})
    assert red_ship.last_action == given, "self-play red action was overridden by heuristic fallback"
    print(f"  OK: {len(records2)} self-play records; survivor metrics present; both sides' actions accepted")

    print("[6/6] Проверка векторной тензорной физики (CPU backend test; CUDA used by hybrid mode)")
    try:
        import torch
        GPUVectorArenaBatch = _load_gpu_batch_class()
        # The same tensor-batched path runs on CPU for portability tests, and on
        # CUDA in the actual hybrid mode. Use 2 concurrent 1v1 battles here.
        batch_policy = ActorCritic(seed=71, width=32, depth=2, device_mode="cpu")
        batch_env = GPUVectorArenaBatch(engine=sys.modules[__name__], policy=batch_policy,
                                        count=2, blue_n=1, red_n=1,
                                        opponent_mode="heuristic", max_decisions=8,
                                        device="cpu", seed=71)
        batch_results = batch_env.collect()
        assert len(batch_results) == 2, f"Expected 2 parallel arenas, got {len(batch_results)}"
        assert all(row[0] and row[5] > 0 for row in batch_results)
        assert all(np.isfinite(float(row[1])) for row in batch_results)
        # Regression test: categorical action 2 must produce forward acceleration;
        # neutral action 1 must not be accidentally shifted to reverse thrust.
        batch_env.vx.zero_(); batch_env.vy.zero_(); batch_env.heading.zero_()
        batch_env.active_env[:] = True
        batch_env.alive[:] = batch_env.present
        batch_env.hp[:] = batch_env.present.float() * MAX_HP
        probe_actions = torch.ones((batch_env.E, batch_env.S, 4), dtype=torch.long, device=batch_env.dev)
        probe_actions[:, 0, 1] = 2
        batch_env._physics_tick(probe_actions, torch.ones(batch_env.E, dtype=torch.bool, device=batch_env.dev))
        assert bool((batch_env.vx[:, 0] > 0).all()), "Categorical forward thrust (2) did not accelerate forward"
        assert bool((batch_env.vx[:, 1].abs() < 1e-5).all()), "Neutral thrust (1) changed forward velocity"
        packed_for_test = PackedExperience.concatenate([episode[0] for episode in batch_results])
        ppo_loss = batch_policy.update_ppo(packed_for_test, epochs=1, batch_size=32, device="cpu")
        assert math.isfinite(ppo_loss)

        # One-tick physics parity check against the scalar Arena. The two paths
        # use float32 vs Python float, so allow a small numeric tolerance.
        ref_env = Arena(seed=714)
        ref_env.reset(2, 2, seed=714)
        parity_batch = GPUVectorArenaBatch(engine=sys.modules[__name__], policy=batch_policy,
                                           count=1, blue_n=2, red_n=2,
                                           opponent_mode="heuristic", max_decisions=8,
                                           device="cpu", seed=714)
        ship_to_slot = {}
        red_index = 0
        for ship in ref_env.ships:
            if ship.team == 0:
                slot = sum(1 for other in ref_env.ships if other.team == 0 and other.sid < ship.sid)
            else:
                slot = 5 + red_index
                red_index += 1
            ship_to_slot[ship.sid] = slot
            for tensor_name, attr_name in (("x", "x"), ("y", "y"), ("vx", "vx"), ("vy", "vy"),
                                           ("heading", "heading"), ("hp", "hp"), ("shield", "shield"),
                                           ("cooldown", "cooldown"), ("last_hit_time", "last_hit_time")):
                getattr(parity_batch, tensor_name)[0, slot] = float(getattr(ship, attr_name))
        parity_batch.sim_time[0] = ref_env.sim_time
        parity_batch.active_env[:] = True
        parity_batch.alive[:] = parity_batch.present
        parity_batch.balive.zero_()
        ref_actions = {}
        cat_actions = torch.ones((1, parity_batch.S, 4), dtype=torch.long, device=parity_batch.dev)
        for i, ship in enumerate(ref_env.ships):
            action = Action(turn=(i % 3) - 1, thrust=((i + 1) % 3) - 1,
                            strafe=((i + 2) % 3) - 1, fire=0)
            ref_actions[ship.sid] = action
            slot = ship_to_slot[ship.sid]
            cat_actions[0, slot, 0] = action.turn + 1
            cat_actions[0, slot, 1] = action.thrust + 1
            cat_actions[0, slot, 2] = action.strafe + 1
        ref_env.physics_tick(ref_actions)
        parity_batch._physics_tick(cat_actions, torch.ones((1,), dtype=torch.bool, device=parity_batch.dev))
        errors = []
        for ship in ref_env.ships:
            slot = ship_to_slot[ship.sid]
            for tensor_name, attr_name in (("x", "x"), ("y", "y"), ("vx", "vx"), ("vy", "vy"),
                                           ("heading", "heading"), ("hp", "hp")):
                errors.append(abs(float(getattr(parity_batch, tensor_name)[0, slot].item()) - float(getattr(ship, attr_name))))
        max_error = max(errors) if errors else 0.0
        assert max_error < 1e-3, f"GPU tensor physics diverged from scalar arena by {max_error}"
        print(f"  OK: {len(batch_results)} arenas in one tensor batch; records={sum(len(row[0]) for row in batch_results):,}; PPO loss={ppo_loss:.4f}")
        print(f"  OK: vector physics one-tick parity with scalar Arena; max absolute state error={max_error:.2e}")
    except Exception as exc:
        raise AssertionError(f"GPU vector batch test failed: {exc}") from exc
    print("SELF-TEST PASSED")
    return 0


def main() -> int:
    if "--self-test" in sys.argv:
        return run_self_test()
    try:
        import tkinter as tk
    except Exception as exc:
        print("Не удалось запустить tkinter. Установи обычную версию Python с поддержкой Tcl/Tk.", exc)
        return 2
    root = tk.Tk()
    app = GravitrueXApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
