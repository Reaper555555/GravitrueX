# -*- coding: utf-8 -*-
"""Tensor-batched GravitrueX arenas for CUDA rollouts.

The class intentionally receives the running game module as `engine`, avoiding
an import of the UI script under a second module name when launched as __main__.
It batches independent arena physics and policy inference on one device. CPU
ProcessPool workers can run ordinary Arena rollouts concurrently in parallel.

Stage 5 "fast path" changes (all of them are behaviour preserving for the
physics itself, the observation layout and the reward formula):

* One shared per-decision context. `_pairwise()` used to run four times per
  decision (observe, aim before, heuristic, aim after) and `_bullet_rel()`
  twice. Both are now computed once before the physics ticks and once (light
  variant) after them.
* Six `topk` calls per decision collapsed to two stable sorts plus one topk.
  The "3 nearest" and "2 extra" selections now share a single ranking.
* Projectile slot allocation no longer uses `torch.where(condition)`, i.e.
  `nonzero()`. That call has a data dependent output shape, so on CUDA it
  forces a device->host synchronization on *every* physics tick. The new
  implementation ranks free slots with a cumsum and scatters into a reserved
  trash slot, so the tick stays completely asynchronous.
* Categorical actions are sampled with the Gumbel-max trick instead of
  `torch.multinomial`. It draws all three action heads in one random tensor,
  is cheaper, and is safe to capture inside a CUDA graph.
* Optional CUDA graph capture of a whole decision. The per-decision body is
  branch free and synchronization free, so it can be replayed with a single
  launch instead of several hundred kernel launches.
* The per-arena numpy packing at the end of a rollout is vectorized instead of
  running eight fancy-index copies per arena.
* Optional "blue only" observations: in heuristic mode only the blue ships are
  trained, so the enemy observations were computed and thrown away.

The physics is fixed-timestep and uses the same major constants as the scalar
Arena, but is implemented as batched tensor operations. This is an experimental
accelerator path; parity tests should be used when physics rules are changed.
"""
from __future__ import annotations

import math
import os
import random
import time
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Attributes that hold mutable simulation state. Used to snapshot the arena
# around CUDA graph warmup so capture does not consume real episode steps. Keep
# this list complete: alive is essential because warmup can kill a zero-HP ship.
_STATE_ATTRS = (
    "x", "y", "vx", "vy", "heading", "hp", "shield", "cooldown",
    "last_hit_time", "last_turn", "damage_dealt", "damage_taken", "alive",
    "kills", "shots", "deaths", "bx", "by", "bvx", "bvy", "bteam",
    "bowner", "bttl", "balive", "sim_time", "step_count", "active_env",
)

# CUDA graph capture failures are memoized per shape/configuration in auto mode.
# This avoids paying warmup + sync costs on every short-lived rollout batch.
_CUDA_GRAPH_FAILED_KEYS = set()
_CUDA_GRAPH_FAILED_LOCK = threading.Lock()


def _reward_coefficients(engine: Any) -> dict:
    """Read the selected reward profile; keep lightweight comparison/test stubs compatible."""
    provider = getattr(engine, "training_reward_coefficients", None)
    if callable(provider):
        return provider()
    # Stage 6.14 fallback for engine stubs used by the vector-physics regression harness.
    return {
        "damage_dealt": float(getattr(engine, "REWARD_DAMAGE_DEALT", 0.10)),
        "damage_taken": float(getattr(engine, "REWARD_DAMAGE_TAKEN", 0.035)),
        "kill": float(getattr(engine, "REWARD_PER_KILL", 13.0)),
        "death": float(getattr(engine, "REWARD_DEATH", 7.0)),
        "enemy_dead": 1.8, "ally_dead": 2.4, "team_progress": 0.0,
        "terminal_win": 24.0, "terminal_kill": 8.0, "terminal_survivor": 6.0,
        "terminal_damage_div": 65.0, "terminal_damage_cap": 3.0, "terminal_bias": -1.5,
    }


def _as_python_float(value: Any) -> float:
    """Convert a configuration scalar to a Python float before CUDA capture.

    A scalar torch.Tensor on CPU can otherwise be treated as a CPU-to-CUDA copy
    when it appears on the RHS of an indexed assignment during graph capture.
    Extracting the value here (constructor time, outside capture) is safe.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu") and not isinstance(value, (int, float, np.number)):
        try:
            value = value.cpu()
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, (int, float, np.number)):
        value = value.item()
    return float(value)


class _Ctx:
    """Everything the observe / aim / heuristic paths need for one moment."""

    __slots__ = ("dx", "dy", "rvx", "rvy", "dist", "enemy_valid", "ally_valid",
                 "target_idx", "has_target", "tx", "ty", "tvx", "tvy", "tdist",
                 "thp", "tshield", "bdx", "bdy", "bvx", "bvy", "bdist", "bvalid")

    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


class GPUVectorArenaBatch:
    """Multiple independent toroidal arenas evaluated together with torch tensors.

    The training observations preserve the current 96-feature layout. Physics is
    fixed-timestep and uses the same major constants as the scalar Arena, but is
    implemented as batched tensor operations. This is an experimental accelerator
    path; parity tests should be used when physics rules are changed.
    """

    def __init__(self, engine: Any, policy: Any, count: int = 8,
                 blue_n: int = 1, red_n: int = 1, random_compositions: bool = False,
                 opponent_mode: str = "heuristic", max_decisions: int = 240,
                 device: str = "cuda", seed: Optional[int] = None,
                 use_cuda_graph: Optional[bool] = None,
                 blue_only_observations: Optional[bool] = None,
                 max_records: Optional[int] = None):
        import torch
        self.torch = torch
        self.e = engine
        self.policy = policy
        self.device = torch.device(device)
        self.dev = self.device
        self.t = torch
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("GPU-пакет арен требует доступную CUDA-видеокарту и CUDA-сборку PyTorch.")
        self.requested_count = max(1, min(4096, int(count)))
        self.count = self.requested_count
        self.max_decisions = max(8, int(max_decisions))
        self.opponent_mode = str(opponent_mode)
        if self.opponent_mode not in ("heuristic", "selfplay"):
            raise RuntimeError("GPU-пакетная физика поддерживает только эвристического противника и self-play на первом этапе.")
        self.rng = random.Random(seed if seed is not None else time.time_ns())
        if random_compositions:
            self.blue_counts = np.asarray([self.rng.randint(1, 5) for _ in range(self.count)], dtype=np.int64)
            self.red_counts = np.asarray([self.rng.randint(1, 5) for _ in range(self.count)], dtype=np.int64)
        else:
            self.blue_counts = np.full(self.count, max(1, min(5, int(blue_n))), dtype=np.int64)
            self.red_counts = np.full(self.count, max(1, min(5, int(red_n))), dtype=np.int64)
        self.max_records_budget = int(max_records) if max_records else 0
        self.record_budget_decisions = self.max_decisions
        self.record_budget_reduced_arenas = 0
        # Preserve the user's requested episode horizon. If the record cap cannot
        # hold all requested arenas for that horizon, reduce the arena count instead
        # of silently chopping every game short. Random team counts are sampled first
        # so the limit uses the exact records needed by this batch.
        if max_records:
            per_arena_slots = (self.blue_counts + self.red_counts
                               if self.opponent_mode == "selfplay" else self.blue_counts)
            cumulative_slots = np.cumsum(per_arena_slots, dtype=np.int64)
            fits = np.flatnonzero(cumulative_slots * self.max_decisions <= int(max_records))
            if len(fits) == 0:
                raise RuntimeError(
                    f"Лимит записей {int(max_records):,} не вмещает даже одну арену с горизонтом {self.max_decisions} "
                    f"и составом {int(self.blue_counts[0])}v{int(self.red_counts[0])}. Уменьши горизонт или увеличь бюджет."
                )
            effective_count = int(fits[-1]) + 1
            self.record_budget_reduced_arenas = max(0, self.requested_count - effective_count)
            self.count = effective_count
            self.blue_counts = self.blue_counts[:effective_count].copy()
            self.red_counts = self.red_counts[:effective_count].copy()
        self.E, self.S, self.B = self.count, 10, 256
        # Resolve the projectile TTL to a host scalar once, before graph capture.
        # The reusable CUDA tensor prevents an implicit CPU-scalar copy in the
        # advanced indexed assignment used by projectile spawning.
        self._bullet_ttl_value = _as_python_float(getattr(engine, "BULLET_TTL", 2.15))
        self._bullet_ttl_spawn_values = torch.full(
            (self.E, self.S), self._bullet_ttl_value,
            dtype=torch.float32, device=self.dev,
        )
        # One extra column per arena acts as a trash slot: projectiles that
        # are not spawned are written there, so the scatter never touches a
        # live projectile and never needs a nonzero() (which would sync CUDA).
        self.BW = self.B + 1
        self._torch_generator = None
        try:
            self._torch_generator = torch.Generator(device=self.device)
            self._torch_generator.manual_seed(self.rng.randrange(1, 2**31 - 1))
        except Exception:
            self._torch_generator = None
        # Keep rollout inference tensors separate when PPO is configured for the
        # other device. Calling policy._ensure_torch("cuda") here would reset the
        # CPU AdamW state every hybrid cycle when the user deliberately trains PPO
        # on CPU. Use the learner's parameters only when they already live on this
        # exact device; otherwise make an inference-only snapshot from NumPy weights.
        learner_params = getattr(self.policy, "_torch_params", None)
        learner_device = str(getattr(self.policy, "_torch_device", ""))
        if learner_params is not None and learner_device == str(self.device):
            self.p = learner_params
            self._owns_policy_snapshot = False
        else:
            self.p = {k: torch.as_tensor(np.asarray(v).copy(), dtype=torch.float32, device=self.device)
                      for k, v in self.policy.params.items()}
            self._owns_policy_snapshot = True
        self.width = int(getattr(self.policy, "width", 0))
        self.depth = int(getattr(self.policy, "depth", 0))
        self.large_residual_active = bool(self.policy._large_residual_active()) if hasattr(self.policy, "_large_residual_active") else False
        # Bound rollout records using the actual configured composition, not the
        # theoretical maximum of 5 ships per side. The old E*5/E*10 estimate cut
        # 1v1 episodes unnecessarily short (especially at 256+ GPU arenas), which
        # discarded useful tactical history while leaving much of the record budget unused.
        if self.opponent_mode == "selfplay":
            records_per_decision = int(np.sum(self.blue_counts + self.red_counts))
        else:
            records_per_decision = int(np.sum(self.blue_counts))
        self.train_slots = max(1, int(round(records_per_decision / max(1, self.E))))
        self.requested_max_decisions = self.max_decisions
        # `max_decisions` is intentionally never shortened here: record-budget
        # adaptation takes place by reducing self.count above.
        self.last_cuda_graph_error = None
        self.last_cuda_graph_traceback = None
        self.last_cuda_graph_attempted = False
        self._graph_failure_key = (
            str(self.dev), self.E, self.S, self.B, self.max_decisions,
            self.width, self.depth, self.large_residual_active, self.opponent_mode,
        )
        if use_cuda_graph is None:
            graph_env = os.environ.get("GRAVITRUEX_CUDA_GRAPH", "auto").strip().lower()
            if graph_env in {"0", "false", "no", "off", "disable", "disabled"}:
                self.use_cuda_graph = False
            elif graph_env in {"1", "true", "yes", "on", "enable", "enabled"}:
                # Explicit on deliberately bypasses the failure cache for diagnosis.
                self.use_cuda_graph = self.dev.type == "cuda"
            else:
                with _CUDA_GRAPH_FAILED_LOCK:
                    previously_failed = self._graph_failure_key in _CUDA_GRAPH_FAILED_KEYS
                self.use_cuda_graph = self.dev.type == "cuda" and not previously_failed
                if previously_failed:
                    self.last_cuda_graph_error = "Skipped in auto mode: capture previously failed for this shape/configuration in this process. Set GRAVITRUEX_CUDA_GRAPH=1 to retry."
        else:
            self.use_cuda_graph = bool(use_cuda_graph)
        self.blue_only_observations = (
            (self.opponent_mode != "selfplay") if blue_only_observations is None else bool(blue_only_observations)
        )
        self._graph = None
        self._graph_mode = False
        self._graph_seed = self.rng.randrange(1, 2**31 - 1)
        self._initialize_state()

    # ------------------------------------------------------------------ setup
    def _rand(self, shape, lo=0.0, hi=1.0):
        # Inside a CUDA graph the default generator is the only capture safe
        # option: its philox offset lives on the device and advances on every
        # replay, while a custom generator would freeze the same values.
        if self._graph_mode or self._torch_generator is None:
            r = self.t.rand(shape, device=self.dev)
        else:
            r = self.t.rand(shape, generator=self._torch_generator, device=self.dev)
        return lo + (hi - lo) * r

    def _initialize_state(self):
        t, E, S, B, dev, e = self.t, self.E, self.S, self.B, self.dev, self.e
        self.ship_slot = t.arange(S, device=dev).view(1, S).expand(E, S)
        self.team = (t.arange(S, device=dev) >= 5).long().view(1, S).expand(E, S)
        bn = t.as_tensor(self.blue_counts, dtype=t.long, device=dev).view(E, 1)
        rn = t.as_tensor(self.red_counts, dtype=t.long, device=dev).view(E, 1)
        slots = t.arange(5, device=dev).view(1, 5)
        self.present = t.cat((slots < bn, slots < rn), dim=1)
        model_slots = self.present if self.opponent_mode == "selfplay" else (self.present & (self.team == 0))
        self.policy_indices = t.where(model_slots.reshape(-1))[0]
        self.alive = self.present.clone()
        self.x = t.zeros((E, S), dtype=t.float32, device=dev)
        self.y = t.zeros_like(self.x)
        self.vx = (self._rand((E, S), -10.0, 10.0) * self.present.float())
        self.vy = (self._rand((E, S), -10.0, 10.0) * self.present.float())
        # Spawn roughly in the center engagement zone; ship IDs have fixed team slots.
        center_x, center_y = e.WORLD_W * .5, e.WORLD_H * .5
        off = min(500.0, e.WORLD_W * .16)
        for s in range(S):
            if s < 5:
                rank = float(s)
                count = bn[:, 0].float()
                valid = self.present[:, s]
                base_x = center_x - off
                base_y = center_y + (rank - (count - 1.0) * .5) * 78.0
                self.x[:, s] = (base_x + self._rand((E,), -55, 55)) * valid.float()
                self.y[:, s] = (base_y + self._rand((E,), -32, 32)) * valid.float()
            else:
                rank = float(s - 5)
                count = rn[:, 0].float()
                valid = self.present[:, s]
                base_x = center_x + off
                base_y = center_y + (rank - (count - 1.0) * .5) * 78.0
                self.x[:, s] = (base_x + self._rand((E,), -55, 55)) * valid.float()
                self.y[:, s] = (base_y + self._rand((E,), -32, 32)) * valid.float()
        self.heading = t.zeros((E, S), dtype=t.float32, device=dev)
        self.heading[:, :5] = self._rand((E, 5), -.18, .18)
        self.heading[:, 5:] = math.pi + self._rand((E, 5), -.18, .18)
        self.hp = self.present.float() * e.MAX_HP
        self.shield = self.present.float() * e.MAX_SHIELD
        self.cooldown = self._rand((E, S), .05, .35) * self.present.float()
        self.last_hit_time = t.full((E, S), -100.0, dtype=t.float32, device=dev)
        self.last_turn = t.zeros((E, S), dtype=t.float32, device=dev)
        self.damage_dealt = t.zeros((E, S), dtype=t.float32, device=dev)
        self.damage_taken = t.zeros_like(self.damage_dealt)
        self.kills = t.zeros((E, S), dtype=t.float32, device=dev)
        self.shots = t.zeros((E, S), dtype=t.float32, device=dev)
        self.deaths = t.zeros((E, S), dtype=t.float32, device=dev)
        self.bx = t.zeros((E, self.BW), dtype=t.float32, device=dev)
        self.by = t.zeros_like(self.bx)
        self.bvx = t.zeros_like(self.bx)
        self.bvy = t.zeros_like(self.bx)
        self.bteam = t.zeros((E, self.BW), dtype=t.long, device=dev)
        self.bowner = t.full((E, self.BW), -1, dtype=t.long, device=dev)
        self.bttl = t.zeros_like(self.bx)
        self.balive = t.zeros((E, self.BW), dtype=t.bool, device=dev)
        self.sim_time = t.zeros(E, dtype=t.float32, device=dev)
        self.step_count = t.zeros(E, dtype=t.long, device=dev)
        self.active_env = t.ones(E, dtype=t.bool, device=dev)
        self.slot_team = (t.arange(S, device=dev) >= 5).long().view(1, S).expand(E, S)
        self._identity = t.eye(S, dtype=t.bool, device=dev).view(1, S, S)
        # Slots that need an observation vector. In heuristic mode only the blue
        # ships are trained, so the enemy observations were pure waste.
        if self.blue_only_observations:
            self.obs_slots = t.arange(5, device=dev)
        else:
            self.obs_slots = t.arange(S, device=dev)

    # ---------------------------------------------------------------- helpers
    def _wrap(self, delta, size):
        return self.t.where(delta > size * .5, delta - size,
                            self.t.where(delta < -size * .5, delta + size, delta))

    def _angle(self, a):
        return self.t.atan2(self.t.sin(a), self.t.cos(a))

    def _pairwise(self):
        # [E, observing ship i, observed ship j]
        dx = self._wrap(self.x[:, None, :] - self.x[:, :, None], self.e.WORLD_W)
        dy = self._wrap(self.y[:, None, :] - self.y[:, :, None], self.e.WORLD_H)
        rvx = self.vx[:, None, :] - self.vx[:, :, None]
        rvy = self.vy[:, None, :] - self.vy[:, :, None]
        dist = self.t.sqrt(dx * dx + dy * dy + 1e-8)
        pair_alive = self.alive[:, :, None] & self.alive[:, None, :]
        enemy = pair_alive & (self.team[:, :, None] != self.team[:, None, :])
        ally = pair_alive & (self.team[:, :, None] == self.team[:, None, :]) & ~self._identity
        return dx, dy, rvx, rvy, dist, enemy, ally

    def _pair_gather(self, mat, idx):
        return self.t.gather(mat, 2, idx)

    def _rank_ships(self, dist, valid, k):
        """k nearest ships per observer, via one stable sort.

        `torch.topk` was called six times per decision (3 nearest enemies, 2
        extra enemies, 3 nearest allies, 2 extra allies and two bullet
        rankings). A single stable sort per relation gives the whole ranking,
        so the "nearest" and the "extra" selections share one computation.
        """
        masked = self.t.where(valid, dist, self.t.full_like(dist, float("inf")))
        values, indices = self.t.sort(masked, dim=-1, stable=True)
        indices = indices[..., :k]
        good = self.t.isfinite(values[..., :k])
        return indices, good

    def _rank_bullets(self, dist, valid, k):
        masked = self.t.where(valid, dist, self.t.full_like(dist, float("inf")))
        values, indices = self.t.topk(masked, k=k, dim=-1, largest=False, sorted=True)
        good = self.t.isfinite(values)
        return indices, good

    def _bullet_rel(self):
        # [E,S,B], vector from ship to bullet and velocity relative to ship.
        dx = self._wrap(self.bx[:, None, :] - self.x[:, :, None], self.e.WORLD_W)
        dy = self._wrap(self.by[:, None, :] - self.y[:, :, None], self.e.WORLD_H)
        rvx = self.bvx[:, None, :] - self.vx[:, :, None]
        rvy = self.bvy[:, None, :] - self.vy[:, :, None]
        dist = self.t.sqrt(dx * dx + dy * dy + 1e-8)
        valid = (self.balive[:, None, :] & self.alive[:, :, None] &
                 (self.bteam[:, None, :] != self.team[:, :, None]))
        return dx, dy, rvx, rvy, dist, valid

    def _target(self, dx, dy, rvx, rvy, dist, enemy_valid):
        """Nearest enemy selection shared by the observation, aim and heuristic."""
        t, e = self.t, self.e
        score = dist * (.82 + .18 * self.hp[:, None, :] / e.MAX_HP)
        score = t.where(enemy_valid, score, t.full_like(score, float("inf")))
        target_idx = score.argmin(dim=-1)
        has_target = enemy_valid.any(dim=-1)
        tx = self._pair_gather(dx, target_idx[..., None]).squeeze(-1)
        ty = self._pair_gather(dy, target_idx[..., None]).squeeze(-1)
        tvx = self._pair_gather(rvx, target_idx[..., None]).squeeze(-1)
        tvy = self._pair_gather(rvy, target_idx[..., None]).squeeze(-1)
        tdist = t.sqrt(tx * tx + ty * ty + 1e-8)
        thp = self.hp.gather(1, target_idx) / e.MAX_HP
        tshield = self.shield.gather(1, target_idx) / e.MAX_SHIELD
        return target_idx, has_target, tx, ty, tvx, tvy, tdist, thp, tshield

    def _build_context(self, with_bullets: bool = True) -> _Ctx:
        dx, dy, rvx, rvy, dist, enemy, ally = self._pairwise()
        target_idx, has_target, tx, ty, tvx, tvy, tdist, thp, tshield = self._target(
            dx, dy, rvx, rvy, dist, enemy)
        ctx = _Ctx(dx=dx, dy=dy, rvx=rvx, rvy=rvy, dist=dist,
                   enemy_valid=enemy, ally_valid=ally,
                   target_idx=target_idx, has_target=has_target,
                   tx=tx, ty=ty, tvx=tvx, tvy=tvy, tdist=tdist,
                   thp=thp, tshield=tshield)
        if with_bullets:
            bdx, bdy, bvx, bvy, bdist, bvalid = self._bullet_rel()
            ctx.bdx, ctx.bdy, ctx.bvx, ctx.bvy = bdx, bdy, bvx, bvy
            ctx.bdist, ctx.bvalid = bdist, bvalid
        return ctx

    # ------------------------------------------------------------ observation
    def _observe(self, ctx: _Ctx):
        t, e = self.t, self.e
        slots = self.obs_slots
        if slots.shape[0] != self.S:
            dx = ctx.dx.index_select(1, slots)
            dy = ctx.dy.index_select(1, slots)
            rvx = ctx.rvx.index_select(1, slots)
            rvy = ctx.rvy.index_select(1, slots)
            dist = ctx.dist.index_select(1, slots)
            enemy_valid = ctx.enemy_valid.index_select(1, slots)
            ally_valid = ctx.ally_valid.index_select(1, slots)
            target_idx = ctx.target_idx.index_select(1, slots)
            has_target = ctx.has_target.index_select(1, slots)
            tx = ctx.tx.index_select(1, slots)
            ty = ctx.ty.index_select(1, slots)
            tvx = ctx.tvx.index_select(1, slots)
            tvy = ctx.tvy.index_select(1, slots)
            tdist = ctx.tdist.index_select(1, slots)
            thp = ctx.thp.index_select(1, slots)
            tshield = ctx.tshield.index_select(1, slots)
            bdx = ctx.bdx.index_select(1, slots)
            bdy = ctx.bdy.index_select(1, slots)
            bvx = ctx.bvx.index_select(1, slots)
            bvy = ctx.bvy.index_select(1, slots)
            bdist = ctx.bdist.index_select(1, slots)
            bvalid = ctx.bvalid.index_select(1, slots)
            own_hp = self.hp.index_select(1, slots)
            own_shield = self.shield.index_select(1, slots)
            own_cooldown = self.cooldown.index_select(1, slots)
            own_last_turn = self.last_turn.index_select(1, slots)
            own_vx = self.vx.index_select(1, slots)
            own_vy = self.vy.index_select(1, slots)
            own_heading = self.heading.index_select(1, slots)
            own_team = self.team.index_select(1, slots)
            alive_slots = self.alive.index_select(1, slots)
        else:
            dx, dy, rvx, rvy = ctx.dx, ctx.dy, ctx.rvx, ctx.rvy
            dist, enemy_valid, ally_valid = ctx.dist, ctx.enemy_valid, ctx.ally_valid
            target_idx, has_target = ctx.target_idx, ctx.has_target
            tx, ty, tvx, tvy, tdist = ctx.tx, ctx.ty, ctx.tvx, ctx.tvy, ctx.tdist
            thp, tshield = ctx.thp, ctx.tshield
            bdx, bdy, bvx, bvy = ctx.bdx, ctx.bdy, ctx.bvx, ctx.bvy
            bdist, bvalid = ctx.bdist, ctx.bvalid
            own_hp, own_shield = self.hp, self.shield
            own_cooldown, own_last_turn = self.cooldown, self.last_turn
            own_vx, own_vy = self.vx, self.vy
            own_heading = self.heading
            own_team, alive_slots = self.team, self.alive

        c, s = t.cos(own_heading), t.sin(own_heading)
        local_vx = (own_vx * c + own_vy * s) / e.MAX_SPEED
        local_vy = (-own_vx * s + own_vy * c) / e.MAX_SPEED
        # The ally ranking refers to ship ids, so the gather below needs the
        # full speed table, not the per-observed-slot one.
        full_speed = t.sqrt(self.vx * self.vx + self.vy * self.vy)
        speed = full_speed if slots.shape[0] == self.S else full_speed.index_select(1, slots)
        terr = self._angle(t.atan2(ty, tx) - own_heading)
        closing_dot = (tvx * tx + tvy * ty) / (tdist * e.MAX_SPEED * 2)
        closing_dot = t.clamp(closing_dot, -1, 1)
        feat = [
            t.clamp(local_vx, -1, 1), t.clamp(local_vy, -1, 1),
            t.clamp(speed / e.MAX_SPEED, 0, 1), own_hp / e.MAX_HP,
            own_shield / e.MAX_SHIELD, t.clamp(own_cooldown / e.FIRE_COOLDOWN, 0, 1),
            own_last_turn, t.ones_like(local_vx),
            t.where(has_target, t.clamp(tx / e.OBS_POS_X, -1, 1), 0),
            t.where(has_target, t.clamp(ty / e.OBS_POS_Y, -1, 1), 0),
            t.where(has_target, t.clamp(tvx / (e.MAX_SPEED * 2), -1, 1), 0),
            t.where(has_target, t.clamp(tvy / (e.MAX_SPEED * 2), -1, 1), 0),
            t.where(has_target, t.sin(terr), 0), t.where(has_target, t.cos(terr), 0),
            t.where(has_target, t.clamp(tdist / 650.0, 0, 1), 0),
            t.where(has_target, thp, 0), t.where(has_target, tshield, 0),
            t.where(has_target, closing_dot, 0),
        ]
        # Three nearest enemies. The "two additional closest enemies" block
        # below re-uses rank 0 and 1 of the very same ranking: the scalar
        # Arena does exactly that, so a deeper ranking would only waste work.
        ei, egood = self._rank_ships(dist, enemy_valid, 3)
        for j in range(3):
            idx = ei[..., j]
            good = egood[..., j]
            for source, denom in ((dx, e.OBS_POS_X), (dy, e.OBS_POS_Y),
                                  (rvx, e.MAX_SPEED * 2), (rvy, e.MAX_SPEED * 2)):
                v = self._pair_gather(source, idx.unsqueeze(-1)).squeeze(-1)
                feat.append(t.where(good, t.clamp(v / denom, -1, 1), 0))
            hp_j = self.hp.gather(1, idx) / e.MAX_HP
            sh_j = self.shield.gather(1, idx) / e.MAX_SHIELD
            feat.extend([t.where(good, hp_j, 0), t.where(good, sh_j, 0)])
        # Three nearest allies; the extra ally block re-uses rank 0 and 1.
        ai, agood = self._rank_ships(dist, ally_valid, 3)
        for j in range(3):
            idx = ai[..., j]
            good = agood[..., j]
            for source, denom in ((dx, e.OBS_POS_X), (dy, e.OBS_POS_Y)):
                v = self._pair_gather(source, idx.unsqueeze(-1)).squeeze(-1)
                feat.append(t.where(good, t.clamp(v / denom, -1, 1), 0))
            ahp = self.hp.gather(1, idx) / e.MAX_HP
            aspeed = t.gather(full_speed, 1, idx) / e.MAX_SPEED
            feat.extend([t.where(good, ahp, 0), t.where(good, t.clamp(aspeed, 0, 1), 0)])
        # Incoming projectiles: the two nearest ones cover both the detailed
        # block and the extra bullet block below.
        bi, bgood = self._rank_bullets(bdist, bvalid, 2)
        for j in range(2):
            idx = bi[..., j]
            good = bgood[..., j]
            for source, denom in ((bdx, e.OBS_POS_X), (bdy, e.OBS_POS_Y),
                                  (bvx, e.BULLET_SPEED * 1.5), (bvy, e.BULLET_SPEED * 1.5)):
                v = t.gather(source, 2, idx.unsqueeze(-1)).squeeze(-1)
                feat.append(t.where(good, t.clamp(v / denom, -1, 1), 0))
        # Team/urgency summary.
        blue_alive = ((self.team == 0) & self.alive).sum(dim=1).float()
        red_alive = ((self.team == 1) & self.alive).sum(dim=1).float()
        own_count = t.where(own_team == 0, blue_alive[:, None], red_alive[:, None])
        opp_count = t.where(own_team == 0, red_alive[:, None], blue_alive[:, None])
        own_mask = (self.team[:, None, :] == own_team[:, :, None]) & self.alive[:, None, :]
        opp_mask = (self.team[:, None, :] != own_team[:, :, None]) & self.alive[:, None, :]
        own_f = own_mask.float()
        opp_f = opp_mask.float()
        own_mean = (self.hp[:, None, :] / e.MAX_HP * own_f).sum(dim=2) / own_f.sum(dim=2).clamp_min(1.0)
        opp_mean = (self.hp[:, None, :] / e.MAX_HP * opp_f).sum(dim=2) / opp_f.sum(dim=2).clamp_min(1.0)
        closest_dist, closest_idx = bdist.masked_fill(~bvalid, float("inf")).min(dim=-1)
        closest_has = t.isfinite(closest_dist)
        c_bdx = t.gather(bdx, 2, closest_idx.unsqueeze(-1)).squeeze(-1)
        c_bdy = t.gather(bdy, 2, closest_idx.unsqueeze(-1)).squeeze(-1)
        c_bvx = t.gather(bvx, 2, closest_idx.unsqueeze(-1)).squeeze(-1)
        c_bvy = t.gather(bvy, 2, closest_idx.unsqueeze(-1)).squeeze(-1)
        c_dist = closest_dist.clamp_min(1.0)
        min_closing = -((c_bvx * c_bdx + c_bvy * c_bdy) / c_dist) / e.MAX_SPEED
        bullet_t = t.clamp(closest_dist / 240.0, 0, 1)
        bullet_t = t.where(closest_has, bullet_t, 1.0)
        min_closing = t.where(closest_has, t.clamp(min_closing, -1, 1), 0.0)
        feat.extend([
            own_count / 5.0, opp_count / 5.0, own_mean, opp_mean,
            bullet_t, min_closing,
            t.where(has_target, t.clamp(terr / math.pi, -1, 1), 0),
            t.where(has_target, t.clamp(tdist / 650.0, 0, 1), 1),
        ])
        # Tactical target context.
        trvx, trvy = tvx, tvy
        tdist_safe = tdist.clamp_min(1.0)
        closing = -(trvx * tx + trvy * ty) / tdist_safe
        ttc = t.where(closing > 0, tdist / closing.clamp_min(1.0), 5.0)
        lead_t = t.clamp(tdist / e.BULLET_SPEED, 0, .72)
        lead_err = self._angle(t.atan2(ty + trvy * lead_t, tx + trvx * lead_t) - own_heading)
        vel_angle = self._angle(t.atan2(trvy, trvx) - own_heading)
        vel_nonzero = (trvx.abs() + trvy.abs()) > 1e-5
        feat.extend([
            t.where(has_target, t.clamp(tdist / 1800.0, 0, 1), 0),
            t.where(has_target, t.clamp(closing / (e.MAX_SPEED * 1.5), -1, 1), 0),
            t.where(has_target, t.clamp(ttc / 5.0, 0, 1), 0),
            t.where(has_target, t.sin(lead_err), 0), t.where(has_target, t.cos(lead_err), 0),
            t.where(has_target, t.clamp(t.sqrt(trvx * trvx + trvy * trvy) / (e.MAX_SPEED * 2), 0, 1), 0),
            t.where(has_target & vel_nonzero, t.sin(vel_angle), 0),
            t.where(has_target & vel_nonzero, t.cos(vel_angle), 0),
        ])
        # Two more allies. The scalar Arena repeats the two nearest allies
        # here instead of skipping the ones used above, so ranks 0 and 1 are
        # used on purpose: changing this would change the observation layout
        # of every already trained policy.
        for j in (0, 1):
            idx = ai[..., j]
            good = agood[..., j]
            for source, denom in ((dx, 1500.0), (dy, 1500.0), (rvx, e.MAX_SPEED * 2), (rvy, e.MAX_SPEED * 2)):
                v = self._pair_gather(source, idx.unsqueeze(-1)).squeeze(-1)
                feat.append(t.where(good, t.clamp(v / denom, -1, 1), 0))
            vhp = t.gather((self.hp / e.MAX_HP)[:, None, :].expand(-1, self.S, -1), 2, idx.unsqueeze(-1)).squeeze(-1)
            feat.append(t.where(good, vhp, 0))
        # Two additional closest enemies: same rank 0 and 1 as above.
        for j in (0, 1):
            idx = ei[..., j]
            good = egood[..., j]
            edx = self._pair_gather(dx, idx.unsqueeze(-1)).squeeze(-1)
            edy = self._pair_gather(dy, idx.unsqueeze(-1)).squeeze(-1)
            evx = self._pair_gather(rvx, idx.unsqueeze(-1)).squeeze(-1)
            evy = self._pair_gather(rvy, idx.unsqueeze(-1)).squeeze(-1)
            vv = evx * evx + evy * evy
            tca = t.clamp(-(edx * evx + edy * evy) / vv.clamp_min(1e-5), 0, 2.5)
            miss = t.sqrt((edx + evx * tca) ** 2 + (edy + evy * tca) ** 2)
            d = t.sqrt(edx * edx + edy * edy).clamp_min(1.0)
            c_e = -(edx * evx + edy * evy) / d
            angle_e = self._angle(t.atan2(edy, edx) - own_heading)
            feat.extend([
                t.where(good, t.clamp(tca / 2.5, 0, 1), 0),
                t.where(good, t.clamp(miss / 500.0, 0, 1), 0),
                t.where(good, t.clamp(c_e / (e.MAX_SPEED * 1.5), -1, 1), 0),
                t.where(good, t.sin(angle_e), 0),
            ])
        # Bullet closest approach context. Same two nearest projectiles as above.
        for j in (0, 1):
            idx = bi[..., j]
            good = bgood[..., j]
            edx = t.gather(bdx, 2, idx.unsqueeze(-1)).squeeze(-1)
            edy = t.gather(bdy, 2, idx.unsqueeze(-1)).squeeze(-1)
            evx = t.gather(bvx, 2, idx.unsqueeze(-1)).squeeze(-1)
            evy = t.gather(bvy, 2, idx.unsqueeze(-1)).squeeze(-1)
            vv = evx * evx + evy * evy
            tca = t.clamp(-(edx * evx + edy * evy) / vv.clamp_min(1e-5), 0, 2.5)
            miss = t.sqrt((edx + evx * tca) ** 2 + (edy + evy * tca) ** 2)
            feat.extend([t.where(good, t.clamp(tca / 2.5, 0, 1), 1.0),
                         t.where(good, t.clamp(miss / 60.0, 0, 1), 1.0)])
        # Nearby ally/enemy counts within 600 units.
        nearby_a = ((ally_valid & (dist < 600.0)).sum(dim=-1).float() / 4.0).clamp(0, 1)
        nearby_e = ((enemy_valid & (dist < 600.0)).sum(dim=-1).float() / 5.0).clamp(0, 1)
        feat.extend([nearby_a, nearby_e])
        obs = t.stack(feat, dim=-1).to(dtype=t.float32)
        if obs.shape[-1] != e.OBS_SIZE:
            raise RuntimeError(f"GPU observations have {obs.shape[-1]} features, expected {e.OBS_SIZE}")
        if slots.shape[0] != self.S:
            # Scatter the observed slots back into the full [E, S, OBS] layout.
            full = t.zeros((self.E, self.S, e.OBS_SIZE), dtype=t.float32, device=self.dev)
            full.index_copy_(1, slots, obs)
            obs = full
        return obs

    # ------------------------------------------------------------------- aim
    def _aim(self, ctx: _Ctx):
        t, e = self.t, self.e
        has = ctx.has_target
        tx, ty, tvx, tvy, tdist = ctx.tx, ctx.ty, ctx.tvx, ctx.tvy, ctx.tdist
        d = t.sqrt(tx * tx + ty * ty).clamp_min(1.0)
        lead = t.clamp(d / e.BULLET_SPEED, 0, .72)
        angle = t.atan2(ty + tvy * lead, tx + tvx * lead)
        aim = t.cos(self._angle(angle - self.heading))
        aim = t.where(has, aim, -1.0)
        d = t.where(has, d, t.full_like(d, float("inf")))
        return aim, d

    def _policy_forward(self, x):
        """Inference from this rollout's device-local weight snapshot."""
        t, p = self.t, self.p
        h = t.tanh(x @ p["W1"] + p["b1"])
        h = t.tanh(h @ p["W2"] + p["b2"])
        logits = {
            "turn": h @ p["Wt"] + p["bt"],
            "thrust": h @ p["Wf"] + p["bf"],
            "strafe": h @ p["Ws"] + p["bs"],
            "fire": (h @ p["Wfire"] + p["bfire"]).squeeze(-1),
        }
        value = ((h @ p["Wv"] + p["bv"]).squeeze(-1)
                 if getattr(self.policy, "architecture", "shared_residual") == "shared_residual" else None)
        if self.large_residual_active:
            g = x
            for i in range(self.depth):
                g = t.tanh(g @ p[f"G_W{i}"] + p[f"G_b{i}"])
            logits["turn"] = logits["turn"] + g @ p["G_Wt"] + p["G_bt"]
            logits["thrust"] = logits["thrust"] + g @ p["G_Wf"] + p["G_bf"]
            logits["strafe"] = logits["strafe"] + g @ p["G_Ws"] + p["G_bs"]
            logits["fire"] = logits["fire"] + (g @ p["G_Wfire"] + p["G_bfire"]).squeeze(-1)
            if getattr(self.policy, "architecture", "shared_residual") == "shared_residual":
                value = value + (g @ p["G_Wv"] + p["G_bv"]).squeeze(-1)
        if getattr(self.policy, "architecture", "shared_residual") == "separate_critic":
            c = x
            for i in range(self.depth):
                c = t.tanh(c @ p[f"C_W{i}"] + p[f"C_b{i}"])
            value = (c @ p["C_Wv"] + p["C_bv"]).squeeze(-1)
        # Match learner/live inference priors: steer toward the lead angle, dodge imminent
        # collision courses, and fire when the weapon is ready in a clean firing window.
        logits["turn"] = self.e.apply_target_turn_prior_torch(t, logits["turn"], x)
        logits["strafe"] = self.e.apply_dodge_prior_torch(t, logits["strafe"], x)
        logits["fire"] = self.e.apply_fire_prior_torch(t, logits["fire"], x)
        return logits, value

    def _sample_categorical(self, logits_stack):
        """Gumbel-max sampling for all categorical heads at once.

        `torch.multinomial` needs a normalization pass, a cumulative sum and a
        search per head. Gumbel-max needs one random tensor and one argmax, and
        it stays valid inside a CUDA graph, where multinomial does not.
        """
        t = self.t
        u = self._rand(tuple(logits_stack.shape), 1e-9, 1.0)
        gumbel = -t.log(-t.log(u))
        return (logits_stack + gumbel).argmax(dim=-1)

    def _policy_sample(self, obs):
        t = self.t
        flat_obs = obs.reshape(-1, self.e.OBS_SIZE)
        # The set of controlled slots is fixed for a rollout batch, so we avoid
        # running the large policy on empty slots or heuristic-controlled enemies.
        x = flat_obs.index_select(0, self.policy_indices)
        with t.no_grad():
            logits, values = self._policy_forward(x)
            head_logits = t.stack((logits["turn"], logits["thrust"], logits["strafe"]), dim=1)
            picks = self._sample_categorical(head_logits)
            logp = t.zeros(x.shape[0], dtype=t.float32, device=self.dev)
            for head in range(3):
                lp = t.log_softmax(head_logits[:, head], dim=-1)
                a = picks[:, head]
                logp = logp + lp.gather(1, a[:, None]).squeeze(-1)
            fire_logits = logits["fire"].reshape(-1)
            fire_p = t.sigmoid(fire_logits)
            fire = (self._rand((x.shape[0],)) < fire_p).long()
            logp = logp + -t.nn.functional.binary_cross_entropy_with_logits(
                fire_logits, fire.float(), reduction="none")
            selected_actions = t.cat((picks, fire[:, None]), dim=-1)
            selected_values = values.reshape(-1)
        # Full-size outputs simplify physics integration; unselected slots receive coast.
        full_actions = t.zeros((self.E * self.S, 4), dtype=t.long, device=self.dev)
        full_actions[:, :3] = 1
        full_logp = t.zeros(self.E * self.S, dtype=t.float32, device=self.dev)
        full_values = t.zeros_like(full_logp)
        full_actions.index_copy_(0, self.policy_indices, selected_actions)
        full_logp.index_copy_(0, self.policy_indices, logp)
        full_values.index_copy_(0, self.policy_indices, selected_values)
        return (full_actions.reshape(self.E, self.S, 4),
                full_logp.reshape(self.E, self.S), full_values.reshape(self.E, self.S))

    def _heuristic_actions(self, ctx: _Ctx):
        t, e = self.t, self.e
        has = ctx.has_target
        tx, ty, tvx, tvy = ctx.tx, ctx.ty, ctx.tvx, ctx.tvy
        d = t.sqrt(tx * tx + ty * ty).clamp_min(1.0)
        lead = t.clamp(d / e.BULLET_SPEED, 0, .72)
        desired = t.atan2(ty + tvy * lead, tx + tvx * lead)
        err = self._angle(desired - self.heading)
        turn = t.where(err.abs() < .045, 0, t.where(err > 0, 1, -1))
        thrust = t.where(d > 285, 1, t.where(d < 185, -1, 0))
        slots = t.arange(self.S, device=self.dev).view(1, self.S).expand(self.E, self.S)
        strafe = t.where(slots % 2 == 0, 1, -1)
        bdx, bdy, bvx, bvy = ctx.bdx, ctx.bdy, ctx.bvx, ctx.bvy
        bvalid = ctx.bvalid
        vv = bvx * bvx + bvy * bvy
        tc = -(bdx * bvx + bdy * bvy) / vv.clamp_min(1e-5)
        tc = tc.clamp(0, .72)
        miss_x = bdx + bvx * tc
        miss_y = bdy + bvy * tc
        miss = t.sqrt(miss_x * miss_x + miss_y * miss_y)
        danger = bvalid & (vv > 1e-5) & (-(bdx * bvx + bdy * bvy) / vv.clamp_min(1e-5) > 0) & (miss < 27)
        score_d = t.where(danger, (27 - miss) / tc.clamp_min(.08), t.zeros_like(miss))
        best_score, bi = score_d.max(dim=-1)
        cx = t.gather(miss_x, 2, bi.unsqueeze(-1)).squeeze(-1)
        cy = t.gather(miss_y, 2, bi.unsqueeze(-1)).squeeze(-1)
        nx, ny = -t.sin(self.heading), t.cos(self.heading)
        dot = nx * (-cx) + ny * (-cy)
        dodge = best_score > 0
        strafe = t.where(dodge, t.where(dot >= 0, 1, -1), strafe)
        thrust = t.where(dodge & (d < 230), 1, thrust)
        fire = (err.abs() < .105) & (d < 720) & (self.cooldown <= .06) & has
        turn = t.where(has, turn, 0); thrust = t.where(has, thrust, 0); strafe = t.where(has, strafe, 0)
        # Return categorical action indices for consistency with PPO records.
        return t.stack((turn + 1, thrust + 1, strafe + 1, fire.long()), dim=-1)

    # --------------------------------------------------------------- physics
    def _physics_tick(self, acts, tick_active):
        t, e, dt = self.t, self.e, self.e.DT
        tick_active = tick_active & self.active_env
        active_ships = self.alive & tick_active[:, None]
        self.sim_time += tick_active.float() * dt
        self.step_count += tick_active.long()
        turn = acts[:, :, 0].float() - 1.0
        thrust = acts[:, :, 1].float() - 1.0
        strafe = acts[:, :, 2].float() - 1.0
        fire = acts[:, :, 3] > 0
        self.heading = self._angle(self.heading + turn * e.TURN_RATE * dt * active_ships.float())
        fx, fy = t.cos(self.heading), t.sin(self.heading)
        lx, ly = -fy, fx
        self.vx = self.vx + (fx * thrust * e.ACCEL_FORWARD + lx * strafe * e.ACCEL_STRAFE) * dt * active_ships.float()
        self.vy = self.vy + (fy * thrust * e.ACCEL_FORWARD + ly * strafe * e.ACCEL_STRAFE) * dt * active_ships.float()
        drag = math.exp(-e.DRAG * dt)
        vx_new, vy_new = self.vx * drag, self.vy * drag
        speed = t.sqrt(vx_new * vx_new + vy_new * vy_new).clamp_min(1e-6)
        scale = t.clamp(e.MAX_SPEED / speed, max=1.0)
        self.vx = t.where(active_ships, vx_new * scale, self.vx)
        self.vy = t.where(active_ships, vy_new * scale, self.vy)
        self.x = t.where(active_ships, t.remainder(self.x + self.vx * dt, e.WORLD_W), self.x)
        self.y = t.where(active_ships, t.remainder(self.y + self.vy * dt, e.WORLD_H), self.y)
        self.cooldown = t.where(active_ships, (self.cooldown - dt).clamp_min(0), self.cooldown)
        regen = active_ships & (self.shield < e.MAX_SHIELD) & ((self.sim_time[:, None] - self.last_hit_time) > e.SHIELD_DELAY)
        self.shield = t.where(regen, (self.shield + e.SHIELD_REGEN * dt).clamp_max(e.MAX_SHIELD), self.shield)
        # Allocate simultaneous shots to distinct free projectile slots by ranks.
        can_fire = active_ships & fire & (self.cooldown <= 1e-6)
        # Rank the free projectile slots with a cumsum. The previous topk over
        # 256 entries was one of the most expensive kernels in the tick.
        free_slots = ~self.balive[:, :self.B]
        free_count = free_slots.sum(dim=1)
        bidx = t.arange(self.B, device=self.dev).view(1, self.B).expand(self.E, self.B)
        free_rank = free_slots.long().cumsum(dim=1) - 1
        slot_table = t.full((self.E, self.S + 1), self.B, dtype=t.long, device=self.dev)
        rank_index = t.where(free_slots & (free_rank < self.S),
                              free_rank.clamp_min(0),
                              t.full_like(free_rank, self.S))
        slot_table.scatter_(1, rank_index, bidx)
        free_sorted = slot_table[:, :self.S]
        fire_rank = can_fire.long().cumsum(dim=1) - 1
        slot_idx = free_sorted.gather(1, fire_rank.clamp_min(0).clamp_max(self.S - 1))
        has_room = fire_rank < free_count[:, None]
        can_fire = can_fire & has_room & (slot_idx < self.B)
        # Spawn without a data dependent index list. `torch.where(condition)`
        # with a single argument is a nonzero() and forces a device->host
        # synchronization on every tick; ships that do not fire are pointed at
        # the trash column instead.
        slot_padded = t.where(can_fire, slot_idx, t.full_like(slot_idx, self.B))
        env_index = t.arange(self.E, device=self.dev).view(self.E, 1).expand(self.E, self.S)
        ship_index = t.arange(self.S, device=self.dev).view(1, self.S).expand(self.E, self.S)
        spawn_x = t.remainder(self.x + t.cos(self.heading) * (e.SHIP_RADIUS + 4), e.WORLD_W)
        spawn_y = t.remainder(self.y + t.sin(self.heading) * (e.SHIP_RADIUS + 4), e.WORLD_H)
        spawn_vx = self.vx + t.cos(self.heading) * e.BULLET_SPEED
        spawn_vy = self.vy + t.sin(self.heading) * e.BULLET_SPEED
        self.bx[env_index, slot_padded] = spawn_x
        self.by[env_index, slot_padded] = spawn_y
        self.bvx[env_index, slot_padded] = spawn_vx
        self.bvy[env_index, slot_padded] = spawn_vy
        self.bteam[env_index, slot_padded] = self.team
        self.bowner[env_index, slot_padded] = ship_index
        # Use a same-device, preallocated value tensor. During CUDA graph capture,
        # indexed assignment from a CPU scalar/tensor can raise:
        # "Cannot copy between CPU and CUDA tensors ... unless the CPU tensor is pinned".
        # Every spawned projectile gets the same TTL; writes to the trash slot are
        # harmless because that column is never marked alive.
        self.bttl.scatter_(1, slot_padded, self._bullet_ttl_spawn_values)
        self.balive[env_index, slot_padded] = can_fire
        self.shots = self.shots + can_fire.float()
        self.cooldown = t.where(can_fire, t.full_like(self.cooldown, e.FIRE_COOLDOWN), self.cooldown)
        active_bullets = self.balive & tick_active[:, None]
        self.bx = t.where(active_bullets, t.remainder(self.bx + self.bvx * dt, e.WORLD_W), self.bx)
        self.by = t.where(active_bullets, t.remainder(self.by + self.bvy * dt, e.WORLD_H), self.by)
        self.bttl = t.where(active_bullets, self.bttl - dt, self.bttl)
        self.balive = self.balive & (self.bttl > 0)
        # Simultaneous projectile/ship collision using [arena, bullet, ship] tensors.
        cdx = self._wrap(self.x[:, None, :] - self.bx[:, :, None], e.WORLD_W)
        cdy = self._wrap(self.y[:, None, :] - self.by[:, :, None], e.WORLD_H)
        d2 = cdx * cdx + cdy * cdy
        slot_ids = t.arange(self.S, device=self.dev).view(1, 1, self.S)
        eligible = (self.alive[:, None, :] & tick_active[:, None, None] & self.balive[:, :, None] &
                    (self.team[:, None, :] != self.bteam[:, :, None]) & (slot_ids != self.bowner[:, :, None]))
        hit_matrix = eligible & (d2 <= (e.SHIP_RADIUS + e.BULLET_RADIUS) ** 2)
        hit_any = hit_matrix.any(dim=-1)
        hit_ship_idx = t.where(hit_matrix, d2, t.full_like(d2, float("inf"))).argmin(dim=-1)
        per_ship_damage = t.zeros((self.E, self.S), dtype=t.float32, device=self.dev)
        per_ship_damage.scatter_add_(1, hit_ship_idx, hit_any.float() * e.BULLET_DAMAGE)
        had_damage = per_ship_damage > 0
        old_alive = self.alive.clone()
        shield_damage = t.minimum(self.shield, per_ship_damage)
        self.shield = (self.shield - shield_damage).clamp_min(0)
        self.hp = self.hp - (per_ship_damage - shield_damage)
        self.damage_taken += per_ship_damage
        self.last_hit_time = t.where(had_damage, self.sim_time[:, None], self.last_hit_time)
        # Attribute damage to projectile owners.
        owner_damage = t.zeros((self.E, self.S), dtype=t.float32, device=self.dev)
        owner_valid = hit_any & (self.bowner >= 0)
        owner_damage.scatter_add_(1, self.bowner.clamp_min(0), owner_valid.float() * e.BULLET_DAMAGE)
        self.damage_dealt += owner_damage
        self.balive = self.balive & ~hit_any
        just_dead = self.alive & (self.hp <= 0) & tick_active[:, None]
        self.hp = t.where(just_dead, t.zeros_like(self.hp), self.hp)
        self.alive = self.alive & ~just_dead
        self.deaths += just_dead.float()
        # Kill credit: assign each newly dead ship to the first projectile that hit it this tick.
        hit_by_target = hit_matrix.permute(0, 2, 1)  # [E,S,B]
        b_ids = t.arange(self.BW, device=self.dev).view(1, 1, self.BW).expand(self.E, self.S, self.BW)
        first_b = t.where(hit_by_target, b_ids, self.BW).amin(dim=-1).clamp(max=self.B - 1)
        owner_for_ship = t.gather(self.bowner, 1, first_b)
        credit_mask = just_dead & (owner_for_ship >= 0)
        self.kills.scatter_add_(1, owner_for_ship.clamp_min(0), credit_mask.float())
        # Episode stops when one side has no ships left; later ticks do no work for it.
        blue_live = ((self.team == 0) & self.alive).sum(dim=1)
        red_live = ((self.team == 1) & self.alive).sum(dim=1)
        eliminated = (blue_live == 0) | (red_live == 0)
        self.active_env = self.active_env & ~eliminated

    # --------------------------------------------------------- decision step
    def _step(self):
        """One policy decision: observe -> act -> physics -> reward."""
        t, e = self.t, self.e
        ctx = self._build_context(with_bullets=True)
        valid = self.alive & self.active_env[:, None]
        obs = self._observe(ctx)
        # Both teams contribute records in self-play; otherwise only blue is trained.
        if self.opponent_mode == "selfplay":
            train_mask = valid
        else:
            train_mask = valid & (self.team == 0)
        acts, logp, values = self._policy_sample(obs)
        # The rule-based pilot is a teacher only in heuristic-opponent mode. Keep
        # the labels aligned with the exact state used to sample the policy action.
        heuristic = self._heuristic_actions(ctx)
        self._last_teacher_actions = heuristic
        if self.opponent_mode != "selfplay":
            # In teacher-opponent mode the red side is heuristic-controlled and blue
            # remains the sampled policy. In self-play both sides remain policy-driven,
            # but both still receive heuristic targets during the warm-start phase.
            acts = t.where((self.team == 1)[:, :, None], heuristic, acts)
        before_alive = self.alive.clone()
        before_damage_dealt = self.damage_dealt.clone()
        before_damage_taken = self.damage_taken.clone()
        before_kills = self.kills.clone()
        before_shots = self.shots.clone()
        aim0, distance0 = self._aim(ctx)
        # `acts` remains categorical: 0/1/2 map to -1/0/+1 inside
        # _physics_tick. Subtracting here as well would shift every action
        # twice (and largely disable forward thrust and positive turning).
        self.last_turn = t.where(valid, acts[:, :, 0].float() - 1.0, self.last_turn)
        for _ in range(e.ACTION_REPEAT):
            self._physics_tick(acts, self.active_env.clone())
        died = before_alive & ~self.alive
        blue_dead = ((died) & (self.team == 0)).sum(dim=1).float()
        red_dead = ((died) & (self.team == 1)).sum(dim=1).float()
        d_dealt = self.damage_dealt - before_damage_dealt
        d_taken = self.damage_taken - before_damage_taken
        d_kills = self.kills - before_kills
        d_shots = self.shots - before_shots
        post_ctx = self._build_context(with_bullets=False)
        aim1, distance1 = self._aim(post_ctx)
        range_factor = t.clamp((1000.0 - distance0) / 800.0, 0, 1)
        range_after = t.clamp((1000.0 - distance1) / 800.0, 0, 1)
        reward_cfg = _reward_coefficients(e)
        reward = (reward_cfg["damage_dealt"] * d_dealt - reward_cfg["damage_taken"] * d_taken
                  + reward_cfg["kill"] * d_kills - reward_cfg["death"] * died.float())
        enemy_dead = t.where(self.team == 0, red_dead[:, None], blue_dead[:, None])
        ally_dead = t.where(self.team == 0, blue_dead[:, None], red_dead[:, None])
        reward = reward + reward_cfg["enemy_dead"] * enemy_dead - reward_cfg["ally_dead"] * ally_dead
        blue_team_net = (d_dealt * (self.team == 0).float()).sum(dim=1) - (d_taken * (self.team == 0).float()).sum(dim=1)
        red_team_net = (d_dealt * (self.team == 1).float()).sum(dim=1) - (d_taken * (self.team == 1).float()).sum(dim=1)
        blue_div = t.as_tensor(self.blue_counts, dtype=t.float32, device=self.dev).view(self.E, 1).clamp_min(1.0)
        red_div = t.as_tensor(self.red_counts, dtype=t.float32, device=self.dev).view(self.E, 1).clamp_min(1.0)
        team_div = t.where(self.team == 0, blue_div, red_div)
        team_net = t.where(self.team == 0, blue_team_net[:, None], red_team_net[:, None])
        reward = reward + reward_cfg["team_progress"] * team_net / team_div
        reward = reward + .065 * (aim1 - aim0)
        reward = reward + .010 * (aim1.clamp_min(0) ** 2) * range_after
        finite_dist = t.isfinite(distance0) & t.isfinite(distance1)
        desired_range = float(getattr(e, "TRAINING_DESIRED_RANGE", 480.0))
        if desired_range > 0.0:
            before_p = (distance0 - desired_range).abs()
            after_p = (distance1 - desired_range).abs()
            # Missing targets yield infinite distances; mask before subtracting to avoid NaNs.
            safe_before = t.where(finite_dist, before_p, t.zeros_like(before_p))
            safe_after = t.where(finite_dist, after_p, t.zeros_like(after_p))
            reward = reward + .016 * t.clamp((safe_before - safe_after) / 30.0, -1, 1) * finite_dist.float()
                # Previous coefficients made almost every plausible shot negative unless
        # aim was nearly perfect, teaching the policy to suppress fire. Reward
        # moderately well-aimed shots while retaining a small blind-fire cost.
        shot_signal = (e.REWARD_SHOT_AIM_BONUS * (aim0.clamp_min(0) ** 2)
                       * range_factor - e.REWARD_SHOT_BASE_COST)
        reward = reward + d_shots * shot_signal
        reward = reward - e.REWARD_STEP_COST
        reward = t.where(train_mask, reward, t.zeros_like(reward))
        # Environment ends by elimination or at horizon. Mark each last valid record terminal.
        ended_now = ~self.active_env
        done = died | ended_now[:, None]
        # The stored mask is the *training* mask, not the raw alive mask. In
        # heuristic mode only the blue ships are trained, so the records of the
        # heuristic opponent must never reach the PPO update.
        return obs, acts, logp, values, reward, train_mask, done

    # ------------------------------------------------------------ CUDA graph
    def _snapshot_state(self):
        return {name: getattr(self, name).clone() for name in _STATE_ATTRS}

    def _restore_state(self, snapshot):
        for name, value in snapshot.items():
            getattr(self, name)[...] = value

    def _alloc_graph_buffers(self):
        t = self.t
        E, S = self.E, self.S
        self._g_obs = t.zeros((E, S, self.e.OBS_SIZE), dtype=t.float32, device=self.dev)
        self._g_actions = t.zeros((E, S, 4), dtype=t.long, device=self.dev)
        self._g_logp = t.zeros((E, S), dtype=t.float32, device=self.dev)
        self._g_values = t.zeros((E, S), dtype=t.float32, device=self.dev)
        self._g_reward = t.zeros((E, S), dtype=t.float32, device=self.dev)
        self._g_valid = t.zeros((E, S), dtype=t.bool, device=self.dev)
        self._g_done = t.zeros((E, S), dtype=t.bool, device=self.dev)
        self._g_teacher_actions = t.zeros((E, S, 4), dtype=t.long, device=self.dev)

    def _step_into_buffers(self):
        obs, acts, logp, values, reward, valid, done = self._step()
        self._g_obs.copy_(obs)
        self._g_actions.copy_(acts)
        self._g_logp.copy_(logp)
        self._g_values.copy_(values)
        self._g_reward.copy_(reward)
        self._g_valid.copy_(valid)
        self._g_done.copy_(done)
        self._g_teacher_actions.copy_(self._last_teacher_actions)

    def _build_cuda_graph(self):
        """Capture one decision into a CUDA graph, or give up quietly.

        The captured body is branch free and free of device synchronizations, so
        a replay performs the whole decision with a single launch instead of
        several hundred small kernel launches.
        """
        t = self.t
        self.last_cuda_graph_attempted = True
        self.last_cuda_graph_error = None
        self.last_cuda_graph_traceback = None
        if self.dev.type != "cuda" or not hasattr(t, "cuda"):
            self.last_cuda_graph_error = f"CUDA Graph requires CUDA device; current device is {self.dev}"
            return None
        snapshot = None
        try:
            self._alloc_graph_buffers()
            snapshot = self._snapshot_state()
            self._graph_mode = True
            t.manual_seed(self._graph_seed)
            stream = t.cuda.Stream()
            stream.wait_stream(t.cuda.current_stream())
            with t.cuda.stream(stream):
                for _ in range(3):
                    self._step_into_buffers()
            t.cuda.current_stream().wait_stream(stream)
            t.cuda.synchronize(self.dev)
            graph = t.cuda.CUDAGraph()
            with t.cuda.graph(graph):
                self._step_into_buffers()
            t.cuda.synchronize(self.dev)
            # The warmup advanced the simulation; put the arena back to step zero.
            self._restore_state(snapshot)
            self._graph = graph
            self._graph_mode = False
            self.last_cuda_graph_error = None
            return graph
        except Exception as exc:
            # Preserve the actual reason; silently swallowing it made the CUDA
            # benchmark report only "capture returned None" and hid the root cause.
            self.last_cuda_graph_error = f"{type(exc).__name__}: {exc}"
            self.last_cuda_graph_traceback = traceback.format_exc()
            with _CUDA_GRAPH_FAILED_LOCK:
                first_failure = self._graph_failure_key not in _CUDA_GRAPH_FAILED_KEYS
                _CUDA_GRAPH_FAILED_KEYS.add(self._graph_failure_key)
            if first_failure:
                print(
                    f"[GravitrueX CUDA Graph] Capture failed for {self.E} arenas; "
                    f"falling back to eager. {self.last_cuda_graph_error}\n"
                    f"Set GRAVITRUEX_CUDA_GRAPH=1 to retry explicitly or =0 to disable.\n"
                    f"Details:\n{self.last_cuda_graph_traceback}",
                    flush=True,
                )
            # Any capture problem falls back to eager. Restore state so a failed
            # warmup cannot corrupt the episode. Auto mode will remember this shape.
            self._graph = None
            self._graph_mode = False
            if snapshot is not None:
                try:
                    self._restore_state(snapshot)
                except Exception as restore_exc:
                    self.last_cuda_graph_error += f"; state restore also failed: {restore_exc}"
            return None

    # ---------------------------------------------------------------- collect
    def collect(self, stop_event=None):
        t, e = self.t, self.e
        E, S = self.E, self.S
        rollout_start = time.perf_counter()
        device = str(self.dev)
        cuda_start = cuda_end = None
        if self.dev.type == "cuda":
            try:
                cuda_start = t.cuda.Event(enable_timing=True)
                cuda_end = t.cuda.Event(enable_timing=True)
                cuda_start.record()
            except Exception:
                cuda_start = cuda_end = None
        self.policy.last_device = f"GPU vector rollout / {device} / {self.E} arenas"
        self.policy.last_inference_backend = "CUDA / vectorized battles"
        T = self.max_decisions
        # Preallocated per-step storage keeps the rollout copy free of device
        # synchronizations and lets a CUDA graph write straight into it.
        obs_store = t.zeros((T, E, S, e.OBS_SIZE), dtype=t.float32, device=self.dev)
        act_store = t.zeros((T, E, S, 4), dtype=t.long, device=self.dev)
        logp_store = t.zeros((T, E, S), dtype=t.float32, device=self.dev)
        value_store = t.zeros((T, E, S), dtype=t.float32, device=self.dev)
        reward_store = t.zeros((T, E, S), dtype=t.float32, device=self.dev)
        valid_store = t.zeros((T, E, S), dtype=t.bool, device=self.dev)
        done_store = t.zeros((T, E, S), dtype=t.bool, device=self.dev)
        teacher_action_store = t.zeros((T, E, S, 4), dtype=t.long, device=self.dev)
        teacher_action_store[..., :3] = 1  # neutral categorical labels by default
        teacher_mask_store = t.zeros((T, E, S), dtype=t.bool, device=self.dev)
        # Reuse a graph already captured by diagnostics/tests for this exact batch
        # instance. In normal training each rollout owns a fresh batch, so graph
        # capture still happens once per rollout; avoiding a duplicate capture here
        # prevents an accidental second warmup when collect() follows a preflight.
        graph = self._graph if self.use_cuda_graph and self._graph is not None else (
            self._build_cuda_graph() if self.use_cuda_graph else None
        )
        self.last_cuda_graph_used = graph is not None
        if self.use_cuda_graph and graph is None and self.last_cuda_graph_error:
            self.policy.last_inference_backend = f"CUDA eager fallback ({self.last_cuda_graph_error[:160]})"
        steps_done = 0
        for decision in range(T):
            if stop_event is not None and stop_event.is_set():
                self.active_env[:] = False
                break
            # Checking a CUDA tensor in a Python `if` forces a device sync. Poll
            # less frequently: checking every 32 decisions substantially reduces
            # host/device barriers in long vectorized batches. Inactive arenas are
            # masked, so the occasional overshoot is harmless.
            if decision % 32 == 0 and not bool(self.active_env.any()):
                break
            if graph is not None:
                graph.replay()
                obs, acts, logp, values = (self._g_obs, self._g_actions,
                                           self._g_logp, self._g_values)
                reward, valid, done = self._g_reward, self._g_valid, self._g_done
                teacher_actions = self._g_teacher_actions
            else:
                obs, acts, logp, values, reward, valid, done = self._step()
                teacher_actions = self._last_teacher_actions
            # Environment ends by elimination or at horizon. Mark each last valid
            # record terminal. The horizon marker is a host side write, so it must
            # stay outside the captured graph.
            horizon = decision == T - 1
            if horizon:
                done = done | self.active_env[:, None]
                self.active_env[:] = False
            obs_store[decision].copy_(obs)
            act_store[decision].copy_(acts)
            logp_store[decision].copy_(logp)
            value_store[decision].copy_(values)
            reward_store[decision].copy_(reward)
            teacher_action_store[decision].copy_(teacher_actions)
            if self.opponent_mode == "heuristic":
                teacher_mask_store[decision].copy_(valid & (self.team == 0))
            elif self.opponent_mode == "selfplay":
                teacher_mask_store[decision].copy_(valid)
            valid_store[decision].copy_(valid)
            done_store[decision].copy_(done)
            steps_done = decision + 1
        if steps_done == 0:
            return []
        obs_t = obs_store[:steps_done]
        act_t = act_store[:steps_done]
        logp_t = logp_store[:steps_done]
        value_t = value_store[:steps_done]
        rewards = reward_store[:steps_done]
        valid_t = valid_store[:steps_done]
        dones_t = done_store[:steps_done]
        # Terminal score mirrors the scalar Arena rollout objective.
        blue_alive = ((self.team == 0) & self.alive).sum(dim=1).float()
        red_alive = ((self.team == 1) & self.alive).sum(dim=1).float()
        blue_kills = (self.kills * (self.team == 0).float()).sum(dim=1)
        red_kills = (self.kills * (self.team == 1).float()).sum(dim=1)
        blue_damage = (self.damage_dealt * (self.team == 0).float()).sum(dim=1)
        red_damage = (self.damage_dealt * (self.team == 1).float()).sum(dim=1)
        blue_survivors, red_survivors = blue_alive.clone(), red_alive.clone()
        outcome = t.where((blue_alive > 0) & (red_alive == 0), 0,
                          t.where((blue_alive == 0) & (red_alive > 0), 1, -1))
        reward_cfg = _reward_coefficients(e)
        terminal_blue = ((blue_kills - red_kills) * reward_cfg["terminal_kill"]
                         + (blue_alive - red_alive) * reward_cfg["terminal_survivor"]
                         + t.clamp((blue_damage - red_damage) / reward_cfg["terminal_damage_div"],
                                   -reward_cfg["terminal_damage_cap"], reward_cfg["terminal_damage_cap"])
                         + reward_cfg["terminal_bias"])
        terminal_red = ((red_kills - blue_kills) * reward_cfg["terminal_kill"]
                        + (red_alive - blue_alive) * reward_cfg["terminal_survivor"]
                        + t.clamp((red_damage - blue_damage) / reward_cfg["terminal_damage_div"],
                                  -reward_cfg["terminal_damage_cap"], reward_cfg["terminal_damage_cap"])
                        + reward_cfg["terminal_bias"])
        terminal_blue = t.where(outcome == 0, reward_cfg["terminal_win"], t.where(outcome == 1, -reward_cfg["terminal_win"], terminal_blue))
        terminal_red = t.where(outcome == 1, reward_cfg["terminal_win"], t.where(outcome == 0, -reward_cfg["terminal_win"], terminal_red))
        # Add terminal objective to the last reward in each env/team trajectory.
        rewards = rewards.clone()
        dones_t = dones_t.clone()
        T_used = rewards.shape[0]
        time_axis = t.arange(T_used, device=self.dev).view(T_used, 1, 1).expand(T_used, E, S)
        last_t_by_ship = t.where(valid_t, time_axis, -1).amax(dim=0)  # [E,S], -1 if no records
        has_record = last_t_by_ship >= 0
        is_blue = self.team == 0
        term_per_ship = t.where(is_blue, terminal_blue[:, None], terminal_red[:, None])
        blue_div = t.as_tensor(self.blue_counts, dtype=t.float32, device=self.dev).view(E, 1)
        red_div = t.as_tensor(self.red_counts, dtype=t.float32, device=self.dev).view(E, 1)
        divisor = t.where(is_blue, blue_div, red_div).clamp_min(1.0)
        bonus = term_per_ship / divisor * has_record.float()
        last_index = last_t_by_ship.clamp_min(0)
        env_index = t.arange(E, device=self.dev).view(E, 1).expand(E, S)
        ship_index = t.arange(S, device=self.dev).view(1, S).expand(E, S)
        rewards[last_index, env_index, ship_index] = (
            rewards[last_index, env_index, ship_index] + bonus)
        dones_t[last_index, env_index, ship_index] = (
            dones_t[last_index, env_index, ship_index] | has_record)
        # Batched GAE over time; done/invalid boundaries prevent leakage across ships.
        adv_t = t.zeros_like(rewards)
        gae = t.zeros((E, S), dtype=t.float32, device=self.dev)
        gamma, lam = .993, .94
        for ti in range(T_used - 1, -1, -1):
            if ti + 1 < T_used:
                next_valid = valid_t[ti + 1]
                next_value = value_t[ti + 1]
            else:
                next_valid = t.zeros((E, S), dtype=t.bool, device=self.dev)
                next_value = t.zeros((E, S), dtype=t.float32, device=self.dev)
            nonterminal = (~dones_t[ti]) & next_valid
            delta = rewards[ti] + gamma * next_value * nonterminal.float() - value_t[ti]
            gae = delta + gamma * lam * nonterminal.float() * gae
            gae = t.where(valid_t[ti], gae, t.zeros_like(gae))
            adv_t[ti] = gae
        ret_t = adv_t + value_t
        # Collect episode summaries on-device; put all CUDA->CPU copies on the queue
        # and synchronize only once. Repeated pageable .cpu().numpy() calls can create
        # multiple host/device barriers and inflate wall time when the model is large.
        blue_mask_f = (self.team == 0).float()
        red_mask_f = (self.team == 1).float()
        summary_tensor = t.stack((
            (self.shots * blue_mask_f).sum(dim=1),
            (self.shots * red_mask_f).sum(dim=1),
            blue_damage, red_damage, blue_kills, red_kills,
            blue_survivors, red_survivors, outcome.float(), self.step_count.float(),
        ), dim=1)
        if cuda_end is not None:
            try:
                cuda_end.record()
            except Exception:
                cuda_end = None

        teacher_act_t = teacher_action_store[:steps_done]
        teacher_mask_t = teacher_mask_store[:steps_done]
        tensors_for_host = (valid_t, obs_t, act_t, logp_t, value_t, rewards,
                            dones_t, adv_t, ret_t, teacher_act_t, teacher_mask_t, summary_tensor)
        if self.dev.type == "cuda":
            host_buffers = []
            for src in tensors_for_host:
                host = t.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
                host.copy_(src, non_blocking=True)
                host_buffers.append(host)
            t.cuda.synchronize(self.dev)
            (valid_cpu, obs_cpu, act_cpu, logp_cpu, value_cpu, reward_cpu,
             done_cpu, adv_cpu, ret_cpu, teacher_act_cpu, teacher_mask_cpu, summary_cpu) = host_buffers
            if cuda_start is not None and cuda_end is not None:
                try:
                    self.last_cuda_device_seconds = float(cuda_start.elapsed_time(cuda_end)) / 1000.0
                except Exception:
                    self.last_cuda_device_seconds = 0.0
        else:
            (valid_cpu, obs_cpu, act_cpu, logp_cpu, value_cpu, reward_cpu,
             done_cpu, adv_cpu, ret_cpu, teacher_act_cpu, teacher_mask_cpu, summary_cpu) = tensors_for_host
            self.last_cuda_device_seconds = 0.0
        valid_np = valid_cpu.numpy()
        obs_np = obs_cpu.numpy()
        act_np = act_cpu.numpy()
        logp_np = logp_cpu.numpy()
        value_np = value_cpu.numpy()
        reward_np = reward_cpu.numpy()
        done_np = done_cpu.numpy()
        adv_np = adv_cpu.numpy()
        ret_np = ret_cpu.numpy()
        teacher_act_np = teacher_act_cpu.numpy()
        teacher_mask_np = teacher_mask_cpu.numpy()
        summary_np = summary_cpu.numpy()
        reward_slots = slice(None) if self.opponent_mode == "selfplay" else slice(0, 5)
        episode_reward_np = reward_np[:, :, reward_slots].sum(axis=(0, 2))
        # Pack every arena in one pass. The old code ran eight fancy-index copies
        # per arena, which does not scale to hundreds of arenas.
        T_used = valid_np.shape[0]
        # Pack every arena with one global selection instead of eight
        # fancy-index copies per arena. The valid mask of a ship is a
        # contiguous run along the time axis, so after transposing to
        # (arena, time, ship) the records of one arena form one contiguous
        # block. The contiguity is verified, and the slow path is used if a
        # future change ever breaks it.
        valid_env = valid_np.transpose(1, 0, 2)
        contiguous_ok = bool(np.all(np.diff(valid_env.astype(np.int8), axis=1) >= 0))
        results = []
        if contiguous_ok:
            counts = valid_env.sum(axis=(1, 2))
            offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
            sel = np.flatnonzero(valid_env.reshape(-1))
            obs_sel = obs_np.transpose(1, 0, 2, 3).reshape(-1, e.OBS_SIZE)[sel]
            act_sel = act_np.transpose(1, 0, 2, 3).reshape(-1, 4)[sel]
            logp_sel = logp_np.transpose(1, 0, 2).reshape(-1)[sel]
            value_sel = value_np.transpose(1, 0, 2).reshape(-1)[sel]
            reward_sel = reward_np.transpose(1, 0, 2).reshape(-1)[sel]
            done_sel = done_np.transpose(1, 0, 2).reshape(-1)[sel]
            adv_sel = adv_np.transpose(1, 0, 2).reshape(-1)[sel]
            ret_sel = ret_np.transpose(1, 0, 2).reshape(-1)[sel]
            teacher_act_sel = teacher_act_np.transpose(1, 0, 2, 3).reshape(-1, 4)[sel]
            teacher_mask_sel = teacher_mask_np.transpose(1, 0, 2).reshape(-1)[sel]
        else:
            obs_sel = act_sel = logp_sel = value_sel = None
            reward_sel = done_sel = adv_sel = ret_sel = None
            teacher_act_sel = teacher_mask_sel = None
        for env_i in range(E):
            if contiguous_ok:
                sl = slice(int(offsets[env_i]), int(offsets[env_i + 1]))
                packed = self.e.PackedExperience(
                    obs_sel[sl].astype(np.float32, copy=False),
                    act_sel[sl].astype(np.int64, copy=False),
                    logp_sel[sl].astype(np.float32, copy=False),
                    value_sel[sl].astype(np.float32, copy=False),
                    reward_sel[sl].astype(np.float32, copy=False),
                    done_sel[sl].astype(np.bool_, copy=False),
                    adv_sel[sl].astype(np.float32, copy=False),
                    ret_sel[sl].astype(np.float32, copy=False),
                    teacher_action=teacher_act_sel[sl].astype(np.int64, copy=False),
                    teacher_mask=teacher_mask_sel[sl].astype(np.bool_, copy=False),
                )
            else:
                mask = valid_np[:, env_i, :]
                packed = self.e.PackedExperience(
                    obs_np[:, env_i][mask].astype(np.float32, copy=True),
                    act_np[:, env_i][mask].astype(np.int64, copy=True),
                    logp_np[:, env_i][mask].astype(np.float32, copy=True),
                    value_np[:, env_i][mask].astype(np.float32, copy=True),
                    reward_np[:, env_i][mask].astype(np.float32, copy=True),
                    done_np[:, env_i][mask].astype(np.bool_, copy=True),
                    adv_np[:, env_i][mask].astype(np.float32, copy=True),
                    ret_np[:, env_i][mask].astype(np.float32, copy=True),
                    teacher_action=teacher_act_np[:, env_i][mask].astype(np.int64, copy=True),
                    teacher_mask=teacher_mask_np[:, env_i][mask].astype(np.bool_, copy=True),
                )
            (bshots, rshots, bdamage, rdamage, bkills, rkills,
             bs, rs, outv_f, steps_f) = summary_np[env_i]
            bshots, rshots = float(bshots), float(rshots)
            bdamage, rdamage = float(bdamage), float(rdamage)
            bkills, rkills = float(bkills), float(rkills)
            bs, rs = int(bs), int(rs)
            outv = int(outv_f)
            out = None if outv < 0 else outv
            ep_reward = float(episode_reward_np[env_i])
            env_valid = valid_np[:, env_i, :].astype(bool, copy=False)
            env_fire = act_np[:, env_i, :, 3][env_valid]
            env_teacher_mask = teacher_mask_np[:, env_i, :].astype(bool, copy=False) & env_valid
            env_teacher_fire = teacher_act_np[:, env_i, :, 3][env_teacher_mask]
            policy_fire_rate = float(np.mean(env_fire)) if env_fire.size else 0.0
            teacher_fire_rate = float(np.mean(env_teacher_fire)) if env_teacher_fire.size else 0.0
            metrics = {
                "blue_damage": bdamage, "red_damage": rdamage,
                "policy_fire_rate": policy_fire_rate, "teacher_fire_rate": teacher_fire_rate,
                "blue_shots": bshots, "red_shots": rshots,
                "blue_kills": bkills, "red_kills": rkills,
                "blue_survivors": bs, "red_survivors": rs,
                "blue_hit_rate": min(1.0, bdamage / max(1.0, bshots * e.BULLET_DAMAGE)),
                "red_hit_rate": min(1.0, rdamage / max(1.0, rshots * e.BULLET_DAMAGE)),
                "rollout_backend": "GPU tensor physics",
            }
            results.append((packed, ep_reward, out, int(self.blue_counts[env_i]),
                            int(self.red_counts[env_i]), int(round(float(steps_f))), metrics))
        self.last_collect_wall_seconds = max(0.001, time.perf_counter() - rollout_start)
        return results
