

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from algorithm.explain.obs_schema import entity_aliases
from algorithm.explain.policy import baseline_action
from algorithm.explain.rollout import METRICS, run_episode
from algorithm.explain.shapley import (coalition_label, exact_interactions, exact_shapley,
                                       permutation_shapley_from_values, sample_permutations)

__all__ = ["Player", "make_players", "CoalitionEvaluator", "AgentShapleyResult",
           "run_agent_shapley"]


@dataclass(frozen=True)
class Player:
    name: str
    agents: Tuple[int, ...]


def make_players(env, num_adversaries: int, mode: str) -> List[Player]:
    world = env.world
    aliases = entity_aliases(world)
    defenders = list(range(int(num_adversaries), int(env.n)))
    if mode == "defenders":
        return [Player(aliases[i], (i,)) for i in defenders]
    if mode == "roles":
        groups: "OrderedDict[str, List[int]]" = OrderedDict()
        for i in defenders:
            sub = str(getattr(world.agents[i], "subtype", None) or "defender")
            groups.setdefault(sub, []).append(i)
        return [Player(sub + "s", tuple(ix)) for sub, ix in groups.items()]
    raise ValueError("未知玩家模式 {!r} (可选 defenders / roles)".format(mode))


def _learned_act(adapter, rng, obs):
    return adapter.act(obs, rng)


def _baseline_act(blocks, kind, rng, obs):
    return baseline_action(blocks, kind, rng)


class CoalitionEvaluator:

    def __init__(self, env, trainers, adapters: Dict[int, object], num_adversaries: int,
                 max_len: int, baseline: str = "random",
                 seed_fn: Optional[Callable[[int], None]] = None,
                 step_stats_fn: Optional[Callable] = None,
                 cost_fn: Optional[Callable] = None):
        if baseline not in ("random", "noop"):
            raise ValueError("未知基线 {!r} (可选 random / noop)".format(baseline))
        self.env = env
        self.trainers = list(trainers)
        self.adapters = dict(adapters)
        self.num_adversaries = int(num_adversaries)
        self.max_len = int(max_len)
        self.baseline = baseline
        self.seed_fn = seed_fn
        self.step_stats_fn = step_stats_fn
        self.cost_fn = cost_fn
        self.defenders = list(range(self.num_adversaries, len(self.trainers)))
        missing = [i for i in self.defenders if i not in self.adapters]
        if missing:
            raise ValueError("缺少防守方 adapter: {}".format(missing))
        self.prepare_fns = [tr.prepare_step for tr in self.trainers[:self.num_adversaries]
                            if getattr(tr, "_on_policy", False) and hasattr(tr, "prepare_step")]
        self._cache: Dict[Tuple, Dict[str, np.ndarray]] = {}
        self.episodes_run = 0
        self.seconds = 0.0

    def action_fns(self, learned, seed: int) -> List[Callable]:
        fns: List[Callable] = []
        for i, tr in enumerate(self.trainers):
            if i < self.num_adversaries:
                fns.append(tr.action)
                continue
            ad = self.adapters[i]
            rng = np.random.default_rng([int(seed), int(i), 0x5EED])
            if i in learned:
                fns.append(partial(_learned_act, ad, rng))
            else:
                fns.append(partial(_baseline_act, list(ad.blocks), self.baseline, rng))
        return fns

    def run(self, learned, seeds: Sequence[int], on_obs: Optional[Callable] = None
            ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, float]]]:
        learned = frozenset(int(i) for i in learned)
        per = {m: np.zeros(len(seeds), dtype=np.float64) for m in METRICS}
        metas: List[Dict[str, float]] = []
        for e, s in enumerate(seeds):
            if self.seed_fn is not None:
                self.seed_fn(int(s))
            fns = self.action_fns(learned, int(s))
            hook = None if on_obs is None else partial(on_obs, e)
            t0 = time.time()
            m, meta = run_episode(self.env, fns, self.max_len, self.num_adversaries,
                                  self.step_stats_fn, self.prepare_fns, hook, self.cost_fn)
            self.seconds += time.time() - t0
            self.episodes_run += 1
            for k in METRICS:
                per[k][e] = m[k]
            metas.append(meta)
        return per, metas

    def evaluate(self, learned, seeds: Sequence[int]) -> Dict[str, np.ndarray]:
        key = (frozenset(int(i) for i in learned), tuple(int(s) for s in seeds))
        if key not in self._cache:
            self._cache[key] = self.run(key[0], key[1])[0]
        return self._cache[key]


@dataclass
class AgentShapleyResult:
    mode: str
    players: List[Player]
    agent_labels: List[str]
    metrics: List[str]
    seeds: np.ndarray
    baseline: str
    method: str
    masks: List[int]
    values: Dict[str, np.ndarray]                  
    phi: Dict[str, np.ndarray]                     
    interactions: Optional[Dict[str, np.ndarray]]  
    seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.players)

    @property
    def full_mask(self) -> int:
        return (1 << self.n) - 1

    def row(self, mask: int) -> Optional[int]:
        try:
            return self.masks.index(int(mask))
        except ValueError:
            return None


def run_agent_shapley(evaluator: CoalitionEvaluator, players: Sequence[Player],
                      seeds: Sequence[int], mode: str, exact_max: int = 8,
                      n_perm: int = 64, rng: Optional[np.random.Generator] = None,
                      log: Optional[Callable[[str], None]] = print) -> AgentShapleyResult:
    players = list(players)
    n = len(players)
    if n == 0:
        raise ValueError("没有可作为玩家的防守方")
    names = [p.name for p in players]
    full = (1 << n) - 1

    def learned_of(mask: int):
        out = set()
        for i in range(n):
            if (mask >> i) & 1:
                out.update(players[i].agents)
        return out

    perms = None
    if n <= int(exact_max):
        masks = list(range(1 << n))
        method = "exact(2^{}={} coalitions)".format(n, len(masks))
    else:
        rng = rng if rng is not None else np.random.default_rng(0)
        perms = sample_permutations(n, n_perm, rng)
        need = {0, full}
        for p in perms:
            mk = 0
            for i in p:
                mk |= (1 << int(i))
                need.add(mk)
        for i in range(n):                       
            need.add(1 << i)
            need.add(full & ~(1 << i))
        masks = sorted(need)
        method = "permutation({} perms, {} coalitions)".format(len(perms), len(masks))

    order = sorted(masks, key=lambda mk: (0 if mk in (0, full) else 1, bin(mk).count("1"), mk))
    table: Dict[int, Dict[str, np.ndarray]] = {}
    measured: List[float] = []
    t_start = time.time()
    every = max(1, len(order) // 16)
    for k, mk in enumerate(order):
        t1 = time.time()
        table[mk] = evaluator.evaluate(learned_of(mk), seeds)
        dt = time.time() - t1
        if dt > 1e-3:
            measured.append(dt)
        if log is not None and (k < 4 or (k + 1) % every == 0 or k + 1 == len(order)):
            eta = float(np.mean(measured)) * (len(order) - k - 1) if measured else 0.0
            log("  [agents:{}] {}/{} {} | reward {:+.3f} | intrusion {:.3f} | {:.1f}s | ETA {:.0f}s"
                .format(mode, k + 1, len(order), coalition_label(mk, names),
                        float(np.mean(table[mk]["Average_Episode_Reward"])),
                        float(np.mean(table[mk]["Intrusion_Success_Rate"])), dt, eta))

    metrics = list(METRICS)
    values = {m: np.stack([table[mk][m] for mk in masks], axis=0) for m in metrics}
    if perms is None:
        phi = {m: exact_shapley(values[m], n) for m in metrics}
        inter = {m: exact_interactions(values[m], n) for m in metrics}
    else:
        index = {mk: r for r, mk in enumerate(masks)}
        phi = {m: permutation_shapley_from_values(
            perms, (lambda mk, _m=m: values[_m][index[mk]]), n) for m in metrics}
        inter = None
    return AgentShapleyResult(
        mode=mode, players=players, agent_labels=entity_aliases(evaluator.env.world),
        metrics=metrics, seeds=np.asarray(seeds), baseline=evaluator.baseline, method=method,
        masks=list(masks), values=values, phi=phi, interactions=inter,
        seconds=time.time() - t_start)
