

from __future__ import annotations

import numpy as np


try:                                        
    from multiagent.scenarios.ioa_infection import DIST_CLIP as _DEFAULT_DIST_CLIP
except Exception:                           
    _DEFAULT_DIST_CLIP = 2.0

COST_NAMES = ("intrusion", "infection", "service")
NUM_COSTS = len(COST_NAMES)


REQUIRED_WORLD_ATTRS = ("agents", "_intrusion_run", "intrusion_persist_steps",
                        "n_infected", "l_true")

__all__ = ["COST_NAMES", "NUM_COSTS", "REQUIRED_WORLD_ATTRS",
           "security_costs", "assert_world_supported", "describe_costs"]


def _benign(world):
    return [a for a in getattr(world, "agents", []) if not getattr(a, "adversary", False)]


def security_costs(world) -> np.ndarray:
    benign = _benign(world)
    n = max(1, len(benign))

    run = int(getattr(world, "_intrusion_run", 0))
    persist = max(1, int(getattr(world, "intrusion_persist_steps", 1)))
    c_int = 1.0 if run >= persist else 0.0

    c_inf = float(getattr(world, "n_infected", 0)) / n
    c_inf = float(np.clip(c_inf, 0.0, 1.0))

    lt = getattr(world, "l_true", None)
    clip = float(getattr(world, "cost_dist_clip", _DEFAULT_DIST_CLIP))
    if lt is not None and benign and clip > 0.0:
        target = np.asarray(lt.state.p_pos, dtype=np.float64)
        d = [float(np.sqrt(np.sum(np.square(
            np.asarray(a.state.p_pos, dtype=np.float64) - target)))) for a in benign]
        c_svc = float(np.clip(float(np.mean(d)) / clip, 0.0, 1.0))
    else:
        c_svc = 0.0

    return np.array([c_int, c_inf, c_svc], dtype=np.float32)


def assert_world_supported(world) -> None:
    missing = [k for k in REQUIRED_WORLD_ATTRS if not hasattr(world, k)]
    if missing:
        raise ValueError(
            "当前场景的 world 缺少 {} —— security constraints (模块③) 依赖 "
            "ioa_infection 系列 world 暴露的入侵/感染状态。在不支持的场景上打开 "
            "--enable-security-constraints 会让三项 cost 恒为 0, λ 恒为 0, "
            "模块静默失效。请改用 --scenario ioa_infection / ioa_unknown_malgent。"
            .format(", ".join(missing)))
    if not _benign(world):
        raise ValueError("world 里没有良性体 (adversary=False), c_inf / c_svc 的"
                         "分母无意义。")


def describe_costs(d=None) -> str:
    if d is None:
        return "costs K={} {}".format(NUM_COSTS, list(COST_NAMES))
    return "costs K={} {}".format(
        NUM_COSTS, ", ".join("{}<={:.3g}".format(k, v) for k, v in zip(COST_NAMES, d)))
