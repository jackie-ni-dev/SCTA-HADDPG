

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from algorithm.safety.security_costs import COST_NAMES

__all__ = ["LagrangeMultiplier", "LagrangeMultiplierSet"]


class LagrangeMultiplier:

    def __init__(self, init: float = 0.0, lr: float = 1e-3,
                 max_value: float = 10.0, threshold: float = 0.0,
                 name: str = "cost"):
        self.name = str(name)
        self.lr = float(lr)
        self.max = float(max_value)
        self.threshold = float(threshold)
        if self.lr < 0.0:
            raise ValueError("--lambda-lr 必须 >= 0 (0 = 冻结 λ, 这是③ 的核心对照组)")
        if self.max < 0.0:
            raise ValueError("--lambda-max 必须 >= 0")
        if self.threshold < 0.0:
            raise ValueError("d_{} 必须 >= 0".format(self.name))
        self.init = float(init)
        self.v = min(max(0.0, self.init), self.max)
        self.n_updates = 0
        self.last_gap = 0.0          

    def value(self) -> float:
        return self.v

    def update(self, j_cost: float) -> float:
        gap = float(j_cost) - self.threshold
        self.last_gap = gap
        self.v = min(max(0.0, self.v + self.lr * gap), self.max)
        self.n_updates += 1
        return self.v

    def reset(self) -> None:
        self.v = min(max(0.0, self.init), self.max)
        self.n_updates = 0
        self.last_gap = 0.0

    def state_dict(self) -> Dict[str, float]:
        return {"v": self.v, "threshold": self.threshold,
                "n_updates": self.n_updates, "name": self.name}

    def load_state_dict(self, sd: Dict) -> None:
        self.v = min(max(0.0, float(sd["v"])), self.max)
        self.threshold = float(sd["threshold"])
        self.n_updates = int(sd.get("n_updates", 0))

    def check_threshold(self, expected: float, tol: float = 1e-9) -> None:
        if abs(self.threshold - float(expected)) > tol:
            raise ValueError(
                "checkpoint 的 d_{}={} 与当前 --d-{} {} 不符。约束阈值变了就不是"
                "同一组实验了: 续训出来的 λ 曲线没法和原来的拼在一张图上。"
                .format(self.name, self.threshold, self.name, expected))

    def describe(self) -> str:
        return "{}: lambda={:.4f} d={:.4g} lr={:.3g} max={:.3g}".format(
            self.name, self.v, self.threshold, self.lr, self.max)

    def __repr__(self) -> str:                      
        return "<LagrangeMultiplier %s>" % self.describe()


class LagrangeMultiplierSet:

    def __init__(self, thresholds: Sequence[float], lr: float = 1e-3,
                 max_value: float = 10.0, init: float = 0.0,
                 names: Iterable[str] = COST_NAMES):
        names = tuple(names)
        thresholds = tuple(float(x) for x in thresholds)
        if len(thresholds) != len(names):
            raise ValueError("thresholds 有 {} 个, names 有 {} 个".format(
                len(thresholds), len(names)))
        self.names = names
        self.items: List[LagrangeMultiplier] = [
            LagrangeMultiplier(init=init, lr=lr, max_value=max_value,
                               threshold=d, name=nm)
            for nm, d in zip(names, thresholds)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, k) -> LagrangeMultiplier:
        return self.items[k]

    def values(self) -> List[float]:
        return [m.value() for m in self.items]

    def thresholds(self) -> List[float]:
        return [m.threshold for m in self.items]

    def gaps(self) -> List[float]:
        return [m.last_gap for m in self.items]

    def update(self, j_costs: Sequence[float]) -> List[float]:
        if len(j_costs) != len(self.items):
            raise ValueError("J_C 有 {} 项, 乘子有 {} 个 —— 顺序/数量必须与 "
                             "COST_NAMES 一致".format(len(j_costs), len(self.items)))
        return [m.update(j) for m, j in zip(self.items, j_costs)]

    def any_active(self) -> bool:
        return any(m.value() > 0.0 for m in self.items)

    def state_dict(self) -> Dict:
        return {"items": [m.state_dict() for m in self.items],
                "names": list(self.names)}

    def load_state_dict(self, sd: Dict) -> None:
        saved = sd["items"]
        if len(saved) != len(self.items):
            raise ValueError("checkpoint 有 {} 个乘子, 当前配置 {} 个".format(
                len(saved), len(self.items)))
        for m, s in zip(self.items, saved):
            m.load_state_dict(s)

    def describe(self) -> str:
        return "lambda=[{}] d=[{}]".format(
            ", ".join("{:.3f}".format(v) for v in self.values()),
            ", ".join("{:.3g}".format(d) for d in self.thresholds()))
