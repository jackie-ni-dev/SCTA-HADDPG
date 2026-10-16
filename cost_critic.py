

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm.safety.security_costs import COST_NAMES, NUM_COSTS
from algorithm.trainer.maddpg import MLP, hard_update, soft_update

__all__ = ["CostCriticSet", "ConstrainedCriticView", "private_torch_rng"]


@contextlib.contextmanager
def private_torch_rng(seed: Optional[int], device=None):
    if seed is None:
        yield
        return
    devices: List[int] = []
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "cuda" and torch.cuda.is_available():
        devices = [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        yield


class CostCriticSet:

    def __init__(self, critic_in: int, num_units: int, lr: float, device,
                 num_costs: int = NUM_COSTS, grad_clip: float = 0.5,
                 names: Sequence[str] = COST_NAMES, seed: Optional[int] = None):
        self.device = torch.device(device)
        self.num_costs = int(num_costs)
        self.critic_in = int(critic_in)
        self.grad_clip = float(grad_clip)
        self.names = tuple(names)[: self.num_costs]
        if len(self.names) != self.num_costs:
            raise ValueError("names 只有 {} 个, 需要 {} 个".format(
                len(self.names), self.num_costs))

        with private_torch_rng(seed, self.device):
            self.critics = nn.ModuleList(
                [MLP(self.critic_in, 1, num_units) for _ in range(self.num_costs)]
            ).to(self.device)
            self.targets = nn.ModuleList(
                [MLP(self.critic_in, 1, num_units) for _ in range(self.num_costs)]
            ).to(self.device)

        for c, t in zip(self.critics, self.targets):
            hard_update(t, c)

        self.optims = [torch.optim.Adam(c.parameters(), lr=float(lr))
                       for c in self.critics]

    def __len__(self) -> int:
        return self.num_costs

    def parameters(self):
        return self.critics.parameters()

    def named_parameters(self):
        return self.critics.named_parameters()

    def q(self, k: int, x: torch.Tensor) -> torch.Tensor:
        return self.critics[k](x).squeeze(-1)

    def target_q(self, k: int, x: torch.Tensor) -> torch.Tensor:
        return self.targets[k](x).squeeze(-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.critics.parameters())

    def td_update(self, q_in: torch.Tensor,
                  cost_targets: Sequence[torch.Tensor]) -> Dict[str, float]:
        if len(cost_targets) != self.num_costs:
            raise ValueError("cost_targets 有 {} 项, cost critic 有 {} 个".format(
                len(cost_targets), self.num_costs))
        out: Dict[str, float] = {}
        for k, (cc, opt) in enumerate(zip(self.critics, self.optims)):
            loss = F.mse_loss(cc(q_in).squeeze(-1), cost_targets[k])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(cc.parameters(), self.grad_clip)
            opt.step()
            out["cost_%s_loss" % self.names[k]] = float(loss.detach().item())
        return out

    def soft_update_targets(self) -> None:
        for c, t in zip(self.critics, self.targets):
            soft_update(t, c)

    def zero_grad(self) -> None:
        for opt in self.optims:
            opt.zero_grad()

    def state_dict(self) -> Dict:
        return {
            "critics": self.critics.state_dict(),
            "targets": self.targets.state_dict(),
            "optims": [o.state_dict() for o in self.optims],
            "critic_in": self.critic_in,
            "num_costs": self.num_costs,
        }

    def load_state_dict(self, sd: Dict) -> None:
        if int(sd.get("critic_in", self.critic_in)) != self.critic_in:
            raise ValueError(
                "checkpoint 的 cost critic 输入维度 {} 与当前 {} 不符 —— 中心化 "
                "critic 的输入 = sum(obs) + sum(act), 通常是场景的实体数变了 "
                "(--ioa-num-* / --ioa-max-malgents)。".format(
                    sd["critic_in"], self.critic_in))
        if int(sd.get("num_costs", self.num_costs)) != self.num_costs:
            raise ValueError("checkpoint 有 {} 个 cost critic, 当前 {} 个".format(
                sd["num_costs"], self.num_costs))
        self.critics.load_state_dict(sd["critics"])
        self.targets.load_state_dict(sd["targets"])
        for o, s in zip(self.optims, sd["optims"]):
            o.load_state_dict(s)

    def describe(self) -> str:
        return "cost critic in={} K={} params={}".format(
            self.critic_in, self.num_costs, self.num_params())


class ConstrainedCriticView:

    def __init__(self, reward_critic, cost_set: CostCriticSet,
                 lambdas: Sequence[float]):
        if len(lambdas) != len(cost_set):
            raise ValueError("λ 有 {} 个, cost critic 有 {} 个".format(
                len(lambdas), len(cost_set)))
        self.reward_critic = reward_critic
        self.cost_set = cost_set
        self.lambdas = [float(x) for x in lambdas]
        self.n_calls = 0
        self.last_reward_q = 0.0
        self.last_cost_q = [0.0] * len(self.lambdas)
        self.last_penalty = 0.0

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        qr = self.reward_critic(x)
        out = qr
        penalty = 0.0
        for k, lam in enumerate(self.lambdas):
            if lam <= 0.0:
                self.last_cost_q[k] = 0.0
                continue
            qc = self.cost_set.critics[k](x)
            out = out - lam * qc
            m = float(qc.detach().mean().item())
            self.last_cost_q[k] = m
            penalty += lam * m
        self.last_reward_q = float(qr.detach().mean().item())
        self.last_penalty = penalty
        return out

    def assert_called_once(self) -> None:
        if self.n_calls != 1:
            raise RuntimeError(
                "actor 更新里 critic1 被调用了 {} 次 (期望 1 次)。ConstrainedCriticView "
                "是靠『父类 pg_loss 恰好调一次 critic1』这个前提工作的; 父类改了写法就"
                "必须同步改这里, 否则 cost penalty 会被漏算或重复计入。".format(self.n_calls))
