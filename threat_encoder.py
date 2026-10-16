

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from algorithm.trainer.maddpg import MLP

__all__ = ["ObsLayout", "ThreatEncoder", "ThreatAwareActor"]


@dataclass
class ObsLayout:

    obs_dim: int
    context_dim: int      
    entity_dim: int       
    max_entities: int     
    tail_dim: int         
    has_mask: bool        

    def __post_init__(self):
        for f in ("obs_dim", "context_dim", "entity_dim", "max_entities", "tail_dim"):
            setattr(self, f, int(getattr(self, f)))
        self.has_mask = bool(self.has_mask)
        if self.total != self.obs_dim:
            raise ValueError(
                "ObsLayout 与 obs_dim 不符: context({}) + entity({}x{}) + mask({}) "
                "+ tail({}) = {} != obs_dim={}".format(
                    self.context_dim, self.max_entities, self.entity_dim,
                    self.max_entities if self.has_mask else 0,
                    self.tail_dim, self.total, self.obs_dim))

    @property
    def mask_dim(self) -> int:
        return self.max_entities if self.has_mask else 0

    @property
    def total(self) -> int:
        return (self.context_dim + self.entity_dim * self.max_entities
                + self.mask_dim + self.tail_dim)

    def head_input_dim(self, threat_dim: int) -> int:
        return self.context_dim + int(threat_dim) + self.tail_dim

    @staticmethod
    def infer_ioa(obs_dim: int, n_agents: int, act_blocks: Sequence[int],
                  *, has_mask: Optional[bool] = None,
                  max_entities: Optional[int] = None) -> "ObsLayout":
        language = len(act_blocks) > 1
        dim_c = int(act_blocks[1]) if language else 0
        m = int(n_agents) - 1 if max_entities is None else int(max_entities)
        kw = dict(
            obs_dim=int(obs_dim),
            context_dim=12,
            entity_dim=3 + dim_c if language else 3,
            max_entities=m,
            tail_dim=2 * dim_c + 2 if language else 0,
        )
        if has_mask is not None:
            return ObsLayout(has_mask=bool(has_mask), **kw)

        candidates = []
        for flag in (False, True):
            try:
                return ObsLayout(has_mask=flag, **kw)
            except ValueError as exc:          
                candidates.append(str(exc))
        raise ValueError(
            "无法为 obs_dim={} (n_agents={}, act_blocks={}) 推断 ObsLayout。\n  {}\n"
            "常见原因: 场景改了 observation() 的分块顺序, 或 --ioa-num-* 与 "
            "checkpoint 不一致。".format(obs_dim, n_agents, list(act_blocks),
                                        "\n  ".join(candidates)))

    def split(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                torch.Tensor, torch.Tensor]:
        if obs.ndim != 2:
            raise ValueError("obs 必须是 [batch, obs_dim], 收到 {}".format(tuple(obs.shape)))
        if obs.shape[-1] != self.obs_dim:
            raise ValueError("obs 最后一维 {} != layout.obs_dim {}".format(
                obs.shape[-1], self.obs_dim))
        b = obs.shape[0]
        e0 = self.context_dim
        e1 = e0 + self.entity_dim * self.max_entities
        ctx = obs[:, :e0]
        ent = obs[:, e0:e1].reshape(b, self.max_entities, self.entity_dim)
        if self.has_mask:
            mask = obs[:, e1:e1 + self.max_entities]
            tail = obs[:, e1 + self.max_entities:]
        else:
            mask = torch.ones(b, self.max_entities, dtype=obs.dtype, device=obs.device)
            tail = obs[:, e1:]
        return ctx, ent, mask, tail

    def describe(self) -> str:
        return ("obs={} = ctx{} + {}x{} entities{} + tail{}".format(
            self.obs_dim, self.context_dim, self.max_entities, self.entity_dim,
            " + mask{}".format(self.max_entities) if self.has_mask else "",
            self.tail_dim))


class ThreatEncoder(nn.Module):

    def __init__(self, layout: ObsLayout, num_units: int = 64, threat_dim: int = 16):
        super().__init__()
        self.max_entities = int(layout.max_entities)
        self.entity_dim = int(layout.entity_dim)
        self.threat_dim = int(threat_dim)
        self.phi = nn.Sequential(
            nn.Linear(self.entity_dim, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU())
        self.rho = nn.Sequential(
            nn.Linear(2 * num_units + 1, self.threat_dim), nn.ReLU())

    def forward(self, entities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.phi(entities)                                
        m = mask.unsqueeze(-1).to(h.dtype)                    
        cnt = m.sum(dim=1).clamp_min(1.0)                     
        mean = (h * m).sum(dim=1) / cnt                       

        neg = torch.finfo(h.dtype).min
        mx = h.masked_fill(m == 0, neg).max(dim=1).values      
        alive = mask.sum(dim=1, keepdim=True) > 0             
        mx = torch.where(alive, mx, torch.zeros_like(mx))

        density = cnt / float(max(1, self.max_entities))       
        return self.rho(torch.cat([mean, mx, density], dim=-1))


class ThreatAwareActor(nn.Module):

    def __init__(self, layout: ObsLayout, act_dim: int, num_units: int = 64,
                 threat_dim: int = 16, enabled: bool = True):
        super().__init__()
        self.layout = layout
        self.enabled = bool(enabled)
        if self.enabled:
            self.encoder = ThreatEncoder(layout, num_units, threat_dim)
            head_in = layout.head_input_dim(threat_dim)
        else:
            self.encoder = None
            head_in = layout.obs_dim
        self.head = MLP(head_in, int(act_dim), num_units)
        self.last_threat: Optional[torch.Tensor] = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.head(obs)
        ctx, ent, mask, tail = self.layout.split(obs)
        z = self.encoder(ent, mask)
        self.last_threat = z.detach()
        return self.head(torch.cat([ctx, z, tail], dim=-1))

    def describe(self) -> str:
        if not self.enabled:
            return "threat encoder OFF (plain MLP over {} dims)".format(self.layout.obs_dim)
        return "threat_dim={} max_entities={} entity_dim={} head_in={}".format(
            self.encoder.threat_dim, self.layout.max_entities,
            self.layout.entity_dim, self.layout.head_input_dim(self.encoder.threat_dim))
