

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from algorithm.explain.obs_schema import ObsSchema
from algorithm.explain.shapley import GroupShapExplainer

__all__ = ["decision_heads", "FeatureExplanation", "explain_agent_features",
           "importance_rows", "split_samples", "subtype_overall", "IMPORTANCE_FIELDS"]

IMPORTANCE_FIELDS = ["agent", "agent_name", "subtype", "grouping", "head", "group",
                     "mean_abs", "mean_signed", "std", "share", "mc_se", "rank",
                     "method", "n_explain", "n_background"]


def decision_heads(blocks: Sequence[int], world, agent) -> Tuple[List[str], np.ndarray]:
    blocks = [int(b) for b in (blocks or [])]
    A = int(sum(blocks))
    if A <= 0:
        raise ValueError("动作维度为 0")
    names: List[str] = []
    rows: List[np.ndarray] = []
    dim_p = int(getattr(world, "dim_p", 2))
    off, bi = 0, 0
    if bool(getattr(agent, "movable", True)) and dim_p == 2 and blocks and blocks[0] == 5:
        for nm, (p, q) in (("move_x", (1, 2)), ("move_y", (3, 4))):
            r = np.zeros(A)
            r[p], r[q] = 1.0, -1.0
            names.append(nm)
            rows.append(r)
        off, bi = blocks[0], 1
    if (not bool(getattr(agent, "silent", True))) and bi < len(blocks):
        width = blocks[bi]
        if getattr(agent, "adversary", False):
            sym, nm = getattr(world, "mal_symbol", None), "mal_msg"
        else:
            sym, nm = getattr(world, "heal_symbol", None), "heal_msg"
        if sym is not None and 0 <= int(sym) < width:
            r = np.zeros(A)
            r[off + int(sym)] = 1.0
            names.append(nm)
            rows.append(r)
    if not rows:
        return ["a{}".format(k) for k in range(A)], np.eye(A)
    return names, np.stack(rows, axis=0)


@dataclass
class FeatureExplanation:
    agent_index: int
    alias: str
    agent_name: str
    subtype: str
    grouping: str
    group_names: List[str]
    head_names: List[str]
    feature_group: np.ndarray      
    phi: np.ndarray                
    mc_se: np.ndarray              
    base: np.ndarray               
    fx: np.ndarray                 
    phi_raw: np.ndarray            
    base_raw: np.ndarray           
    fx_raw: np.ndarray             
    X: np.ndarray                  
    method: str
    efficiency_gap: float
    n_background: int


def explain_agent_features(adapter, schema: ObsSchema, heads: Tuple[List[str], np.ndarray],
                           X_explain: np.ndarray, X_background: np.ndarray, grouping: str,
                           exact_max: int = 10, n_perm: int = 32,
                           rng: Optional[np.random.Generator] = None,
                           max_rows: int = 131072) -> FeatureExplanation:
    head_names, M = heads
    M = np.asarray(M, dtype=np.float64)
    A = int(M.shape[1])

    def f(Z):
        P = adapter.probs(Z)
        if P.shape[1] != A:
            raise RuntimeError("动作宽度 {} != 决策头矩阵宽度 {}".format(P.shape[1], A))
        return np.concatenate([P, P @ M.T], axis=1)

    gnames, gidx = schema.groups(grouping)
    ex = GroupShapExplainer(f, X_background, gidx, exact_max=exact_max, n_perm=n_perm,
                            rng=rng, max_rows=max_rows)
    res = ex.explain(np.asarray(X_explain))
    return FeatureExplanation(
        agent_index=schema.agent_index, alias=schema.alias, agent_name=schema.agent_name,
        subtype=schema.subtype, grouping=grouping, group_names=list(gnames),
        head_names=list(head_names), feature_group=schema.feature_group_ids(grouping),
        phi=res.phi[:, :, A:], mc_se=res.mc_se[:, :, A:], base=res.base[A:],
        fx=res.fx[:, A:], phi_raw=res.phi[:, :, :A], base_raw=res.base[:A],
        fx_raw=res.fx[:, :A], X=np.asarray(X_explain, dtype=np.float32),
        method=res.method, efficiency_gap=res.efficiency_gap(),
        n_background=int(np.asarray(X_background).shape[0]))


def _ranks(v: np.ndarray) -> np.ndarray:
    order = np.argsort(-v, kind="stable")
    r = np.empty(len(v), dtype=np.int64)
    r[order] = np.arange(1, len(v) + 1)
    return r


def importance_rows(fe: FeatureExplanation, extra: Optional[Dict] = None) -> List[Dict]:
    rows: List[Dict] = []
    shares, totals = [], []
    common = OrderedDict([("agent", fe.alias), ("agent_name", fe.agent_name),
                          ("subtype", fe.subtype), ("grouping", fe.grouping)])
    tail = {"method": fe.method, "n_explain": int(fe.phi.shape[0]),
            "n_background": int(fe.n_background)}
    for h, head in enumerate(fe.head_names):
        a = np.abs(fe.phi[:, :, h])
        mean_abs = a.mean(axis=0)
        tot = float(mean_abs.sum())
        share = mean_abs / tot if tot > 0 else np.zeros_like(mean_abs)
        shares.append(share)
        totals.append(mean_abs)
        signed = fe.phi[:, :, h].mean(axis=0)
        std = fe.phi[:, :, h].std(axis=0)
        mc = fe.mc_se[:, :, h].mean(axis=0)
        rk = _ranks(share)
        for g, gname in enumerate(fe.group_names):
            row = OrderedDict(common)
            row.update(head=head, group=gname, mean_abs=float(mean_abs[g]),
                       mean_signed=float(signed[g]), std=float(std[g]),
                       share=float(share[g]), mc_se=float(mc[g]), rank=int(rk[g]), **tail)
            rows.append(row)
    if shares:
        overall = np.mean(np.stack(shares), axis=0)
        total_abs = np.sum(np.stack(totals), axis=0)
        rk = _ranks(overall)
        for g, gname in enumerate(fe.group_names):
            row = OrderedDict(common)
            row.update(head="overall", group=gname, mean_abs=float(total_abs[g]),
                       mean_signed=float("nan"), std=float("nan"), share=float(overall[g]),
                       mc_se=float("nan"), rank=int(rk[g]), **tail)
            rows.append(row)
    if extra:
        rows = [OrderedDict(list(extra.items()) + list(r.items())) for r in rows]
    return rows


def split_samples(data: np.ndarray, n_explain: int, n_background: int,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data)
    n = int(data.shape[0])
    if n < 2:
        raise ValueError("观测太少 ({} 条), 无法切分背景/解释样本".format(n))
    nb = max(1, min(int(n_background), n // 2))
    ne = max(1, min(int(n_explain), n - nb))
    perm = rng.permutation(n)
    return data[perm[nb:nb + ne]], data[perm[:nb]]


def subtype_overall(rows: Sequence[Dict], grouping: Optional[str] = None,
                    head: str = "overall") -> Dict[str, List[Tuple[str, float, float]]]:
    acc: "OrderedDict[str, OrderedDict[str, List[float]]]" = OrderedDict()
    for r in rows:
        if str(r["head"]) != head:
            continue
        if grouping is not None and str(r["grouping"]) != grouping:
            continue
        acc.setdefault(str(r["subtype"]), OrderedDict()).setdefault(
            str(r["group"]), []).append(float(r["share"]))
    out = {}
    for sub, d in acc.items():
        items = [(g, float(np.mean(v)), float(np.std(v))) for g, v in d.items()]
        items.sort(key=lambda t: -t[1])
        out[sub] = items
    return out
