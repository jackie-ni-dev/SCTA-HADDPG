

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from algorithm.explain import report  
from algorithm.explain.agent_shapley import (CoalitionEvaluator, make_players,  
                                             run_agent_shapley)
from algorithm.explain.feature_shap import (IMPORTANCE_FIELDS, decision_heads,  
                                            explain_agent_features, importance_rows,
                                            split_samples, subtype_overall)
from algorithm.explain.obs_schema import (build_obs_schema, generic_schema,  
                                          parse_groupings, verify_schema)
from algorithm.explain.policy import PolicyAdapter  
from algorithm.explain.rollout import PRIMARY_METRICS  
from algorithm.explain.shapley import coalition_label  

POSTHOC_GROUPINGS = ("semantic", "kind")
POSTHOC_SAMPLES = 256
POSTHOC_BACKGROUND = 64



_TRAINING_ONLY = ("--shap-every", "--num-episodes", "--save-rate", "--log-rate", "--restore",
                  "--display", "--benchmark", "--benchmark-iters", "--benchmark-dir")


def build_explain_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="IoA_explain.py", add_help=False, allow_abbrev=False,
        description="训练后的可解释性分析。除下列参数外, 其余参数与 IoA_train.py 完全相同, "
                    "且必须与训练时一致 (否则网络结构对不上 checkpoint)。")
    g = p.add_argument_group("IoA_explain 专属参数")
    g.add_argument("-h", "--help", action="store_true", help="显示本帮助")
    g.add_argument("--explain", choices=["all", "features", "agents"], default="all",
                   help="features = 特征级 SHAP (论文 Fig.20); agents = agent 级 Shapley "
                        "(论文 Fig.8/12/16); all = 两者 (默认)")
    g.add_argument("--explain-dir", default=None, help="输出目录 (默认 ./explain/<exp-name>)")
    g.add_argument("--explain-seed", type=int, default=None,
                   help="事后分析的随机种子 (默认 = --seed)。决定评估回合的初始状态与抽样。")
    g.add_argument("--collect-episodes", type=int, default=20,
                   help="特征级: 用完整的学到的团队跑多少回合来收集观测 (默认 20)")
    g.add_argument("--agent-players", choices=["both", "roles", "defenders"], default="both",
                   help="agent 级玩家: roles = healers/guardians 两个角色 (4 个联盟); "
                        "defenders = 每个防守方 (6 人 -> 64 个联盟); both (默认, 共享缓存)")
    g.add_argument("--agent-episodes", type=int, default=50,
                   help="agent 级: 每个联盟评估多少回合 (公共随机数, 默认 50)")
    g.add_argument("--agent-baseline", choices=["random", "noop"], default="random",
                   help="联盟之外的防守方执行什么: random = 均匀随机动作 (默认); "
                        "noop = 不动且不发话")
    g.add_argument("--agent-exact-max", type=int, default=8,
                   help="玩家数 <= 此值时精确枚举 2^n 个联盟, 否则置换采样 (默认 8)")
    g.add_argument("--agent-permutations", type=int, default=64,
                   help="置换采样时的置换数 (antithetic 成对, 默认 64)")
    g.add_argument("--no-plots", action="store_true", help="只写数据文件, 不出图")
    g.add_argument("--fig-format", default="pdf", metavar="FMT[,FMT]",
                   help="图片格式, 逗号分隔: pdf (矢量, 默认) / svg / eps (矢量) / png "
                        "(300 dpi 位图, 仅预览)。例: pdf,png = 矢量图 + 预览图。"
                        "对 --plot-trace / --compare 同样生效")
    g.add_argument("--allow-untrained", action="store_true",
                   help="checkpoint 不存在时仍然运行 (仅冒烟测试: 解释随机初始化的网络没有意义)")
    g.add_argument("--plot-trace", default=None, metavar="CSV",
                   help="子命令: 把训练期 --shap-every 写出的 CSV 画成演化曲线 (不需要训练参数)")
    g.add_argument("--compare", nargs="+", default=None, metavar="DIR",
                   help="子命令: 多个 IoA_explain 输出目录并排比较 (不需要训练参数)")
    g.add_argument("--compare-labels", nargs="+", default=None, metavar="LABEL",
                   help="--compare 各目录的图例名 (默认用目录名)")
    g.add_argument("--compare-players", choices=["roles", "defenders"], default="roles")
    g.add_argument("--compare-out", default=os.path.join(".", "explain", "compare"),
                   help="--compare 的输出目录")
    return p


def split_argv(argv):
    xp = build_explain_parser()
    xargs, rest = xp.parse_known_args(list(argv))
    return xp, xargs, rest


def parse_training_args(rest):
    import IoA_train
    saved = sys.argv
    try:
        sys.argv = ["IoA_train.py"] + list(rest)
        return IoA_train.parse_args()
    finally:
        sys.argv = saved


def _given(argv, flag) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def _apply_fig_format(xargs) -> list:
    spec = getattr(xargs, "fig_format", None)
    if spec is None:
        return report.figure_formats()
    try:
        return report.set_figure_formats(spec)
    except ValueError as e:
        raise SystemExit("--fig-format: {}".format(e))


def _validate_xargs(xargs) -> None:
    if xargs.collect_episodes < 1:
        raise SystemExit("--collect-episodes 必须 >= 1")
    if xargs.agent_episodes < 1:
        raise SystemExit("--agent-episodes 必须 >= 1")
    if xargs.agent_exact_max < 0:
        raise SystemExit("--agent-exact-max 必须 >= 0")
    if xargs.agent_permutations < 2:
        raise SystemExit("--agent-permutations 必须 >= 2")


def resolve_settings(arglist, xargs) -> dict:
    s = getattr(arglist, "shap_samples", None)
    b = getattr(arglist, "shap_background", None)
    em = getattr(arglist, "shap_exact_max", 10)
    return {
        "groupings": parse_groupings(getattr(arglist, "shap_grouping", None),
                                     default=POSTHOC_GROUPINGS),
        "samples": int(s) if s is not None else POSTHOC_SAMPLES,
        "background": int(b) if b is not None else POSTHOC_BACKGROUND,
        "permutations": int(getattr(arglist, "shap_permutations", 32) or 32),
        "exact_max": int(10 if em is None else em),
        "seed": (int(xargs.explain_seed) if xargs.explain_seed is not None
                 else int(getattr(arglist, "seed", 0) or 0)),
        "collect_episodes": int(xargs.collect_episodes),
        "agent_players": xargs.agent_players,
        "agent_episodes": int(xargs.agent_episodes),
        "agent_baseline": xargs.agent_baseline,
        "agent_exact_max": int(xargs.agent_exact_max),
        "agent_permutations": int(xargs.agent_permutations),
        "max_episode_len": int(arglist.max_episode_len),
    }


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    xp, xargs, rest = split_argv(argv)
    if xargs.help:
        xp.print_help()
        print("\n其余参数与 IoA_train.py 完全相同 (python IoA_train.py -h), 且必须与训练时一致。")
        return 0
    _apply_fig_format(xargs)
    if xargs.plot_trace:
        out = xargs.explain_dir or os.path.dirname(os.path.abspath(xargs.plot_trace))
        paths = report.plot_trace(xargs.plot_trace, out)
        print("[explain] SHAP 记录曲线 (训练期或 benchmark 时的 --shap-every): " + (", ".join(paths) if paths
                                             else "(没有生成; matplotlib 未安装或 CSV 为空)"))
        return 0
    if xargs.compare:
        paths = report.compare_runs(xargs.compare, xargs.compare_labels, xargs.compare_out,
                                    players=xargs.compare_players)
        print("[explain] 比较结果:\n  " + "\n  ".join(paths))
        return 0
    _validate_xargs(xargs)
    arglist = parse_training_args(rest)
    run_explain(arglist, xargs, rest)
    return 0


def run_explain(arglist, xargs, train_argv) -> dict:
    import train as base
    import IoA_train
    from other.runtime.runtime import configure_runtime, seed_torch
    from algorithm.safety.security_costs import REQUIRED_WORLD_ATTRS, security_costs

    ignored = [f for f in _TRAINING_ONLY if _given(train_argv, f)]
    if ignored:
        print("[explain] 以下训练/评估专用参数在事后分析里不使用: " + " ".join(ignored))
    if hasattr(arglist, "freeze_threat_prototype"):
        arglist.freeze_threat_prototype = True

    device, info = configure_runtime(arglist.device, arglist.cpu_threads, arglist.allow_tf32)
    arglist.device = str(device)
    print("[runtime] " + info)
    random.seed(arglist.seed)
    np.random.seed(arglist.seed)
    seed_torch(arglist.seed, device)

    env = base.make_env(arglist.scenario, arglist, False)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    base_num_adv = getattr(arglist, "num_adversaries", None)
    world_attackers = getattr(env.world, "num_attackers", None)
    if world_attackers is not None:
        num_adversaries = min(env.n, int(world_attackers))
    else:
        num_adversaries = min(env.n, int(base_num_adv or 0))
    arglist.num_adversaries = num_adversaries
    trainers = base.get_trainers(env, num_adversaries, obs_shape_n, arglist)

    load_dir = arglist.load_dir if arglist.load_dir else arglist.save_dir
    ckpt = os.path.join(load_dir, arglist.exp_name + ".pt")
    if os.path.isfile(ckpt):
        base.load_checkpoint(trainers, ckpt)
        ckpt_info = {"path": os.path.abspath(ckpt), "modified": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt)))}
        print("[explain] 已加载 checkpoint: {}".format(ckpt))
    elif xargs.allow_untrained:
        print("[explain][warn] 找不到 {} —— --allow-untrained: 继续解释【随机初始化】的网络, "
              "结果只能用于冒烟测试。".format(ckpt))
        ckpt_info = {"path": None, "untrained": True}
    else:
        raise SystemExit(
            "[explain] 找不到 checkpoint {}\n"
            "  IoA_explain.py 需要与训练时【完全相同】的参数 (尤其 --exp-name / --save-dir / "
            "--good-policy / 各模块开关)。\n  先跑完对应的训练; 只想冒烟测试流程可加 "
            "--allow-untrained。".format(ckpt))

    def seed_fn(s):
        random.seed(s)
        np.random.seed(s)
        seed_torch(s, device)

    cost_fn = (security_costs if all(hasattr(env.world, a) for a in REQUIRED_WORLD_ATTRS)
               else None)
    out_dir = xargs.explain_dir or os.path.join(".", "explain", arglist.exp_name)
    header = {"exp_name": arglist.exp_name, "scenario": arglist.scenario,
              "good_policy": arglist.good_policy, "adv_policy": arglist.adv_policy,
              "device": str(device), "checkpoint": ckpt_info, "argv": list(sys.argv),
              "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    return explain_loaded(env, trainers, num_adversaries, arglist, xargs, out_dir, seed_fn,
                          IoA_train._ioa_step_stats, cost_fn, header)


def explain_loaded(env, trainers, num_adversaries, arglist, xargs, out_dir, seed_fn,
                   step_stats_fn, cost_fn=None, header=None, log=print) -> dict:
    t_all = time.time()
    st = resolve_settings(arglist, xargs)
    st["fig_formats"] = _apply_fig_format(xargs)
    report.ensure_dir(out_dir)
    world = env.world
    defenders = list(range(int(num_adversaries), int(env.n)))
    if not defenders:
        raise SystemExit("[explain] 场景里没有防守方 (num_adversaries == env.n)")
    adapters = {i: PolicyAdapter(trainers[i]) for i in defenders}

    obs_n = env.reset()
    schemas, heads = {}, {}
    for i in defenders:
        D = int(env.observation_space[i].shape[0])
        sch = build_obs_schema(world, i, D)
        problems = verify_schema(sch, world, obs_n[i])
        if problems:
            log("[explain][warn] {} 的观测命名核对失败 ({} 处, 例: {}); 退化为 obs[k] 命名".format(
                sch.alias, len(problems), problems[0]))
            sch = generic_schema(D, i, sch.agent_name, sch.alias, sch.subtype, note="核对失败")
        elif sch.note:
            log("[explain][warn] " + sch.describe())
        schemas[i] = sch
        if not adapters[i].blocks:
            adapters[i].forward_f32(np.asarray(obs_n[i])[None, :])
        heads[i] = decision_heads(adapters[i].blocks, world, world.agents[i])

    bar = "=" * 76
    log(bar)
    log("IoA_explain | exp {} | scenario {} | good {} / adv {}".format(
        arglist.exp_name, arglist.scenario, arglist.good_policy, arglist.adv_policy))
    for i in defenders:
        log("  [schema] {} | heads: {}".format(schemas[i].describe(), ",".join(heads[i][0])))
    log("  特征级: groupings={} samples={} background={} exact<={} perms={} | "
        "agent 级: players={} episodes={} baseline={} | seed={}".format(
            ",".join(st["groupings"]), st["samples"], st["background"], st["exact_max"],
            st["permutations"], st["agent_players"], st["agent_episodes"],
            st["agent_baseline"], st["seed"]))
    log("  输出目录: {} | 图片格式: {}".format(os.path.abspath(out_dir),
                                            ",".join(st["fig_formats"])))
    log(bar)

    ss = np.random.SeedSequence(st["seed"])
    ss_collect, ss_agents, ss_feat, ss_perm = ss.spawn(4)
    collect_seeds = [int(s) for s in ss_collect.generate_state(st["collect_episodes"])]
    agent_seeds = [int(s) for s in ss_agents.generate_state(st["agent_episodes"])]
    evaluator = CoalitionEvaluator(env, trainers, adapters, num_adversaries,
                                   st["max_episode_len"], st["agent_baseline"], seed_fn,
                                   step_stats_fn, cost_fn)
    plots = not bool(getattr(xargs, "no_plots", False))
    if plots and not report.have_matplotlib():
        log("[explain] 未安装 matplotlib: 只写 CSV/NPZ/JSON, 不出图 (pip install matplotlib)")
        plots = False
    summary = {"header": header or {}, "settings": st, "features": {}, "agents": {}}

    if xargs.explain in ("all", "features"):
        t0 = time.time()
        store = {i: [] for i in defenders}

        def on_obs(e, obs_now, t):
            for i in defenders:
                store[i].append(np.asarray(obs_now[i], dtype=np.float32).reshape(-1))

        per, _metas = evaluator.run(set(defenders), collect_seeds, on_obs=on_obs)
        team = {m: float(np.nanmean(per[m])) for m in PRIMARY_METRICS}
        log("[features] 完整团队 {} 回合: reward {:+.3f} | intrusion {:.3f} | infected {:.3f} "
            "(可与训练日志同名列对照, 确认 checkpoint 对得上)".format(
                st["collect_episodes"], team["Average_Episode_Reward"],
                team["Intrusion_Success_Rate"], team["Mean_Infected"]))
        summary["features"]["team_check"] = team
        feat_rng = np.random.default_rng(ss_feat)
        splits = {i: split_samples(np.stack(store[i], axis=0), st["samples"],
                                   st["background"], feat_rng) for i in defenders}
        for grouping in st["groupings"]:
            rows, gaps = [], {}
            for i in defenders:
                X_ex, X_bg = splits[i]
                t1 = time.time()
                fe = explain_agent_features(adapters[i], schemas[i], heads[i], X_ex, X_bg,
                                            grouping, st["exact_max"], st["permutations"],
                                            feat_rng)
                rows.extend(importance_rows(fe))
                gaps[fe.alias] = fe.efficiency_gap
                report.save_feature_npz(os.path.join(
                    out_dir, "features", "shap_{}_{}.npz".format(fe.alias, grouping)),
                    fe, schemas[i])
                log("  [features:{}] {} | {} 组 | {} | 效率自检 {:.1e} | {:.1f}s".format(
                    grouping, fe.alias, len(fe.group_names), fe.method, fe.efficiency_gap,
                    time.time() - t1))
            report.write_csv(os.path.join(out_dir, "feature_importance_{}.csv".format(grouping)),
                             rows, IMPORTANCE_FIELDS)
            agg = subtype_overall(rows, grouping)
            for sub, items in agg.items():
                log("  [features:{}] {} 最重要: {}".format(grouping, sub, ", ".join(
                    "{} {:.2f}".format(g, s) for g, s, _ in items[:5])))
            if plots:
                report.plot_feature_overall(rows, grouping, os.path.join(
                    out_dir, "fig_feature_importance_{}".format(grouping)))
                report.plot_feature_heads(rows, grouping, os.path.join(
                    out_dir, "fig_feature_{}".format(grouping)))
            summary["features"][grouping] = {
                "efficiency_gap_max": max(gaps.values()) if gaps else None,
                "top": {sub: items[:8] for sub, items in agg.items()}}
        summary["features"]["seconds"] = time.time() - t0

    if xargs.explain in ("all", "agents"):
        modes = ["roles", "defenders"] if st["agent_players"] == "both" else [st["agent_players"]]
        perm_rng = np.random.default_rng(ss_perm)
        for mode in modes:
            players = make_players(env, num_adversaries, mode)
            log("[agents:{}] 玩家 {} | 基线 {} | 每个联盟 {} 回合 (公共随机数)".format(
                mode, ", ".join(p.name for p in players), st["agent_baseline"],
                st["agent_episodes"]))
            res = run_agent_shapley(evaluator, players, agent_seeds, mode,
                                    st["agent_exact_max"], st["agent_permutations"],
                                    perm_rng, log)
            coal, shap, inter, eff = report.agent_tables(res)
            report.write_csv(os.path.join(out_dir, "agent_coalitions_{}.csv".format(mode)), coal)
            report.write_csv(os.path.join(out_dir, "agent_shapley_{}.csv".format(mode)), shap)
            if inter:
                report.write_csv(os.path.join(out_dir, "agent_interactions_{}.csv".format(mode)),
                                 inter)
            report.save_agent_npz(os.path.join(out_dir, "agent_{}.npz".format(mode)), res)
            _log_agent_summary(res, coal, shap, inter, log)
            if plots:
                for m in PRIMARY_METRICS:
                    report.plot_agent_shapley(shap, coal, inter, m, os.path.join(
                        out_dir, "fig_agent_shapley_{}_{}".format(mode, m)), mode)
                    report.plot_coalitions(coal, m, os.path.join(
                        out_dir, "fig_agent_coalitions_{}_{}".format(mode, m)), res.baseline)
                    if inter and res.n >= 3:
                        report.plot_interactions(inter, [p.name for p in players], m, os.path.join(
                            out_dir, "fig_agent_interactions_{}_{}".format(mode, m)))
            summary["agents"][mode] = {
                "method": res.method,
                "players": {p.name: [res.agent_labels[a] for a in p.agents] for p in players},
                "efficiency_gap": eff,
                "phi": {m: {r["player"]: {"phi": r["phi"], "se": r["se"]}
                            for r in shap if r["metric"] == m} for m in PRIMARY_METRICS},
                "seconds": res.seconds}
        summary["agents"]["episodes_run"] = evaluator.episodes_run

    summary["seconds_total"] = time.time() - t_all
    report.write_json(os.path.join(out_dir, "summary.json"), summary)
    log("[explain] 完成, 用时 {:.1f}s | 输出目录: {}".format(summary["seconds_total"],
                                                      os.path.abspath(out_dir)))
    return summary


def _log_agent_summary(res, coal, shap, inter, log) -> None:
    names = [p.name for p in res.players]
    full_label = coalition_label(res.full_mask, names)
    for m in PRIMARY_METRICS:
        c = {r["coalition"]: r for r in coal if r["metric"] == m}
        v0, vn = c.get("{}"), c.get(full_label)
        if v0 is not None and vn is not None:
            log("  {}: v(空)={:+.3f}  v(全体)={:+.3f}  团队增益={:+.3f}".format(
                m, v0["mean"], vn["mean"], vn["mean"] - v0["mean"]))
        for r in shap:
            if r["metric"] == m:
                log("    {:<12s} phi={:+.3f} ±{:.3f} | solo={:+.3f} | leave-one-out={:+.3f}".format(
                    r["player"], r["phi"], 1.96 * r["se"], r["solo"], r["loo"]))
        ii = [r for r in inter if r["metric"] == m and np.isfinite(r["interaction"])]
        ii.sort(key=lambda r: -abs(r["interaction"]))
        for r in ii[:3]:
            log("    交互 {} x {}: {:+.3f} ±{:.3f} (>0 协同, <0 冗余)".format(
                r["player_i"], r["player_j"], r["interaction"], 1.96 * r["se"]))


if __name__ == "__main__":
    sys.exit(main())
