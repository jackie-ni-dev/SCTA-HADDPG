

import inspect
import os
import random
import time
import pickle

import numpy as np
import torch



import train as base
from other.runtime.runtime import configure_runtime, seed_torch
from algorithm.safety import REQUIRED_WORLD_ATTRS, assert_world_supported, security_costs
from algorithm.trainer.sequential_defender_update import make_defender_updater


_BASE_DEFAULT_SCENARIO = "simple_adversary_hetero"
_BASE_DEFAULT_MAX_EP_LEN = 25
_BASE_DEFAULT_GOOD_POLICY = "hatd3"
_BASE_DEFAULT_ADV_POLICY = "hatd3"
_BASE_DEFAULT_EXP_NAME = "default_exp"


def _agent_role(a):
    return getattr(a, "role", "defender")


def _c_value(agent, idx):
    c = np.asarray(getattr(agent.state, "c", np.zeros(idx + 1)), dtype=float)
    return float(c[idx]) if c.size > idx else 0.0


def _joint_rescue_coverage(world):
    if not bool(getattr(world, "joint_coordination", False)):
        return float("nan")
    benign = [a for a in world.agents if not getattr(a, "adversary", False)]
    healers = [a for a in benign
               if getattr(a, "subtype", None) == "healer"
               and _agent_role(a) == "defender"]
    guardians = [a for a in benign
                 if getattr(a, "subtype", None) == "guardian"
                 and _agent_role(a) == "defender"]
    th = float(getattr(world, "joint_rescue_risk_threshold", 0.15))
    risky = [(b, float(getattr(b, "corruption", 0.0)))
             for b in benign if float(getattr(b, "corruption", 0.0)) >= th]
    if not risky:
        return 1.0
    total = sum(r for _, r in risky) + 1e-8
    covered = 0.0
    hrad = float(getattr(world, "comm_range", 0.5))
    grad = float(getattr(world, "guard_heal_radius", 0.40))
    for b, r in risky:
        h_ok = any(h is not b and float(np.sqrt(np.sum(np.square(
            h.state.p_pos - b.state.p_pos)))) < hrad for h in healers)
        g_ok = any(g is not b and float(np.sqrt(np.sum(np.square(
            g.state.p_pos - b.state.p_pos)))) < grad for g in guardians)
        if h_ok and g_ok:
            covered += r
    return float(covered / total)


def _ioa_step_stats(world, prev_roles):
    agents = world.agents
    mal = int(getattr(world, "mal_symbol", 0))
    heal = int(getattr(world, "heal_symbol", 1))

    benign = [a for a in agents if not getattr(a, "adversary", False)]
    attackers = [a for a in agents if getattr(a, "adversary", False)]

    n_infected = int(getattr(
        world, "n_infected",
        sum(1 for a in benign if _agent_role(a) == "infected")))

    if benign:
        mean_corruption = float(np.mean(
            [float(getattr(a, "corruption", 0.0)) for a in benign]))
    else:
        mean_corruption = 0.0

    recoveries = 0
    new_infections = 0
    cur_roles = {}
    for a in agents:
        r = _agent_role(a)
        cur_roles[a.name] = r
        pr = prev_roles.get(a.name, r)
        if pr == "infected" and r == "defender":
            recoveries += 1
        elif pr == "defender" and r == "infected":
            new_infections += 1

    mal_volume = float(sum(max(0.0, _c_value(a, mal)) for a in agents
                           if _agent_role(a) in ("attacker", "infected")))
    heal_volume = float(sum(max(0.0, _c_value(a, heal)) for a in agents
                            if _agent_role(a) == "defender"))

    reach = float("nan")
    if attackers and benign:
        per_atk = []
        for atk in attackers:
            d = min(float(np.sqrt(np.sum(np.square(
                atk.state.p_pos - b.state.p_pos)))) for b in benign)
            per_atk.append(d)
        reach = float(np.mean(per_atk))

    return {
        "n_infected": n_infected,
        "mean_corruption": mean_corruption,
        "recoveries": recoveries,
        "new_infections": new_infections,
        "mal_volume": mal_volume,
        "heal_volume": heal_volume,
        "reach": reach,
        "joint_rescue_coverage": _joint_rescue_coverage(world),
        "roles": cur_roles,
    }


def _security_metrics(sc_agents):
    nan = float("nan")
    names = ("Intrusion", "Infection", "Service")
    if not sc_agents:
        return dict([("Lambda_%s" % n, nan) for n in names]
                    + [("Cost_%s" % n, nan) for n in names])
    ref = sc_agents[0]
    for ag in sc_agents[1:]:
        if any(abs(a - b) > 1e-9 for a, b in zip(ag.last_costs, ref.last_costs)):
            raise RuntimeError(
                "defender {} 与 {} 的 J_C 不同 ({} vs {}) —— 它们本应共享同一批 "
                "cost。多半是 stage_cost 漏掉了部分 trainer, 或 buffer 不同步。"
                .format(ag.agent_index, ref.agent_index, ag.last_costs, ref.last_costs))
    lam = ref.lagrange.values()
    return {**{("Lambda_%s" % n): lam[k] for k, n in enumerate(names)},
            **{("Cost_%s" % n): ref.last_costs[k] for k, n in enumerate(names)}}


def _beta_trainers(trainers):
    return [ag for ag in trainers if getattr(ag, "use_threat_aware_beta", False)]


def _beta_metrics(beta_agents):
    nan = float("nan")
    if not beta_agents:
        return {"Mean_Beta": nan, "Mean_OOD_Score": nan}
    measured = [ag for ag in beta_agents
                if int(getattr(ag, "_beta_update_count", 0)) > 0]
    if not measured:
        return {"Mean_Beta": nan, "Mean_OOD_Score": nan}
    return {"Mean_Beta": float(np.mean([float(ag.last_beta) for ag in measured])),
            "Mean_OOD_Score": float(np.mean([float(ag.last_ood) for ag in measured]))}


def _defender_critic_probe(trainers, num_adversaries):
    nan = float("nan")
    for ag in trainers[int(num_adversaries):]:
        if bool(getattr(ag, "_on_policy", False)):
            continue          
        c1 = getattr(ag, "critic1", None)
        if c1 is not None:
            n_critics = 2 if getattr(ag, "critic2", None) is not None else 1
            net = getattr(c1, "net", None)
            if net is not None:
                return float(n_critics), float(net[0].in_features)
            continue
        c = getattr(ag, "critic", None)      
        net = getattr(c, "net", None)
        if net is not None:
            return 1.0, float(net[0].in_features)
    return nan, nan


def _describe_critic_family(trainers, num_adversaries, arglist, coordinated):
    n_critics, in_dim = _defender_critic_probe(trainers, num_adversaries)
    if not np.isfinite(n_critics):
        return None
    if n_critics == 1:
        fam = "single (DDPG 语义: target = Q')"
    else:
        fam = "twin  (TD3 语义: target = min(Q1', Q2'))"
    line = "defender Q-critic = {} | input dim = {}".format(fam, int(in_dim))

    if coordinated:
        ref = coordinated[0]
        period = getattr(ref, "policy_update_every", None)
        if period is None:
            period = int(arglist.update_rate) * int(arglist.policy_update_rate)
            how = "= --update-rate {} x --policy-update-rate {}".format(
                arglist.update_rate, arglist.policy_update_rate)
        else:
            period = int(period)
            how = ("trainer 自报 policy_update_every = --update-rate {}; "
                   "--policy-update-rate 不参与".format(arglist.update_rate))
        line += " | actor 每 {} 训练步更新一次 [{}]".format(period, how)
    return line


def _window_mean(hist, window):
    if window <= 0 or len(hist) == 0:
        return float("nan")
    arr = np.asarray(hist[-window:], dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


class IoAMetricsLogger:
    METRIC_NAMES = [
        "Average_Episode_Reward",     
        "Intrusion_Success_Rate",     
        "Cumulative_Defense_Count",   
        "Attacker_Episode_Reward",    
        "Mean_Infected",              
        "Peak_Infected",              
        "Mean_Corruption",            
        "Recovery_Count",             
        "Malicious_Msg_Volume",       
        "Healing_Msg_Volume",         
        "Attacker_Reach",             
        "Lambda_Intrusion", "Lambda_Infection", "Lambda_Service",
        "Cost_Intrusion", "Cost_Infection", "Cost_Service",
        "Mean_Beta", "Mean_OOD_Score",
        "Critic_Input_Dim",
        "Joint_Rescue_Coverage",
        "Num_Critics",
    ]

    def __init__(self, data_dir, scenario, good_policy, adv_policy, timestamp=None,
                 exp_name=None):
        self.data_dir = data_dir
        self.scenario = scenario
        self.good_policy = good_policy
        self.adv_policy = adv_policy
        self.timestamp = timestamp or time.strftime("%Y%m%d_%H%M%S")
        self._exp = str(exp_name) if exp_name else ""
        self._use_exp = bool(self._exp)

        self.paths = {}
        for m in self.METRIC_NAMES:
            d = os.path.join(data_dir, m)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, self.fname_for(m))
            with open(p, "w") as f:
                f.write("step,value\n")
            self.paths[m] = p

        total_dir = os.path.join(data_dir, "total")
        os.makedirs(total_dir, exist_ok=True)
        self.total_path = os.path.join(total_dir, self.fname_for("total"))
        with open(self.total_path, "w") as f:
            f.write("step," + ",".join(self.METRIC_NAMES) + "\n")

    def fname_for(self, metric_label):
        if self._use_exp:
            return f"{self._exp}_{metric_label}_{self.timestamp}.csv"
        return (f"{self.scenario}_{metric_label}_{self.good_policy}_"
                f"{self.adv_policy}_{self.timestamp}.csv")

    def log(self, step, values):
        for m in self.METRIC_NAMES:
            v = values.get(m, float("nan"))
            with open(self.paths[m], "a") as f:
                f.write(f"{step},{v}\n")
        with open(self.total_path, "a") as f:
            row = ",".join(str(values.get(m, float("nan"))) for m in self.METRIC_NAMES)
            f.write(f"{step},{row}\n")


def parse_args():
    args = base.parse_args()  

    if args.scenario == _BASE_DEFAULT_SCENARIO:
        args.scenario = "ioa_infection"
    if args.max_episode_len == _BASE_DEFAULT_MAX_EP_LEN:
        args.max_episode_len = 50
    if args.good_policy == _BASE_DEFAULT_GOOD_POLICY:
        args.good_policy = "hatd3"
    if args.adv_policy == _BASE_DEFAULT_ADV_POLICY:
        args.adv_policy = "hatd3"
    if args.exp_name == _BASE_DEFAULT_EXP_NAME:
        args._exp_name_user_set = False
        args.exp_name = "ioa_infection"
    else:
        args._exp_name_user_set = True
    return args


def train(arglist):
    resolved_device, runtime_info = configure_runtime(
        arglist.device, arglist.cpu_threads, arglist.allow_tf32
    )
    arglist.device = str(resolved_device)
    print("[runtime] " + runtime_info)

    random.seed(arglist.seed)
    np.random.seed(arglist.seed)
    seed_torch(arglist.seed, resolved_device)

    env = base.make_env(arglist.scenario, arglist, arglist.benchmark)
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

    base_num_adv = getattr(arglist, "num_adversaries", None)
    world_attackers = getattr(env.world, "num_attackers", None)
    if world_attackers is not None:
        num_adversaries = min(env.n, int(world_attackers))
    else:
        num_adversaries = min(env.n, int(base_num_adv or 0))
    arglist.num_adversaries = num_adversaries

    trainers = base.get_trainers(env, num_adversaries, obs_shape_n, arglist)

    n_benign = env.n - num_adversaries
    is_ioa = hasattr(env.world, "intrusion_succeeded")
    print("=" * 72)
    print("IoA dedicated trainer  |  scenario: {}".format(arglist.scenario))
    print("  good(defender) policy: {}   adv(attacker) policy: {}".format(
        arglist.good_policy, arglist.adv_policy))
    print("  agents: {}  (attackers={}, benign={})".format(
        env.n, num_adversaries, n_benign))
    print("  obs dims per agent: {}".format([int(s[0]) for s in obs_shape_n]))
    print("  action convention : {}".format(
        base.describe_action_convention(arglist, trainers)))
    for _line in base.describe_update_semantics(arglist, trainers, num_adversaries):
        print("  [update] " + _line)
    if (base_num_adv is not None) and (world_attackers is not None) \
            and (int(base_num_adv) != num_adversaries):
        print("  [auto-sync] num_adversaries {} -> {} (= scenario num_attackers); "
              "no need to edit train.py's NUM_ADVERSARIES".format(
                  base_num_adv, num_adversaries))
    if not is_ioa:
        print("  [warn] scenario does not expose world.intrusion_succeeded; "
              "IoA-specific columns will be empty/NaN.")
    _full_line = base.describe_full_innovation(arglist, trainers)
    if _full_line:
        print("  [full] " + _full_line)
    _beta_line = base.describe_threat_beta(arglist, trainers)
    if _beta_line:
        print("  [full] " + _beta_line)
    _sc_line = base.describe_security_constraints(arglist, trainers)
    if _sc_line:
        print("  [full] " + _sc_line)
        assert_world_supported(env.world)
    _khop_line = base.describe_khop_security(arglist, trainers)
    if _khop_line:
        print("  [full] " + _khop_line)
    _bw_line = base.describe_div_bandwidth(arglist, trainers)
    if _bw_line:
        print("  [div] " + _bw_line)
    _smooth_line = base.describe_intent_smoothing(arglist, trainers)
    if _smooth_line:
        print("  [div] " + _smooth_line)
    for _w in base.warn_ignored_smoothing_flags(arglist):
        print("  " + _w)
    for _w in base.warn_ignored_bandwidth_flags(arglist):
        print("  " + _w)
    for _w in base.warn_ignored_security_flags(arglist):
        print("  " + _w)
    for _w in base.warn_ignored_shap_flags(arglist):
        print("  " + _w)
    if bool(getattr(env.world, "role_synergy", False)):
        w = env.world
        print("  [env] role synergy = ON | guardian intercept r={:.2f} strength={:.2f} "
              "| guarded HEAL r={:.2f} boost=+{:.2f} | triage_w={:.2f}".format(
                  w.guard_intercept_radius, w.guard_intercept_strength,
                  w.guard_heal_radius, w.heal_synergy_boost, w.triage_w))
        if bool(getattr(w, "joint_coordination", False)):
            print("  [env] joint coordination v2 = ON | unguarded HEAL x{:.2f} "
                  "| joint_rescue_w={:.2f} risk_th={:.2f}".format(
                      w.unguarded_heal_factor, w.joint_rescue_w,
                      w.joint_rescue_risk_threshold))
    if hasattr(env.world, "max_malgents"):
        w = env.world
        print("  [full] malgent slots: M_max={} active∈[{},{}] sampling={} "
              "attacker={} | OOD strength={} surface={}".format(
                  w.max_malgents, w.active_min, w.active_max, w.slot_sampling,
                  w.attacker_mode, w.attack_strength, w.attack_surface))
        if w.attacker_mode == "learned" and w.slot_sampling == "ordered" \
                and w.active_max < w.max_malgents:
            print("  [warn] slot-sampling=ordered + learned attacker: slot {}..{} "
                  "在训练期永远不会被激活。评估区间一旦覆盖它们, 你测到的是"
                  "『攻击者变弱』而不是『防守方泛化』。改用 --ioa-malgent-slot-sampling "
                  "random 或 --ioa-attacker-mode scripted。".format(
                      w.active_max, w.max_malgents - 1))
    print("  tip: use --good-policy maddpg to let defenders exploit the "
          "language buffer for detection.")
    print("=" * 72)

    if arglist.load_dir == "":
        arglist.load_dir = arglist.save_dir
    ckpt_path = os.path.join(arglist.load_dir, arglist.exp_name + ".pt")
    if arglist.display or arglist.restore or arglist.benchmark:
        if os.path.isfile(ckpt_path):
            print("Loading previous state from {}".format(ckpt_path))
            base.load_checkpoint(trainers, ckpt_path)
        else:
            print("No checkpoint found at {} - starting fresh".format(ckpt_path))

    metrics_logger = IoAMetricsLogger(
        arglist.data_dir, arglist.scenario, arglist.good_policy, arglist.adv_policy,
        exp_name=(arglist.exp_name if getattr(arglist, "_exp_name_user_set", False)
                  else None))
    print(f"Per-metric CSV: {arglist.data_dir}/<metric>/{metrics_logger.fname_for('<metric>')}")
    print(f"Combined CSV:   {metrics_logger.total_path}")

    shap_tracer = None
    if int(getattr(arglist, "shap_every", 0) or 0) > 0:
        from algorithm.explain.trace import ShapTracer
        shap_tracer = ShapTracer(
            env, trainers, num_adversaries, arglist,
            csv_path=os.path.join(arglist.data_dir, "shap_trace",
                                  metrics_logger.fname_for("shap_trace")))
        print("  [shap] " + shap_tracer.describe())

    episode_rewards = [0.0]                       
    defender_episode_rewards = [0.0]              
    adversary_episode_rewards = [0.0]             
    agent_rewards = [[0.0] for _ in range(env.n)]
    final_ep_rewards, final_ep_ag_rewards = [], []
    agent_info = [[[]]]

    intrusion_successes = []
    cumulative_defense = 0
    episode_intrusion = False

    mean_infected_hist = []
    peak_infected_hist = []
    mean_corr_hist = []
    recovery_hist = []
    mal_vol_hist = []
    heal_vol_hist = []
    reach_hist = []
    joint_rescue_hist = []

    ep_steps = 0
    ep_inf_sum = 0.0
    ep_inf_peak = 0
    ep_corr_sum = 0.0
    ep_recov = 0
    ep_mal_sum = 0.0
    ep_heal_sum = 0.0
    ep_reach_sum = 0.0
    ep_reach_n = 0
    ep_joint_rescue_sum = 0.0
    ep_joint_rescue_n = 0

    eval_costs_on = bool(arglist.benchmark) and all(
        hasattr(env.world, k) for k in REQUIRED_WORLD_ATTRS)
    ep_cost_sum = np.zeros(3, dtype=np.float64)
    eval_cost_hist = []                           
    bench_rows = 0                                

    obs_n = env.reset()
    prev_roles = {a.name: _agent_role(a) for a in env.world.agents}
    episode_step = 0
    train_step = 0
    t_start = time.time()

    on_policy_agents = [a for a in trainers if getattr(a, "_on_policy", False)]
    off_policy_agents = [a for a in trainers if not getattr(a, "_on_policy", False)]
    rng = np.random.default_rng(arglist.seed)

    defender_updater, coordinated_agents = make_defender_updater(
        trainers, num_adversaries, rng=rng)
    coordinated_ids = {id(ag) for ag in coordinated_agents}
    if defender_updater is not None:
        print("  [migration] shared-batch sequential defender updater enabled "
              "(order=random, HARL paper semantics)")
        print("  [migration] defenders={} div_coef={} div_weight={}".format(
            len(coordinated_agents), arglist.div_coef, arglist.div_weight))

    critic_n, critic_in_dim = _defender_critic_probe(trainers, num_adversaries)
    _critic_line = _describe_critic_family(
        trainers, num_adversaries, arglist, coordinated_agents)
    if _critic_line:
        print("  [critic] " + _critic_line)

    sc_agents = [ag for ag in trainers
                 if getattr(ag, "use_security_constraints", False)]
    sc_checked = False

    beta_agents = _beta_trainers(trainers)

    def _metric_values(window):
        def_reward = (float(np.mean(defender_episode_rewards[-window - 1:-1]))
                      if window > 0 else 0.0)
        atk_reward = (float(np.mean(adversary_episode_rewards[-window - 1:-1]))
                      if window > 0 else 0.0)
        intrusion_rate = (float(np.mean(intrusion_successes[-window:]))
                          if window > 0 else 0.0)
        values = {
            "Average_Episode_Reward": def_reward,
            "Intrusion_Success_Rate": intrusion_rate,
            "Cumulative_Defense_Count": cumulative_defense,
            "Attacker_Episode_Reward": atk_reward,
            "Mean_Infected": _window_mean(mean_infected_hist, window),
            "Peak_Infected": _window_mean(peak_infected_hist, window),
            "Mean_Corruption": _window_mean(mean_corr_hist, window),
            "Recovery_Count": _window_mean(recovery_hist, window),
            "Malicious_Msg_Volume": _window_mean(mal_vol_hist, window),
            "Healing_Msg_Volume": _window_mean(heal_vol_hist, window),
            "Attacker_Reach": _window_mean(reach_hist, window),
            **_security_metrics(sc_agents),
            **_beta_metrics(beta_agents),
            "Critic_Input_Dim": critic_in_dim,
            "Joint_Rescue_Coverage": _window_mean(joint_rescue_hist, window),
            "Num_Critics": critic_n,
        }
        if eval_costs_on:
            recent = eval_cost_hist[-window:] if window > 0 else []
            for k, n in enumerate(("Intrusion", "Infection", "Service")):
                values["Cost_%s" % n] = _window_mean([c[k] for c in recent], window)
        return values

    if arglist.benchmark:
        print("  [benchmark] 评估指标每 {} 个回合写一行 CSV (--log-rate); step 列 = 已完成的"
              "评估回合数; Cost_* = 评估期逐步实测均值{}; Lambda_* = checkpoint 里的 λ "
              "(评估期不更新)".format(
                  arglist.log_rate, "" if eval_costs_on else " (本场景不支持, 写 nan)"))

    print("Starting iterations...")
    while True:
        for agent in on_policy_agents:
            agent.prepare_step(obs_n)

        action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
        if shap_tracer is not None:
            shap_tracer.observe(obs_n)          
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = episode_step >= arglist.max_episode_len
        done = done or terminal

        w = env.world
        if bool(getattr(w, "intrusion_succeeded", False)):
            episode_intrusion = True

        st = _ioa_step_stats(w, prev_roles)
        prev_roles = st["roles"]
        ep_steps += 1
        ep_inf_sum += st["n_infected"]
        ep_inf_peak = max(ep_inf_peak, st["n_infected"])
        ep_corr_sum += st["mean_corruption"]
        ep_recov += st["recoveries"]
        ep_mal_sum += st["mal_volume"]
        ep_heal_sum += st["heal_volume"]
        if not np.isnan(st["reach"]):
            ep_reach_sum += st["reach"]
            ep_reach_n += 1
        if not np.isnan(st["joint_rescue_coverage"]):
            ep_joint_rescue_sum += st["joint_rescue_coverage"]
            ep_joint_rescue_n += 1
        if eval_costs_on:
            ep_cost_sum += security_costs(w)      

        if not (arglist.benchmark or arglist.display):
            if sc_agents:
                _costs = security_costs(w)
                for _ag in sc_agents:
                    _ag.stage_cost(_costs)

            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done, terminal)

            if sc_agents and not sc_checked \
                    and len(sc_agents[0].replay_buffer) >= sc_agents[0].min_replay_buffer_len:
                sc_checked = True
                for _ag in sc_agents:
                    _cov = _ag.assert_cost_wired(0.99)
                print("  [full] cost 接线自检通过: coverage={:.3f} ({})".format(
                    _cov, sc_agents[0].replay_buffer.describe()))

        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew
            if i >= num_adversaries:
                defender_episode_rewards[-1] += rew
            else:
                adversary_episode_rewards[-1] += rew

        if done or terminal:
            intrusion_successes.append(1 if episode_intrusion else 0)
            if not episode_intrusion:
                cumulative_defense += 1
            episode_intrusion = False

            denom = max(1, ep_steps)
            mean_infected_hist.append(ep_inf_sum / denom)
            peak_infected_hist.append(float(ep_inf_peak))
            mean_corr_hist.append(ep_corr_sum / denom)
            recovery_hist.append(float(ep_recov))
            mal_vol_hist.append(ep_mal_sum / denom)
            heal_vol_hist.append(ep_heal_sum / denom)
            reach_hist.append(ep_reach_sum / ep_reach_n if ep_reach_n > 0 else float("nan"))
            joint_rescue_hist.append(
                ep_joint_rescue_sum / ep_joint_rescue_n
                if ep_joint_rescue_n > 0 else float("nan"))
            if eval_costs_on:
                eval_cost_hist.append(ep_cost_sum / denom)
                ep_cost_sum = np.zeros(3, dtype=np.float64)

            ep_steps = 0
            ep_inf_sum = 0.0
            ep_inf_peak = 0
            ep_corr_sum = 0.0
            ep_recov = 0
            ep_mal_sum = 0.0
            ep_heal_sum = 0.0
            ep_reach_sum = 0.0
            ep_reach_n = 0
            ep_joint_rescue_sum = 0.0
            ep_joint_rescue_n = 0

            obs_n = env.reset()
            prev_roles = {a.name: _agent_role(a) for a in env.world.agents}
            episode_step = 0
            episode_rewards.append(0)
            defender_episode_rewards.append(0)
            adversary_episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])
            if shap_tracer is not None:
                shap_tracer.maybe_explain(len(episode_rewards) - 1, train_step + 1)

        train_step += 1

        if arglist.benchmark:
            for i, info in enumerate(info_n["n"]):
                while len(agent_info[-1]) <= i:
                    agent_info[-1].append([])
                agent_info[-1][i].append(info)
            n_eval = len(intrusion_successes)
            if (done or terminal) and n_eval % arglist.log_rate == 0:
                metrics_logger.log(n_eval, _metric_values(arglist.log_rate))
                bench_rows += 1
            if train_step > arglist.benchmark_iters and (done or terminal):
                if bench_rows == 0 and n_eval > 0:
                    metrics_logger.log(n_eval, _metric_values(n_eval))
                    bench_rows += 1
                s_all = _metric_values(n_eval)
                d_all = np.asarray(defender_episode_rewards[:-1], dtype=float)
                print("[benchmark] 全部 {} 个评估回合 | def_rew {:.2f} ± {:.2f} | atk_rew {:.2f} | "
                      "infected {:.2f}/{:.1f} of {} | recover {:.2f} | intrusion {:.1%} | "
                      "cost int/inf/svc {:.3f}/{:.3f}/{:.3f}".format(
                          n_eval, s_all["Average_Episode_Reward"],
                          float(np.std(d_all)) if d_all.size else float("nan"),
                          s_all["Attacker_Episode_Reward"], s_all["Mean_Infected"],
                          s_all["Peak_Infected"], n_benign, s_all["Recovery_Count"],
                          s_all["Intrusion_Success_Rate"], s_all["Cost_Intrusion"],
                          s_all["Cost_Infection"], s_all["Cost_Service"]))
                print("[benchmark] 评估 CSV 共 {} 行: {}".format(
                    bench_rows, metrics_logger.total_path))
                os.makedirs(arglist.benchmark_dir, exist_ok=True)
                file_name = os.path.join(arglist.benchmark_dir, arglist.exp_name + ".pkl")
                print("Finished benchmarking, saving to {}".format(file_name))
                with open(file_name, "wb") as fp:
                    pickle.dump(agent_info[:-1], fp)
                break
            continue

        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        for agent in off_policy_agents:
            agent.preupdate()

        ordinary_off_policy = [
            ag for ag in off_policy_agents if id(ag) not in coordinated_ids
        ]

        update_order = base.randomized_harl_order(ordinary_off_policy, rng)
        for agent in update_order:
            agent.update(trainers, train_step)

        if defender_updater is not None:
            defender_updater.update(trainers, train_step)

        if on_policy_agents:
            groups = {}
            for ag in on_policy_agents:
                groups.setdefault(type(ag), []).append(ag)
            for cls, group in groups.items():
                if not group[0].is_ready_to_update():
                    continue
                for ag in group:
                    ag.prepare_step(obs_n)
                params = inspect.signature(cls.joint_update).parameters
                kwargs = {"rng": rng}
                if "all_agents" in params:
                    kwargs["all_agents"] = trainers
                cls.joint_update(group, **kwargs)

        if terminal and (len(episode_rewards) % arglist.log_rate == 0):
            window = min(arglist.log_rate, len(intrusion_successes))
            metrics_logger.log(len(episode_rewards), _metric_values(window))

        if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            base.save_checkpoint(trainers, ckpt_path)
            sw = min(arglist.save_rate, len(intrusion_successes))
            def_rew = (float(np.mean(defender_episode_rewards[-sw - 1:-1]))
                       if sw > 0 else 0.0)
            atk_rew = (float(np.mean(adversary_episode_rewards[-sw - 1:-1]))
                       if sw > 0 else 0.0)
            intrusion_rate = (float(np.mean(intrusion_successes[-sw:]))
                              if sw > 0 else 0.0)
            inf_mean = _window_mean(mean_infected_hist, sw)
            inf_peak = _window_mean(peak_infected_hist, sw)
            recov = _window_mean(recovery_hist, sw)
            mal_v = _window_mean(mal_vol_hist, sw)
            heal_v = _window_mean(heal_vol_hist, sw)
            reach = _window_mean(reach_hist, sw)
            print(
                "steps {} | eps {} | def_rew {:.2f} atk_rew {:.2f} | "
                "infected {:.2f}/{:.1f} of {} | recover {:.2f} | "
                "mal/heal {:.2f}/{:.2f} | reach {:.2f} | "
                "intrusion {:.1%} | defense {} | {:.1f}s".format(
                    train_step, len(episode_rewards), def_rew, atk_rew,
                    inf_mean, inf_peak, n_benign, recov,
                    mal_v, heal_v, reach,
                    intrusion_rate, cumulative_defense, time.time() - t_start))
            t_start = time.time()
            final_ep_rewards.append(def_rew)
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:-1]))

        if len(episode_rewards) > arglist.num_episodes:
            os.makedirs(arglist.plots_dir, exist_ok=True)
            with open(os.path.join(arglist.plots_dir, arglist.exp_name + "_rewards.pkl"), "wb") as fp:
                pickle.dump(final_ep_rewards, fp)
            with open(os.path.join(arglist.plots_dir, arglist.exp_name + "_agrewards.pkl"), "wb") as fp:
                pickle.dump(final_ep_ag_rewards, fp)
            with open(os.path.join(arglist.plots_dir, arglist.exp_name + "_args.pkl"), "wb") as fp:
                pickle.dump(arglist, fp)
            stats = {
                "intrusion_successes": intrusion_successes,
                "cumulative_defense": cumulative_defense,
                "mean_infected": mean_infected_hist,
                "peak_infected": peak_infected_hist,
                "mean_corruption": mean_corr_hist,
                "recovery_count": recovery_hist,
                "malicious_msg_volume": mal_vol_hist,
                "healing_msg_volume": heal_vol_hist,
                "attacker_reach": reach_hist,
                "joint_rescue_coverage": joint_rescue_hist,
            }
            with open(os.path.join(arglist.plots_dir, arglist.exp_name + "_stats.pkl"), "wb") as fp:
                pickle.dump(stats, fp)

            final_window = min(arglist.log_rate, len(intrusion_successes))
            final_rate = (float(np.mean(intrusion_successes[-final_window:]))
                          if final_window > 0 else 0.0)
            print("...Finished total of {} episodes.".format(len(episode_rewards)))
            print(f"Final intrusion rate: {final_rate:.2%}")
            print(f"Cumulative defense: {cumulative_defense}")
            print(f"Final mean infected: {_window_mean(mean_infected_hist, final_window):.3f} "
                  f"of {n_benign} benign")
            break


if __name__ == "__main__":
    train(parse_args())