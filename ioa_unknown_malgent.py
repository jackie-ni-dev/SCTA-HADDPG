

import collections

import numpy as np

from multiagent.core import Agent, Landmark
from multiagent.scenarios.ioa_infection import (
    DIM_C, IoAWorld, MIN_POLE_SEP, Scenario as IoAInfectionScenario,
    ROLE_SYNERGY, GUARD_INTERCEPT_RADIUS, GUARD_INTERCEPT_STRENGTH,
    GUARD_HEAL_RADIUS, HEAL_SYNERGY_BOOST, TRIAGE_W,
    JOINT_COORDINATION, UNGUARDED_HEAL_FACTOR, JOINT_RESCUE_W,
    JOINT_RESCUE_RISK_THRESHOLD)


MAX_MALGENTS = 6           
ACTIVE_MIN = 1             
ACTIVE_MAX = 3             
SLOT_SAMPLING = "random"   
ATTACKER_MODE = "learned"  
PARK_ORIGIN = 3.0          

OOD_ATTACK_STRENGTH = 1.0  
OOD_ATTACK_SURFACE = 1.0   


def _ov(args, attr, default):
    if args is None:
        return default
    v = getattr(args, attr, None)
    return default if v is None else v


class UnknownMalgentWorld(IoAWorld):

    def __init__(self):
        super(UnknownMalgentWorld, self).__init__()
        self.max_malgents = MAX_MALGENTS
        self.active_min = ACTIVE_MIN
        self.active_max = ACTIVE_MAX
        self.slot_sampling = SLOT_SAMPLING
        self.attacker_mode = ATTACKER_MODE
        self.attack_strength = OOD_ATTACK_STRENGTH
        self.attack_surface = OOD_ATTACK_SURFACE
        self.n_active_attackers = 0

    @staticmethod
    def is_active(agent):
        return bool(getattr(agent, "active", True))

    def active_attackers(self):
        return [a for a in self.agents
                if getattr(a, "adversary", False) and self.is_active(a)]

    def step(self):
        if self.attacker_mode == "scripted":
            self._script_attacker_actions()
        self._silence_dormant_actions()
        super(UnknownMalgentWorld, self).step()
        self._pin_dormant()

    def _script_attacker_actions(self):
        benign = [a for a in self.agents if not getattr(a, "adversary", False)]
        for agent in self.agents:
            if not getattr(agent, "adversary", False) or not self.is_active(agent):
                continue
            u = np.zeros(self.dim_p)
            if benign:
                tgt = min(benign, key=lambda b: float(np.sum(np.square(
                    b.state.p_pos - agent.state.p_pos))))
                d = tgt.state.p_pos - agent.state.p_pos
                nrm = float(np.sqrt(np.sum(np.square(d)))) + 1e-8
                u = (d / nrm) * (agent.accel if agent.accel is not None else 5.0)
            agent.action.u = u
            c = np.zeros(self.dim_c)
            c[self.mal_symbol] = 1.0
            agent.action.c = c

    def _silence_dormant_actions(self):
        for agent in self.agents:
            if self.is_active(agent):
                continue
            agent.action.u = np.zeros(self.dim_p)
            agent.action.c = np.zeros(self.dim_c)

    def _pin_dormant(self):
        for agent in self.agents:
            if self.is_active(agent):
                continue
            agent.state.p_pos = np.array(agent.park_pos, dtype=float).copy()
            agent.state.p_vel = np.zeros(self.dim_p)
            agent.state.c = np.zeros(self.dim_c)
            agent.corruption = 1.0

    def apply_action_force(self, p_force):
        p_force = super(UnknownMalgentWorld, self).apply_action_force(p_force)
        for i, agent in enumerate(self.agents):
            if not self.is_active(agent):
                p_force[i] = np.zeros(self.dim_p)
        return p_force

    def _route_messages(self):
        n = len(self.agents)
        self._incoming_mal = np.zeros(n)
        self._incoming_heal = np.zeros(n)
        for j, sender in enumerate(self.agents):
            if not self.is_active(sender):
                continue
            content = np.asarray(sender.state.c, dtype=float)
            mal_level = float(content[self.mal_symbol])
            heal_level = float(content[self.heal_symbol])
            srole = getattr(sender, "role", "defender")
            corr = float(getattr(sender, "corruption", 0.0))
            reach = self.comm_range * (
                self.attack_surface if srole == "attacker" else 1.0)
            for i, receiver in enumerate(self.agents):
                if i == j or not self.is_active(receiver):
                    continue
                dist = float(np.sqrt(np.sum(np.square(
                    sender.state.p_pos - receiver.state.p_pos))))
                if dist >= reach:
                    continue
                if not sender.silent:
                    receiver.msg_buffer.append({
                        "sender": sender.name,
                        "sender_idx": j,
                        "content": content.copy(),
                        "mal_level": mal_level,
                        "dist": dist,
                        "step": self.step_count,
                    })
                if srole == "attacker":
                    self._incoming_mal[i] += (
                        (self.attack_base + mal_level) * self.attack_strength *
                        self._mal_intercept_factor(sender))
                elif srole == "infected":
                    self._incoming_mal[i] += (
                        (self.attack_base + corr) * self.infected_efficacy *
                        self._mal_intercept_factor(sender))
                elif srole == "defender":
                    self._incoming_heal[i] += (
                        heal_level * self._heal_synergy_factor(receiver))


class Scenario(IoAInfectionScenario):

    def attackers(self, world):
        return [a for a in world.agents
                if a.role == "attacker" and UnknownMalgentWorld.is_active(a)]

    def make_world(self, args=None):
        world = UnknownMalgentWorld()
        world.dim_c = DIM_C
        assert world.dim_c >= 2, "dim_c 必须 >=2 以容纳 MAL/HEAL 符号"

        m_max = int(_ov(args, "ioa_max_malgents", MAX_MALGENTS))
        a_min = int(_ov(args, "ioa_active_malgent_min", ACTIVE_MIN))
        a_max = int(_ov(args, "ioa_active_malgent_max", ACTIVE_MAX))
        num_healers = int(_ov(args, "ioa_num_healers", 3))
        num_guardians = int(_ov(args, "ioa_num_guardians", 3))
        if m_max < 1:
            raise ValueError("--ioa-max-malgents 必须 >= 1")
        if not 0 <= a_min <= a_max <= m_max:
            raise ValueError(
                "需要 0 <= --ioa-active-malgent-min({}) <= --ioa-active-malgent-max({}) "
                "<= --ioa-max-malgents({})".format(a_min, a_max, m_max))

        world.max_malgents = m_max
        world.active_min = a_min
        world.active_max = a_max
        world.slot_sampling = str(_ov(args, "ioa_malgent_slot_sampling", SLOT_SAMPLING))
        world.attacker_mode = str(_ov(args, "ioa_attacker_mode", ATTACKER_MODE))
        world.attack_strength = float(_ov(args, "ood_attack_strength", OOD_ATTACK_STRENGTH))
        world.attack_surface = float(_ov(args, "ood_attack_surface", OOD_ATTACK_SURFACE))
        world.role_synergy = bool(_ov(args, "ioa_role_synergy", ROLE_SYNERGY))
        world.joint_coordination = bool(_ov(args, "ioa_joint_coordination", JOINT_COORDINATION))
        if world.joint_coordination:
            world.role_synergy = True
        world.guard_intercept_radius = float(_ov(args, "ioa_guard_intercept_radius", GUARD_INTERCEPT_RADIUS))
        world.guard_intercept_strength = float(_ov(args, "ioa_guard_intercept_strength", GUARD_INTERCEPT_STRENGTH))
        world.guard_heal_radius = float(_ov(args, "ioa_guard_heal_radius", GUARD_HEAL_RADIUS))
        world.heal_synergy_boost = float(_ov(args, "ioa_heal_synergy_boost", HEAL_SYNERGY_BOOST))
        world.triage_w = float(_ov(args, "ioa_triage_w", TRIAGE_W))
        world.unguarded_heal_factor = float(_ov(args, "ioa_unguarded_heal_factor", UNGUARDED_HEAL_FACTOR))
        world.joint_rescue_w = float(_ov(args, "ioa_joint_rescue_w", JOINT_RESCUE_W))
        world.joint_rescue_risk_threshold = float(_ov(args, "ioa_joint_rescue_risk_threshold", JOINT_RESCUE_RISK_THRESHOLD))
        if world.guard_intercept_radius < 0 or world.guard_heal_radius < 0:
            raise ValueError("IoA role-synergy radius 必须 >= 0")
        if not 0.0 <= world.guard_intercept_strength <= 0.95:
            raise ValueError("--ioa-guard-intercept-strength 必须在 [0, 0.95]")
        if world.heal_synergy_boost < 0 or world.triage_w < 0 or world.joint_rescue_w < 0:
            raise ValueError("--ioa-heal-synergy-boost / --ioa-triage-w / --ioa-joint-rescue-w 必须 >= 0")
        if not 0.0 <= world.unguarded_heal_factor <= 1.0:
            raise ValueError("--ioa-unguarded-heal-factor 必须在 [0,1]")
        if not 0.0 <= world.joint_rescue_risk_threshold <= 1.0:
            raise ValueError("--ioa-joint-rescue-risk-threshold 必须在 [0,1]")
        if world.slot_sampling not in ("random", "ordered"):
            raise ValueError("--ioa-malgent-slot-sampling 只能是 random / ordered")
        if world.attacker_mode not in ("learned", "scripted"):
            raise ValueError("--ioa-attacker-mode 只能是 learned / scripted")

        num_benign = num_healers + num_guardians
        num_agents = m_max + num_benign
        if args is not None and getattr(args, "ioa_num_attackers", None) is not None:
            print("  [warn] ioa_unknown_malgent 忽略 --ioa-num-attackers={}; "
                  "请改用 --ioa-max-malgents (当前 {}) 与 "
                  "--ioa-active-malgent-min/max (当前 [{},{}])。".format(
                      args.ioa_num_attackers, m_max, a_min, a_max))
        world.num_agents = num_agents
        world.num_attackers = m_max
        world.num_benign = num_benign
        world.num_healers = num_healers
        world.num_guardians = num_guardians
        world.intrusion_min_infected = max(1, num_benign // 2)

        benign_speeds = np.linspace(1.0, 1.4, num_benign).tolist()
        benign_accels = np.linspace(3.0, 4.5, num_benign).tolist()

        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = "agent_%d" % i
            agent.collide = False
            is_attacker = (i < m_max)
            agent.adversary = is_attacker
            agent.role = "attacker" if is_attacker else "defender"
            agent.size = 0.06 if is_attacker else 0.05
            if is_attacker:
                agent.subtype = "attacker"
                agent.silent = False
                agent.max_speed = 1.3
                agent.accel = 4.0
                agent.slot = i
                agent.park_pos = np.array([PARK_ORIGIN + 0.25 * i, PARK_ORIGIN])
                agent.active = True
            else:
                k = i - m_max
                agent.max_speed = benign_speeds[k]
                agent.accel = benign_accels[k]
                agent.subtype = "healer" if k < num_healers else "guardian"
                agent.silent = (agent.subtype == "guardian")
                agent.slot = -1
                agent.park_pos = None
                agent.active = True
            agent.corruption = 0.0
            agent.msg_buffer = collections.deque(maxlen=world.buffer_size)

        world.landmarks = [Landmark() for _ in range(2)]
        for lm in world.landmarks:
            lm.collide = False
            lm.movable = False
            lm.size = 0.08
        world.landmarks[0].name = "consensus_true"
        world.landmarks[1].name = "consensus_mal"

        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.step_count = 0
        world.intrusion_succeeded = False
        world._intrusion_run = 0
        world.n_infected = 0
        world.n_clean = world.num_benign

        world.l_true = world.landmarks[0]
        world.l_mal = world.landmarks[1]
        world.l_true.color = np.array([0.15, 0.70, 0.20])
        world.l_mal.color = np.array([0.55, 0.10, 0.10])
        world.l_true.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
        cand = np.random.uniform(-0.9, +0.9, world.dim_p)
        for _ in range(100):
            if np.sqrt(np.sum(np.square(cand - world.l_true.state.p_pos))) >= MIN_POLE_SEP:
                break
            cand = np.random.uniform(-0.9, +0.9, world.dim_p)
        world.l_mal.state.p_pos = cand
        for lm in world.landmarks:
            lm.state.p_vel = np.zeros(world.dim_p)

        m_max = world.max_malgents
        n_active = int(np.random.randint(world.active_min, world.active_max + 1))
        if world.slot_sampling == "random":
            active_slots = set(int(s) for s in
                               np.random.choice(m_max, size=n_active, replace=False))
        else:
            active_slots = set(range(n_active))
        world.n_active_attackers = n_active

        for i, agent in enumerate(world.agents):
            agent.role = "attacker" if agent.adversary else "defender"
            agent.corruption = 1.0 if agent.adversary else 0.0
            agent.color = UnknownMalgentWorld.color_for(agent.role)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.action.c = np.zeros(world.dim_c)
            agent.action.u = np.zeros(world.dim_p)
            agent.msg_buffer.clear()
            if agent.adversary:
                agent.active = (agent.slot in active_slots)
                if agent.active:
                    agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                else:
                    agent.state.p_pos = np.array(agent.park_pos, dtype=float).copy()
                    agent.color = np.array([0.75, 0.75, 0.75])   
            else:
                agent.active = True
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)

    def reward(self, agent, world):
        if getattr(agent, "adversary", False) and not UnknownMalgentWorld.is_active(agent):
            return 0.0
        return super(Scenario, self).reward(agent, world)

    def observation(self, agent, world):
        dim_c = world.dim_c
        language_aware = (not agent.silent)
        n_other = len(world.agents) - 1
        e_dim = (3 + dim_c) if language_aware else 3

        if not UnknownMalgentWorld.is_active(agent):
            tail = (2 * dim_c + 2) if language_aware else 0
            return np.zeros(12 + n_other * e_dim + n_other + tail)

        role_onehot = np.array([
            1.0 if agent.role == "attacker" else 0.0,
            1.0 if agent.role == "defender" else 0.0,
            1.0 if agent.role == "infected" else 0.0,
        ])
        context = np.concatenate([
            agent.state.p_vel,                    
            agent.state.p_pos,                    
            role_onehot,                          
            [float(agent.corruption)],            
            *[lm.state.p_pos - agent.state.p_pos for lm in world.landmarks],  
        ])

        entities = np.zeros((n_other, e_dim))
        mask = np.zeros(n_other)
        k = 0
        for other in world.agents:
            if other is agent:
                continue
            if UnknownMalgentWorld.is_active(other):
                rel = other.state.p_pos - agent.state.p_pos
                dist = float(np.sqrt(np.sum(np.square(rel))))
                in_range = 1.0 if dist < world.comm_range else 0.0
                if language_aware:
                    entities[k] = np.concatenate(
                        [rel, np.asarray(other.state.c, dtype=float), [in_range]])
                else:
                    entities[k] = np.concatenate([rel, [in_range]])
                mask[k] = 1.0
            k += 1

        blocks = [context, entities.reshape(-1), mask]
        if language_aware:
            blocks.append(self._buffer_summary(agent, world))
        return np.concatenate(blocks)

    def benchmark_data(self, agent, world):
        base = super(Scenario, self).benchmark_data(agent, world)
        return tuple(base) + (int(world.n_active_attackers),)


