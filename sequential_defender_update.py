

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from algorithm.trainer.hatd3 import HATD3AgentTrainer
from algorithm.trainer.hatd3_diverse import DiverseHATD3AgentTrainer
from algorithm.trainer.maddpg import to_tensor


def make_defender_updater(trainers, num_adversaries: int, rng=None, verbose: bool = True):
    coordinated = [
        ag for ag in trainers[int(num_adversaries):]
        if isinstance(ag, DiverseHATD3AgentTrainer)
    ]
    if not coordinated:
        return None, []

    if verbose:
        stray = [ag for ag in trainers[: int(num_adversaries)]
                 if isinstance(ag, DiverseHATD3AgentTrainer)]
        if stray:
            print("  [warn] hatd3_diverse is a defender-side trainer; {} attacker(s) "
                  "using it fall back to the plain HATD3 update (no Mutual-PDM term)."
                  .format(len(stray)))

    return DefenderSequentialUpdater(num_adversaries, rng=rng), coordinated


class DefenderSequentialUpdater:

    def __init__(self, num_adversaries: int, rng=None):
        self.num_adversaries = int(num_adversaries)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_info = {}

    @staticmethod
    def _diverse_defenders(agents, num_adversaries):
        return [
            ag
            for ag in agents[num_adversaries:]
            if isinstance(ag, DiverseHATD3AgentTrainer)
        ]

    @staticmethod
    def _actor_order(defenders, rng):
        defender_ids = [int(ag.agent_index) for ag in defenders]
        if len(defender_ids) <= 1:
            return defender_ids
        return [int(x) for x in rng.permutation(defender_ids)]

    @staticmethod
    def _ready(defenders, train_step: int) -> bool:
        if not defenders:
            return False
        ref = defenders[0]
        if len(ref.replay_buffer) < ref.min_replay_buffer_len:
            return False
        return train_step % ref.args.update_rate == 0

    @staticmethod
    def _build_shared_batch(agents, index, device) -> Dict[str, List[torch.Tensor]]:
        batch_size = len(index)
        obs_n, obs_next_n, act_n, rew_n, done_n = [], [], [], [], []
        for ag in agents:
            obs, act, rew, obs_next, done = ag.replay_buffer.sample_index(index)
            obs_n.append(to_tensor(obs, device).reshape(batch_size, -1))
            obs_next_n.append(to_tensor(obs_next, device).reshape(batch_size, -1))
            act_n.append(to_tensor(act, device))
            rew_n.append(to_tensor(rew, device))
            done_n.append(to_tensor(done, device))
        return {
            "obs_n": obs_n,
            "obs_next_n": obs_next_n,
            "act_n": act_n,
            "rew_n": rew_n,
            "done_n": done_n,
            "index": index,
        }

    @staticmethod
    def _current_action(ag, obs_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return HATD3AgentTrainer._peer_current_action(ag, obs_t).detach()

    def update(self, agents, train_step: int):
        defenders = self._diverse_defenders(agents, self.num_adversaries)
        if not self._ready(defenders, train_step):
            return None

        ref = defenders[0]
        index = ref.replay_buffer.generate_sample_indices(ref.args.batch_size)
        for ag in defenders:
            ag.replay_sample_index = index
        batch = self._build_shared_batch(agents, index, ref.device)

        critic_info = {}
        for ag in defenders:
            critic_info[ag.agent_index] = ag.critic_update_from_batch(batch, agents)

        actor_info = {}
        period = getattr(ref, "policy_update_every", None)
        if period is None:
            period = ref.args.update_rate * ref.args.policy_update_rate
        actor_due = train_step % int(period) == 0
        order = []
        if actor_due:
            action_cache = [
                self._current_action(ag, batch["obs_n"][i]) for i, ag in enumerate(agents)
            ]
            order = self._actor_order(defenders, self.rng)
            by_id = {ag.agent_index: ag for ag in defenders}

            preceding = None
            updated_prefix = set()
            for agent_id in order:
                ag = by_id[agent_id]
                ag._harl_prefix_updated_indices = frozenset(updated_prefix)
                actor_info[agent_id] = ag.actor_update_from_batch(
                    batch,
                    action_cache,
                    preceding_agent=preceding,
                )
                action_cache[agent_id] = self._current_action(
                    ag, batch["obs_n"][agent_id]
                )
                updated_prefix.add(int(agent_id))
                preceding = ag

            for ag in defenders:
                ag.soft_update_targets()

        self.last_info = {
            "shared_index": index,
            "defender_order": order,
            "critic": critic_info,
            "actor": actor_info,
        }
        return self.last_info
