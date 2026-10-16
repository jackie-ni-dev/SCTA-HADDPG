

from __future__ import annotations

import torch

from algorithm.common.action_blocks import resolve_action_blocks
from algorithm.modules.threat_encoder import ObsLayout
from algorithm.safety.cost_critic import private_torch_rng
from algorithm.security.k_hop_context import KHopCriticContext
from algorithm.trainer.haddpg_diverse import DiverseHADDPGAgentTrainer
from algorithm.trainer.hatd3_full import FullInnovationHATD3AgentTrainer
from algorithm.trainer.maddpg import MLP, hard_update, resolve_actor_critic_lr

__all__ = ["FullInnovationHADDPGAgentTrainer"]


class FullInnovationHADDPGAgentTrainer(FullInnovationHATD3AgentTrainer,
                                       DiverseHADDPGAgentTrainer):

    def _build_khop(self, cfg) -> None:
        blocks_n = [resolve_action_blocks(pdt, self.act_param_size_n[j])
                    for j, pdt in enumerate(self.act_pdtype_n)]
        layouts = [ObsLayout.infer_ioa(self.obs_size_n[j], self.n, blocks_n[j])
                   for j in range(self.n)]
        self.khop = KHopCriticContext(
            agent_index=self.agent_index, n_agents=self.n, layouts=layouts,
            act_size_n=self.act_size_n, intent_adapter=self.intent_adapter,
            kappa=self.kappa)

        critic_in = self.khop.output_dim
        _actor_lr, critic_lr = resolve_actor_critic_lr(cfg)
        with private_torch_rng(self._khop_seed(cfg), self.device):
            self.critic1 = MLP(critic_in, 1, self.num_units).to(self.device)
            self.target_critic1 = MLP(critic_in, 1, self.num_units).to(self.device)
        hard_update(self.target_critic1, self.critic1)
        self.critic1_optim = torch.optim.Adam(
            self.critic1.parameters(), lr=critic_lr)

        self.critic2 = None
        self.target_critic2 = None
        self.critic2_optim = None

    def describe_full(self) -> str:
        return "[single-critic/HADDPG] " + super().describe_full()
