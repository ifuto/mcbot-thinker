"""Tiny feed-forward actor-critic for the PvP agent.

Multi-discrete action heads (move, sprint, sneak, jump, attack). Small enough
to run many forward passes per second on a 2-core CPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mcbot.sim import consts as C


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=C.NOBS, action_space=None, hidden=(64, 64),
                 no_bn=True):
        super().__init__()
        action_space = action_space or [C.NMOVE, 2, 2, 2, 2]
        self.action_space = list(action_space)

        layers = []
        dims = [obs_dim] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        self.trunk_dim = dims[-1]

        # actor heads
        self.heads = nn.ModuleList(
            [nn.Linear(self.trunk_dim, n) for n in self.action_space]
        )
        # critic
        self.critic = nn.Linear(self.trunk_dim, 1)

    def _forward(self, obs):
        return self.trunk(obs)

    def get_logits(self, obs):
        feat = self._forward(obs)
        return [head(feat) for head in self.heads], feat

    def forward_value(self, obs):
        feat = self._forward(obs)
        return self.critic(feat).squeeze(-1)

    def dist_entropy(self, logits):
        ent = 0.0
        for lg in logits:
            ent = ent + torch.distributions.Categorical(
                logits=lg).entropy().mean()
        return ent

    def sample_actions(self, obs, greedy=False):
        """Returns (actions (B,5) int64, logprob (B,), value (B,))."""
        logits, feat = self.get_logits(obs)
        acts = torch.empty(obs.shape[0], len(logits), dtype=torch.int64)
        logps = torch.zeros(obs.shape[0], dtype=torch.float32)
        for i, lg in enumerate(logits):
            d = torch.distributions.Categorical(logits=lg)
            a = d.probs.argmax(-1) if greedy else d.sample()
            acts[:, i] = a
            logps += d.log_prob(a)
        val = self.critic(feat).squeeze(-1)
        return acts, logps, val

    def evaluate(self, obs, acts):
        """Given obs (B,NOBS) and actions (B,5), return (logp, entropy, value)."""
        logits, feat = self.get_logits(obs)
        logps = torch.zeros(obs.shape[0], dtype=torch.float32)
        ents = 0.0
        for i, lg in enumerate(logits):
            d = torch.distributions.Categorical(logits=lg)
            logps += d.log_prob(acts[:, i])
            ents += d.entropy()
        val = self.critic(feat).squeeze(-1)
        return logps, ents, val


def count_params(model):
    return sum(p.numel() for p in model.parameters())
