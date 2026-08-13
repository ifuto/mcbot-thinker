"""Hybrid actor-critic: scripted swordsman as base, learned policy on top.

The action space is augmented with an extra head `trust_scripted` (0/1). When
the agent chooses `trust_scripted=1`, the final action equals the scripted
bot's recommendation; when `0`, the policy's own movement/attack actions are
used. This guarantees a competent base behavior (scripted) while letting the
policy learn to deviate when it predicts a better outcome.

Action layout (per agent per tick):
  slot 0      move (0..8)   <- policy move (only used when not trusting scripted)
  slot 1      sprint 0/1
  slot 2      sneak  0/1
  slot 3      jump   0/1
  slot 4      attack 0/1
  slot 5      trust_scripted 0/1   (NEW)
NACT_TOTAL = 6

During training/eval the scripted action vector is supplied externally; the
network just outputs `trust_scripted`. The environment itself does NOT know
about the scripted base — the hybrid wrapper composes the final action.
"""
import torch
import torch.nn as nn

from mcbot.sim import consts as C

NACT_TOTAL = C.NACT + 1  # 6
TRUST_SLOT = C.NACT      # 5


class HybridActorCritic(nn.Module):
    def __init__(self, obs_dim=C.NOBS, action_space=None, hidden=(64, 64)):
        super().__init__()
        # base heads: move(9), sprint(2), sneak(2), jump(2), attack(2)
        action_space = action_space or [C.NMOVE, 2, 2, 2, 2]
        self.action_space = list(action_space)

        layers = []
        dims = [obs_dim] + list(hidden)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)
        td = dims[-1]

        self.base_heads = nn.ModuleList([nn.Linear(td, n) for n in action_space])
        self.trust_head = nn.Linear(td, 2)  # trust scripted? 0/1
        self.critic = nn.Linear(td, 1)

    def _feat(self, obs):
        return self.trunk(obs)

    def get_logits(self, obs):
        feat = self._feat(obs)
        base = [h(feat) for h in self.base_heads]
        trust = self.trust_head(feat)
        return base + [trust], feat

    def forward_value(self, obs):
        return self.critic(self._feat(obs)).squeeze(-1)

    def sample_actions(self, obs, greedy=False):
        """Returns (actions (B,6) int64, logprob (B,), value (B,))."""
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
        """acts: (B,6). Returns (logp, entropy, value)."""
        logits, feat = self.get_logits(obs)
        logps = torch.zeros(obs.shape[0], dtype=torch.float32)
        ents = 0.0
        for i, lg in enumerate(logits):
            d = torch.distributions.Categorical(logits=lg)
            logps += d.log_prob(acts[:, i])
            ents += d.entropy()
        val = self.critic(feat).squeeze(-1)
        return logps, ents, val

    @staticmethod
    def compose(base_acts, trust, scripted_acts):
        """Merge policy base actions with scripted base via trust flags.
        base_acts (B,5), trust (B,) 0/1, scripted_acts (B,5) -> (B,5) final."""
        final = base_acts.clone()
        mask = trust.bool()
        final[mask] = scripted_acts[mask]
        return final
