"""Policy wrapper for inference: a Bot is a policy + optional behavior presets."""
import numpy as np
import torch

from mcbot.rl.network import ActorCritic
from mcbot.behavior import presets as P
from mcbot.sim import consts as C


class Bot:
    def __init__(self, policy: ActorCritic, presets=None, name="bot", greedy=False):
        self.policy = policy
        self.presets = list(presets) if presets else []
        self.name = name
        self.greedy = greedy

    def act(self, obs_np):
        """obs_np: (B, NOBS) float32. Returns (B,5) int64 actions."""
        obs = torch.from_numpy(obs_np)
        if self.presets:
            acts, _ = P.sample_presets(self.policy, obs, self.presets,
                                       greedy=self.greedy)
            return acts
        with torch.no_grad():
            acts, _, _ = self.policy.sample_actions(obs, greedy=self.greedy)
        return acts

    @staticmethod
    def load(path, presets=None, name=None, greedy=False, hidden=(64, 64)):
        policy = ActorCritic(C.NOBS, [C.NMOVE, 2, 2, 2, 2], hidden)
        ck = torch.load(path, map_location="cpu")
        policy.load_state_dict(ck["learner"])
        policy.eval()
        return Bot(policy, presets=presets, name=name or path, greedy=greedy)


class ScriptedBot:
    """Deterministic heuristic swordsman: close distance, sprint, swing on full
    charge. `style` adds a bit of variety (strafe/crit) so BC warm-start data
    isn't degenerate. Use it as a fixed opponent or to bootstrap the RL policy."""

    def __init__(self, style="aggro", name="scripted"):
        self.name = name
        self.style = style
        self._rng = None

    def act(self, obs_np, rng=None):
        """obs_np (B,NOBS) for a single side. Returns (B,5) int64.
        styles:
          aggro   - always approach + sprint, attack at full charge
          strafer - approach + random A/D strafing
          critter - jump before attacking for crits (vs sprintless)
          kiter   - back away to out-space, attack opportunistically
          mixer   - random style each tick (unpredictable)
        """
        rng = rng or self._rng or __import__("numpy").random.default_rng(0)
        B = obs_np.shape[0]
        a = np.zeros((B, C.NACT), dtype=np.int64)
        charge = obs_np[:, C.O_CHARGE]
        dist = obs_np[:, C.O_DIST]
        in_reach = dist <= 3.0
        opp_ground = obs_np[:, C.O_OPP_GROUND]
        opp_sprint = obs_np[:, C.O_OPP_SPRINT]

        style = self.style
        if style == "mixer":
            style = rng.choice(["aggro", "strafer", "critter", "kiter"])

        mv = np.full(B, C.M_FWD, dtype=np.int64)   # default forward
        sprint = np.ones(B, dtype=np.int64)
        jump = np.zeros(B, dtype=np.int64)
        attack = np.zeros(B, dtype=np.int64)

        if style == "aggro":
            ready = charge >= 0.85
            attack = (ready & (in_reach | (dist <= 3.5))).astype(np.int64)
        elif style == "strafer":
            r = rng.random(B)
            mv[r < 0.2] = C.M_LFT
            mv[(r >= 0.2) & (r < 0.4)] = C.M_RGT
            ready = charge >= 0.85
            attack = (ready & (in_reach | (dist <= 3.5))).astype(np.int64)
        elif style == "critter":
            # jump right before swinging to land crits; doesn't sprint much
            sprint[:] = 0
            ready = charge >= 0.9
            jump = ready.astype(np.int64)          # start rising
            attack = (ready & in_reach & (obs_np[:, C.O_GROUND] < 0.5)).astype(np.int64)
        elif style == "kiter":
            # back away to keep distance (out-space), attack when opp is close
            mv = np.full(B, C.M_BACK, dtype=np.int64)
            sprint = (opp_sprint > 0.5).astype(np.int64)   # retreat faster if opp sprints
            ready = charge >= 0.9
            attack = (ready & in_reach & (dist >= 1.5)).astype(np.int64)

        a[:, C.A_MOVE] = mv
        a[:, C.A_SPRINT] = sprint
        a[:, C.A_JUMP] = jump
        a[:, C.A_ATTACK] = attack
        return a


class HybridBot:
    """A hybrid swordsman: scripted base + learned override via a trust gate.

    Loads a HybridActorCritic checkpoint. Each tick the network outputs its own
    action plus `trust_scripted`; when trusting, the scripted action is used.
    The scripted base can be `aggro` (default) or `strafer`."""

    def __init__(self, policy, base_style="aggro", name="hybrid", greedy=False):
        from mcbot.rl.hybrid_network import HybridActorCritic, TRUST_SLOT
        self.policy = policy
        self.base_style = base_style
        self.base = ScriptedBot(base_style)
        self.name = name
        self.greedy = greedy
        self.TRUST_SLOT = TRUST_SLOT

    def act(self, obs_np):
        """obs_np (B,NOBS). Returns (B,5) int64."""
        obs = torch.from_numpy(obs_np)
        with torch.no_grad():
            a6, _, _ = self.policy.sample_actions(obs, greedy=self.greedy)
        base = a6[:, :C.NACT].numpy()
        trust = a6[:, self.TRUST_SLOT].numpy()
        scripted = self.base.act(obs_np)
        return np.where(trust[:, None] == 1, scripted, base).astype(np.int64)

    @staticmethod
    def load(path, base_style="aggro", name=None, greedy=False, hidden=(64, 64)):
        from mcbot.rl.hybrid_network import HybridActorCritic
        policy = HybridActorCritic(C.NOBS, [C.NMOVE, 2, 2, 2, 2], hidden)
        ck = torch.load(path, map_location="cpu")
        policy.load_state_dict(ck["learner"])
        policy.eval()
        return HybridBot(policy, base_style=base_style, name=name or path, greedy=greedy)

