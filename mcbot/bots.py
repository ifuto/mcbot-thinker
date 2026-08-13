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
        """obs_np (B,NOBS) for a single side. Returns (B,5) int64."""
        rng = rng or self._rng or __import__("numpy").random.default_rng(0)
        B = obs_np.shape[0]
        a = np.zeros((B, C.NACT), dtype=np.int64)
        charge = obs_np[:, C.O_CHARGE]
        dist = obs_np[:, C.O_DIST]
        opp_ground = obs_np[:, C.O_OPP_GROUND]
        in_reach = dist <= 3.0
        # move toward opponent; occasionally strafe (style)
        mv = np.full(B, C.M_FWD, dtype=np.int64)
        if self.style == "strafer":
            r = rng.random(B)
            mv[r < 0.2] = C.M_LFT
            mv[(r >= 0.2) & (r < 0.4)] = C.M_RGT
        a[:, C.A_MOVE] = mv
        a[:, C.A_SPRINT] = 1
        # attack at full charge when in reach (or close)
        ready = charge >= 0.85
        a[:, C.A_ATTACK] = (ready & (in_reach | (dist <= 3.5))).astype(np.int64)
        return a

