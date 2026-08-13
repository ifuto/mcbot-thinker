"""PPO training loop with self-play (learner vs. slowly-updated opponent).

The learner controls agent 0 of every match; a separate, polyak-averaged copy
of the same architecture controls agent 1. This is fictitious self-play: the
learner improves against a slightly-behind version of itself, which avoids
policy-collapse while producing skills that generalize to an equal opponent.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from mcbot.sim.env import SimEnv
from mcbot.sim import consts as C
from mcbot.rl.network import ActorCritic


def polyak_update(target, source, tau):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


class PPOTrainer:
    def __init__(self, cfg: "RLConfig", rlcfg=None, build_env=True):
        self.cfg = cfg
        self.nmatches = cfg.nmatches
        self.nagents = cfg.nmatches * C.NAG

        if build_env:
            self.env = SimEnv(cfg.nmatches, seed=cfg.seed)
            self.env.set_rewards(cfg.reward)
        else:
            self.env = None

        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)
        act_space = cfg.action_space()
        self.learner = ActorCritic(C.NOBS, act_space, cfg.hidden).to(self.device)
        self.opponent = ActorCritic(C.NOBS, act_space, cfg.hidden).to(self.device)
        self.opponent.load_state_dict(self.learner.state_dict())
        self.optim = torch.optim.Adam(self.learner.parameters(), lr=cfg.lr)

        self.obs_dim = C.NOBS
        self._buf = None

    # ------------------------------------------------------------------ run
    def _action_batch(self, learner_obs, opponent_obs, greedy=False):
        """Sample actions for both agent-0 (learner) and agent-1 (opponent)."""
        lo = learner_obs.to(self.device)
        oo = opponent_obs.to(self.device)
        with torch.no_grad():
            actL, logpL, valL = self.learner.sample_actions(lo, greedy=greedy)
            actO, _, _ = self.opponent.sample_actions(oo, greedy=greedy)
        full = torch.empty(self.nmatches, C.NAG, C.NACT, dtype=torch.int64)
        full[:, 0, :] = actL.cpu()
        full[:, 1, :] = actO.cpu()
        return full, logpL, valL

    def collect_rollout(self, greedy=False):
        n = self.nmatches
        r = self.cfg.rollout_len
        D = self.obs_dim
        # buffers (learner transitions)
        obs = np.zeros((r, n, D), dtype=np.float32)
        acts = np.zeros((r, n, C.NACT), dtype=np.int64)
        logps = np.zeros((r, n), dtype=np.float32)
        vals = np.zeros((r, n), dtype=np.float32)
        rews = np.zeros((r, n), dtype=np.float32)
        dones = np.zeros((r, n), dtype=np.float32)

        stats = {"dmg_dealt": 0.0, "dmg_taken": 0.0, "wins": 0, "losses": 0, "draws": 0}

        prev = None
        for t in range(r):
            obs_np = self.env.obs.copy()            # (nmatches, 2, D) BEFORE step
            prev = obs_np
            obsL = torch.from_numpy(obs_np[:, 0, :])
            obsO = torch.from_numpy(obs_np[:, 1, :])
            full, logpL, valL = self._action_batch(obsL, obsO, greedy=greedy)

            obs[t] = obs_np[:, 0, :]
            acts[t] = full[:, 0, :].cpu().numpy()
            logps[t] = logpL.cpu().numpy()
            vals[t] = valL.cpu().numpy()

            obs2, rew, done, outcome = self.env.step(full.cpu().numpy(),
                                                     nticks=self.cfg.frame_skip)
            rews[t] = rew[:, 0]
            dones[t] = done

            # damage accounting for logging
            d_done = np.maximum(prev[:, 0, C.O_HP] - obs2[:, 0, C.O_HP], 0).sum()  # taken by learner
            d_dealt = np.maximum(prev[:, 1, C.O_HP] - obs2[:, 1, C.O_HP], 0).sum()  # dealt by learner
            stats["dmg_taken"] += d_done
            stats["dmg_dealt"] += d_dealt
            for m in np.nonzero(done)[0]:
                if outcome[m] == 1: stats["wins"] += 1
                elif outcome[m] == 2: stats["losses"] += 1
                elif outcome[m] == 3: stats["draws"] += 1

        # bootstrap last value
        obs_next = torch.from_numpy(self.env.obs[:, 0, :])
        with torch.no_grad():
            v_next = self.learner.forward_value(obs_next.to(self.device)).cpu().numpy()

        return (obs, acts, logps, vals, rews, dones, v_next), stats

    def _compute_gae(self, data):
        obs, acts, logps, vals, rews, dones, v_next = data
        r, n = vals.shape
        g = self.cfg.gamma
        lam = self.cfg.gae_lambda
        adv = np.zeros((r, n), dtype=np.float32)
        ret = np.zeros((r, n), dtype=np.float32)
        last_adv = np.zeros(n, dtype=np.float32)
        last_v = v_next
        last_done = np.ones(n, dtype=np.float32)
        for t in reversed(range(r)):
            mask = 1 - dones[t]
            delta = rews[t] + g * last_v * (1 - dones[t]) - vals[t]
            last_adv = delta + g * lam * mask * last_adv
            last_v = vals[t]
            adv[t] = last_adv
            ret[t] = adv[t] + vals[t]
        return adv, ret

    def update(self, data):
        obs, acts, logps, vals, rews, dones, v_next = data
        adv, ret = self._compute_gae(data)
        r, n = obs.shape[:2]
        obs_f = obs.reshape(-1, self.obs_dim)
        act_f = acts.reshape(-1, C.NACT)
        logp_f = logps.reshape(-1)
        adv_f = adv.reshape(-1)
        ret_f = ret.reshape(-1)

        if self.cfg.norm_adv:
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        if self.cfg.norm_returns:
            ret_v = (ret_f - ret_f.mean()) / (ret_f.std() + 1e-8)
        else:
            ret_v = ret_f

        idx = np.arange(obs_f.shape[0])
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), self.cfg.minibatch):
                bi = idx[s:s + self.cfg.minibatch]
                ob = torch.from_numpy(obs_f[bi])
                ac = torch.from_numpy(act_f[bi])
                lp = torch.from_numpy(logp_f[bi])
                a = torch.from_numpy(adv_f[bi])
                r = torch.from_numpy(ret_v[bi])
                logp, ent, val = self.learner.evaluate(ob, ac)
                ratio = torch.exp(logp - lp)
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * a
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(val, r)
                entropy = ent.mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.learner.parameters(),
                                               self.cfg.max_grad_norm)
                self.optim.step()

        # self-play: update opponent
        if self.cfg.opp_reset_every and self._iter % self.cfg.opp_reset_every == 0:
            self.opponent.load_state_dict(self.learner.state_dict())
        else:
            polyak_update(self.opponent, self.learner, self.cfg.opp_tau)

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    # ------------------------------------------------------------------ api
    def iterate(self):
        data, stats = self.collect_rollout()
        loss = self.update(data)
        return loss, stats

    def save(self, path):
        torch.save({
            "learner": self.learner.state_dict(),
            "opponent": self.opponent.state_dict(),
            "optim": self.optim.state_dict(),
        }, path)

    def load(self, path, load_optim=True):
        ck = torch.load(path, map_location="cpu")
        self.learner.load_state_dict(ck["learner"])
        self.opponent.load_state_dict(ck["opponent"])
        if load_optim and "optim" in ck:
            self.optim.load_state_dict(ck["optim"])
        return ck
