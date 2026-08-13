"""Population-based hybrid training: scripted base + learned override, trained
against a DIVERSE pool of opponents (anti-overfitting).

Design
------
- The learner is a HybridActorCritic: it outputs its own actions PLUS a
  `trust_scripted` gate. When trusting, the scripted swordsman's action is used;
  otherwise the policy's own action. Guarantees competent base behavior.
- Opponent pool (sample one per episode): scripted:aggro, scripted:strafer,
  random, self-play EMA copy of the learner, and the hybrid itself. Training
  against all of these prevents overfitting to any single opponent.

Usage
-----
    python -m mcbot.rl.train_hybrid --iterations 0 --nmatches 512 --rollout 128 \
        --save-every 100 --log-every 20 --eval-every 50 --max-minutes 340
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from config import RLConfig
from mcbot.sim.env import SimEnv
from mcbot.sim import consts as C
from mcbot.rl.hybrid_network import HybridActorCritic, NACT_TOTAL, TRUST_SLOT
from mcbot.bots import ScriptedBot
from mcbot.rl.train_io import Log


class HybridTrainer:
    def __init__(self, cfg: RLConfig, opp_tau=0.01):
        self.cfg = cfg
        self.nmatches = cfg.nmatches
        self.env = SimEnv(cfg.nmatches, seed=cfg.seed)
        self.env.set_rewards(cfg.reward)
        self.learner = HybridActorCritic(C.NOBS, cfg.action_space(), cfg.hidden)
        self.ema = HybridActorCritic(C.NOBS, cfg.action_space(), cfg.hidden)
        self.ema.load_state_dict(self.learner.state_dict())
        self.opp_tau = opp_tau
        self.optim = torch.optim.Adam(self.learner.parameters(), lr=cfg.lr)
        self._rng = np.random.default_rng(cfg.seed)
        self._it = 0
        self._bots = {st: ScriptedBot(st) for st in
                      ["aggro", "strafer", "critter", "kiter", "mixer"]}
        self._pool_keys = ["aggro", "strafer", "critter", "kiter", "mixer",
                           "random", "self", "hybrid"]
        self._init_opp_winrates()

    def _opp_act(self, obs_np_side, opponent):
        """Get opponent action (B,5) for agent-1 from obs[:,1,:]."""
        B = obs_np_side.shape[0]
        if isinstance(opponent, str):
            if opponent == "random":
                a = np.zeros((B, C.NACT), dtype=np.int64)
                a[:, C.A_MOVE] = self._rng.integers(0, C.NMOVE, size=B)
                a[:, C.A_ATTACK] = self._rng.integers(0, 2, size=B)
                return a
            return self._bots[opponent].act(obs_np_side)
        # a policy (EMA copy or hybrid self): need (B,5) from its sample_actions
        with torch.no_grad():
            acts, _, _ = opponent.sample_actions(torch.from_numpy(obs_np_side))
        return acts[:, :C.NACT].numpy()

    # ---- PFSP: prioritized fictitious self-play opponent sampling -----------
    # Track win-rate vs each opponent and sample the HARDEST (lowest win-rate)
    # opponents most often. From AlphaStar / Minimax-Exploiter: focus training
    # on the opponents you currently lose to, instead of uniform sampling.
    def _init_opp_winrates(self):
        self._opp_wr = {k: (0.9 if k == "random" else 0.5) for k in self._pool_keys}
        self._opp_games = {k: 0 for k in self._pool_keys}

    def _sample_opponent(self, it):
        """PFSP: sample opponent ∝ loss-rate (1 - win_rate), softened with a
        small uniform floor so no opponent is ever starved out."""
        base = {k: (1.0 - wr) for k, wr in self._opp_wr.items()}
        # add a small floor so every opponent keeps getting some games
        for k in base:
            base[k] += 0.1
        tot = sum(base.values())
        keys = list(base.keys())
        p = [base[k] / tot for k in keys]
        return self._rng.choice(keys, p=p)

    def _update_opp_winrate(self, key, outcome):
        """outcome: 1 = learner won, 2 = learner lost, 3 = draw."""
        if outcome == 1:
            self._opp_wr[key] = self._opp_wr[key] * 0.9 + 1.0 * 0.1
        elif outcome == 2:
            self._opp_wr[key] = self._opp_wr[key] * 0.9 + 0.0 * 0.1
        # draws leave the estimate roughly unchanged

    # ------------------------------------------------------------------ run
    def collect_rollout(self):
        r = self.cfg.rollout_len
        n = self.nmatches
        obs = np.zeros((r, n, C.NOBS), dtype=np.float32)
        acts = np.zeros((r, n, NACT_TOTAL), dtype=np.int64)   # includes trust
        logps = np.zeros((r, n), dtype=np.float32)
        vals = np.zeros((r, n), dtype=np.float32)
        rews = np.zeros((r, n), dtype=np.float32)
        dones = np.zeros((r, n), dtype=np.float32)
        stats = {"dmg_dealt": 0.0, "dmg_taken": 0.0, "wins": 0, "losses": 0, "draws": 0,
                 "trust_rate": 0.0}

        for t in range(r):
            obs_np = self.env.obs.copy()
            with torch.no_grad():
                a6, lp, v = self.learner.sample_actions(torch.from_numpy(obs_np[:, 0, :]))
            a6 = a6.numpy()
            trust = a6[:, TRUST_SLOT]
            base = a6[:, :C.NACT]
            # scripted recommendation for agent-0
            scripted = self._bots["aggro"].act(obs_np[:, 0, :])
            final_a0 = np.where(trust[:, None] == 1, scripted, base)
            full_a0 = np.concatenate([final_a0, trust[:, None]], axis=1)  # (n,6) for storage

            # opponent (sample one per rollout step) via curriculum schedule
            opp_key = self._sample_opponent(self._it)
            if opp_key == "self":
                opp_act = self._opp_act(obs_np[:, 1, :], self.ema)
            elif opp_key == "hybrid":
                with torch.no_grad():
                    o6, _, _ = self.learner.sample_actions(torch.from_numpy(obs_np[:, 1, :]))
                obase = o6[:, :C.NACT].numpy()
                oscript = self._bots["aggro"].act(obs_np[:, 1, :])
                otrust = o6[:, TRUST_SLOT].numpy()
                opp_act = np.where(otrust[:, None] == 1, oscript, obase)
            else:
                opp_act = self._opp_act(obs_np[:, 1, :], opp_key)

            full = np.zeros((n, C.NAG, C.NACT), dtype=np.int64)
            full[:, 0, :] = final_a0
            full[:, 1, :] = opp_act

            obs[t] = obs_np[:, 0, :]
            acts[t] = full_a0
            logps[t] = lp.numpy()
            vals[t] = v.numpy()

            obs2, rew, done, outcome = self.env.step(full, 1)
            rews[t] = rew[:, 0]
            dones[t] = done
            stats["dmg_dealt"] += float(np.maximum(obs_np[:, 1, C.O_HP] - obs2[:, 1, C.O_HP], 0).sum())
            stats["dmg_taken"] += float(np.maximum(obs_np[:, 0, C.O_HP] - obs2[:, 0, C.O_HP], 0).sum())
            stats["trust_rate"] += float((trust == 1).mean())
            for m in np.nonzero(done)[0]:
                if outcome[m] == 1: stats["wins"] += 1
                elif outcome[m] == 2: stats["losses"] += 1
                else: stats["draws"] += 1
                self._update_opp_winrate(opp_key, int(outcome[m]))

        with torch.no_grad():
            v_next = self.learner.forward_value(torch.from_numpy(self.env.obs[:, 0, :])).numpy()
        stats["trust_rate"] /= max(1, r)
        return (obs, acts, logps, vals, rews, dones, v_next), stats

    def update(self, data, entropy_coef):
        obs, acts, logps, vals, rews, dones, v_next = data
        r, n = obs.shape[:2]
        g, lam = self.cfg.gamma, self.cfg.gae_lambda

        adv = np.zeros((r, n), dtype=np.float32)
        last_adv = np.zeros(n, dtype=np.float32)
        last_v = v_next
        for t in reversed(range(r)):
            delta = rews[t] + g * last_v * (1 - dones[t]) - vals[t]
            last_adv = delta + g * lam * (1 - dones[t]) * last_adv
            last_v = vals[t]
            adv[t] = last_adv
        ret = adv + vals

        obs_f = obs.reshape(-1, C.NOBS)
        act_f = acts.reshape(-1, NACT_TOTAL)
        logp_f = logps.reshape(-1)
        adv_f = adv.reshape(-1)
        ret_f = ret.reshape(-1)
        if self.cfg.norm_adv:
            adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        ret_v = (ret_f - ret_f.mean()) / (ret_f.std() + 1e-8)

        idx = np.arange(obs_f.shape[0])
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), self.cfg.minibatch):
                bi = idx[s:s + self.cfg.minibatch]
                logp, ent, val = self.learner.evaluate(
                    torch.from_numpy(obs_f[bi]), torch.from_numpy(act_f[bi]))
                ratio = torch.exp(logp - torch.from_numpy(logp_f[bi]))
                a = torch.from_numpy(adv_f[bi])
                surr1 = ratio * a
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * a
                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = F.mse_loss(val, torch.from_numpy(ret_v[bi]))
                loss = p_loss + self.cfg.value_coef * v_loss - entropy_coef * ent.mean()
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.learner.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

        # polyak update EMA opponent
        with torch.no_grad():
            for tp, sp in zip(self.ema.parameters(), self.learner.parameters()):
                tp.mul_(1 - self.opp_tau).add_(sp, alpha=self.opp_tau)
        return {"policy_loss": float(p_loss.item()), "value_loss": float(v_loss.item()),
                "entropy": float(ent.mean().item())}

    def save(self, path):
        torch.save({"learner": self.learner.state_dict(),
                    "ema": self.ema.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        self.learner.load_state_dict(ck["learner"])
        if "ema" in ck:
            self.ema.load_state_dict(ck["ema"])


def eval_vs_population(trainer, nmatches=48, ticks=2400, seed=7):
    """Evaluate learner (hybrid composed) against each pool opponent."""
    env = SimEnv(nmatches, seed=seed)
    idle = np.zeros((nmatches, C.NAG, C.NACT), dtype=np.int64)
    idle[:, :, C.A_MOVE] = C.M_NONE
    env.step(idle, 1)
    rng = np.random.default_rng(seed)
    results = {}
    for key in ["aggro", "strafer", "critter", "kiter", "mixer", "random"]:
        wins = losses = 0
        for _ in range(ticks):
            obs = env.obs.copy()
            with torch.no_grad():
                a6, _, _ = trainer.learner.sample_actions(torch.from_numpy(obs[:, 0, :]))
            base = a6[:, :C.NACT].numpy()
            trust = a6[:, TRUST_SLOT].numpy()
            scripted = trainer._bots["aggro"].act(obs[:, 0, :])
            a0 = np.where(trust[:, None] == 1, scripted, base)
            if key == "random":
                a1 = np.zeros((nmatches, C.NACT), dtype=np.int64)
                a1[:, C.A_MOVE] = rng.integers(0, C.NMOVE, size=nmatches)
                a1[:, C.A_ATTACK] = rng.integers(0, 2, size=nmatches)
            else:
                a1 = trainer._bots[key].act(obs[:, 1, :])
            full = np.zeros((nmatches, C.NAG, C.NACT), dtype=np.int64)
            full[:, 0, :] = a0
            full[:, 1, :] = a1
            _, _, done, outcome = env.step(full, 1)
            for m in np.nonzero(done)[0]:
                if outcome[m] == 1: wins += 1
                elif outcome[m] == 2: losses += 1
        results[key] = wins / max(1, wins + losses)
    env.close()
    return results


def main():
    ap = argparse.ArgumentParser(description="Population-based hybrid self-play training")
    ap.add_argument("--nmatches", type=int, default=512)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--run", default="sword1v1")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-matches", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--entropy", type=float)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    cfg = RLConfig(nmatches=args.nmatches)
    if args.lr: cfg.lr = args.lr
    if args.entropy: cfg.entropy_coef = args.entropy
    if args.seed: cfg.seed = args.seed
    cfg.rollout_len = args.rollout

    os.makedirs(args.ckpt_dir, exist_ok=True)
    trainer = HybridTrainer(cfg)

    def entropy_schedule(it):
        if it < cfg.entropy_decay_iters:
            return cfg.entropy_coef + (cfg.entropy_end - cfg.entropy_coef) * (it / cfg.entropy_decay_iters)
        return cfg.entropy_end

    print(f"[mcbot] hybrid PPO (scripted base + learned override) vs opponent pool "
          f"{['aggro','strafer','random','self','hybrid']} | {cfg.nmatches} matches", flush=True)
    it = 0
    total = args.iterations or float("inf")
    t_start = time.time()
    try:
        while it < total:
            ent = entropy_schedule(it)
            data, stats = trainer.collect_rollout()
            loss = trainer.update(data, ent)
            it += 1
            trainer._it = it
            win_rate = stats["wins"] / max(1, stats["wins"] + stats["losses"])
            if it % cfg.log_every == 0 or it == 1:
                print(f"[mcbot] it {it:4d} | {time.time()-t_start:.0f}s | win {win_rate:.2f} "
                      f"| W/L/D {stats['wins']}/{stats['losses']}/{stats['draws']} "
                      f"| dmg {stats['dmg_dealt']:.0f}/{stats['dmg_taken']:.0f} "
                      f"| trust {stats['trust_rate']:.2f} | ent {ent:.3f}", flush=True)
            if it % cfg.save_every == 0:
                trainer.save(os.path.join(args.ckpt_dir, f"{args.run}_it{it}.pt"))
            if args.eval_every and it % args.eval_every == 0:
                res = eval_vs_population(trainer, nmatches=args.eval_matches)
                avg = sum(res.values()) / len(res)
                print(f"[mcbot] eval @it{it}: " + " ".join(f"{k}={v:.2f}" for k, v in res.items())
                      + f" | avg {avg:.2f}", flush=True)
                trainer.save(os.path.join(args.ckpt_dir, f"{args.run}_eval.pt"))
            if args.max_minutes and (time.time() - t_start) / 60.0 >= args.max_minutes:
                print(f"\n[mcbot] reached --max-minutes={args.max_minutes}; saving final", flush=True)
                trainer.save(os.path.join(args.ckpt_dir, f"{args.run}_it{it}.pt"))
                break
    except KeyboardInterrupt:
        print("\n[mcbot] interrupted; saving", flush=True)
        trainer.save(os.path.join(args.ckpt_dir, f"{args.run}_it{it}.pt"))


if __name__ == "__main__":
    main()
