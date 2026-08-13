"""PPO training against a fixed, strong scripted swordsman.

Why this instead of self-play or BC:
  - Self-play oscillates: the EMA opponent adapts, so win rate swings and the
    policy can collapse below the warm-start.
  - BC of a timing-dependent attacker collapses into spam-attack / never-attack
    (cross-entropy can't capture cooldown-gated swings).

Fighting a FIXED strong opponent with reward shaping (damage dealt, win) lets
the policy learn attack timing directly and monotonically improve toward
beating that opponent. The learner is agent-0; agent-1 is a ScriptedBot.

Usage:
    python -m mcbot.rl.train_vs_scripted --iterations 0 --nmatches 512 \
        --rollout 128 --eval-every 50 --save-every 100 --max-minutes 340
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
from mcbot.rl.network import ActorCritic
from mcbot.bots import ScriptedBot
from mcbot.rl.train_io import Log


class ScriptedPPOTrainer:
    def __init__(self, cfg: RLConfig, style="aggro"):
        self.cfg = cfg
        self.nmatches = cfg.nmatches
        self.env = SimEnv(cfg.nmatches, seed=cfg.seed)
        self.env.set_rewards(cfg.reward)
        self.bot = ScriptedBot(style=style)
        self.learner = ActorCritic(C.NOBS, cfg.action_space(), cfg.hidden)
        self.optim = torch.optim.Adam(self.learner.parameters(), lr=cfg.lr)

    # ------------------------------------------------------------------ run
    def collect_rollout(self):
        r = self.cfg.rollout_len
        n = self.nmatches
        obs = np.zeros((r, n, C.NOBS), dtype=np.float32)
        acts = np.zeros((r, n, C.NACT), dtype=np.int64)
        logps = np.zeros((r, n), dtype=np.float32)
        vals = np.zeros((r, n), dtype=np.float32)
        rews = np.zeros((r, n), dtype=np.float32)
        dones = np.zeros((r, n), dtype=np.float32)
        stats = {"dmg_dealt": 0.0, "dmg_taken": 0.0, "wins": 0, "losses": 0, "draws": 0}

        for t in range(r):
            obs_np = self.env.obs.copy()
            with torch.no_grad():
                a0, lp0, v0 = self.learner.sample_actions(
                    torch.from_numpy(obs_np[:, 0, :]))
            a0 = a0.numpy()
            a1 = self.bot.act(obs_np[:, 1, :])
            full = np.zeros((n, C.NAG, C.NACT), dtype=np.int64)
            full[:, 0, :] = a0
            full[:, 1, :] = a1

            obs[t] = obs_np[:, 0, :]
            acts[t] = a0
            logps[t] = lp0.numpy()
            vals[t] = v0.numpy()

            obs2, rew, done, outcome = self.env.step(full, 1)
            rews[t] = rew[:, 0]
            dones[t] = done
            stats["dmg_dealt"] += float(np.maximum(obs_np[:, 1, C.O_HP] - obs2[:, 1, C.O_HP], 0).sum())
            stats["dmg_taken"] += float(np.maximum(obs_np[:, 0, C.O_HP] - obs2[:, 0, C.O_HP], 0).sum())
            for m in np.nonzero(done)[0]:
                if outcome[m] == 1: stats["wins"] += 1
                elif outcome[m] == 2: stats["losses"] += 1
                else: stats["draws"] += 1

        with torch.no_grad():
            v_next = self.learner.forward_value(
                torch.from_numpy(self.env.obs[:, 0, :])).numpy()
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
        act_f = acts.reshape(-1, C.NACT)
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
        return {"policy_loss": float(p_loss.item()), "value_loss": float(v_loss.item()),
                "entropy": float(ent.mean().item())}

    def save(self, path):
        torch.save({"learner": self.learner.state_dict(),
                    "opponent": self.learner.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        self.learner.load_state_dict(ck["learner"])


def eval_vs_scripted(trainer, nmatches=48, ticks=3000, seed=7):
    env = SimEnv(nmatches, seed=seed)
    idle = np.zeros((nmatches, C.NAG, C.NACT), dtype=np.int64)
    idle[:, :, C.A_MOVE] = C.M_NONE
    env.step(idle, 1)
    wins = losses = 0
    for _ in range(ticks):
        obs = env.obs.copy()
        with torch.no_grad():
            a0, _, _ = trainer.learner.sample_actions(torch.from_numpy(obs[:, 0, :]))
        a0 = a0.numpy()
        a1 = trainer.bot.act(obs[:, 1, :])
        full = np.zeros((nmatches, C.NAG, C.NACT), dtype=np.int64)
        full[:, 0, :] = a0
        full[:, 1, :] = a1
        _, _, done, outcome = env.step(full, 1)
        for m in np.nonzero(done)[0]:
            if outcome[m] == 1: wins += 1
            elif outcome[m] == 2: losses += 1
    env.close()
    return wins / max(1, wins + losses)


def main():
    ap = argparse.ArgumentParser(description="PPO vs fixed scripted swordsman")
    ap.add_argument("--nmatches", type=int, default=512)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--iterations", type=int, default=0)
    ap.add_argument("--style", default="aggro")
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
    cfg.run_name = args.run
    cfg.ckpt_dir = args.ckpt_dir
    cfg.rollout_len = args.rollout

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    trainer = ScriptedPPOTrainer(cfg, style=args.style)
    log = Log(cfg.ckpt_dir, cfg.run_name)

    def entropy_schedule(it):
        if it < cfg.entropy_decay_iters:
            return cfg.entropy_coef + (cfg.entropy_end - cfg.entropy_coef) * (it / cfg.entropy_decay_iters)
        return cfg.entropy_end

    print(f"[mcbot] PPO vs scripted:{args.style} | {cfg.nmatches} matches | "
          f"hidden {cfg.hidden} | start entropy {cfg.entropy_coef}", flush=True)
    best_wr = -1.0
    it = 0
    total = args.iterations or float("inf")
    t_start = time.time()
    try:
        while it < total:
            ent = entropy_schedule(it)
            data, stats = trainer.collect_rollout()
            loss = trainer.update(data, ent)
            it += 1
            win_rate = stats["wins"] / max(1, stats["wins"] + stats["losses"])
            if it % cfg.log_every == 0 or it == 1:
                print(f"[mcbot] it {it:4d} | {time.time()-t_start:.0f}s | "
                      f"win {win_rate:.2f} | W/L/D {stats['wins']}/{stats['losses']}/{stats['draws']} "
                      f"| dmg {stats['dmg_dealt']:.0f}/{stats['dmg_taken']:.0f} "
                      f"| ent {ent:.3f}", flush=True)
            if it % cfg.save_every == 0:
                trainer.save(os.path.join(cfg.ckpt_dir, f"{args.run}_it{it}.pt"))
            if args.eval_every and it % args.eval_every == 0:
                wr = eval_vs_scripted(trainer, nmatches=args.eval_matches)
                print(f"[mcbot] eval @it{it}: win-vs-scripted {wr:.3f}", flush=True)
                if wr > best_wr:
                    best_wr = wr
                    trainer.save(os.path.join(cfg.ckpt_dir, f"{args.run}_best.pt"))
                    print(f"[mcbot] new best ({wr:.3f}) -> {args.run}_best.pt", flush=True)
            if args.max_minutes and (time.time() - t_start) / 60.0 >= args.max_minutes:
                print(f"\n[mcbot] reached --max-minutes={args.max_minutes}; saving final", flush=True)
                trainer.save(os.path.join(cfg.ckpt_dir, f"{args.run}_it{it}.pt"))
                break
    except KeyboardInterrupt:
        print("\n[mcbot] interrupted; saving", flush=True)
        trainer.save(os.path.join(cfg.ckpt_dir, f"{args.run}_it{it}.pt"))


if __name__ == "__main__":
    main()
