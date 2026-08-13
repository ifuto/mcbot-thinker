"""Behavior-cloning warm start.

Bootstrap the RL policy to a basic swordsman (approach + sprint + swing on full
charge) by imitating a scripted opponent under self-play. This guarantees the
agent fights immediately, so PPO refines tactics instead of spending thousands
of iterations rediscovering "walk toward the enemy".

Usage:
    python -m mcbot.rl.warmstart --iter 300 --run sword1v1 --ckpt-dir checkpoints
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


def collect(cfg, nsteps, style="aggro"):
    """Run scripted self-play and return (obs, actions) for agent-0 side."""
    env = SimEnv(cfg.nmatches, seed=cfg.seed)
    env.set_rewards(cfg.reward)
    bot = ScriptedBot(style=style)
    rng = np.random.default_rng(cfg.seed)
    obs_buf = np.zeros((nsteps, C.NOBS), dtype=np.float32)
    act_buf = np.zeros((nsteps, C.NACT), dtype=np.int64)
    filled = 0
    idle = np.zeros((cfg.nmatches, C.NAG, C.NACT), dtype=np.int64)
    idle[:, :, C.A_MOVE] = C.M_NONE
    env.step(idle, 1)
    while filled < nsteps:
        obs = env.obs.copy()
        a0 = bot.act(obs[:, 0, :], rng)
        a1 = bot.act(obs[:, 1, :], rng)
        full = np.zeros((cfg.nmatches, C.NAG, C.NACT), dtype=np.int64)
        full[:, 0, :] = a0
        full[:, 1, :] = a1
        n = min(nsteps - filled, cfg.nmatches)
        obs_buf[filled:filled + n] = obs[:n, 0, :]
        act_buf[filled:filled + n] = a0[:n]
        filled += n
        env.step(full, 1)
    env.close()
    return obs_buf, act_buf


def bc_train(cfg, obs, acts, epochs=3, lr=1e-3, minibatch=2048):
    """Behavior-cloning with class-weighting so minority actions (attack) are
    actually learned instead of being collapsed away by greedy argmax."""
    net = ActorCritic(C.NOBS, cfg.action_space(), cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    N = obs.shape[0]

    # per-head class weights (attack=1 is the minority -> upweight it)
    nclasses = [C.NMOVE, 2, 2, 2, 2]
    weights = []
    for i in range(C.NACT):
        nc = nclasses[i]
        if i == C.A_ATTACK:
            counts = np.bincount(acts[:, i], minlength=2)[:2].astype(np.float32)
            # balance: rare attack class gets w = n0/max(1,n1), capped
            w1 = min(10.0, max(0.0, counts[0]) / max(1.0, counts[1]))
            weights.append(torch.tensor([1.0, w1]))
        else:
            weights.append(torch.ones(nc))

    idx = np.arange(N)
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot = 0.0
        nb = 0
        for s in range(0, N, minibatch):
            bi = idx[s:s + minibatch]
            ob = torch.from_numpy(obs[bi])
            ac = torch.from_numpy(acts[bi])
            logits, _ = net.get_logits(ob)
            loss = 0.0
            for i, lg in enumerate(logits):
                w = weights[i].to(lg.device)
                loss = loss + F.cross_entropy(lg, ac[:, i], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item())
            nb += 1
        print(f"[warmstart] epoch {ep+1}/{epochs} bc_loss {tot/max(1,nb):.3f} "
              f"(attack_weight={weights[C.A_ATTACK][1].item():.1f})")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, default=300, help="steps of scripted data")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--run", default="sword1v1")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--nmatches", type=int, default=512)
    ap.add_argument("--style", default="aggro")
    args = ap.parse_args()

    cfg = RLConfig(nmatches=args.nmatches)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    obs, acts = collect(cfg, args.iter, style=args.style)
    print(f"[warmstart] collected {obs.shape[0]} transitions")
    net = bc_train(cfg, obs, acts, epochs=args.epochs)
    path = os.path.join(args.ckpt_dir, f"{args.run}_warm.pt")
    torch.save({"learner": net.state_dict(), "opponent": net.state_dict()}, path)
    print(f"[warmstart] saved init -> {path}")


if __name__ == "__main__":
    main()
