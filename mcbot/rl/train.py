"""Main training loop. `python -m mcbot.rl.train [opts]`

Runs PPO self-play on the C simulator and checkpoints the learner. Designed to
be left running for hours/days: it resumes from the latest checkpoint, logs to
console + a CSV, and saves incrementally.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from config import RLConfig, TrainConfig
from mcbot.sim import consts as C
from mcbot.rl.ppo import PPOTrainer
from mcbot.rl import train_io


def eval_vs_random(trainer, nmatches=48, ticks=2400, seed=7):
    """Fixed-baseline evaluation: learner (agent0) vs a random bot (agent1),
    returns win rate. Used to keep the BEST checkpoint (unlike self-play win
    rate vs the adapting EMA opponent, which oscillates)."""
    from mcbot.sim.env import SimEnv
    env = SimEnv(nmatches, seed=seed)
    idle = np.zeros((nmatches, C.NAG, C.NACT), dtype=np.int64)
    idle[:, :, C.A_MOVE] = C.M_NONE
    env.step(idle, 1)
    rng = np.random.default_rng(seed)
    wins = losses = 0
    for _ in range(ticks):
        obs = env.obs.copy()
        with torch.no_grad():
            a0, _, _ = trainer.learner.sample_actions(
                torch.from_numpy(obs[:, 0, :]), greedy=False)
        a0 = a0.numpy()
        a1 = np.zeros((nmatches, C.NACT), dtype=np.int64)
        a1[:, C.A_MOVE] = rng.integers(0, C.NMOVE, size=nmatches)
        a1[:, C.A_ATTACK] = rng.integers(0, 2, size=nmatches)
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
    ap = argparse.ArgumentParser(description="mcbot-thinker sword self-play training")
    ap.add_argument("--nmatches", type=int)
    ap.add_argument("--frame-skip", type=int)
    ap.add_argument("--rollout", type=int)
    ap.add_argument("--iterations", type=int, default=None,
                    help="total iterations; omit for infinite (until stopped)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--hidden", type=str, default=None, help="e.g. '64,64'")
    ap.add_argument("--opp-tau", type=float)
    ap.add_argument("--run", default="sword1v1")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--log-every", type=int)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--warmstart", type=int, default=0,
                    help="BC-bootstrap steps before PPO (recommended ~6000)")
    ap.add_argument("--init", default=None,
                    help="path to an init checkpoint (e.g. *_warm.pt) to start from")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="every N iters evaluate learner vs random bot & keep best")
    ap.add_argument("--eval-matches", type=int, default=6)
    ap.add_argument("--reward", default=None,
                    help="comma list of 9 reward weights in slot order: "
                         "dmg_dealt,dmg_taken,crit,sprint_kb,combo,miss,win,lose,draw "
                         "e.g. '1,1,0.3,0.4,0.5,0.005,5,8,2'")
    args = ap.parse_args()

    rl = RLConfig()
    if args.nmatches: rl.nmatches = args.nmatches
    if args.frame_skip: rl.frame_skip = args.frame_skip
    if args.rollout: rl.rollout_len = args.rollout
    if args.lr: rl.lr = args.lr
    if args.hidden: rl.hidden = [int(x) for x in args.hidden.split(",")]
    if args.opp_tau: rl.opp_tau = args.opp_tau
    if args.seed: rl.seed = args.seed
    rl.run_name = args.run
    rl.ckpt_dir = args.ckpt_dir
    if args.save_every: rl.save_every = args.save_every
    if args.log_every: rl.log_every = args.log_every
    if args.reward:
        vals = [float(x) for x in args.reward.split(",")]
        assert len(vals) == len(C.DEFAULT_REW), \
            f"--reward needs {len(C.DEFAULT_REW)} values, got {len(vals)}"
        rl.reward = vals
        print(f"[mcbot] reward set: {dict(zip(C.RW_NAMES, vals))}", flush=True)

    tc = TrainConfig()
    if args.iterations is not None:
        tc.iterations = args.iterations

    os.makedirs(rl.ckpt_dir, exist_ok=True)

    trainer = PPOTrainer(rl)
    start_iter = 0
    if args.warmstart:
        from mcbot.rl import warmstart
        print(f"[mcbot] behavior-cloning warm start ({args.warmstart} steps)...",
              flush=True)
        obs, acts = warmstart.collect(rl, args.warmstart)
        net = warmstart.bc_train(rl, obs, acts)
        trainer.learner.load_state_dict(net.state_dict())
        trainer.opponent.load_state_dict(net.state_dict())
        print("[mcbot] warm-start complete; beginning PPO", flush=True)
    elif args.init:
        ck = torch.load(args.init, map_location="cpu")
        trainer.learner.load_state_dict(ck["learner"])
        trainer.opponent.load_state_dict(ck["opponent"])
        print(f"[mcbot] initialized from {args.init}", flush=True)
    else:
        start_iter = train_io.try_resume(trainer, rl.ckpt_dir, rl.run_name) \
            if (args.resume and not args.no_resume) else 0
    trainer._iter = start_iter

    def entropy_schedule(it):
        c = rl
        if it < c.entropy_decay_iters:
            return c.entropy_coef + (c.entropy_end - c.entropy_coef) * (it / c.entropy_decay_iters)
        return c.entropy_end

    log = train_io.Log(rl.ckpt_dir, rl.run_name)
    print(f"[mcbot] self-play PPO | {rl.nmatches} matches | hidden {rl.hidden} "
          f"| frame_skip {rl.frame_skip} | start_iter {start_iter}", flush=True)
    print(f"[mcbot] ~throughput target: {rl.nmatches*rl.rollout_len//2:.0f} "
          f"agent-steps/iter", flush=True)

    ema_win = None
    best_wr = -1.0
    t_start = time.time()
    it = start_iter
    total = tc.iterations or float("inf")
    try:
        while it < total:
            trainer.cfg.entropy_coef = entropy_schedule(it)
            loss, stats = trainer.iterate()
            win_rate = stats["wins"] / max(1, stats["wins"] + stats["losses"])
            it += 1
            trainer._iter = it
            log.record(it, time.time() - t_start, win_rate, loss, stats)
            if it % rl.log_every == 0 or it == start_iter + 1:
                log.print(it, win_rate, loss, stats)
                log.write_csv(it, time.time() - t_start, win_rate, loss, stats)
            if (not args.no_save) and (it % rl.save_every == 0 or it == total):
                log.save_checkpoint(trainer, it)
            # fixed-baseline eval: keep the genuinely best checkpoint
            if args.eval_every and it % args.eval_every == 0:
                wr = eval_vs_random(trainer, nmatches=args.eval_matches)
                print(f"[mcbot] baseline eval @it{it}: win-vs-random {wr:.3f}", flush=True)
                if wr > best_wr and not args.no_save:
                    best_wr = wr
                    trainer.save(os.path.join(rl.ckpt_dir, f"{rl.run_name}_best.pt"))
                    print(f"[mcbot] new best ({wr:.3f}) saved -> {rl.run_name}_best.pt",
                          flush=True)
            if ema_win is None:
                ema_win = win_rate
            else:
                ema_win = 0.98 * ema_win + 0.02 * win_rate
    except KeyboardInterrupt:
        print("\n[mcbot] interrupted; saving checkpoint", flush=True)
        log.save_checkpoint(trainer, it)
        sys.exit(0)


if __name__ == "__main__":
    main()
