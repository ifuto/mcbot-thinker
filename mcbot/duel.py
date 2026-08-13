"""Bot-vs-bot duel runner (exhibition / evaluation / replay recording).

Usage:
    python -m mcbot.duel --a checkpoints/sword1v1_it100.pt \\
                        --b checkpoints/sword1v1_it100.pt --games 20
    # with behavior presets on A (e.g. combo-only, no A/D strafing):
    --a-presets combo_only,no_adtap
    # record one match to an interactive HTML replay:
    --replay out/replay.html
"""
import argparse
import os

import numpy as np
import torch

from mcbot.sim.env import SimEnv
from mcbot.sim import consts as C
from mcbot.bots import Bot, ScriptedBot, HybridBot
from mcbot.viz.replay_html import write_replay


def make_side(spec, presets, hidden, greedy):
    """Accept a checkpoint path, 'scripted:<style>', or 'hybrid:<style>:<path>'."""
    if spec.startswith("scripted:"):
        return ScriptedBot(style=spec.split(":", 1)[1], name=spec), "scripted"
    if spec.startswith("hybrid:"):
        parts = spec.split(":")
        style = parts[1] if len(parts) > 2 else "aggro"
        path = parts[2] if len(parts) > 2 else parts[1]
        return HybridBot.load(path, base_style=style, name=spec, greedy=greedy,
                              hidden=hidden), "hybrid"
    return Bot.load(spec, presets=presets, name=spec, greedy=greedy, hidden=hidden), "policy"


class Dueler:
    def __init__(self, botA: Bot, botB: Bot, nmatches=1, seed=1, max_ticks=1200):
        self.a, self.b = botA, botB
        self.env = SimEnv(nmatches, seed=seed)
        self.max_ticks = max_ticks

    def _decide(self, obs_np):
        obs_a = obs_np[:, 0, :]
        obs_b = obs_np[:, 1, :]
        full = np.zeros((obs_np.shape[0], C.NAG, C.NACT), dtype=np.int64)
        full[:, 0, :] = np.asarray(self.a.act(obs_a))
        full[:, 1, :] = np.asarray(self.b.act(obs_b))
        return full

    def fight(self, games=1, record=False, name="duel"):
        """Play `games` matches (with auto-reset). Returns stats dict, optional
        recorded trajectory for the first match."""
        env = self.env
        wins_a = wins_b = draws = 0
        dmg_a = dmg_b = 0.0
        trace = None
        if record:
            trace = {"name": name, "a": self.a.name, "b": self.b.name,
                     "frames": [], "max_ticks": self.max_ticks}
        prev = None
        for g in range(games):
            ticks = 0
            while True:
                obs_np = env.obs.copy()
                prev = obs_np
                if record and ticks < self.max_ticks:
                    trace["frames"].append(obs_np.copy())
                full = self._decide(obs_np)
                obs2, rew, done, outcome = env.step(full, 1)
                dmg_b += float(np.maximum(prev[:, 1, C.O_HP] - obs2[:, 1, C.O_HP], 0).sum())
                dmg_a += float(np.maximum(prev[:, 0, C.O_HP] - obs2[:, 0, C.O_HP], 0).sum())
                ticks += 1
                if done.any():
                    for m in np.nonzero(done)[0]:
                        if outcome[m] == 1: wins_a += 1
                        elif outcome[m] == 2: wins_b += 1
                        else: draws += 1
                    break
        if record:
            trace["frames"] = trace["frames"][:self.max_ticks]
        return {"wins_a": wins_a, "wins_b": wins_b, "draws": draws,
                "games": games, "dmg_a": dmg_a, "dmg_b": dmg_b}, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", default=None, help="defaults to --a (mirror match)")
    ap.add_argument("--a-presets", default="")
    ap.add_argument("--b-presets", default="")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()

    hidden = [int(x) for x in args.hidden.split(",")]
    b = args.b or args.a
    a_bot, _ = make_side(args.a, args.a_presets.split(",") if args.a_presets else [],
                         hidden, args.greedy)
    b_bot, _ = make_side(b, args.b_presets.split(",") if args.b_presets else [],
                         hidden, args.greedy)

    d = Dueler(a_bot, b_bot, seed=args.seed)
    record = bool(args.replay)
    stats, trace = d.fight(games=args.games, record=record)
    wr_a = stats["wins_a"] / max(1, stats["wins_a"] + stats["wins_b"])
    print(f"[duel] {a_bot.name} vs {b_bot.name}")
    print(f"[duel] {stats['games']} games -> A {stats['wins_a']}  "
          f"B {stats['wins_b']}  draw {stats['draws']}")
    print(f"[duel] A win-rate {wr_a:.2f} | dmg A {stats['dmg_a']:.1f} / B {stats['dmg_b']:.1f}")

    if record and trace:
        os.makedirs(os.path.dirname(args.replay) or ".", exist_ok=True)
        write_replay(trace, args.replay)
        print(f"[duel] replay written -> {args.replay}")


if __name__ == "__main__":
    main()
