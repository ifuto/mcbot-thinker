"""Checkpointing + logging helpers for the training loop."""
import glob
import os
import re

import torch


def _ckpt_path(ckpt_dir, run, it):
    return os.path.join(ckpt_dir, f"{run}_it{it}.pt")


def _latest(ckpt_dir, run):
    pat = os.path.join(ckpt_dir, f"{run}_it*.pt")
    files = glob.glob(pat)
    if not files:
        return None, -1
    def num(f):
        m = re.search(r"_it(\d+)\.pt$", f)
        return int(m.group(1)) if m else -1
    best = max(files, key=num)
    return best, num(best)


def try_resume(trainer, ckpt_dir, run):
    path, it = _latest(ckpt_dir, run)
    if path and it >= 0:
        trainer.load(path)
        print(f"[mcbot] resumed from {path} (iter {it})", flush=True)
        return it
    return 0


class Log:
    def __init__(self, ckpt_dir, run):
        self.ckpt_dir = ckpt_dir
        self.run = run
        self.csv = os.path.join(ckpt_dir, f"{run}_log.csv")
        if not os.path.exists(self.csv):
            with open(self.csv, "w") as f:
                f.write("iter,sec,winrate,policy_loss,value_loss,entropy,"
                        "reward_mean,dmg_dealt,dmg_taken\n")

    def record(self, it, sec, win_rate, loss, stats):
        self._last = (it, sec, win_rate, loss, stats)

    def print(self, it, win_rate, loss, stats):
        sec = self._last[1] if hasattr(self, "_last") else 0
        print(f"[mcbot] it {it:5d} | {sec:6.1f}s | win {win_rate:5.2f} "
              f"| W/L/D {int(stats['wins'])}/{int(stats['losses'])}/{int(stats['draws'])} "
              f"| dmg {stats['dmg_dealt']:.0f}/{stats['dmg_taken']:.0f} "
              f"| pl {loss['policy_loss']:.3f} vl {loss['value_loss']:.3f} "
              f"ent {loss['entropy']:.3f}", flush=True)

    def write_csv(self, it, sec, win_rate, loss, stats):
        with open(self.csv, "a") as f:
            f.write(f"{it},{sec:.1f},{win_rate:.4f},{loss['policy_loss']:.4f},"
                    f"{loss['value_loss']:.4f},{loss['entropy']:.4f},"
                    f"{stats.get('dmg_dealt',0):.1f},{stats.get('dmg_taken',0):.1f}\n")

    def save_checkpoint(self, trainer, it):
        path = os.path.join(self.ckpt_dir, f"{self.run}_it{it}.pt")
        trainer.save(path)
        # keep only latest 3 to avoid disk bloat on long runs
        keep = 3
        for old in sorted(glob.glob(os.path.join(self.ckpt_dir, f"{self.run}_it*.pt")),
                          key=os.path.getmtime)[:-keep]:
            try:
                os.remove(old)
            except OSError:
                pass
        print(f"[mcbot] saved {path}", flush=True)
