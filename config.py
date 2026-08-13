"""Central training / physics configuration.

Everything the user might want to tweak (parallelism, network, learning,
self-play, rewards, physics knobs) lives here. Pass --cfg on the CLI or import
and mutate before building components.
"""
from dataclasses import dataclass, field
from typing import List

from mcbot.sim import consts as C


@dataclass
class SimConfig:
    # arena / kit
    reach: float = 3.0
    sword_dmg: float = 7.0
    armor_points: float = 20.0          # full diamond, no enchants
    bound: float = 12.0                 # arena half-size
    max_ticks: int = 600                # episode timeout (30s)


@dataclass
class RLConfig:
    # parallelism / throughput
    nmatches: int = 512                 # concurrent self-play matches
    frame_skip: int = 1                 # ticks per policy decision (1 = full fidelity)
    rollout_len: int = 128              # ticks of experience collected per update
    seed: int = 0

    # network
    hidden: List[int] = field(default_factory=lambda: [64, 64])
    entropy_coef: float = 0.03        # start value (annealed toward entropy_end)
    entropy_end: float = 0.004
    entropy_decay_iters: int = 1500

    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch: int = 1024
    norm_adv: bool = True
    norm_returns: bool = True       # normalize returns before value loss (stability)

    # self-play (learner vs slowly-updated opponent)
    opp_tau: float = 0.01               # polyak EMA rate for opponent policy
    opp_reset_every: int = 0            # 0 = EMA; else hard-copy learner->opp every N iters

    # reward shaping (slot order matches consts.DEFAULT_REW)
    reward: List[float] = field(default_factory=lambda: list(C.DEFAULT_REW))

    # device
    device: str = "cpu"

    # logging / checkpointing
    log_every: int = 25
    save_every: int = 200
    ckpt_dir: str = "checkpoints"
    run_name: str = "sword1v1"

    def action_space(self):
        return [C.NMOVE, 2, 2, 2, 2]


@dataclass
class TrainConfig:
    iterations: int = 20000             # total PPO iterations (0 = run forever)
    device: str = "cpu"
