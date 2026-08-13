"""Post-training behavior modifiers.

After learning, you can constrain the trained policy at inference time (no
re-training needed) by forcing/forbidding whole actions. Presets are applied as
logit masks, so the policy only ever samples the allowed actions. Combine any
number of presets: their constraints are intersected per action head.

Heads / allowed-value domains (see mcbot.sim.consts):
  move    0..8  F,B,L,R,FL,FR,BL,BR,NONE
  sprint  0/1   sneak 0/1  jump 0/1  attack 0/1
"""
import numpy as np
import torch

from mcbot.sim import consts as C

# movement sub-sets
_FWD   = {C.M_FWD, C.M_FL, C.M_FR}
_BACK  = {C.M_BACK, C.M_BL, C.M_BR}
_STRF  = {C.M_LFT, C.M_RGT, C.M_FL, C.M_FR, C.M_BL, C.M_BR}  # any A/D component
_NOAD  = {C.M_FWD, C.M_BACK, C.M_NONE}                        # no A/D tap at all
_ALLM  = set(range(C.NMOVE))

# ---------------- preset definitions ---------------------------------------
# each maps a head name -> set of allowed values; None/absent head = unconstrained
PRESETS = {
    "combo_only": {        # only sprint-reset knockback combos: never jump, always sprint
        "jump": {0},
        "sprint": {1},
    },
    "crit_only": {         # crit spam: always jumping, never sprinting
        "jump": {1},
        "sprint": {0},
    },
    "no_jump": {"jump": {0}},
    "always_jump": {"jump": {1}},
    "no_sprint": {"sprint": {0}},
    "always_sprint": {"sprint": {1}},
    "no_sneak": {"sneak": {0}},
    "no_attack": {"attack": {0}},
    "always_attack": {"attack": {1}},
    "no_adtap": {"move": _NOAD},          # disable A/D strafing entirely
    "no_strafe": {"move": _NOAD},
    "strafe_only": {"move": _STRF},       # only A/D, no W/S
    "walk_only": {"sprint": {0}},
    "back_pedal": {"move": _BACK},        # only move away
    "charge": {"move": _FWD, "sprint": {1}, "attack": {1}},  # straight sprint attack
    "passive": {"attack": {0}},
}


def resolve(names):
    """names: str or iterable of preset keys. Returns dict head->allowed set."""
    if isinstance(names, str):
        names = [n for n in names.split(",") if n]
    merged = {}
    for name in names:
        name = name.strip()
        if not name:
            continue
        if name not in PRESETS:
            raise KeyError(f"unknown preset '{name}'. available: {sorted(PRESETS)}")
        for head, allowed in PRESETS[name].items():
            cur = merged.setdefault(head, set(range(_HEAD_N[head])))
            merged[head] = cur & set(allowed)
    return merged


_HEAD_N = {"move": C.NMOVE, "sprint": 2, "sneak": 2, "jump": 2, "attack": 2}
_HEAD_IDX = {"move": C.A_MOVE, "sprint": C.A_SPRINT, "sneak": C.A_SNEAK,
             "jump": C.A_JUMP, "attack": C.A_ATTACK}


def mask_logits(logits, allowed):
    """logits: list[Tensor(B,n_i)] per head. allowed: resolve() dict.
    Returns masked logits (disallowed actions set to -inf)."""
    masked = []
    for head_name, idx in _HEAD_IDX.items():
        lg = logits[idx]
        if head_name in allowed:
            m = torch.full_like(lg, float("-inf"))
            allow = sorted(allowed[head_name])
            m[:, allow] = lg[:, allow]
            masked.append(m)
        else:
            masked.append(lg)
    return masked


def sample_presets(policy, obs, preset_names, greedy=False):
    """Sample actions from `policy` subject to the given presets.
    Returns (actions (B,5) long tensor, logprob, value)."""
    logits, _ = policy.get_logits(obs)
    allowed = resolve(preset_names)
    if allowed:
        logits = mask_logits(logits, allowed)
    acts = torch.empty(obs.shape[0], len(logits), dtype=torch.int64)
    logps = torch.zeros(obs.shape[0])
    for i, lg in enumerate(logits):
        d = torch.distributions.Categorical(logits=lg)
        a = d.probs.argmax(-1) if greedy else d.sample()
        acts[:, i] = a
        logps += d.log_prob(a)
    return acts, logps
