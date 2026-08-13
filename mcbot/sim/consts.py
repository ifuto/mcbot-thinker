"""Shared constants / action-space helpers (mirror core.h)."""
import numpy as np

NAG = 2
NOBS = 27
NACT = 5
NMOVE = 9

# A_MOVE codes
M_FWD, M_BACK, M_LFT, M_RGT = 0, 1, 2, 3
M_FL, M_FR, M_BL, M_BR = 4, 5, 6, 7
M_NONE = 8

# action slots
A_MOVE, A_SPRINT, A_SNEAK, A_JUMP, A_ATTACK = 0, 1, 2, 3, 4

# observation slots (must match core.h)
O_X, O_Z, O_Y = 0, 1, 2
O_VX, O_VZ, O_VY = 3, 4, 5
O_HP, O_CHARGE = 6, 7
O_SPRINT, O_SNEAK, O_GROUND, O_AIRTICKS = 8, 9, 10, 11
O_DX, O_DZ, O_DY = 12, 13, 14
O_DVX, O_DVZ, O_DVY = 15, 16, 17
O_DIST = 18
O_OPP_HP, O_OPP_SPRINT, O_OPP_SNEAK, O_OPP_GROUND = 19, 20, 21, 22
O_TIME, O_RECENT_ATK, O_IMMUNE, O_SPRINT_CD = 23, 24, 25, 26

# reward parameter slots (mirror core.c RW_*)
RW_DMG, RW_TAKEN, RW_CRIT, RW_KB, RW_COMBO, RW_MISS, RW_WIN, RW_LOSE, RW_DRAW = range(9)
RW_NAMES = ["dmg_dealt", "dmg_taken", "crit", "sprint_kb", "combo", "miss",
            "win", "lose", "draw"]
DEFAULT_REW = [1.0, 1.0, 0.3, 0.4, 0.5, 0.005, 5.0, 8.0, 2.0]

MOVE_NAMES = ["F", "B", "L", "R", "FL", "FR", "BL", "BR", "IDLE"]

# observation field names (for debugging/visualization)
OBS_NAMES = [
    "x", "z", "y", "vx", "vz", "vy", "hp", "charge", "sprint",
    "sneak", "ground", "airticks", "dx", "dz", "dy", "dvx", "dvz", "dvy",
    "dist", "opp_hp", "opp_sprint", "opp_sneak", "opp_ground",
    "time", "recent_atk", "immune", "sprint_cd",
]


def random_action(nagents, rng):
    """Random discrete actions (nagents, 5). For sanity / untrained baseline."""
    a = np.zeros((nagents, NACT), dtype=np.int32)
    a[:, A_MOVE] = rng.integers(0, NMOVE, size=nagents)
    a[:, A_SPRINT] = rng.integers(0, 2, size=nagents)
    a[:, A_SNEAK] = rng.integers(0, 2, size=nagents)
    a[:, A_JUMP] = rng.integers(0, 2, size=nagents)
    a[:, A_ATTACK] = rng.integers(0, 2, size=nagents)
    return a
