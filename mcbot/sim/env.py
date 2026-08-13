"""Vectorized self-play environment backed by the C core (ctypes).

The core steps `nmatches` 1v1 matches in lock-step. This wrapper owns the C
arrays, exposes numpy views for observations/actions/rewards, and handles
auto-reset (finished matches are restarted in the C core).

Action layout (per agent per tick), see core.h:
  A_MOVE 0..8  FWD,BACK,LEFT,RIGHT,FL,FR,BL,BR,NONE
  A_SPRINT 0/1, A_SNEAK 0/1, A_JUMP 0/1, A_ATTACK 0/1
"""
import ctypes as C
import os
import numpy as np

from . import build as _build

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---- ctypes signatures -----------------------------------------------------
def _lib():
    _build.build()                      # build once if missing
    lib = C.CDLL(os.path.join(_HERE, "core.so"))
    lib.sim_create.restype = C.c_void_p
    lib.sim_create.argtypes = [C.c_int, C.c_uint]
    lib.sim_destroy.argtypes = [C.c_void_p]
    lib.sim_reset_match.argtypes = [C.c_void_p, C.c_int]
    lib.sim_teleport.argtypes = [C.c_void_p, C.c_int, C.c_int,
                                 C.c_float, C.c_float, C.c_float, C.c_float]
    lib.sim_set_rewards.argtypes = [C.c_void_p,
                                    C.POINTER(C.c_float), C.c_int]
    lib.sim_step.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_int),      # actions
        C.c_int,                 # nticks
        C.POINTER(C.c_float),    # obs_out
        C.POINTER(C.c_float),    # reward_out
        C.POINTER(C.c_int),      # done_out
        C.POINTER(C.c_int),      # outcome
    ]
    return lib

_NAG = 2
NOBS = 27
NACT = 5
NMOVE = 9  # A_MOVE range


class SimEnv:
    def __init__(self, nmatches, seed=1):
        self.nmatches = nmatches
        self.nagents = nmatches * _NAG
        self.lib = _lib()
        self.ctx = self.lib.sim_create(nmatches, seed)

        self._act = np.zeros(self.nagents * NACT, dtype=np.int32)
        self._obs = np.zeros(self.nagents * NOBS, dtype=np.float32)
        self._rew = np.zeros(self.nagents, dtype=np.float32)
        self._done = np.zeros(nmatches, dtype=np.int32)
        self._outcome = np.zeros(nmatches, dtype=np.int32)

        # numpy views for ergonomic use
        self.obs = self._obs.reshape(nmatches, _NAG, NOBS)
        self.actions = self._act.reshape(nmatches, _NAG, NACT)
        self.outcome = self._outcome

    # ---- stepping ----------------------------------------------------------
    def step(self, actions, nticks=1):
        """actions: (nmatches, 2, 5) int array in 0..(moves) ranges.
        Returns (obs, reward, done, outcome) numpy arrays.
        outcome: 0 ongoing, 1 agent0 won, 2 agent1 won, 3 draw (when done)."""
        a = np.ascontiguousarray(actions, dtype=np.int32).reshape(-1)
        assert a.size == self.nagents * NACT, "actions shape mismatch"
        self._act[:] = a
        self.lib.sim_step(
            self.ctx,
            self._act.ctypes.data_as(C.POINTER(C.c_int)),
            int(nticks),
            self._obs.ctypes.data_as(C.POINTER(C.c_float)),
            self._rew.ctypes.data_as(C.POINTER(C.c_float)),
            self._done.ctypes.data_as(C.POINTER(C.c_int)),
            self._outcome.ctypes.data_as(C.POINTER(C.c_int)),
        )
        return (self.obs.copy(),
                self._rew.reshape(self.nmatches, _NAG).copy(),
                self._done.copy(),
                self._outcome.copy())

    def reset_match(self, match):
        self.lib.sim_reset_match(self.ctx, int(match))

    def set_rewards(self, rew):
        """rew: length-8 float list/array in slot order [DMG,TAKEN,CRIT,KB,COMBO,MISS,WIN,LOSE]."""
        arr = np.ascontiguousarray(rew, dtype=np.float32)
        self.lib.sim_set_rewards(self.ctx, arr.ctypes.data_as(C.POINTER(C.c_float)),
                                 int(arr.size))

    def teleport(self, match, agent, x, z, y, hp):
        self.lib.sim_teleport(self.ctx, int(match), int(agent),
                              float(x), float(z), float(y), float(hp))

    def obs_flat(self):
        return self._obs  # (nmatches*2, NOBS) view, own agent index == match*2+i

    def close(self):
        self.lib.sim_destroy(self.ctx)
        self.ctx = None

    def __del__(self):
        try:
            if getattr(self, "ctx", None):
                self.close()
        except Exception:
            pass
