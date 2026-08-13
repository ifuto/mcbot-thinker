"""Physics correctness tests against researched Minecraft Java 1.9+ values."""
import numpy as np
import pytest

from mcbot.sim.env import SimEnv
from mcbot.sim import consts as C

def _env():
    e = SimEnv(1, seed=1)
    idle = np.zeros((1, 2, 5), np.int32); idle[:, :, C.A_MOVE] = C.M_NONE
    e.step(idle, 1)
    return e, idle

def _step(e, idle, overrides=None):
    a = idle.copy()
    for agent, slots in (overrides or {}).items():
        for k, v in slots.items():
            a[0, agent, k] = v
    obs, rew, done, _ = e.step(a, 1)
    return obs, rew, done

def _one_attack(e, idle):
    return _step(e, idle, overrides={0: {C.A_ATTACK: 1}})

# ---------------------------------------------------------------- movement
def test_walk_speed():
    e, idle = _env()
    sp = []
    for _ in range(150):
        obs, _, _ = _step(e, idle, {0: {C.A_MOVE: C.M_RGT}})
        sp.append(np.hypot(obs[0,0,C.O_VX], obs[0,0,C.O_VZ]))
    assert abs(np.mean(sp[-60:]) - 0.2167) < 0.01, np.mean(sp[-60:])

def test_sneak_speed():
    e, idle = _env()
    sp = []
    for _ in range(150):
        obs, _, _ = _step(e, idle, {0: {C.A_MOVE: C.M_RGT, C.A_SNEAK: 1}})
        sp.append(np.hypot(obs[0,0,C.O_VX], obs[0,0,C.O_VZ]))
    assert abs(np.mean(sp[-60:]) - 0.065) < 0.005, np.mean(sp[-60:])

def test_sprint_speed():
    e, idle = _env()
    peak = 0.0
    for _ in range(40):
        obs, _, _ = _step(e, idle, {0: {C.A_MOVE: C.M_FWD, C.A_SPRINT: 1}})
        s = np.hypot(obs[0,0,C.O_VX], obs[0,0,C.O_VZ])
        peak = max(peak, s)
    assert abs(peak - 0.28) < 0.01, peak

def test_jump_apex():
    e, idle = _env()
    peak = 0.0
    for _ in range(60):
        obs, _, _ = _step(e, idle, {0: {C.A_JUMP: 1}})
        peak = max(peak, obs[0,0,C.O_Y])
    assert abs(peak - 1.2522) < 0.06, peak

# ------------------------------------------------------------------- combat
def test_reach_horizontal():
    e, idle = _env()
    e.teleport(0, 0, 0, 0, 0, 20)   # attacker at origin
    e.teleport(0, 1, 2.0, 0, 0, 20) # victim 2.0 away -> hit
    obs, _, _ = _one_attack(e, idle)
    assert obs[0,1,C.O_HP] < 20, "within reach should hit"
    e.teleport(0, 1, 3.5, 0, 0, 20)
    obs, _, _ = _one_attack(e, idle)
    assert obs[0,1,C.O_HP] == 20, "3.5 blocks should miss"

def test_reach_vertical_asymmetry():
    e, idle = _env()
    e.teleport(0, 0, 0, 0, 0, 20)
    e.teleport(0, 1, 0, 0, -3.0, 20)   # far below -> reachable (feet dist 3.0)
    obs, _, _ = _one_attack(e, idle)
    assert obs[0,1,C.O_HP] < 20, "3-below target should still be hittable"
    e.teleport(0, 1, 0, 0, 3.2, 20)    # far above -> out of vertical reach
    obs, _, _ = _one_attack(e, idle)
    assert obs[0,1,C.O_HP] == 20, "target above vertical reach should miss"

def test_full_charge_damage_with_diamond_armor():
    e, idle = _env()
    e.teleport(0, 0, 0, 0, 0, 20)
    e.teleport(0, 1, 1.0, 0, 0, 20)
    obs, _, _ = _one_attack(e, idle)
    dmg = 20.0 - obs[0,1,C.O_HP]
    # base 7, armor: reduction=20-4*7/8=16.5 -> factor 0.34 -> 2.38
    assert abs(dmg - 2.38) < 0.06, dmg

def test_crit_damage():
    e, idle = _env()
    e.teleport(0, 0, 0, 0, 3.0, 20)  # attacker airborne (falling)
    e.teleport(0, 1, 1.0, 0, 0, 20)
    obs, _, _ = _one_attack(e, idle)
    dmg = 20.0 - obs[0,1,C.O_HP]
    # crit: 10.5, armor factor 0.41 -> 4.305
    assert abs(dmg - 4.305) < 0.08, dmg

def test_invulnerability_frames():
    e, idle = _env()
    e.teleport(0, 0, 0, 0, 0, 20)
    e.teleport(0, 1, 1.0, 0, 0, 20)
    obs, _, _ = _one_attack(e, idle)
    hp1 = obs[0,1,C.O_HP]
    obs, _, _ = _one_attack(e, idle)
    assert obs[0,1,C.O_HP] == hp1, "second hit within i-frames should deal 0"

def test_cooldown_charge_ramp():
    e, idle = _env()
    e.teleport(0, 0, 0, 0, 0, 20)
    e.teleport(0, 1, 3.0, 0, 0, 20)  # out of reach so attack only resets cd
    obs, _, _ = _one_attack(e, idle)
    assert obs[0,0,C.O_CHARGE] < 0.2
    for _ in range(6):
        obs, _, _ = _step(e, idle, {})
    assert 0.4 < obs[0,0,C.O_CHARGE] < 0.7, obs[0,0,C.O_CHARGE]

def test_sprint_knockback_cancels_sprint():
    e, idle = _env()
    e.teleport(0, 0, 0, 0, 0, 20)
    e.teleport(0, 1, 1.0, 0, 0, 20)
    obs, _, _ = _step(e, idle, {0: {C.A_MOVE: C.M_FWD, C.A_SPRINT: 1}})
    assert obs[0,0,C.O_SPRINT] > 0.5
    obs, _, _ = _step(e, idle, {0: {C.A_MOVE: C.M_FWD, C.A_SPRINT: 1, C.A_ATTACK: 1}})
    assert obs[0,0,C.O_SPRINT] < 0.5, "sprint-knockback should cancel sprint"
    assert obs[0,1,C.O_Y] > 0.1 or obs[0,1,C.O_GROUND] < 0.5, "victim knocked up"

def test_wtap_sprint_lock_prevents_instant_resprint():
    """After a sprint-knockback, holding W+sprint must NOT re-sprint for a few
    ticks (sprint lock). The AI has to release W (W-tap) to reset."""
    e, idle = _env()
    e.teleport(0, 0, 0, 0, 0, 20)
    e.teleport(0, 1, 1.0, 0, 0, 20)
    # sprint + attack (sprint-knockback)
    obs, _, _ = _step(e, idle, {0: {C.A_MOVE: C.M_FWD, C.A_SPRINT: 1, C.A_ATTACK: 1}})
    assert obs[0,0,C.O_SPRINT_CD] > 0, "sprint lock should be set after sprint-KB"
    # holding W+sprint next tick should NOT re-enter sprint while locked
    obs, _, _ = _step(e, idle, {0: {C.A_MOVE: C.M_FWD, C.A_SPRINT: 1}})
    assert obs[0,0,C.O_SPRINT] < 0.5, "sprint locked: must not re-sprint yet"
    # after lock expires (and W held) sprint can return
    for _ in range(C.O_SPRINT_CD + 3):
        obs, _, _ = _step(e, idle, {0: {C.A_MOVE: C.M_FWD, C.A_SPRINT: 1}})
    assert obs[0,0,C.O_SPRINT] > 0.5, "sprint should return once lock expires"

def test_jump_reset_reduces_horizontal_knockback():
    """A victim that is airborne when hit (jump-reset) takes less horizontal KB."""
    e, idle = _env()
    e.teleport(0, 0, 0, 0, 0, 20)
    e.teleport(0, 1, 2.5, 0, 0, 20)
    # case 1: victim grounded when hit
    obs, _, _ = _step(e, idle, {0: {C.A_ATTACK: 1}})
    kb_grounded = abs(obs[0,1,C.O_VX]) + abs(obs[0,1,C.O_VZ])
    # case 2: victim is airborne (jumped) when hit
    e.teleport(0, 1, 2.5, 0, 0.6, 20)   # airborne at y=0.6
    obs, _, _ = _step(e, idle, {0: {C.A_ATTACK: 1}})
    kb_airborne = abs(obs[0,1,C.O_VX]) + abs(obs[0,1,C.O_VZ])
    assert kb_airborne < kb_grounded, "airborne victim should take less horizontal KB"
