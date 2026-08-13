/* mcbot-thinker — Minecraft 1.9+ Java PvP physics simulator core (C).
 *
 * Faithful-on-the-axes-that-matter, 3D, single-file. Constants mirror the
 * values researched from minecraft.wiki (see README.md for sources). Knockback
 * magnitudes are tunable constants with defaults close to measured values.
 *
 * State/observation/action layout is fixed and shared with the Python wrapper
 * (sim.py). See core.c for detailed layout docs.
 */
#ifndef MCBOT_CORE_H
#define MCBOT_CORE_H

#define MCBOT_NAGENTS       2       /* agents per match (self-play pair) */

/* per-agent observation fields (index into obs[match*2+agent][NOBS]) */
#define O_X           0
#define O_Z           1
#define O_Y           2
#define O_VX          3
#define O_VZ          4
#define O_VY          5
#define O_HP          6
#define O_CHARGE      7   /* attack cooldown charge 0..1 */
#define O_SPRINT      8
#define O_SNEAK       9
#define O_GROUND      10
#define O_AIRTICKS    11  /* ticks since last grounded */
#define O_DX          12  /* rel opponent */
#define O_DZ          13
#define O_DY          14
#define O_DVX         15
#define O_DVZ         16
#define O_DVY         17
#define O_DIST        18
#define O_OPP_HP      19
#define O_OPP_SPRINT  20
#define O_OPP_SNEAK   21
#define O_OPP_GROUND  22
#define O_TIME        23  /* match time in ticks */
#define O_RECENT_ATK  24  /* ticks since own last attack (for hit-select) */
#define O_IMMUNE      25  /* own i-frames remaining */
#define O_SPRINT_CD   26  /* sprint lock remaining (W-tap reset timing) */
#define MCBOT_NOBS    27

/* action slots per agent: actions[(match*2+agent)*NACT + slot] */
#define A_MOVE    0       /* 0..8: F,B,L,R,FL,FR,BL,BR,none */
#define A_SPRINT  1       /* 0/1 */
#define A_SNEAK   2       /* 0/1 */
#define A_JUMP    3       /* 0/1 */
#define A_ATTACK  4       /* 0/1 */
#define MCBOT_NACT 5

/* move direction codes */
#define M_FWD  0
#define M_BACK 1
#define M_LFT  2
#define M_RGT  3
#define M_FL   4
#define M_FR   5
#define M_BL   6
#define M_BR   7
#define M_NONE 8

#ifdef __cplusplus
extern "C" {
#endif

void *sim_create(int nmatches, unsigned seed);
void  sim_destroy(void *ctx);
void  sim_set_rewards(void *ctx, const float *p, int n);
void  sim_reset_match(void *ctx, int match);
/* teleport an agent (useful for tests, exhibition matches, replays) */
void  sim_teleport(void *ctx, int match, int agent,
                   float x, float z, float y, float hp);
/* step all matches `nticks` ticks given `actions` (size nmatches*2*NACT).
 * obs_out   : float[nmatches*2*NOBS]  (final obs of each agent)
 * reward_out: float[nmatches*2]       (accumulated reward this step)
 * done_out  : int[nmatches]           (1 if match ended & auto-reset)
 * outcome   : int[nmatches] (0=ongoing; when done: 1=agent0 won,
 *            2=agent1 won, 3=draw/timeout)
 */
void  sim_step(void *ctx, const int *actions, int nticks,
               float *obs_out, float *reward_out, int *done_out, int *outcome);

#ifdef __cplusplus
}
#endif
#endif /* MCBOT_CORE_H */
