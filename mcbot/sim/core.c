/* mcbot-thinker — Minecraft 1.9+ Java PvP physics simulator core (C).
 * Single-file. Vectorized over N self-play matches by a tight inner loop so a
 * potato CPU can step millions of ticks/sec.
 *
 * Physics model (3D, ground combat on a flat walled arena):
 *   - Player hitbox 0.6w x 1.8h (1.5h sneaking); reach = distance from the
 *     attacker's feet-center to the CLOSEST POINT on the victim's AABB <= 3.0.
 *     This yields the real vertical asymmetry (easy to hit targets below,
 *     harder to reach ones above).
 *   - Movement: walk 0.2167 b/t, sprint 0.28 b/t, sneak 0.065 b/t; ground
 *     friction 0.546, air drag 0.91, gravity 0.08 b/t^2, jump vy 0.42,
 *     terminal fall -3.92 b/t. Sprint persists while holding W+sprint.
 *   - Sword: cooldown 12.5 ticks, damage 7, crit x1.5 (falling & not
 *     sprinting), charge multiplier 0.2+((t+0.5)/T)^2*0.8.
 *   - Full diamond armor (no enchants): 20 AP / 0 toughness reduction.
 *   - Sprint-knockback attack needs charge>=0.848, cancels attacker sprint,
 *     bigger horizontal + vertical(0.5) KB. 10-tick i-frames. Hit-select
 *     (attack within 2 ticks of being hit) cuts incoming horizontal KB x0.6.
 */
#include "core.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---------------- physics constants (blocks/tick unless noted) ------------ */
#define GRAVITY      0.08f
#define AIR_DRAG     0.91f
#define GROUND_FRIC  0.546f
#define WALK_SPEED   0.2167f
#define SPRINT_SPEED 0.28f
#define SNEAK_SPEED  0.065f
/* accel = speed*(1-fric)/fric so equilibrium under GROUND_FRIC == target speed */
#define ACCEL_WALK   (0.2167f*(1.0f-GROUND_FRIC)/GROUND_FRIC)
#define ACCEL_SPRINT (0.28f  *(1.0f-GROUND_FRIC)/GROUND_FRIC)
#define ACCEL_SNEAK  (0.065f *(1.0f-GROUND_FRIC)/GROUND_FRIC)
/* tuned so apex under gravity+0.98 vertical drag reproduces the 1.2522-block
 * jump (documented Java jump height); terminal fall stays -3.92 b/t (78.4 m/s) */
#define JUMP_VEL     0.508f
#define TERM_FALL    -3.92f
#define WALL_FRIC    0.6f
#define KB_GROUND    0.4f      /* horizontal KB, normal hit */
#define KB_SPRINT    0.6f      /* horizontal KB, sprint-knockback */
#define KB_VERT      0.4f      /* vertical KB, normal ground hit */
#define KB_SPRINT_V  0.5f      /* vertical KB, sprint-knockback */
#define KB_CRIT_MUL  0.5f      /* crits knock back less */
#define HITSELECT    0.6f      /* victim recently attacking: incoming KB factor */
#define JPRESET      0.7f      /* victim airborne when hit (jump-reset): KB factor */
#define SPRINT_LOCK  3         /* ticks attacker must wait to re-sprint (W-tap) */

#define COOLDOWN_TICKS 12.5f
#define SWORD_DMG    7.0f
#define CRIT_MULT    1.5f
#define SPRINT_KB_CHG 0.848f
#define CRIT_CHG     0.848f
#define INVULN       10.0f

#define ARMOR_PTS    20.0f     /* full diamond, no enchants */
#define ARMOR_TOUGH  0.0f      /* diamond toughness */

#define REACH        3.0f
#define HW           0.3f      /* half-width of player box */
#define BOUND        7.0f      /* arena half-size (small -> forced melee) */
#define MAX_TICKS    1200      /* episode timeout (60s) */

/* ---------------- reward parameter slots -------------------------------- */
#define RW_DMG    0   /* per damage dealt              */
#define RW_TAKEN  1   /* per damage taken              */
#define RW_CRIT   2
#define RW_KB     3
#define RW_COMBO  4
#define RW_MISS   5
#define RW_WIN    6
#define RW_LOSE   7
#define RW_DRAW   8   /* penalty when a match times out (both alive) */
#define RW_NPARAM 9

static const float DEFAULT_REW[RW_NPARAM] =
    {1.0f, 1.0f, 0.3f, 0.4f, 0.5f, 0.005f, 5.0f, 8.0f, 2.0f};

typedef struct {
    float x,z,y,vx,vz,vy,hp,charge,imm;
    int sprint,sneak,ground,air,recent_atk,sprint_cd;
} Agent;

typedef struct {
    int nmatches, ticks, done;
    Agent a[MCBOT_NAGENTS];
} Match;

typedef struct {
    int nmatches;
    Match *m;
    unsigned rng;
    float rew[RW_NPARAM];
} Sim;

static unsigned lcg(unsigned *s){ *s = *s*1664525u + 1013904223u; return *s; }
static float frand(unsigned *s){ return (float)((lcg(s)>>8) & 0xFFFF) / 65536.0f; }
static float clampf(float v,float lo,float hi){ return v<lo?lo:(v>hi?hi:v); }

static void reset_match(Sim *sim, Match *mc){
    float sep = 2.0f + frand(&sim->rng)*2.0f;      /* 2..4 blocks apart (in/near reach) */
    float ang = frand(&sim->rng)*6.2831853f;
    mc->a[0].x =  cosf(ang)*sep*0.5f;  mc->a[0].z =  sinf(ang)*sep*0.5f;
    mc->a[1].x = -cosf(ang)*sep*0.5f;  mc->a[1].z = -sinf(ang)*sep*0.5f;
    for (int i=0;i<MCBOT_NAGENTS;i++){
        Agent *a=&mc->a[i];
        a->y=0.f; a->vx=0.f; a->vz=0.f; a->vy=0.f;
        a->hp=20.f; a->charge=COOLDOWN_TICKS; a->imm=0.f;
        a->sprint=0; a->sneak=0; a->ground=1; a->air=0; a->recent_atk=999;
        a->sprint_cd=0;
    }
    mc->ticks=0; mc->done=0;
}

/* armor reduction (Java 1.9+), returns final damage */
static float apply_armor(float dmg){
    float red = ARMOR_PTS - (4.0f*dmg)/(ARMOR_TOUGH+8.0f);
    red = clampf(red, 0.f, ARMOR_PTS);
    return dmg * (1.0f - red/25.0f);
}

static void move_agent(Match *mc, int i, const int *act){
    Agent *a = &mc->a[i];
    Agent *o = &mc->a[1-i];
    int mv = act[A_MOVE], sp = act[A_SPRINT], sn = act[A_SNEAK], jp = act[A_JUMP];

    /* facing yaw toward opponent */
    float dx=o->x-a->x, dz=o->z-a->z;
    float facing = atan2f(dz, dx);
    float fx=cosf(facing), fz=sinf(facing);   /* forward unit */

    /* movement dir (forward = toward opponent, left/right = perpendicular) */
    float lx=-fz, lz=fx;                        /* left unit */
    float mx=0,mz=0;
    int fwd = (mv==M_FWD||mv==M_FL||mv==M_FR);
    int back = (mv==M_BACK||mv==M_BL||mv==M_BR);
    int lft = (mv==M_LFT||mv==M_FL||mv==M_BL);
    int rgt = (mv==M_RGT||mv==M_FR||mv==M_BR);
    if (fwd){ mx+=fx; mz+=fz; }
    if (back){ mx-=fx; mz-=fz; }
    if (lft){ mx+=lx; mz+=lz; }
    if (rgt){ mx-=lx; mz-=lz; }
    float ml = sqrtf(mx*mx+mz*mz);
    if (ml>1e-6f){ mx/=ml; mz/=ml; }

    /* sprint state: enter when grounded & holding W & sprint & no sprint lock.
     * The lock (set by a sprint-knockback attack) forces the attacker to
     * release W for a tick or two before re-sprinting — this is the W-tap
     * reset that makes combos a timing skill the AI must learn. */
    if (fwd && sp && a->ground && a->sprint_cd <= 0){ a->sprint = 1; }
    if (!(fwd && sp)){ a->sprint = 0; }
    if (a->sprint_cd > 0) a->sprint_cd--;
    a->sneak = sn;

    float speed, accel;
    if (a->sneak){ speed = SNEAK_SPEED; accel = ACCEL_SNEAK; }
    else if (a->sprint){ speed = SPRINT_SPEED; accel = ACCEL_SPRINT; }
    else { speed = WALK_SPEED; accel = ACCEL_WALK; }

    if (a->ground){
        /* MC ground motion: additive acceleration then friction, so the
         * equilibrium speed equals `speed` and W-tap stops take a few ticks. */
        if (ml > 1e-6f){
            a->vx += mx*accel; a->vz += mz*accel;
        }
        a->vx *= GROUND_FRIC; a->vz *= GROUND_FRIC;
        if (jp){ a->vy = JUMP_VEL; a->ground = 0; }
    } else {
        a->vx *= AIR_DRAG; a->vz *= AIR_DRAG;
    }

    /* vertical (gravity + vertical air drag 0.98, MC-like) */
    if (!a->ground){
        a->vy = (a->vy - GRAVITY)*0.98f;
        if (a->vy < TERM_FALL) a->vy = TERM_FALL;
        a->air++;
    }
    a->y += a->vy;
    if (a->y <= 0.f && a->vy <= 0.f){ a->y=0.f; a->vy=0.f; a->ground=1; a->air=0; }

    /* walls */
    if (a->x >  BOUND){ a->x =  BOUND; a->vx *= -WALL_FRIC; }
    if (a->x < -BOUND){ a->x = -BOUND; a->vx *= -WALL_FRIC; }
    if (a->z >  BOUND){ a->z =  BOUND; a->vz *= -WALL_FRIC; }
    if (a->z < -BOUND){ a->z = -BOUND; a->vz *= -WALL_FRIC; }

    /* integrate horizontal */
    a->x += a->vx; a->z += a->vz;

    /* timers */
    if (a->charge < COOLDOWN_TICKS) a->charge += 1.0f;
    if (a->imm > 0) a->imm -= 1.0f;
    if (a->recent_atk < 999) a->recent_atk++;
}

/* 3D reach: distance from attacker feet-center to closest point on victim AABB */
static float reach_dist(const Agent *atk, const Agent *v){
    float h = v->sneak ? 1.5f : 1.8f;
    float dx = clampf(atk->x, v->x-HW, v->x+HW) - atk->x;
    float dy = clampf(atk->y, v->y,     v->y+h ) - atk->y;
    float dz = clampf(atk->z, v->z-HW, v->z+HW) - atk->z;
    return sqrtf(dx*dx+dy*dy+dz*dz);
}

/* attack from agent i; mutates victim; returns nothing (rewards go to rw[i]) */
static void do_attack(Sim *sim, Match *mc, int i, float *rw){
    Agent *a=&mc->a[i]; Agent *v=&mc->a[1-i];
    float t = a->charge;
    a->charge = 0.f;             /* swing resets cooldown even on miss */
    a->recent_atk = 0;
    (void)sim;

    float T=COOLDOWN_TICKS;
    float charge = t/T;
    if (charge > 1.f) charge = 1.f;

    int in_reach = reach_dist(a,v) <= REACH;
    int can_sprint_kb = (a->sprint && charge >= SPRINT_KB_CHG);

    /* swing that misses: small penalty */
    if (!in_reach){
        rw[i] -= 1.0f * sim->rew[RW_MISS];
        return;
    }
    /* i-frames: no damage, no knockback */
    if (v->imm > 0) return;

    float mult = 0.2f + ((t+0.5f)/T)*((t+0.5f)/T)*0.8f;
    if (mult>1.f) mult=1.f;
    if (mult<0.2f) mult=0.2f;

    int crit = (!a->ground && a->vy < 0.f && !a->sprint && t >= CRIT_CHG);
    int sprint_atk = can_sprint_kb;

    float base = SWORD_DMG * mult;
    if (crit) base *= CRIT_MULT;
    float dmg = apply_armor(base);
    v->hp -= dmg;
    v->imm = INVULN;

    rw[i] += sim->rew[RW_DMG] * dmg;
    rw[1-i] -= sim->rew[RW_TAKEN] * dmg;
    if (crit) rw[i] += sim->rew[RW_CRIT];
    if (sprint_atk) rw[i] += sim->rew[RW_KB];
    if (!v->ground) rw[i] += sim->rew[RW_COMBO];   /* hitting airborne opp */

    /* ---- knockback ---- */
    float dx=v->x-a->x, dz=v->z-a->z;
    float mag = sqrtf(dx*dx+dz*dz);
    float nx = mag>1e-6f? dx/mag:1.f;
    float nz = mag>1e-6f? dz/mag:0.f;

    float str = sprint_atk ? KB_SPRINT : KB_GROUND;
    if (crit) str *= KB_CRIT_MUL;
    /* hit-select: victim swung recently -> take less KB */
    if (v->recent_atk <= 2) str *= HITSELECT;
    /* jump-reset: victim already airborne (e.g. jumped into the hit) takes
     * less horizontal KB — the AI can learn to jump right before impact */
    if (!v->ground) str *= JPRESET;

    v->vx = v->vx*0.5f + nx*str;
    v->vz = v->vz*0.5f + nz*str;
    float vert = sprint_atk ? KB_SPRINT_V : KB_VERT;
    if (v->ground){ v->vy = vert; v->ground = 0; }   /* knocked airborne */
    else v->vy += vert;

    /* sprint-knockback cancels the attacker's sprint and locks it for a few
     * ticks, so chaining sprint-KB hits requires the W-tap reset rhythm */
    if (sprint_atk){ a->sprint = 0; a->sprint_cd = SPRINT_LOCK; }
}

static void fill_obs(Match *mc, float *obs){
    for (int i=0;i<MCBOT_NAGENTS;i++){
        Agent *a=&mc->a[i]; Agent *o=&mc->a[1-i];
        float *p = obs + i*MCBOT_NOBS;
        p[O_X]=a->x; p[O_Z]=a->z; p[O_Y]=a->y;
        p[O_VX]=a->vx; p[O_VZ]=a->vz; p[O_VY]=a->vy;
        p[O_HP]=a->hp;
        float chg=a->charge/COOLDOWN_TICKS; if(chg>1.f)chg=1.f;
        p[O_CHARGE]=chg;
        p[O_SPRINT]=a->sprint; p[O_SNEAK]=a->sneak; p[O_GROUND]=a->ground;
        p[O_AIRTICKS]=(float)a->air;
        p[O_DX]=o->x-a->x; p[O_DZ]=o->z-a->z; p[O_DY]=o->y-a->y;
        p[O_DVX]=o->vx-a->vx; p[O_DVZ]=o->vz-a->vz; p[O_DVY]=o->vy-a->vy;
        p[O_DIST]=reach_dist(a,o);
        p[O_OPP_HP]=o->hp; p[O_OPP_SPRINT]=o->sprint;
        p[O_OPP_SNEAK]=o->sneak; p[O_OPP_GROUND]=o->ground;
        p[O_TIME]=(float)mc->ticks;
        p[O_RECENT_ATK]=(float)a->recent_atk;
        p[O_IMMUNE]=a->imm;
        p[O_SPRINT_CD]=(float)a->sprint_cd;
    }
}

/* ---------------- public API -------------------------------------------- */
void *sim_create(int nmatches, unsigned seed){
    Sim *sim = calloc(1,sizeof(Sim));
    sim->nmatches = nmatches;
    sim->rng = seed ? seed : 0x9e3779b9u;
    memcpy(sim->rew, DEFAULT_REW, sizeof(DEFAULT_REW));
    sim->m = calloc(nmatches,sizeof(Match));
    for (int i=0;i<nmatches;i++){ sim->m[i].nmatches=nmatches; reset_match(sim,&sim->m[i]); }
    return sim;
}
void sim_destroy(void *ctx){ Sim *s=ctx; free(s->m); free(s); }
void sim_set_rewards(void *ctx,const float*p,int n){
    Sim*s=ctx;
    if (n>RW_NPARAM) n=RW_NPARAM;
    for (int i=0;i<n;i++) s->rew[i]=p[i];
}
void sim_reset_match(void *ctx,int match){ Sim*s=ctx; reset_match(s,&s->m[match]); }
void sim_teleport(void *ctx,int match,int agent,float x,float z,float y,float hp){
    Sim*s=ctx; Agent*a=&s->m[match].a[agent];
    a->x=x; a->z=z; a->y=y; a->hp=hp; a->vx=0; a->vz=0; a->vy=0; a->ground=(y<=0.001f);
}

void sim_step(void *ctx, const int *actions, int nticks,
              float *obs_out, float *reward_out, int *done_out, int *outcome){
    Sim *sim=ctx;
    for (int m=0;m<sim->nmatches;m++) outcome[m]=0;
    for (int k=0;k<nticks;k++){
        for (int m=0;m<sim->nmatches;m++){
            Match *mc=&sim->m[m];
            if (mc->done) continue;
            float rw[MCBOT_NAGENTS]={0,0};
            for (int i=0;i<MCBOT_NAGENTS;i++){
                const int *act = actions + (m*MCBOT_NAGENTS+i)*MCBOT_NACT;
                move_agent(mc,i,act);
            }
            for (int i=0;i<MCBOT_NAGENTS;i++){
                const int *act = actions + (m*MCBOT_NAGENTS+i)*MCBOT_NACT;
                if (act[A_ATTACK]) do_attack(sim,mc,i,rw);
            }
            mc->ticks++;
            for (int i=0;i<MCBOT_NAGENTS;i++){
                reward_out[m*MCBOT_NAGENTS+i] += rw[i];
            }
                /* episode end */
                if (mc->a[0].hp<=0.f || mc->a[1].hp<=0.f || mc->ticks>=MAX_TICKS){
                    if (mc->a[0].hp<=0.f && mc->a[1].hp<=0.f){
                        reward_out[m*2+0]-=sim->rew[RW_LOSE]; reward_out[m*2+1]-=sim->rew[RW_LOSE];
                        outcome[m]=3;
                    } else if (mc->a[0].hp<=0.f){
                        reward_out[m*2+0]-=sim->rew[RW_LOSE]; reward_out[m*2+1]+=sim->rew[RW_WIN];
                        outcome[m]=2;                       /* agent1 won */
                    } else if (mc->a[1].hp<=0.f){
                        reward_out[m*2+0]+=sim->rew[RW_WIN]; reward_out[m*2+1]-=sim->rew[RW_LOSE];
                        outcome[m]=1;                       /* agent0 won */
                    } else {   /* timeout -> draw penalty (both alive) */
                        reward_out[m*2+0]-=sim->rew[RW_DRAW];
                        reward_out[m*2+1]-=sim->rew[RW_DRAW];
                        outcome[m]=3;
                    }
                    mc->done=1;
                }
        }
    }
    /* fill obs + report done + auto-reset finished matches */
    for (int m=0;m<sim->nmatches;m++){
        Match *mc=&sim->m[m];
        fill_obs(mc, obs_out + m*MCBOT_NAGENTS*MCBOT_NOBS);
        done_out[m] = mc->done;
        if (mc->done) reset_match(sim,mc);
    }
}
