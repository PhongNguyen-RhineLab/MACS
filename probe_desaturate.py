"""
probe_desaturate.py — Open Item O8 config selection WITHOUT training.

main_mac.tex, sec:results-ceiling, reports that the k=6 saturating
configuration no longer separates credit rules, and O8 proposes restoring
headroom by either (a) lowering the per-region caps to c_R = 4 or (b)
shortening the horizon to T = 40. Each candidate configuration costs
7 methods x 5 seeds x 4000 episodes to evaluate by training, so the choice
should be made on cheap numpy-only diagnostics first. This script provides
them. No torch, no training, seconds per configuration.

Three reference policies bracket what a learner can do:

  rand   uniform random actions.
  dec    DECENTRALIZED greedy. Each agent picks the action maximizing its
         own marginal gain in F against the current coverage S, blind to
         where teammates are moving this step, tie-broken toward its own
         nearest still-valuable cell. This is the policy class the paper
         enforces architecturally (q_i(s^i, S, a^i), Assumption
         decentralizable), so it is the realistic proxy for a trained
         agent.
  coord  COORDINATED greedy. Agents choose sequentially in random order,
         each seeing what earlier agents already claimed at this step.
         This is per-step greedy submodular maximization, so it is an
         optimistic proxy for a perfectly coordinated team.

and four diagnostics decide whether a configuration can test anything:

  tsat   fraction of the horizon elapsed before `coord` stops earning
         reward. tsat << 1 means the episode is mostly dead time and every
         competent learner will finish at the same value.
  actLt  fraction of last-third steps that still earn reward under `dec`.
         The budget clip acts late, so it needs a live late phase.
  gap    coord% - dec%, the COORDINATION GAP. This is the headroom that
         credit assignment exists to close. gap = 0 means coordination is
         worth nothing in this configuration, so no credit rule can beat
         any other no matter how many seeds are run. This is the single
         most diagnostic number here.
  cap%   what fraction of the credit spread comes from the saturation
         caps rather than from footprint overlap. Prop. ordering gives
             sum_i DR_i <= r_t <= sum_i LOCAL_i,
         so spread_t = (sum_i LOCAL_i - sum_i DR_i)/r_t is exactly zero
         when the three rules coincide. Computing the same spread under a
         MODULAR F(S)=|S| on the identical trajectory isolates the part
         driven purely by overlap; the remainder is cap-driven.
         cap% = (spread_sat - spread_modular)/spread_sat.

         cap% matters because if it is near zero the strictly submodular F
         is decorative: a modular F would separate the rules equally well,
         Prop. modular's linear-time closed form would apply, and the
         exponential/Monte Carlo Shapley machinery would be unnecessary.

MEASURED RESULT (see report): across ~35 candidate configurations, `gap`
and `cap%` are anti-correlated. Every configuration with cap% > 50% has
gap ~ 0; every configuration with gap > 5 has cap% < 10%. Neither O8
lever escapes this. The mechanism is that block-partition caps make the
problem CHEAPER (less total value to collect), which is the opposite of
restoring headroom.

Usage:
  python probe_desaturate.py                     the default candidate set
  python probe_desaturate.py --episodes 20       more episodes per config
  python probe_desaturate.py --family 24 4 0.5 0 60   sweep k over one env
                                                 (size region cap_frac patch T)
"""

import argparse

import numpy as np

import macs_main as M
from macs_v3 import SaturatingCoverageEnv


# =============================================================================
# Reference policies
# =============================================================================

def valuable(env, S):
    """Cells that would still earn value: uncovered AND region below cap."""
    Fs = env.Fsat
    counts = np.bincount(Fs.region_id[S], minlength=Fs.n_regions)
    return (~S) & (counts < Fs.caps)[Fs.region_id]


def _pick(env, i, base, val, rng):
    """Greedy action for agent i against coverage `base`."""
    idx = np.argwhere(val)
    tgt = idx[np.argmin(np.abs(idx - env.pos[i]).sum(1))] if len(idx) else None
    f_base = env.F(base)
    best, best_key = 0, None
    for a, (dx, dy) in enumerate(env.ACTIONS):
        nx = int(np.clip(env.pos[i, 0] + dx, 0, env.size - 1))
        ny = int(np.clip(env.pos[i, 1] + dy, 0, env.size - 1))
        gain = env.F(base | env._patch_mask((nx, ny))) - f_base
        tie = 0.0 if tgt is None else -float(abs(nx - tgt[0]) + abs(ny - tgt[1]))
        key = (gain, tie, rng.random())
        if best_key is None or key > best_key:
            best_key, best = key, a
    return best


def pol_random(env, rng):
    return rng.integers(0, env.n_actions, size=env.k)


def pol_dec(env, rng):
    S = env.covered
    val = valuable(env, S)
    return np.array([_pick(env, i, S, val, rng) for i in range(env.k)])


def pol_coord(env, rng):
    acts = np.zeros(env.k, dtype=np.int64)
    claimed = env.covered.copy()
    for i in rng.permutation(env.k):
        a = _pick(env, i, claimed, valuable(env, claimed), rng)
        acts[i] = a
        dx, dy = env.ACTIONS[a]
        nx = int(np.clip(env.pos[i, 0] + dx, 0, env.size - 1))
        ny = int(np.clip(env.pos[i, 1] + dy, 0, env.size - 1))
        claimed |= env._patch_mask((nx, ny))
    return acts


# =============================================================================
# Per-step credit spread
# =============================================================================

def spread(info, F, k):
    """(sum_i LOCAL_i - sum_i DR_i) / r_t, or None if the step earned nothing."""
    S_prev, patches = info["S_prev"], info["patches"]
    f0 = F(S_prev)
    union = S_prev.copy()
    for pm in patches:
        union |= pm
    r = F(union) - f0
    if r <= 1e-12:
        return None
    loc = sum(F(S_prev | pm) - f0 for pm in patches)
    f_all = F(union)
    dr = 0.0
    for i in range(k):
        w = S_prev.copy()
        for j, pm in enumerate(patches):
            if j != i:
                w |= pm
        dr += f_all - F(w)
    return (loc - dr) / r


# =============================================================================
# Probe
# =============================================================================

def run(cfg, policy, episodes, seed=0, collect=False):
    rng = np.random.default_rng(seed)
    finals, tsat, act_lt, sp_sat, sp_mod = [], [], [], [], []
    F_card = M.F_cardinality
    for ep in range(episodes):
        env = SaturatingCoverageEnv(**cfg, seed=seed * 1000 + ep)
        env.reset()
        T = env.horizon
        late = int(T * 2 / 3)
        last_gain, n_late, n_active = 0, 0, 0
        for t in range(T):
            _, r, done, info = env.step(policy(env, rng))
            if r > 1e-12:
                last_gain = t + 1
            if t >= late:
                n_late += 1
                n_active += int(r > 1e-12)
            if collect:
                a = spread(info, env.F, env.k)
                b = spread(info, F_card, env.k)
                if a is not None:
                    sp_sat.append(a)
                if b is not None:
                    sp_mod.append(b)
            if done:
                break
        finals.append(env.total_coverage)
        tsat.append(last_gain / T)
        act_lt.append(n_active / max(1, n_late))
    mean = lambda x: float(np.mean(x)) if len(x) else 0.0
    s, m = mean(sp_sat), mean(sp_mod)
    return dict(final=mean(finals), tsat=mean(tsat), act_lt=mean(act_lt),
                spread=s, cap_share=0.0 if s <= 1e-9 else max(0.0, (s - m)) / s)


def probe(cfg, episodes=10, seed=0):
    env = SaturatingCoverageEnv(**cfg, seed=seed)
    F_V, cap = env.max_cells, int(env.Fsat.caps[0])
    rr = run(cfg, pol_random, episodes, seed)
    rd = run(cfg, pol_dec, episodes, seed, collect=True)
    rc = run(cfg, pol_coord, episodes, seed)
    d, c = 100 * rd["final"] / F_V, 100 * rc["final"] / F_V
    return dict(F_V=F_V, cap=cap, rand=100 * rr["final"] / F_V, dec=d, coord=c,
                gap=c - d, tsat=rc["tsat"], act_lt=rd["act_lt"],
                spread=rd["spread"], cap_share=rd["cap_share"])


HEADER = (f"{'config':<30} {'cap':>3} {'F(V)':>5} {'rand%':>6} {'dec%':>6} "
          f"{'coord%':>7} {'gap':>5} | {'tsat':>5} {'actLt':>6} | "
          f"{'sprd':>6} {'cap%':>6}")


def show(name, cfg, episodes, seed=0):
    r = probe(cfg, episodes, seed)
    print(f"{name:<30} {r['cap']:>3} {r['F_V']:>5.0f} {r['rand']:>5.1f}% "
          f"{r['dec']:>5.1f}% {r['coord']:>6.1f}% {r['gap']:>5.1f} | "
          f"{r['tsat']:>5.2f} {r['act_lt']:>6.2f} | {r['spread']:>6.3f} "
          f"{100 * r['cap_share']:>5.1f}%")
    return r


CANDIDATES = [
    ("A current: 16x16 r4 c.50 p1 T60",
     dict(size=16, k=6, horizon=60, region=4, cap_frac=0.50, patch=1)),
    ("O8(b) T=40: 16x16 r4 c.50 p1",
     dict(size=16, k=6, horizon=40, region=4, cap_frac=0.50, patch=1)),
    ("O8(a) cap 4: 16x16 r4 c.25 p1",
     dict(size=16, k=6, horizon=60, region=4, cap_frac=0.25, patch=1)),
    ("both: 16x16 r4 c.25 p1 T40",
     dict(size=16, k=6, horizon=40, region=4, cap_frac=0.25, patch=1)),
    ("bigger grid: 24x24 r4 c.50 p1",
     dict(size=24, k=6, horizon=60, region=4, cap_frac=0.50, patch=1)),
    ("biggest: 40x40 r4 c.50 p1 T60",
     dict(size=40, k=6, horizon=60, region=4, cap_frac=0.50, patch=1)),
    ("fine caps: 24x24 r4 c.25 p1",
     dict(size=24, k=6, horizon=60, region=4, cap_frac=0.25, patch=1)),
    ("point cov: 16x16 r4 c.50 p0",
     dict(size=16, k=6, horizon=60, region=4, cap_frac=0.50, patch=0)),
    ("point cov: 20x20 r4 c.50 p0",
     dict(size=20, k=6, horizon=60, region=4, cap_frac=0.50, patch=0)),
    ("point cov: 24x24 r4 c.50 p0",
     dict(size=24, k=6, horizon=60, region=4, cap_frac=0.50, patch=0)),
    ("point+fine: 24x24 r2 c.25 p0",
     dict(size=24, k=6, horizon=60, region=2, cap_frac=0.25, patch=0)),
    ("point+fine: 24x24 r3 c.25 p0",
     dict(size=24, k=6, horizon=60, region=3, cap_frac=0.25, patch=0)),
    ("ref k=3: 12x12 r4 c.50 p1 T40",
     dict(size=12, k=3, horizon=40, region=4, cap_frac=0.50, patch=1)),
    ("ref k=4: 12x12 r4 c.50 p1 T40",
     dict(size=12, k=4, horizon=40, region=4, cap_frac=0.50, patch=1)),
]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", nargs=5, type=float, default=None,
                    metavar=("SIZE", "REGION", "CAP_FRAC", "PATCH", "T"),
                    help="sweep k in {2,3,4,6,8} over one environment")
    args = ap.parse_args()

    print(HEADER)
    print("-" * len(HEADER))

    if args.family:
        size, region, cap_frac, patch, T = args.family
        base = dict(size=int(size), region=int(region), cap_frac=cap_frac,
                    patch=int(patch), horizon=int(T))
        for k in (2, 3, 4, 6, 8):
            show(f"k={k}", dict(base, k=k), args.episodes, args.seed)
    else:
        for name, cfg in CANDIDATES:
            show(name, cfg, args.episodes, args.seed)

    print("\ngap   = coord% - dec%, the coordination headroom credit "
          "assignment exists to close.")
    print("        gap ~ 0 means NO credit rule can separate here.")
    print("cap%  = share of the credit spread coming from saturation caps "
          "rather than overlap.")
    print("        cap% ~ 0 means a modular F would do just as well and the "
          "Shapley machinery is idle.")
    print("tsat  = fraction of horizon before the coordinated oracle stops "
          "earning; << 1 means dead time.")