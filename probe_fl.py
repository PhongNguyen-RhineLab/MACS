"""
probe_fl.py -- config selection for the facility-location suite (O8, option 1).

Same four diagnostics as probe_desaturate.py so the numbers are directly
comparable with the saturating suite:

  gap    coord% - dec%.  How much a coordinated team beats a decentralized
         greedy one.  This is the headroom credit assignment exists to close.
         gap = 0  =>  no credit rule can beat any other on this task.
  tsat   fraction of the horizon before the coordinated oracle stops earning.
         Low tsat = the episode is mostly dead time.
  actLt  fraction of LATE-episode steps (last half) still earning reward.
  nonmod share of the LOCAL-minus-DR credit spread that a modular surrogate
         (F = |S|) cannot reproduce.  In the saturating suite this was
         "cap%", the share attributable to caps binding rather than to
         literal footprint overlap.

Usage:
    python probe_fl.py [--episodes 10]
"""

import sys
import time
import numpy as np

import macs_main as M
from macs_fl import FacilityCoverageEnv, shapley_facility


# ---------------------------------------------------------------- policies

def _next_pos(env, i, a):
    dx, dy = env.ACTIONS[a]
    return (int(np.clip(env.pos[i, 0] + dx, 0, env.size - 1)),
            int(np.clip(env.pos[i, 1] + dy, 0, env.size - 1)))


def pol_random(env, rng):
    return rng.integers(0, env.n_actions, size=env.k)


def pol_dec(env, rng):
    """Each agent maximizes its OWN marginal, ignoring teammates."""
    S = env.covered
    f0 = env.F(S)
    acts = np.zeros(env.k, dtype=np.int64)
    for i in range(env.k):
        best, ties = -np.inf, []
        for a in range(env.n_actions):
            pm = env._patch_mask(_next_pos(env, i, a))
            g = env.F(S | pm) - f0
            if g > best + 1e-12:
                best, ties = g, [a]
            elif abs(g - best) <= 1e-12:
                ties.append(a)
        acts[i] = ties[rng.integers(len(ties))]
    return acts


def pol_coord(env, rng):
    """Sequential greedy: agent i conditions on agents 1..i-1 already placed."""
    run = env.covered.copy()
    acts = np.zeros(env.k, dtype=np.int64)
    for i in range(env.k):
        f0 = env.F(run)
        best, ties, bestpm = -np.inf, [], None
        for a in range(env.n_actions):
            pm = env._patch_mask(_next_pos(env, i, a))
            g = env.F(run | pm) - f0
            if g > best + 1e-12:
                best, ties, bestpm = g, [a], pm
            elif abs(g - best) <= 1e-12:
                ties.append(a)
        acts[i] = ties[rng.integers(len(ties))]
        run |= env._patch_mask(_next_pos(env, i, acts[i]))
    return acts


# ---------------------------------------------------------------- rollout

def run(cfg, policy, episodes, seed, collect=False):
    finals, tsats, actlts = [], [], []
    spreads, mods = [], []
    Fc = M.F_cardinality
    for e in range(episodes):
        rng = np.random.default_rng(seed * 7919 + e)
        env = FacilityCoverageEnv(**cfg, seed=seed * 1000 + e)
        env.reset()
        H = env.horizon
        last_earn, late_earn = 0, 0
        for t in range(H):
            _, r, done, info = env.step(policy(env, rng))
            if r > 1e-9:
                last_earn = t + 1
                if t >= H // 2:
                    late_earn += 1
            if collect and r > 1e-9:
                S, P = info["S_prev"], info["patches"]
                for F, acc in ((env.F, spreads), (Fc, mods)):
                    f0 = F(S)
                    U = S.copy()
                    for pm in P: U |= pm
                    rr = F(U) - f0
                    if rr <= 1e-12:
                        continue
                    loc = sum(F(S | pm) - f0 for pm in P)
                    dr = sum(F(U) - F(np.logical_or.reduce(
                        [S] + [P[j] for j in range(env.k) if j != i]))
                        for i in range(env.k))
                    acc.append((loc - dr) / rr)
        finals.append(env.total_coverage)
        tsats.append(last_earn / H)
        actlts.append(late_earn / (H - H // 2))
    s = float(np.mean(spreads)) if spreads else 0.0
    m = float(np.mean(mods)) if mods else 0.0
    return dict(final=float(np.mean(finals)), tsat=float(np.mean(tsats)),
                act_lt=float(np.mean(actlts)), spread=s,
                nonmod=0.0 if s <= 1e-9 else max(0.0, (s - m)) / s)


def probe(cfg, episodes=10, seed=0):
    F_V = FacilityCoverageEnv(**cfg, seed=seed).max_cells
    rr = run(cfg, pol_random, episodes, seed)
    rd = run(cfg, pol_dec, episodes, seed, collect=True)
    rc = run(cfg, pol_coord, episodes, seed)
    d, c = 100 * rd["final"] / F_V, 100 * rc["final"] / F_V
    return dict(F_V=F_V, rand=100 * rr["final"] / F_V, dec=d, coord=c,
                gap=c - d, tsat=rc["tsat"], act_lt=rd["act_lt"],
                spread=rd["spread"], nonmod=rd["nonmod"])


HEADER = (f"{'config':<34} {'F(V)':>6} {'rand%':>6} {'dec%':>6} {'coord%':>7} "
          f"{'gap':>5} | {'tsat':>5} {'actLt':>6} | {'sprd':>6} {'nonmod':>7}")


def show(name, cfg, episodes, seed=0):
    t0 = time.time()
    r = probe(cfg, episodes, seed)
    print(f"{name:<34} {r['F_V']:>6.0f} {r['rand']:>5.1f}% {r['dec']:>5.1f}% "
          f"{r['coord']:>6.1f}% {r['gap']:>5.1f} | {r['tsat']:>5.2f} "
          f"{r['act_lt']:>6.2f} | {r['spread']:>6.3f} {100*r['nonmod']:>6.1f}%"
          f"   [{time.time()-t0:.0f}s]")
    return r


def base(size, k, horizon, rho, demand="uniform", patch=0):
    return dict(size=size, k=k, horizon=horizon, rho=rho, demand=demand,
                patch=patch)


if __name__ == "__main__":
    eps = 10
    if "--episodes" in sys.argv:
        eps = int(sys.argv[sys.argv.index("--episodes") + 1])

    print("\n=== A. radius sweep, 16x16 k=6 T=60, uniform demand ===")
    print(HEADER); print("-" * 104)
    for rho in (2.0, 3.0, 4.0, 6.0):
        show(f"16x16 k6 rho={rho} uniform", base(16, 6, 60, rho), eps)

    print("\n=== B. grid size at fixed rho=4, k=6 ===")
    print(HEADER); print("-" * 104)
    for size, T in ((16, 60), (20, 60), (24, 60), (24, 80)):
        show(f"{size}x{size} k6 T{T} rho=4 uniform",
             base(size, 6, T, 4.0), eps)

    print("\n=== C. clustered demand (facility location proper) ===")
    print(HEADER); print("-" * 104)
    for size, nc in ((16, 5), (20, 6), (24, 8)):
        cfg = base(size, 6, 60, 4.0, demand="clusters")
        cfg["n_clusters"] = nc
        show(f"{size}x{size} k6 clusters n={nc}", cfg, eps)

    print("\n=== D. k ladder at the best uniform config ===")
    print(HEADER); print("-" * 104)
    for k in (2, 3, 4, 6):
        show(f"20x20 k{k} T60 rho=4 uniform", base(20, k, 60, 4.0), eps)

    print("\n=== E. k ladder at the best clustered config ===")
    print(HEADER); print("-" * 104)
    for k in (2, 3, 4, 6):
        cfg = base(20, k, 60, 4.0, demand="clusters"); cfg["n_clusters"] = 6
        show(f"20x20 k{k} clusters n=6", cfg, eps)