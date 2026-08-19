"""
probe_fl2.py -- follow-up probe.

Two things probe_fl.py could not settle:

  1. `pol_coord` is a ONE-STEP sequential greedy.  It captures collision
     avoidance but not territory assignment ("agent 3 takes the north-west
     quadrant"), which is the coordination that actually matters over a
     horizon.  A one-step oracle can report gap = 0 on a task that has
     large multi-step headroom.  We add pol_terr, a territory oracle:
     k-means on demand-weighted cells at reset, agents matched to
     territories, each agent greedy WITHIN its territory.

  2. Every config in probe_fl.py was over-resourced -- decentralized greedy
     already reached 89-97% of F(V), so there was nothing left to win.  We
     sweep the under-resourced corner (large grid, short horizon) where the
     visited set S stays sparse, which is also the regime where the
     facility kernel keeps interacting at range instead of collapsing to a
     frontier-local phenomenon.

Also reports `dens`, the final fraction of cells visited, to make the
resourcing explicit.
"""

import sys
import time
import numpy as np

import macs_main as M
from macs_fl import FacilityCoverageEnv, shapley_facility
from probe_fl import pol_random, pol_dec, pol_coord, _next_pos


# ------------------------------------------------------- territory oracle

def _kmeans(pts, w, k, rng, iters=25):
    idx = rng.choice(len(pts), size=k, replace=False,
                     p=w / w.sum() if w.sum() > 0 else None)
    cen = pts[idx].astype(np.float64)
    lab = np.zeros(len(pts), dtype=np.int64)
    for _ in range(iters):
        d = ((pts[:, None, :] - cen[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for c in range(k):
            m = lab == c
            if m.any():
                cen[c] = (pts[m] * w[m, None]).sum(0) / max(w[m].sum(), 1e-9)
    return lab, cen


class TerritoryPolicy:
    """Assign each agent a demand cluster at reset, then greedy within it."""

    def __init__(self, env, rng):
        H = env.size
        xs, ys = np.meshgrid(np.arange(H), np.arange(H), indexing="ij")
        pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
        w = env.Ffl.d.ravel().copy()
        lab, cen = _kmeans(pts, w, env.k, rng)
        self.terr = lab.reshape(H, H)
        # greedy matching of agents to centroids by current distance
        free = list(range(env.k))
        self.assign = np.zeros(env.k, dtype=np.int64)
        pairs = sorted(((np.hypot(*(env.pos[i] - cen[c])), i, c)
                        for i in range(env.k) for c in range(env.k)))
        taken_i, taken_c = set(), set()
        for _, i, c in pairs:
            if i in taken_i or c in taken_c:
                continue
            self.assign[i] = c; taken_i.add(i); taken_c.add(c)

    def __call__(self, env, rng):
        run = env.covered.copy()
        acts = np.zeros(env.k, dtype=np.int64)
        for i in range(env.k):
            f0 = env.F(run)
            mine = self.terr == self.assign[i]
            best, ties = -np.inf, []
            for a in range(env.n_actions):
                nx, ny = _next_pos(env, i, a)
                pm = env._patch_mask((nx, ny))
                g = env.F(run | pm) - f0
                # bias toward own territory: outside moves pay a small tax
                if not mine[nx, ny]:
                    g -= 1e-3
                if g > best + 1e-12:
                    best, ties = g, [a]
                elif abs(g - best) <= 1e-12:
                    ties.append(a)
            acts[i] = ties[rng.integers(len(ties))]
            run |= env._patch_mask(_next_pos(env, i, acts[i]))
        return acts


# ------------------------------------------------------------- rollout

def run(cfg, make_policy, episodes, seed, collect=False):
    finals, tsats, actlts, dens = [], [], [], []
    spreads, mods = [], []
    Fc = M.F_cardinality
    for e in range(episodes):
        rng = np.random.default_rng(seed * 7919 + e)
        env = FacilityCoverageEnv(**cfg, seed=seed * 1000 + e)
        env.reset()
        policy = make_policy(env, rng)
        H = env.horizon
        last_earn, late_earn = 0, 0
        for t in range(H):
            _, r, done, info = env.step(policy(env, rng))
            if r > 1e-9:
                last_earn = t + 1
                if t >= H // 2:
                    late_earn += 1
                if collect:
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
        dens.append(env.covered.mean())
    s = float(np.mean(spreads)) if spreads else 0.0
    m = float(np.mean(mods)) if mods else 0.0
    return dict(final=float(np.mean(finals)), tsat=float(np.mean(tsats)),
                act_lt=float(np.mean(actlts)), dens=float(np.mean(dens)),
                spread=s, nonmod=0.0 if s <= 1e-9 else max(0.0, (s - m)) / s)


_wrap = lambda p: (lambda env, rng: p)

HEADER = (f"{'config':<32} {'F(V)':>5} {'rand%':>6} {'dec%':>6} {'seq%':>6} "
          f"{'terr%':>6} {'gapS':>5} {'gapT':>5} | {'dens':>5} {'tsat':>5} | "
          f"{'sprd':>6} {'nonmod':>7}")


def show(name, cfg, episodes, seed=0):
    t0 = time.time()
    F_V = FacilityCoverageEnv(**cfg, seed=seed).max_cells
    rr = run(cfg, _wrap(pol_random), episodes, seed)
    rd = run(cfg, _wrap(pol_dec), episodes, seed, collect=True)
    rc = run(cfg, _wrap(pol_coord), episodes, seed)
    rt = run(cfg, TerritoryPolicy, episodes, seed)
    d = 100 * rd["final"] / F_V
    c = 100 * rc["final"] / F_V
    tt = 100 * rt["final"] / F_V
    print(f"{name:<32} {F_V:>5.0f} {100*rr['final']/F_V:>5.1f}% {d:>5.1f}% "
          f"{c:>5.1f}% {tt:>5.1f}% {c-d:>5.1f} {tt-d:>5.1f} | "
          f"{rd['dens']:>5.2f} {rc['tsat']:>5.2f} | {rd['spread']:>6.3f} "
          f"{100*rd['nonmod']:>6.1f}%   [{time.time()-t0:.0f}s]")
    return dict(gapS=c - d, gapT=tt - d, dens=rd["dens"],
                spread=rd["spread"], nonmod=rd["nonmod"])


def cfg(size, k, T, rho, demand="uniform", **kw):
    c = dict(size=size, k=k, horizon=T, rho=rho, demand=demand, patch=0)
    c.update(kw); return c


if __name__ == "__main__":
    eps = 6
    if "--episodes" in sys.argv:
        eps = int(sys.argv[sys.argv.index("--episodes") + 1])

    print("\n=== F. does the territory oracle find headroom the 1-step one misses? ===")
    print(HEADER); print("-" * 122)
    show("16x16 k6 T60 rho4 (probe_fl A)", cfg(16, 6, 60, 4.0), eps)
    show("20x20 k6 T60 rho4 (probe_fl B)", cfg(20, 6, 60, 4.0), eps)
    show("24x24 k6 T60 rho4 (probe_fl B)", cfg(24, 6, 60, 4.0), eps)

    print("\n=== G. under-resourced corner: S stays sparse ===")
    print(HEADER); print("-" * 122)
    for size, T, rho in ((32, 40, 4.0), (32, 40, 6.0), (32, 60, 4.0),
                         (40, 40, 4.0), (40, 60, 6.0)):
        show(f"{size}x{size} k6 T{T} rho{rho:.0f}", cfg(size, 6, T, rho), eps)

    print("\n=== H. clustered demand, under-resourced ===")
    print(HEADER); print("-" * 122)
    for size, T, nc in ((32, 40, 8), (32, 60, 8), (40, 60, 10)):
        show(f"{size}x{size} k6 T{T} clust{nc}",
             cfg(size, 6, T, 4.0, demand="clusters", n_clusters=nc), eps)