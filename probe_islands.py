"""
probe_islands.py -- does a sparse-demand facility task have credit-assignment
headroom that the uniform / clustered maps do not?

Findings so far (probe_fl / probe_blind / probe_est / probe_free):
  coord - dec ~ 0 on every facility config.  Once S_t is shared, one-step
  greedy is near-optimal, so no credit rule can win asymptotically.
  MACS-shared sits far BELOW dec (233 vs 357 at k=2), so the learners are
  losing to approximation error, not to credit assignment.

Sparse islands change the structure in two ways:
  * between islands the one-step marginal is ~0, so a myopic policy stalls
    -> planning headroom  (plan - dec)
  * two agents that pick the same island waste one of them, and a shared
    team reward cannot tell them which one should leave
    -> assignment headroom (plan - plan_selfish)

Both are measured with F-oracle policies so the numbers are properties of
the task, not of any learner.

Policies (all scored on the true F):
  dec          one-step greedy on shared S_t                  (myopic)
  coord        one-step sequential greedy                      (myopic)
  sweep        fixed serpentine, no state                      (F-free)
  plan_selfish each agent walks to its nearest unfinished island, ignoring
               teammates; re-targets when its island is done  (no coord)
  plan         sequential nearest-island assignment: agent i skips islands
               already claimed this step by agents 1..i-1      (coord)
  plan+greedy  plan, but once inside kernel range of the target island the
               agent switches to one-step greedy on shared S_t

Usage:
    python probe_islands.py [--quick]
"""

import sys
import numpy as np

from probe_fl import pol_random, pol_dec, pol_coord, _next_pos
from probe_free import make_sweep


from macs_fl import IslandEnv  # noqa: E402  (env now lives in macs_fl)


# ---------------------------------------------------------------- planners

def _island_done(env, ci):
    """Island finished when every island cell is served at weight >= 0.7
    (standing on the centre of a radius-1 island at rho=4 serves every
    island cell at 0.75, so this means someone has stood on it)."""
    cx, cy = env.Ffl.centres[ci]
    xs, ys = np.meshgrid(np.arange(env.size), np.arange(env.size),
                         indexing="ij")
    cells = (xs - cx) ** 2 + (ys - cy) ** 2 <= env.Ffl.radius ** 2
    best = env.Ffl.utility_map(env.covered)
    return bool((best[cells] >= 0.7).all())


def _step_toward(env, i, tgt):
    best, ba = None, 4
    for a in range(env.n_actions):
        p = _next_pos(env, i, a)
        d = abs(p[0] - tgt[0]) + abs(p[1] - tgt[1])
        if best is None or d < best:
            best, ba = d, a
    return ba


def _greedy_one(env, i, S, rng):
    f0 = env.F(S)
    best, ties = -np.inf, []
    for a in range(env.n_actions):
        pm = env._patch_mask(_next_pos(env, i, a))
        g = env.F(S | pm) - f0
        if g > best + 1e-12:
            best, ties = g, [a]
        elif abs(g - best) <= 1e-12:
            ties.append(a)
    return ties[rng.integers(len(ties))]


def make_plan(coord=True, local_greedy=False):
    def mk(env):
        def pol(env, rng):
            n_isl = len(env.Ffl.centres)
            done = np.array([_island_done(env, c) for c in range(n_isl)])
            claimed = np.zeros(n_isl, dtype=bool)
            acts = np.full(env.k, 4, dtype=np.int64)
            for i in range(env.k):
                avail = ~done & (~claimed if coord else True)
                if not avail.any():
                    avail = ~done if (~done).any() else np.ones(n_isl, bool)
                dists = np.abs(env.Ffl.centres - env.pos[i]).sum(1)
                dists = np.where(avail, dists, 10 ** 9)
                c = int(np.argmin(dists))
                claimed[c] = True
                tgt = env.Ffl.centres[c]
                near = np.hypot(*(env.pos[i] - tgt)) <= env.Ffl.radius + 1
                if local_greedy and near:
                    acts[i] = _greedy_one(env, i, env.covered, rng)
                else:
                    acts[i] = _step_toward(env, i, tgt)
            return acts
        return pol
    return mk


PER_ENV = {make_sweep}


def rollout(cfg, mk, eps, seed, rand_layout):
    out = []
    for e in range(eps):
        rng = np.random.default_rng(seed * 7919 + e)
        lay = None if rand_layout else 0
        env = IslandEnv(**cfg, layout_seed=lay, seed=seed * 1000 + e)
        env.reset()
        pol = mk(env) if (mk in PER_ENV or getattr(mk, "_per_env", False)) else mk
        for _ in range(env.horizon):
            env.step(pol(env, rng))
        out.append(env.total_coverage / env.max_cells * 100.0)
    return float(np.mean(out)), float(np.std(out))


def pol_random_(env, rng):
    return pol_random(env, rng)


def build_policies():
    P_self = make_plan(coord=False); P_self._per_env = True
    P_coord = make_plan(coord=True); P_coord._per_env = True
    P_cg = make_plan(coord=True, local_greedy=True); P_cg._per_env = True
    return (("random", pol_random_), ("sweep", make_sweep),
            ("dec", pol_dec), ("coord", pol_coord),
            ("plan_self", P_self), ("plan", P_coord), ("plan+g", P_cg))


def table(name, cfg_base, ks, eps, rand_layout):
    pols = build_policies()
    print(f"\n=== {name} ===   (% of F(V), mean over {eps} eps)")
    print(f"{'k':>3} " + " ".join(f"{n:>9}" for n, _ in pols) +
          f" | {'plan-dec':>8} {'plan-self':>9} {'plan-sweep':>10}")
    print("-" * 100)
    for k in ks:
        cfg = dict(k=k, **cfg_base)
        v = {n: rollout(cfg, p, eps, 0, rand_layout)[0] for n, p in pols}
        print(f"{k:>3} " + " ".join(f"{v[n]:>9.1f}" for n, _ in pols) +
              f" | {v['plan']-v['dec']:>8.1f} {v['plan']-v['plan_self']:>9.1f} "
              f"{v['plan']-v['sweep']:>10.1f}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    EPS = 4 if quick else 10
    ks = [2, 3, 4, 6]

    # Canonical island config: no demand floor (myopic greedy stalls between
    # islands), depot spawn (agents start together, so nearest-island
    # choices collide), horizon too short for a sweep to tile the map.
    A = dict(size=24, horizon=30, rho=4.0, n_islands=8, radius=1,
             min_sep=8, eps=0.0)
    table("ISLANDS  24x24 H=30 rho=4  8 islands r=1  depot spawn, "
          "random layout per episode", A, ks, EPS, True)
    table("ISLANDS  same, FIXED layout (layout_seed=0)", A, ks, EPS, False)

    # Controls: which ingredient makes the headroom
    table("control: demand floor eps=0.02 (myopic no longer stalls)",
          dict(A, eps=0.02), ks, EPS, True)
    table("control: H=60 (sweep has time to tile the map)",
          dict(A, horizon=60), ks, EPS, True)