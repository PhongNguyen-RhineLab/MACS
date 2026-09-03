"""
probe_free.py -- is there a policy that beats MACS-shared WITHOUT querying F?

`dec`, `coord` and `blind` all evaluate F(S u patch) for every candidate
action, i.e. they are model-based greedy with an F oracle. The argument that
survives probe_horizon.py is that MACS is model-free and the oracle is not
available in the settings the method targets. That argument only holds if no
cheap F-free heuristic also beats the learner.

Two candidates, both using nothing but the coverage bitmap and the agent
positions. Neither reads the demand field d, and neither calls env.F.

  front  sequential greedy on the distance transform: move to the position
         whose distance to the nearest covered cell is largest. Agents
         commit in order and each writes its choice into a working copy, so
         later agents avoid earlier ones -- the same shared-set information
         `dec` uses, without the objective.

  sweep  hand-designed, no state at all: agent i owns a vertical stripe and
         walks a serpentine through it with rows spaced by rho, so the
         kernel discs tile the stripe. This is the policy a reviewer will
         propose, and it is the strongest thing available for free.

Scored on the true F, against the learned MACS-shared means from the O9 run.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

from macs_fl import FacilityCoverageEnv
from probe_fl import pol_random, pol_dec, pol_coord, _next_pos
from probe_blind import make_blind
from probe_est import _PER_ENV

FAC = dict(size=24, horizon=60, rho=4.0, patch=0, demand="uniform")

# means from o9_summary, arm MACS/shared/none/lr0.0005
MACS_SHARED = {2: 232.75, 3: 300.75, 4: 352.71, 6: 419.12, 8: 457.91}


def pol_front(env, rng):
    work = env.covered.copy()
    acts = np.zeros(env.k, dtype=np.int64)
    for i in range(env.k):
        D = distance_transform_edt(~work)
        best, ties = -np.inf, []
        for a in range(env.n_actions):
            p = _next_pos(env, i, a)
            v = D[p[0], p[1]]
            if v > best + 1e-12:
                best, ties = v, [a]
            elif abs(v - best) <= 1e-12:
                ties.append(a)
        acts[i] = ties[rng.integers(len(ties))]
        work[_next_pos(env, i, acts[i])] = True
    return acts


def make_sweep(env):
    """Serpentine within a per-agent stripe; rows spaced by rho."""
    n, k, step = env.size, env.k, max(1, int(round(env.rho_hint)))
    wpts = []
    for i in range(k):
        x0 = int(round(i * n / k))
        x1 = int(round((i + 1) * n / k)) - 1
        cols = list(range(x0 + step // 2, x1 + 1, step)) or [x0]
        path, up = [], True
        for c in cols:
            ys = list(range(step // 2, n, step))
            path += [(c, y) for y in (ys if up else ys[::-1])]
            up = not up
        wpts.append(path)
    idx = [0] * k

    def pol(env, rng):
        acts = np.zeros(env.k, dtype=np.int64)
        for i in range(env.k):
            path = wpts[i]
            if idx[i] >= len(path):
                idx[i] = 0
            tx, ty = path[idx[i]]
            cx, cy = env.pos[i]
            if (cx, cy) == (tx, ty):
                idx[i] += 1
                tx, ty = path[min(idx[i], len(path) - 1)]
            best, ba = None, 0
            for a in range(env.n_actions):
                p = _next_pos(env, i, a)
                d = abs(p[0] - tx) + abs(p[1] - ty)
                if best is None or d < best:
                    best, ba = d, a
            acts[i] = ba
        return acts
    return pol


_PER_ENV_LOCAL = set(_PER_ENV) | {make_sweep}


def rollout(cfg, mk, eps=8, seed=0):
    out = []
    for e in range(eps):
        rng = np.random.default_rng(seed * 7919 + e)
        env = FacilityCoverageEnv(**cfg, seed=seed * 1000 + e)
        env.rho_hint = cfg["rho"]
        env.reset()
        pol = mk(env) if mk in _PER_ENV_LOCAL else mk
        for _ in range(env.horizon):
            env.step(pol(env, rng))
        out.append(env.total_coverage)
    return float(np.mean(out))


if __name__ == "__main__":
    print("uniform 24x24, H=60.  F-free policies marked *")
    print(f"{'k':>3} {'random*':>9} {'front*':>9} {'sweep*':>9} "
          f"{'MACS-sh':>9} {'blind':>9} {'dec':>9} | {'best* -MACS':>12}")
    print("-" * 82)
    for k in (2, 3, 4, 6, 8):
        cfg = dict(k=k, **FAC)
        r = {}
        for n, p in (("random", pol_random), ("front", pol_front),
                     ("sweep", make_sweep), ("blind", make_blind),
                     ("dec", pol_dec)):
            r[n] = rollout(cfg, p)
        m = MACS_SHARED[k]
        free_best = max(r["random"], r["front"], r["sweep"])
        print(f"{k:>3} {r['random']:>9.1f} {r['front']:>9.1f} "
              f"{r['sweep']:>9.1f} {m:>9.1f} {r['blind']:>9.1f} "
              f"{r['dec']:>9.1f} | {free_best - m:>12.1f}")
