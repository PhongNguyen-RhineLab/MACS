"""
probe_svalue.py -- two loose ends left by probe_horizon.py.

  1. S-value (dec - blind) is the quantity O9 exists to demonstrate. Does it
     keep growing past H=60, or is the current configuration already at its
     maximum? Answer: H=60 is the peak; longer horizons let even a blind
     agent cover the map, and the value of the shared set decays.

  2. pol_coord is sequential greedy WITHIN a timestep, so it cannot see
     multi-step planning headroom. pol_coord2 below adds a 2-step lookahead
     and commits the first move, closing that gap in the argument.
"""
import numpy as np
from macs_fl import FacilityCoverageEnv
from probe_fl import pol_dec, pol_coord, _next_pos
from probe_blind import make_blind
from probe_est import _PER_ENV

def rollout(cfg, mk, eps=8, seed=0):
    out = []
    for e in range(eps):
        rng = np.random.default_rng(seed*7919+e)
        env = FacilityCoverageEnv(**cfg, seed=seed*1000+e); env.reset()
        pol = mk(env) if mk in _PER_ENV else mk
        for _ in range(env.horizon): env.step(pol(env, rng))
        out.append(env.total_coverage)
    return float(np.mean(out))

def pol_coord2(env, rng):
    """Sequential greedy with 2-step lookahead: agent i commits the first
    move of its best 2-move plan, conditioned on 1..i-1 already committed."""
    run = env.covered.copy()
    acts = np.zeros(env.k, dtype=np.int64)
    for i in range(env.k):
        f0 = env.F(run)
        best, ties, bestpm = -np.inf, [], None
        for a in range(env.n_actions):
            p1 = _next_pos(env, i, a)
            m1 = env._patch_mask(p1)
            sub = -np.inf
            for b in range(env.n_actions):
                dx, dy = env.ACTIONS[b]
                p2 = (int(np.clip(p1[0]+dx, 0, env.size-1)),
                      int(np.clip(p1[1]+dy, 0, env.size-1)))
                g = env.F(run | m1 | env._patch_mask(p2)) - f0
                if g > sub: sub = g
            if sub > best + 1e-12: best, ties, bestpm = sub, [a], m1
            elif abs(sub - best) <= 1e-12: ties.append(a)
        acts[i] = ties[rng.integers(len(ties))]
        run |= env._patch_mask(_next_pos(env, i, acts[i]))
    return acts

print("uniform 24x24, k=6 -- S-value vs horizon")
print(f"{'H':>5} {'blind':>8} {'dec':>8} {'S-value':>9} {'%F(V)':>7}")
print("-"*42)
for H in (60, 90, 120, 160):
    cfg = dict(size=24, horizon=H, rho=4.0, patch=0, k=6, demand="uniform")
    F_V = FacilityCoverageEnv(**cfg, seed=0).max_cells
    b, d = rollout(cfg, make_blind), rollout(cfg, pol_dec)
    print(f"{H:>5} {b:>8.1f} {d:>8.1f} {d-b:>9.1f} {100*(d-b)/F_V:>6.1f}%")

print("\nmulti-step planning headroom, uniform 24x24 k=6")
print(f"{'H':>5} {'dec':>8} {'coord':>8} {'coord2':>8} {'c2-dec':>8} {'%F(V)':>7}")
print("-"*50)
for H in (20, 60):
    cfg = dict(size=24, horizon=H, rho=4.0, patch=0, k=6, demand="uniform")
    F_V = FacilityCoverageEnv(**cfg, seed=0).max_cells
    d = rollout(cfg, pol_dec); c = rollout(cfg, pol_coord)
    c2 = rollout(cfg, pol_coord2, eps=4)
    print(f"{H:>5} {d:>8.1f} {c:>8.1f} {c2:>8.1f} {c2-d:>8.1f} "
          f"{100*(c2-d)/F_V:>6.1f}%")
