"""
probe_blind.py -- is the coordination gap small because the map S is SHARED?

probe_fl.py and probe_fl2.py both found gap = coord% - dec% near zero on
every facility-location configuration, and probe_desaturate.py found the
same on every saturating one.  This script tests the explanation.

  pol_dec   each agent maximizes its marginal against the SHARED cumulative
            coverage set S_t.  This is MACS's own execution setting: by
            Prop. `decentralized`, every agent observes S_t.
  pol_blind each agent maximizes its marginal against ITS OWN visited set
            only.  Dynamics and the reported objective are unchanged; only
            the information the policy conditions on is removed.
  pol_coord one-step sequential greedy over the shared S_t (the oracle).

Two differences are then separable:

  S-value = dec - blind    what the shared augmented state buys
  deconf  = coord - dec    what same-step deconfliction buys on top of it

If S-value is large and deconf is ~0, the augmented state is doing the
coordinating and no credit rule can recover further asymptotic policy
value -- which is a statement about MACS's problem class, not about any
particular F.
"""

import numpy as np

from macs_fl import FacilityCoverageEnv
from macs_v3 import SaturatingCoverageEnv
from probe_fl import pol_dec, pol_coord, _next_pos


def make_blind(env):
    """Per-agent private coverage set; no shared map."""
    own = [env._patch_mask(env.pos[i]).copy() for i in range(env.k)]

    def pol(env, rng):
        acts = np.zeros(env.k, dtype=np.int64)
        for i in range(env.k):
            f0 = env.F(own[i])
            best, ties = -np.inf, []
            for a in range(env.n_actions):
                pm = env._patch_mask(_next_pos(env, i, a))
                g = env.F(own[i] | pm) - f0
                if g > best + 1e-12:
                    best, ties = g, [a]
                elif abs(g - best) <= 1e-12:
                    ties.append(a)
            acts[i] = ties[rng.integers(len(ties))]
        for i in range(env.k):
            own[i] |= env._patch_mask(_next_pos(env, i, acts[i]))
        return acts
    return pol


def rollout(EnvCls, cfg, mk, eps, seed):
    out = []
    for e in range(eps):
        rng = np.random.default_rng(seed * 7919 + e)
        env = EnvCls(**cfg, seed=seed * 1000 + e)
        env.reset()
        pol = mk(env) if mk is make_blind else mk
        for _ in range(env.horizon):
            env.step(pol(env, rng))
        out.append(env.total_coverage)
    return float(np.mean(out))


def ladder(name, EnvCls, mk_cfg, ks, eps=8):
    print(f"\n{name}")
    print(f"{'k':>3} {'blind%':>8} {'dec%':>8} {'coord%':>8} | "
          f"{'S-value':>8} {'deconf':>7}")
    print("-" * 52)
    rows = []
    for k in ks:
        cfg = mk_cfg(k)
        F_V = EnvCls(**cfg, seed=0).max_cells
        b = 100 * rollout(EnvCls, cfg, make_blind, eps, 0) / F_V
        d = 100 * rollout(EnvCls, cfg, pol_dec, eps, 0) / F_V
        c = 100 * rollout(EnvCls, cfg, pol_coord, eps, 0) / F_V
        print(f"{k:>3} {b:>7.1f}% {d:>7.1f}% {c:>7.1f}% | "
              f"{d-b:>7.1f} {c-d:>6.1f}")
        rows.append((k, b, d, c))
    return rows


if __name__ == "__main__":
    ladder("FL 24x24 T60 rho=4, uniform demand", FacilityCoverageEnv,
           lambda k: dict(size=24, k=k, horizon=60, rho=4.0, patch=0),
           (2, 3, 4, 6, 8))
    ladder("FL 32x32 T60 rho=4, uniform demand", FacilityCoverageEnv,
           lambda k: dict(size=32, k=k, horizon=60, rho=4.0, patch=0),
           (2, 3, 4, 6, 8))
    ladder("FL 24x24 T60 rho=4, clustered demand", FacilityCoverageEnv,
           lambda k: dict(size=24, k=k, horizon=60, rho=4.0, patch=0,
                          demand="clusters", n_clusters=8),
           (2, 3, 4, 6, 8))
    ladder("SAT (paper suite) 12x12 k<6 / 16x16 k=6", SaturatingCoverageEnv,
           lambda k: dict(size=12 if k < 6 else 16, k=k,
                          horizon=40 if k < 6 else 60,
                          region=4, cap_frac=0.5, patch=1),
           (2, 3, 4, 6))