"""
probe_cells.py -- distinct-cells-visited reference for the O9 private diagnostic.

The private arm scores below `random` at every k. Before reading a private
training run we need to know what |S_H| looks like for policies whose
behaviour we already understand, so that a small number is legible as
collapse rather than as an unfamiliar scale.

Reports mean |S_H| (cells in the shared visited set at the end of the
episode) and the per-agent mean, alongside the true objective F(S_H).
"""

import numpy as np

from macs_fl import FacilityCoverageEnv
from probe_fl import pol_random, pol_dec, pol_coord
from probe_blind import make_blind

FACILITY = dict(size=24, horizon=60, rho=4.0, patch=0, demand="uniform")


def rollout_cells(cfg, mk, eps, seed):
    cells, per_agent, obj = [], [], []
    for e in range(eps):
        rng = np.random.default_rng(seed * 7919 + e)
        env = FacilityCoverageEnv(**cfg, seed=seed * 1000 + e)
        env.reset()
        own = [env._patch_mask(env.pos[i]).copy() for i in range(env.k)]
        pol = mk(env) if mk is make_blind else mk
        for _ in range(env.horizon):
            env.step(pol(env, rng))
            for i in range(env.k):
                own[i] |= env._patch_mask(env.pos[i])
        cells.append(int(env.covered.sum()))
        per_agent.append(float(np.mean([o.sum() for o in own])))
        obj.append(env.total_coverage)
    return (float(np.mean(cells)), float(np.mean(per_agent)),
            float(np.mean(obj)))


if __name__ == "__main__":
    ks = [2, 3, 4, 6, 8]
    H = FACILITY["horizon"]
    print(f"grid {FACILITY['size']}x{FACILITY['size']}, horizon {H}, "
          f"patch {FACILITY['patch']}, F(V) = {FACILITY['size']**2}")
    print("ceiling |S_H| if no agent ever revisits a cell = k*H + k\n")
    print(f"{'k':>3} {'policy':<8} {'|S_H|':>7} {'/agent':>7} "
          f"{'%ceil':>6} {'F(S_H)':>8}")
    print("-" * 46)
    for k in ks:
        cfg = dict(k=k, **FACILITY)
        ceil = k * H + k
        for name, pol in (("random", pol_random), ("blind", make_blind),
                          ("dec", pol_dec), ("coord", pol_coord)):
            c, pa, f = rollout_cells(cfg, pol, 8, 0)
            print(f"{k:>3} {name:<8} {c:>7.1f} {pa:>7.1f} "
                  f"{100*c/ceil:>5.1f}% {f:>8.1f}")
        print()
