"""
probe_est.py -- is there anything for a learner to win once the demand field
is unknown?

`dec`, `coord` and `blind` all call env.F directly, so they read the demand
field d exactly. Changing d from uniform to clustered does not touch that
advantage. The comparison that means something is against a greedy policy
restricted to the same partial information a learner could have:

    dec_est   greedy against F_hat built from d_hat, where
              d_hat = d on cells the team has visited (sensor radius 0)
                    = 1.0 elsewhere (the uninformed prior; FacilityLocationF
                      rescales d so that its mean is exactly 1)

Scored on the TRUE F in every case. Under demand="uniform", d_hat == d
identically, so dec_est must reproduce dec exactly -- that is the unit test.

The number that decides the plan is  dec - dec_est.  If it is small, the
estimation problem is trivial, the environment is still not hard, and
training MACS on it buys nothing.
"""

import copy
import numpy as np

from macs_fl import FacilityCoverageEnv
from probe_fl import pol_random, pol_dec, pol_coord, _next_pos
from probe_blind import make_blind

BASE = dict(size=24, horizon=60, rho=4.0, patch=0)


def make_dec_est(env):
    """Greedy on the shared set, but against an estimated demand field."""
    def pol(env, rng):
        Fhat = copy.copy(env.Ffl)
        Fhat.d = np.where(env.covered, env.Ffl.d, 1.0)
        S = env.covered
        f0 = Fhat(S)
        acts = np.zeros(env.k, dtype=np.int64)
        for i in range(env.k):
            best, ties = -np.inf, []
            for a in range(env.n_actions):
                pm = env._patch_mask(_next_pos(env, i, a))
                g = Fhat(S | pm) - f0
                if g > best + 1e-12:
                    best, ties = g, [a]
                elif abs(g - best) <= 1e-12:
                    ties.append(a)
            acts[i] = ties[rng.integers(len(ties))]
        return acts
    return pol


_PER_ENV = (make_blind, make_dec_est)


def rollout(cfg, mk, eps, seed, rand_field):
    out, sensed = [], []
    for e in range(eps):
        rng = np.random.default_rng(seed * 7919 + e)
        c = dict(cfg)
        if rand_field:
            c["demand_seed"] = 10_000 + e     # a new field every episode
        env = FacilityCoverageEnv(**c, seed=seed * 1000 + e)
        env.reset()
        pol = mk(env) if mk in _PER_ENV else mk
        for _ in range(env.horizon):
            env.step(pol(env, rng))
        out.append(env.total_coverage)
        sensed.append(env.covered.mean())
    return float(np.mean(out)), float(np.std(out)), float(np.mean(sensed))


POLICIES = (("random", pol_random), ("blind", make_blind),
            ("dec_est", make_dec_est), ("dec", pol_dec), ("coord", pol_coord))


def table(name, cfg_base, ks, eps, rand_field):
    print(f"\n=== {name} ===")
    print(f"{'k':>3} {'sensed':>7} " +
          " ".join(f"{n:>9}" for n, _ in POLICIES) +
          f" | {'dec-dec_est':>11} {'as %F(V)':>9}")
    print("-" * 88)
    for k in ks:
        cfg = dict(k=k, **cfg_base)
        F_V = FacilityCoverageEnv(**cfg, seed=0).max_cells
        vals, sens = {}, None
        for n, p in POLICIES:
            m, s, sn = rollout(cfg, p, eps, 0, rand_field)
            vals[n] = m
            if n == "dec":
                sens = sn
        gap = vals["dec"] - vals["dec_est"]
        print(f"{k:>3} {100*sens:>6.1f}% " +
              " ".join(f"{vals[n]:>9.1f}" for n, _ in POLICIES) +
              f" | {gap:>11.1f} {100*gap/F_V:>8.1f}%")


if __name__ == "__main__":
    ks = [2, 3, 4, 6, 8]
    EPS = 12

    table("uniform demand (unit test: dec_est must equal dec)",
          dict(demand="uniform", **BASE), ks, 8, rand_field=False)

    table("clustered demand, new field every episode "
          "(n_clusters=6, sigma=2.5)",
          dict(demand="clusters", n_clusters=6, cluster_sigma=2.5, **BASE),
          ks, EPS, rand_field=True)

    print("\n=== sharpness sweep at k=6 ===")
    print(f"{'n_cl':>5} {'sigma':>6} " +
          " ".join(f"{n:>9}" for n, _ in POLICIES) + f" | {'dec-dec_est':>11}")
    print("-" * 78)
    for ncl, sig in ((3, 1.5), (3, 2.5), (6, 1.5), (6, 2.5),
                     (10, 2.5), (10, 4.0)):
        cfg = dict(k=6, demand="clusters", n_clusters=ncl,
                   cluster_sigma=sig, **BASE)
        vals = {}
        for n, p in POLICIES:
            m, _, _ = rollout(cfg, p, EPS, 0, rand_field=True)
            vals[n] = m
        print(f"{ncl:>5} {sig:>6.1f} " +
              " ".join(f"{vals[n]:>9.1f}" for n, _ in POLICIES) +
              f" | {vals['dec'] - vals['dec_est']:>11.1f}")
