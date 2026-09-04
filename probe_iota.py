"""
probe_iota.py -- empirical credit-coupling coefficient.

Assumption 2 (decentralizability) asks whether Q_i* factors through
(s^i, S, a^i).  It does not hold in general, and it does not hold for modular
F either: with F(S) = sum_v w_v,

    phi_i      = sum_{v in P_i \ S} w_v / n_v      (n_v = # agents covering v)
    phibar_i   = sum_{v in P_i \ S} w_v            (agent i alone)
    phibar_i - phi_i = sum_{v in P_i \ S} w_v (1 - 1/n_v)   >= 0,

so the coupling comes directly from arrival-footprint overlap.  Note that
phibar_i is exactly the LOCAL credit, so the per-agent coupling

    iota_i = phi_i^LOC - phi_i   >= 0     (Prop. `ordering`)

is measurable with no extra machinery.  This probe reports its distribution
along reference-policy rollouts.

SCOPE.  iota measures the CREDIT channel only.  Even at iota = 0, Q_i* can
still depend on teammates through the transition channel, since
S' = S u (union_j P(s'^j)) enters agent i's bootstrap.  iota = 0 is therefore
NECESSARY but not sufficient for Assumption 2, and this probe says nothing
about Assumption 3 (IGM).  Report it as credit coupling, not as a bound on
decentralization error.

Normalizations reported per (suite, k, policy):
    iota/r      per-step coupling relative to the team reward that step
    iota/F(V)   coupling in units of the fully-covered objective
Also reported: overlap rate (fraction of rewarding steps with any overlap),
and the LOCAL-DR spread already used by probe_fl, for continuity.

Usage:
    python probe_iota.py --episodes 8
    python probe_iota.py --suite facility --episodes 4
"""

import argparse
import json
import sys
import time

import numpy as np

import macs_main as M
from macs_main import shapley_exact
from macs_fl import FacilityCoverageEnv, shapley_facility
from macs_v3 import SaturatingCoverageEnv
from shapley_subset import shapley_subset
from probe_fl import pol_random, pol_dec, pol_coord


FACILITY = dict(size=24, horizon=60, rho=4.0, patch=0, demand="uniform")
SATURATING = {2: dict(size=12, horizon=40), 3: dict(size=12, horizon=40),
              4: dict(size=12, horizon=40), 6: dict(size=16, horizon=60),
              8: dict(size=16, horizon=60)}

POLICIES = {"dec": pol_dec, "coord": pol_coord, "random": pol_random}


def _credits(env, info, exact_fn):
    """(phi, phi_loc, phi_dr, r) for one step. All length k."""
    S, P, F = info["S_prev"], info["patches"], env.F
    f0 = F(S)
    U = S.copy()
    for pm in P:
        U |= pm
    r = F(U) - f0
    phi_loc = np.array([F(S | pm) - f0 for pm in P])
    phi_dr = np.array([
        F(U) - F(np.logical_or.reduce([S] + [P[j] for j in range(env.k)
                                             if j != i]))
        for i in range(env.k)])
    phi = exact_fn(env, info)
    return phi, phi_loc, phi_dr, r


def _exact_facility(env, info):
    return shapley_facility(info["S_prev"], info["patches"], env.Ffl)


def _exact_subset(env, info):
    return shapley_subset(info["S_prev"], info["patches"], env.F)


def rollout(make_env, policy, episodes, seed, exact_fn, check=False):
    iota, iota_over_r, spread, ov = [], [], [], []
    finals = []
    for e in range(episodes):
        rng = np.random.default_rng(seed * 7919 + e)
        env = make_env(seed * 1000 + e)
        env.reset()
        for _ in range(env.horizon):
            _, r_team, done, info = env.step(policy(env, rng))
            if r_team <= 1e-9:
                continue
            phi, loc, dr, r = _credits(env, info, exact_fn)
            if check:
                assert abs(phi.sum() - r) < 1e-6, "efficiency violated"
                assert np.all(dr <= phi + 1e-9) and np.all(phi <= loc + 1e-9)
            d = loc - phi                      # iota_i, non-negative
            iota.extend(d.tolist())
            iota_over_r.extend((d / r).tolist())
            spread.append(float((loc.sum() - dr.sum()) / r))
            ov.append(1.0 if d.max() > 1e-9 else 0.0)
        finals.append(env.total_coverage)
    a = np.asarray(iota)
    b = np.asarray(iota_over_r)
    return dict(
        final=float(np.mean(finals)),
        F_V=float(env.F_total()),
        n_steps=int(len(ov)),
        overlap_rate=float(np.mean(ov)) if ov else 0.0,
        iota_mean=float(a.mean()) if a.size else 0.0,
        iota_p90=float(np.quantile(a, 0.90)) if a.size else 0.0,
        iota_p99=float(np.quantile(a, 0.99)) if a.size else 0.0,
        iota_max=float(a.max()) if a.size else 0.0,
        rel_mean=float(b.mean()) if b.size else 0.0,
        rel_p90=float(np.quantile(b, 0.90)) if b.size else 0.0,
        rel_max=float(b.max()) if b.size else 0.0,
        locdr_spread=float(np.mean(spread)) if spread else 0.0,
    )


def suite_facility(ks, episodes, seed, check):
    out = {}
    for k in ks:
        def mk(s, k=k):
            return FacilityCoverageEnv(k=k, seed=s, **FACILITY)
        for name, pol in POLICIES.items():
            t0 = time.time()
            out[f"facility|k={k}|{name}"] = rollout(
                mk, pol, episodes, seed, _exact_facility, check)
            print(f"  facility k={k} {name:6s} "
                  f"{time.time() - t0:5.1f}s", flush=True)
    return out


def suite_saturating(ks, episodes, seed, check):
    out = {}
    for k in ks:
        cfg = SATURATING[k]
        def mk(s, k=k, cfg=cfg):
            return SaturatingCoverageEnv(k=k, patch=1, seed=s, **cfg)
        for name, pol in POLICIES.items():
            t0 = time.time()
            out[f"saturating|k={k}|{name}"] = rollout(
                mk, pol, episodes, seed, _exact_subset, check)
            print(f"  saturating k={k} {name:6s} "
                  f"{time.time() - t0:5.1f}s", flush=True)
    return out


def table(res):
    hdr = (f"{'suite/k/policy':28s} {'steps':>6s} {'ovl%':>6s} "
           f"{'i_mean':>8s} {'i_p90':>7s} {'i_max':>7s} "
           f"{'i/r_mean':>9s} {'i/r_p90':>8s} {'i/F(V)':>8s} {'loc-dr':>7s}")
    print(hdr)
    print("-" * len(hdr))
    for kk, v in res.items():
        print(f"{kk:28s} {v['n_steps']:6d} {100*v['overlap_rate']:6.1f} "
              f"{v['iota_mean']:8.4f} {v['iota_p90']:7.4f} "
              f"{v['iota_max']:7.4f} {v['rel_mean']:9.4f} "
              f"{v['rel_p90']:8.4f} "
              f"{v['iota_mean']/v['F_V']:8.5f} {v['locdr_spread']:7.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suite", default="both",
                    choices=["both", "facility", "saturating"])
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--check", action="store_true",
                    help="assert efficiency and DR<=phi<=LOC every step")
    ap.add_argument("--out", default="logs/iota.json")
    a = ap.parse_args()

    res = {}
    if a.suite in ("both", "saturating"):
        res.update(suite_saturating(a.ks, a.episodes, a.seed, a.check))
    if a.suite in ("both", "facility"):
        res.update(suite_facility(a.ks, a.episodes, a.seed, a.check))
    print()
    table(res)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()