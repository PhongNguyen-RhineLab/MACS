"""
Where can credit rules disagree? numpy only; no training.

Every efficient rule inside the [DR, LOCAL] bracket satisfies the paper's
learning and equilibrium results (Shapley and INT among them). Two rules can
only learn differently on steps where their credits differ, and how much
they can differ is bounded by two training-free quantities, measured on
scripted behaviour:

    kappa    (sum_t sum_i LOC_i - r_t) / sum_t r_t    LOCAL's over-payment
    rho_DR   (sum_t r_t - sum_i DR_i) / sum_t r_t     DR's under-payment

Both are zero exactly when no two new footprints interact. Also reported:

    d(INT, Sh)   sum_t sum_i |INT_i - Sh_i| / sum_t r_t   how far the
                 cheap efficient rule is from Shapley

(The same distance for the endpoints is not reported because it is already
in the table: DR <= Sh <= LOC per agent, so d(DR, Sh) = rho_DR and
d(LOC, Sh) = kappa exactly.)

and the reference floors (random, dec, plan_self, plan as % of F(V)), so
the same table carries the headroom diagnostic of probes.headroom.

The 'crowded' rows put more agents than islands (8) on the map. Agents must
share islands there, so footprints interact and DR is expected to under-pay;
that is the regime in which an efficient rule and DR can separate.

Usage:
  python -m probes.sandwich                        # islands k = 3..16, decoy
  python -m probes.sandwich --ks 6 12 --episodes 10 --json out.json
"""

import argparse
import json
import sys

import numpy as np

from macs import config as CFG
from macs import credits as C
from macs import policies as P
from macs import shapley
from macs.envs import IslandEnv
from macs.rollout import factory, rollout

GAMMA = 0.99     # the discount of every training run; the fixed-point bound
                 # of the paper is stated in discounted units

POLICIES = {"islands": (("dec", P.dec), ("plan_self", P.plan_self),
                        ("plan", P.plan)),
            "decoy": (("dec", P.dec), ("plan_self_v", P.plan_self_v),
                      ("plan_v", P.plan_v))}


def credit_mass(cfg, k, policy, episodes, seed):
    tot = dict(r=0.0, over=0.0, under=0.0, d_int=0.0, d_dr=0.0, d_loc=0.0,
               steps=0, shared_steps=0, gr=0.0, gover=0.0, gunder=0.0)
    for e in range(episodes):
        env = IslandEnv(k=k, **cfg, seed=seed * 1000 + e)
        env.reset()
        pol = policy(env)
        rng = np.random.default_rng(seed * 7919 + e)
        for t in range(env.horizon):
            _, r, done, info = env.step(pol(env, rng))
            if r > 0:
                loc, dr, rr = C.marginal_bounds(info, env.F)
                sh = shapley.compute(info["S_prev"], info["patches"], env.F)
                it = C.interpolated_credit(loc, dr, rr)
                tot["r"] += rr
                tot["over"] += loc.sum() - rr
                tot["under"] += rr - dr.sum()
                tot["d_int"] += np.abs(it - sh).sum()
                tot["d_dr"] += np.abs(dr - sh).sum()
                tot["d_loc"] += np.abs(loc - sh).sum()
                g = GAMMA ** t
                tot["gr"] += g * rr
                tot["gover"] += g * (loc.sum() - rr)
                tot["gunder"] += g * (rr - dr.sum())
                tot["steps"] += 1
                tot["shared_steps"] += int(loc.sum() - dr.sum() > 1e-9)
            if done:
                break
    R = max(tot["r"], 1e-12)
    G = max(tot["gr"], 1e-12)
    return dict(kappa=100 * tot["over"] / R, rho_dr=100 * tot["under"] / R,
                kappa_disc=100 * tot["gover"] / G,
                rho_dr_disc=100 * tot["gunder"] / G,
                d_int=100 * tot["d_int"] / R, d_dr=100 * tot["d_dr"] / R,
                d_loc=100 * tot["d_loc"] / R,
                interact=100 * tot["shared_steps"] / max(tot["steps"], 1))


def floors(cfg, k, names, episodes, seed):
    make = factory(IslandEnv, dict(k=k, **cfg))
    pols = dict(P.ISLAND_LADDER)
    return {n: float(rollout(make, pols[n], episodes, seed,
                             normalize=True)["coverage"].mean())
            for n in ("random",) + names}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("--ks", type=int, nargs="+",
                    default=[3, 4, 6, 8, 12, 16])
    ap.add_argument("--decoy-k", type=int, nargs="*", default=[6])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    rows = []
    suites = [("islands", CFG.ISLANDS, k) for k in args.ks]
    suites += [("decoy", CFG.DECOY, k) for k in args.decoy_k]
    hdr = (f"{'suite':>8} {'k':>3} {'policy':>11} | {'kappa':>6} "
           f"{'rho_DR':>6} {'inter%':>6} | {'d(INT,Sh)':>9} | {'F%':>5}")
    print(hdr)
    print("-" * len(hdr))
    for suite, cfg, k in suites:
        names = tuple(n for n, _ in POLICIES[suite])
        fl = floors(cfg, k, names, args.episodes, args.seed)
        for name, pol in POLICIES[suite]:
            m = credit_mass(cfg, k, pol, args.episodes, args.seed)
            rows.append(dict(suite=suite, k=k, policy=name, floor=fl[name],
                             random=fl["random"], **m))
            print(f"{suite:>8} {k:>3} {name:>11} | {m['kappa']:>5.1f}% "
                  f"{m['rho_dr']:>5.1f}% {m['interact']:>5.1f}% | "
                  f"{m['d_int']:>8.2f}% | {fl[name]:>5.1f}")
        print(f"{'':>8} {k:>3} {'random':>11} | {'':>6} {'':>6} {'':>6} | "
              f"{'':>9} | {fl['random']:>5.1f}")
    print("\nkappa, rho_DR, d(INT,Sh): % of the team reward collected; "
          "inter%: % of rewarded steps\non which at least one agent's "
          "credit depends on arrival order; F%: coverage, % of F(V).")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rows, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
