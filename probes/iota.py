"""
The credit coupling coefficient iota.

Assumption 2 of the draft asks that an agent's credit not depend on its
teammates. An earlier version claimed a modular F satisfies it; that is
false -- Shapley credit depends on teammates whenever arrival footprints
overlap, and a modular F overlaps as readily as any other. What is true is
that the dependence is MEASURABLE, so the assumption can be replaced by a
reported quantity:

    iota_i = phi^LOC_i - phi_i    >= 0 for submodular F

the amount of an agent's first-arrival marginal that Shapley redistributes
to teammates because their footprints overlapped. iota = 0 is necessary but
not sufficient for Assumption 2.

Reported per configuration:

    ovl%       fraction of rewarding steps where any iota_i > 0
    i_mean     mean iota over agent-steps
    i_p90      90th percentile, because the mean hides a heavy tail
    i/r        iota relative to the step reward
    i_mass     total iota over total reward across the run -- the single
               number to quote
    loc-dr     (sum LOC - sum DR) / r, the full spread the credit rules can
               occupy. Shapley sits inside it by DR <= phi <= LOC.

The reference policies matter on islands in a way they do not elsewhere:
plan_self sends every agent to its own nearest island and piles them on top
of each other; plan claims islands and spreads them. The learned arms sit
near plan_self, so its overlap rate is the figure that bears on whether
MACS and LOCAL can differ at all.

--check asserts efficiency and the DR <= phi <= LOC ordering at every step,
which is the cheap way to confirm the exact backend is really in use.

Usage:
  python -m probes.iota --suite islands --ks 3 4 6
  python -m probes.iota --suite saturating --check
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from macs import config as CFG
from macs import policies as P
from macs import shapley
from macs.envs import FacilityCoverageEnv, IslandEnv, SaturatingCoverageEnv

SUITES = ("facility", "saturating", "islands")

BASE_POLICIES = (("random", P.random_policy), ("dec", P.dec),
                 ("coord", P.coord))


def step_credits(env, info):
    """(phi, phi_LOC, phi_DR, r) for one step. All length k."""
    S, patches, F = info["S_prev"], info["patches"], env.F
    f0 = F(S)
    U = S.copy()
    for pm in patches:
        U |= pm
    r = F(U) - f0
    phi_loc = np.array([F(S | pm) - f0 for pm in patches])
    phi_dr = np.array([
        F(U) - F(np.logical_or.reduce(
            [S] + [patches[j] for j in range(env.k) if j != i]))
        for i in range(env.k)])
    # backend="auto" routes facility-location F to the closed form and
    # everything else to an exact enumeration, so this is exact throughout.
    phi = shapley.compute(S, patches, F, backend="auto")
    return phi, phi_loc, phi_dr, r


def measure(env_factory, policy, episodes, seed, check=False):
    iota, rel, spread, overlap, finals = [], [], [], [], []
    tot_iota = tot_r = 0.0
    env = None
    for e in range(episodes):
        rng = np.random.default_rng(seed * 7919 + e)
        env = env_factory(e, seed * 1000 + e)
        env.reset()
        pol = policy(env)
        for _ in range(env.horizon):
            _, r_team, _, info = env.step(pol(env, rng))
            if r_team <= 1e-9:
                continue
            phi, loc, dr, r = step_credits(env, info)
            if check:
                assert abs(phi.sum() - r) < 1e-6, "efficiency violated"
                assert np.all(dr <= phi + 1e-9), "DR > Shapley"
                assert np.all(phi <= loc + 1e-9), "Shapley > LOCAL"
            d = loc - phi                       # iota_i, non-negative
            tot_iota += float(d.sum())
            tot_r += float(r)
            iota.extend(d.tolist())
            rel.extend((d / r).tolist())
            spread.append(float((loc.sum() - dr.sum()) / r))
            overlap.append(1.0 if d.max() > 1e-9 else 0.0)
        finals.append(env.total_coverage)
    a, b = np.asarray(iota), np.asarray(rel)
    return dict(
        final=float(np.mean(finals)),
        F_V=float(env.F_total()),
        n_steps=int(len(overlap)),
        overlap_rate=float(np.mean(overlap)) if overlap else 0.0,
        iota_mean=float(a.mean()) if a.size else 0.0,
        iota_p90=float(np.quantile(a, 0.90)) if a.size else 0.0,
        iota_max=float(a.max()) if a.size else 0.0,
        rel_mean=float(b.mean()) if b.size else 0.0,
        rel_p90=float(np.quantile(b, 0.90)) if b.size else 0.0,
        locdr_spread=float(np.mean(spread)) if spread else 0.0,
        iota_mass=float(tot_iota / tot_r) if tot_r > 0 else 0.0,
    )


def suite_rows(suite, ks):
    """[(row label, env_factory, [(policy name, policy)])]"""
    out = []
    for k in ks:
        if suite == "facility":
            def make(e, s, k=k):
                env = FacilityCoverageEnv(k=k, **CFG.FACILITY, seed=s)
                env.rho_hint = CFG.FACILITY["rho"]
                return env
            pols = BASE_POLICIES
        elif suite == "islands":
            def make(e, s, k=k):
                env = IslandEnv(k=k, **CFG.ISLANDS, seed=s)
                env.rho_hint = CFG.ISLANDS["rho"]
                return env
            pols = BASE_POLICIES + (("plan_self", P.plan_self),
                                    ("plan", P.plan))
        else:
            if k not in CFG.SATURATING:
                continue
            def make(e, s, k=k):
                return SaturatingCoverageEnv(**CFG.saturating_cfg(k), seed=s)
            pols = BASE_POLICIES
        for name, pol in pols:
            out.append((f"{suite}|k={k}|{name}", make, pol))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=SUITES, default="islands")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="assert efficiency and DR <= phi <= LOC every step")
    ap.add_argument("--json", default="logs/iota.json")
    args = ap.parse_args(argv)

    res = {}
    for label, make, pol in suite_rows(args.suite, args.ks):
        t0 = time.time()
        res[label] = measure(make, pol, args.episodes, args.seed, args.check)
        print(f"  {label:<28} {time.time() - t0:5.1f}s", flush=True)

    hdr = (f"\n{'suite/k/policy':<28} {'steps':>6} {'ovl%':>6} "
           f"{'i_mean':>8} {'i_p90':>8} {'i_max':>8} {'i/r':>8} "
           f"{'loc-dr':>8} {'i_mass':>8}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for label, v in res.items():
        print(f"{label:<28} {v['n_steps']:>6} {100 * v['overlap_rate']:>6.1f} "
              f"{v['iota_mean']:>8.4f} {v['iota_p90']:>8.4f} "
              f"{v['iota_max']:>8.4f} {v['rel_mean']:>8.4f} "
              f"{v['locdr_spread']:>8.4f} {v['iota_mass']:>8.4f}")
    print("\niota = 0 is necessary but not sufficient for Assumption 2. "
          "i_mass is the\nnumber to quote: the share of total reward that "
          "Shapley redistributes.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
