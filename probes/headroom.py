"""
Headroom probes: how much room does a benchmark leave for coordination?

Seven probe scripts (probe_blind, probe_est, probe_cells, probe_horizon,
probe_ceiling, probe_free, probe_svalue) were the same program with
different constants: run a ladder of non-learning policies over a grid of
configurations and print a table of gaps. They are one script here, selected
by --suite.

The three gaps that decide whether a benchmark can test a credit rule:

    S-value        dec - blind        what SHARING the coverage set buys.
                   Theorem `sufficient-statistic` predicts this is positive;
                   if it is ~0 the augmented state is not doing anything and
                   the O9 ablation has nothing to measure.
    planning       plan - dec         what LOOKING AHEAD buys over a myopic
                   greedy. Zero on uniform demand: once S_t is shared,
                   myopic greedy is already near-optimal.
    assignment     plan - plan_self   what AGREEING WHO GOES WHERE buys.
                   This is the one a credit rule can plausibly improve, and
                   the reason the islands benchmark exists: the saturating
                   and uniform-facility suites have coord - dec ~ 0, so no
                   credit rule can win on them asymptotically and a
                   separation measured there is about approximation
                   difficulty in learning, not about task headroom.

Suites:
  facility     the S-value ladder over k, on uniform and clustered demand
  islands      the full island ladder including the planners
  saturating   the paper's saturating suite, for comparison
  horizon      the S-value ladder against the episode horizon
  sizes        the same against grid size
  demand       true demand field vs an estimated one (dec vs dec_est)

Usage:
  python -m probes.headroom --suite facility
  python -m probes.headroom --suite islands --ks 3 4 6 --episodes 10
  python -m probes.headroom --suite horizon --json logs/headroom.json
"""

import argparse
import json
import sys
from pathlib import Path

from macs import config as CFG
from macs import policies as P
from macs.envs import FacilityCoverageEnv, IslandEnv, SaturatingCoverageEnv
from macs.rollout import factory, ladder

SUITES = ("facility", "islands", "saturating", "horizon", "sizes", "demand")

#: gaps worth naming, as (label, better, worse)
GAPS = (("S-value", "dec", "blind"),
        ("planning", "plan", "dec"),
        ("assignment", "plan", "plan_self"),
        ("coord-dec", "coord", "dec"),
        ("estimation", "dec", "dec_est"))


def _facility(k, **over):
    cfg = dict(CFG.FACILITY, k=k)
    cfg.update(over)
    return factory(FacilityCoverageEnv, cfg)


def _islands(k, **over):
    cfg = dict(CFG.ISLANDS, k=k)
    cfg.update(over)
    return factory(IslandEnv, cfg)


def _saturating(k):
    return factory(SaturatingCoverageEnv, CFG.saturating_cfg(k))


def rows(suite, ks, episodes, seed):
    """[(row label, env_factory, [(policy name, policy)])]"""
    if suite == "facility":
        return [(f"k={k}", _facility(k), P.LADDER) for k in ks] + \
               [(f"k={k} clustered",
                 _facility(k, demand="clusters", n_clusters=8), P.LADDER)
                for k in ks]
    if suite == "islands":
        return [(f"k={k}", _islands(k), P.ISLAND_LADDER) for k in ks]
    if suite == "saturating":
        return [(f"k={k}", _saturating(k), P.LADDER)
                for k in ks if k in CFG.SATURATING]
    if suite == "horizon":
        return [(f"k={k} H={H}", _facility(k, horizon=H), P.LADDER)
                for k in ks for H in (20, 40, 60, 90)]
    if suite == "sizes":
        return [(f"k={k} {n}x{n}", _facility(k, size=n), P.LADDER)
                for k in ks for n in (16, 24, 32)]
    if suite == "demand":
        return [(f"k={k} uniform", _facility(k), P.LADDER) for k in ks] + \
               [(f"k={k} clustered",
                 _facility(k, demand="clusters", n_clusters=8), P.LADDER)
                for k in ks]
    raise ValueError(suite)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=SUITES, default="facility")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write results here")
    args = ap.parse_args(argv)

    table = rows(args.suite, args.ks, args.episodes, args.seed)
    names = []
    for _, _, pols in table:
        for name, _ in pols:
            if name not in names:
                names.append(name)

    head = f"{'config':<18}" + "".join(f"{n:>11}" for n in names)
    gaps = [g for g in GAPS if g[1] in names and g[2] in names]
    head += "  |" + "".join(f"{g[0]:>12}" for g in gaps)
    print(f"\n=== {args.suite} ({args.episodes} episodes, seed "
          f"{args.seed}, percent of F(V)) ===")
    print(head)
    print("-" * len(head))

    out = {}
    for label, fac, pols in table:
        res = ladder(fac, pols, args.episodes, args.seed, normalize=True)
        out[label] = {n: v[0] for n, v in res.items()}
        line = f"{label:<18}"
        for n in names:
            line += (f"{res[n][0]:>10.1f}%" if n in res else f"{'--':>11}")
        line += "  |"
        for g in gaps:
            line += (f"{res[g[1]][0] - res[g[2]][0]:>12.1f}"
                     if g[1] in res and g[2] in res else f"{'--':>12}")
        print(line)

    print("\nA credit rule can only be tested where the assignment gap is "
          "clearly positive.\nWhere coord-dec is ~0 the benchmark has no "
          "coordination headroom, and any\nseparation measured on it is "
          "about approximation difficulty in learning.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
