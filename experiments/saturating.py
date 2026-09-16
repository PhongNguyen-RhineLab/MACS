"""
The saturating suite, five seeds per arm.

This is the sweep that produced the saturating columns of Table 1. Every
arm runs at k in {2,3,4,6} with the geometry of Table 2, and k=4 also runs
MACS-MC, the forced-Monte-Carlo ablation that isolates approximation error
against an exact ground truth available at that team size.

Read the results with the benchmark's own limitation in view: the
saturating suite has coord - dec ~ 0 across every configuration, so it has
near-zero coordination headroom and cannot by itself validate a
credit-assignment claim. Separation here comes from approximation
difficulty in learning, not from task headroom. The islands benchmark
(experiments/facility.py --exp islands) is the one that can carry that
claim.

Labels use the historic short form, so logs already on disk resume.

Usage:
  python -m experiments.saturating --dry-run
  python -m experiments.saturating --smoke
  python -m experiments.saturating --seeds 0 1 2 3 4
  python -m experiments.saturating --aggregate
  python -m experiments.saturating --inspect
"""

import argparse
import sys
from pathlib import Path

from macs import config as CFG
from macs import sweep
from macs.envs import SaturatingCoverageEnv
from macs.sweep import Run

LOG_DIR = "logs/macs_v3_multiseed"

# k=4 is the largest team where the exact Shapley value is cheap enough to
# run alongside the MC estimator, so it is the only configuration where the
# MACS-MC ablation has an exact reference to be measured against.
EXTRA_MODES = {4: ["MACS-MC"]}

COMPARISONS = [
    ("MACS-CLIP/shared/budget", "LOCAL/shared/none",
     "clip vs strongest baseline"),
    ("MACS-CLIP/shared/budget", "MACS/shared/none",
     "clip ablation, identical credits"),
    ("MACS/shared/none", "MACS-MC/shared/none",
     "exact Shapley vs Monte Carlo"),
]


def factory(k, size, horizon):
    return lambda: SaturatingCoverageEnv(size=size, horizon=horizon, k=k,
                                         **CFG.SATURATING_KW)


def plan(ks, seeds, modes, smoke=False):
    runs = []
    for k in ks:
        geom = CFG.SATURATING[k]
        size = 8 if smoke else geom["size"]
        horizon = 15 if smoke else geom["horizon"]
        name = f"Saturating {size}x{size}"
        for mode in list(CFG.SATURATING_MODES) + EXTRA_MODES.get(k, []):
            if modes and mode not in modes:
                continue
            clip = "budget" if mode == "MACS-CLIP" else "none"
            for s in seeds:
                runs.append(Run(name, k, mode, s, "shared", clip, None,
                                factory(k, size, horizon), compact=True))
    return runs


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=CFG.DEFAULT_SEEDS)
    ap.add_argument("--ks", type=int, nargs="+",
                    default=sorted(CFG.SATURATING))
    ap.add_argument("--modes", nargs="+", default=None)
    ap.add_argument("--shapley", default="auto",
                    choices=("auto", "permutation", "subset", "mc"),
                    help="'subset' is the exact backend at k=6, where auto "
                         "falls back to Monte Carlo")
    ap.add_argument("--mc-m", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--log-dir", default=LOG_DIR)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--unpaired", action="store_true")
    args = ap.parse_args(argv)

    log_dir = args.log_dir
    if args.inspect:
        sweep.inspect(log_dir)
        return 0
    if args.aggregate:
        sweep.aggregate(COMPARISONS, log_dir, window=args.window,
                        paired=not args.unpaired,
                        out_path=Path(log_dir) / "saturating_summary.txt")
        return 0

    cfg = dict(CFG.SMOKE_CFG if args.smoke else CFG.TRAIN_CFG)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    if args.smoke:
        log_dir = log_dir + "_smoke"

    runs = plan(args.ks, seeds, args.modes, smoke=args.smoke)
    print(f"\nsaturating: {len(runs)} run(s), ks={args.ks}, seeds={seeds}, "
          f"log_dir={log_dir}")
    sweep.execute(runs, cfg, log_dir, dry_run=args.dry_run,
                  backend=args.shapley, mc_m=args.mc_m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
