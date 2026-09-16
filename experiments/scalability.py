"""
Scalability: does the Shapley credit computation become the bottleneck?

Trains the full method list at k in {8, 10} and records, per run, the
wall-clock share spent inside the credit rule, seconds per episode, and
peak resident memory.

The cost question the table answers: exact Shapley by subset enumeration
costs 2^k evaluations of F, SHARED across all k agents -- not k * 2^(k-1),
which was the figure an earlier draft used to justify Monte Carlo. At k=6
the subset form is roughly 6.6x cheaper than MC with m=64, so the
justification inverts; by k=10 the 1024 subset evaluations are back above
MC's 640, which is the crossover this sweep measures directly.

Timing is by dependency injection: `train(credit_fn=...)` takes a wrapped
dispatcher. The previous version monkey-patched the module-level dispatcher
and restored it afterwards, which meant an exception mid-run left the
process with a timing wrapper still installed.

Peak RSS is process-wide and monotone within a session, so the per-method
figure only means something when each run is launched in a fresh process.
Run one method per invocation for clean memory numbers.

Usage:
  python -m experiments.scalability --dry-run
  python -m experiments.scalability --methods MACS --ks 8
  python -m experiments.scalability --aggregate
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from macs import config as CFG
from macs import credits as C
from macs import runlog, sweep
from macs.envs import SaturatingCoverageEnv
from macs.sweep import Run

LOG_DIR = "logs/macs_v3_scale"
CONFIGS = {8: dict(size=20, horizon=80), 10: dict(size=20, horizon=80)}
METHODS = ["SHARED", "LOCAL", "DR", "VDN", "QMIX", "MACS", "MACS-CLIP"]
SMOKE_CFG = dict(n_episodes=20, warmup=200, eval_every=10, n_eval=2,
                 log_every=10, target_every=10)


class CreditTimer:
    """Wraps the credit dispatcher and accumulates wall time and calls."""

    def __init__(self):
        self.seconds = 0.0
        self.calls = 0

    def __call__(self, *a, **kw):
        t0 = time.perf_counter()
        out = C.compute_credits(*a, **kw)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return out


def peak_rss_mb():
    """Peak resident set size of this process in MB (Linux reports KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def factory(k, size, horizon):
    return lambda: SaturatingCoverageEnv(size=size, horizon=horizon, k=k,
                                         **CFG.SATURATING_KW)


def plan(ks, seeds, methods, smoke=False):
    runs = []
    for k in ks:
        geom = CONFIGS[k]
        size = 12 if smoke else geom["size"]
        horizon = 15 if smoke else geom["horizon"]
        name = f"Saturating {size}x{size}"
        for mode in methods:
            clip = "budget" if mode == "MACS-CLIP" else "none"
            for s in seeds:
                runs.append(Run(name, k, mode, s, "shared", clip, None,
                                factory(k, size, horizon), compact=True))
    return runs


def execute(runs, cfg, log_dir, backend="auto", mc_m=64, dry_run=False):
    from macs.train import train
    n_ep, log_ev = cfg["n_episodes"], cfg["log_every"]
    todo = [r for r in runs
            if not runlog.is_complete(r.label, log_dir, n_ep, log_ev)]
    print(f"Plan: {len(runs)} runs, {len(runs) - len(todo)} complete, "
          f"{len(todo)} to run.")
    for i, r in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {r.label}")
        if dry_run:
            continue
        timer = CreditTimer()
        t0 = time.perf_counter()
        train(r.factory, env_name=r.env_name, mode=r.mode, seed=r.seed,
              clip=r.clip, label=r.label, log_dir=log_dir,
              backend=backend, mc_m=mc_m, credit_fn=timer, **cfg)
        wall = time.perf_counter() - t0
        stats = dict(k=r.k, method=r.mode, seed=r.seed, label=r.label,
                     wall_seconds=wall,
                     sec_per_episode=wall / cfg["n_episodes"],
                     credit_seconds=timer.seconds,
                     credit_calls=timer.calls,
                     credit_share=(timer.seconds / wall if wall else 0.0),
                     peak_rss_mb=peak_rss_mb())
        out = Path(log_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"runtime_{r.label}.json").write_text(
            json.dumps(stats, indent=2))
        print(f"  credit {100 * stats['credit_share']:.1f}% of wall, "
              f"{stats['sec_per_episode']:.3f} s/episode, "
              f"peak {stats['peak_rss_mb']:.0f} MB")


def aggregate(log_dir=LOG_DIR, window=10):
    perf = {}
    for p in sorted(Path(log_dir).glob("*seed*.json")):
        if p.name.startswith("runtime_"):
            continue
        d = runlog.load(p)
        c = d["config"]
        ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
        if len(ev) >= window:
            perf.setdefault((c["k"], c["method"]), []).append(
                float(np.mean(ev[-window:])))

    runt = {}
    for p in sorted(Path(log_dir).glob("runtime_*.json")):
        d = json.loads(p.read_text())
        runt.setdefault((d["k"], d["method"]), []).append(d)

    if not perf and not runt:
        print(f"Nothing to aggregate in {log_dir}.")
        return

    for k in sorted({k for k, _ in list(perf) + list(runt)}):
        print(f"\n=== k = {k} ===")
        print(f"{'method':<10} {'final F':>16} {'s/episode':>10} "
              f"{'credit %':>9} {'peak MB':>8}")
        for m in METHODS:
            pv, rv = perf.get((k, m)), runt.get((k, m))
            if not pv and not rv:
                continue
            fstr = "--"
            if pv:
                a = np.array(pv)
                sd = a.std(ddof=1) if len(a) > 1 else 0.0
                fstr = f"{a.mean():7.2f} +- {sd:5.2f}"
            spe = (f"{np.mean([r['sec_per_episode'] for r in rv]):.3f}"
                   if rv else "--")
            csh = (f"{100 * np.mean([r['credit_share'] for r in rv]):.1f}"
                   if rv else "--")
            mem = (f"{np.mean([r['peak_rss_mb'] for r in rv]):.0f}"
                   if rv else "--")
            print(f"{m:<10} {fstr:>16} {spe:>10} {csh:>9} {mem:>8}")
    print("\nNote: peak RSS is process-wide and monotone within a session; "
          "the per-method figure is only meaningful when each run is "
          "launched in a fresh process.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", type=int, nargs="+", default=sorted(CONFIGS))
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--shapley", default="auto",
                    choices=("auto", "permutation", "subset", "mc"))
    ap.add_argument("--mc-m", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--log-dir", default=LOG_DIR)
    args = ap.parse_args(argv)

    if args.aggregate:
        aggregate(args.log_dir)
        return 0

    cfg = dict(SMOKE_CFG if args.smoke else CFG.TRAIN_CFG)
    seeds = args.seeds[:1] if args.smoke else args.seeds
    log_dir = args.log_dir + ("_smoke" if args.smoke else "")
    runs = plan(args.ks, seeds, args.methods, smoke=args.smoke)
    execute(runs, cfg, log_dir, backend=args.shapley, mc_m=args.mc_m,
            dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
