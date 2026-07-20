"""
run_scalability.py — scalability to k in {8, 10} plus runtime/memory report.

Covers two "Rat nen lam" checklist items:

  1. SCALABILITY. Saturating coverage on a 20x20 grid (25 regions of 4x4,
     F(V) = 200) with k = 8 and k = 10 agents, horizon 80. Exact Shapley
     would need 8! = 40,320 and 10! = 3.6M permutations per step, so MACS
     and MACS-CLIP run in Monte Carlo mode (m = 64 -> 512 and 640 F
     evaluations per env step). The MC path is justified by the k = 4
     sweep of run_mc_sweep.py, where exact ground truth exists.

  2. RUNTIME / MEMORY OVERHEAD. Every run records wall-clock time, peak
     RSS (resource.getrusage), and — separately — the cumulative time
     spent inside credit computation, obtained by timing the
     compute_credits dispatcher. This isolates the cost of Shapley from
     the cost of the DQN pipeline, which is what a reviewer will ask:
     "how much does the credit rule itself cost?" The answer is written to
     logs/macs_v3_scale/runtime_<label>.json and summarized by
     --aggregate as a per-method table (seconds/episode, credit share %,
     peak MB).

Default methods exclude QMIX (its mixer input grows with k and it already
underperformed at small k); include it explicitly with --methods if wanted.

Usage:
  python run_scalability.py --dry-run
  python run_scalability.py --seeds 0 1 2          # 3 seeds to start
  python run_scalability.py --ks 8                 # one team size
  python run_scalability.py --aggregate
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

import macs_v3 as V3
import macs_main as M
from macs_v3 import SaturatingCoverageEnv

LOG_DIR = "logs/macs_v3_scale"
CONFIGS = [
    dict(k=8,  size=20, horizon=80),
    dict(k=10, size=20, horizon=80),
]
METHODS = ["SHARED", "LOCAL", "DR", "VDN", "MACS", "MACS-CLIP"]
TRAIN_CFG = dict(n_episodes=4000, eval_every=100, log_every=100)
SMOKE_CFG = dict(n_episodes=20, warmup=200, eval_every=10, n_eval=2,
                 log_every=10, target_every=10)


# =============================================================================
# Credit-timing instrumentation
# =============================================================================

class CreditTimer:
    """Wraps the compute_credits dispatcher and accumulates wall time."""

    def __init__(self):
        self.seconds = 0.0
        self.calls = 0

    def install(self):
        inner = V3.compute_credits_v3

        def timed(mode, r_team, info, F, k, mc_m=V3.MC_M, rng=None):
            t0 = time.perf_counter()
            out = inner(mode, r_team, info, F, k, mc_m=mc_m, rng=rng)
            self.seconds += time.perf_counter() - t0
            self.calls += 1
            return out

        M.compute_credits = timed

    @staticmethod
    def uninstall():
        M.compute_credits = V3.compute_credits_v3


def peak_rss_mb():
    """Peak resident set size of this process in MB (Linux: KB units)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# =============================================================================
# Runner
# =============================================================================

def is_complete(label, cfg, log_dir=LOG_DIR):
    p = Path(log_dir) / f"{label}.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return len(d["per_block"]["episode"]) >= \
            cfg["n_episodes"] // cfg["log_every"]
    except (json.JSONDecodeError, KeyError):
        return False


def build_plan(seeds, ks, methods, smoke=False):
    for c in CONFIGS:
        if c["k"] not in ks:
            continue
        size = 12 if smoke else c["size"]
        horizon = 15 if smoke else c["horizon"]
        env_name = f"Saturating {size}x{size}"
        for mode in methods:
            for s in seeds:
                yield dict(env_name=env_name, k=c["k"], size=size,
                           horizon=horizon, mode=mode, seed=s,
                           label=f"{env_name.replace(' ', '_')}"
                                 f"_k{c['k']}_{mode}_seed{s}")


def execute(plan, cfg, log_dir=LOG_DIR):
    plan = list(plan)
    todo = [r for r in plan if not is_complete(r["label"], cfg, log_dir)]
    print(f"Plan: {len(plan)} runs, {len(plan) - len(todo)} complete, "
          f"{len(todo)} to run.")
    for i, r in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {r['label']}")
        factory = (lambda r=r: SaturatingCoverageEnv(
            size=r["size"], horizon=r["horizon"], k=r["k"], patch=1,
            region=4, cap_frac=0.5))

        timer = CreditTimer()
        timer.install()
        rss0 = peak_rss_mb()
        t0 = time.perf_counter()
        M.train_macs(factory, env_name=r["env_name"], mode=r["mode"],
                     seed=r["seed"], label=r["label"], log_dir=log_dir,
                     **cfg)
        wall = time.perf_counter() - t0
        timer.uninstall()

        rt = dict(label=r["label"], method=r["mode"], k=r["k"],
                  size=r["size"], n_episodes=cfg["n_episodes"],
                  wall_seconds=round(wall, 1),
                  sec_per_episode=round(wall / cfg["n_episodes"], 4),
                  credit_seconds=round(timer.seconds, 1),
                  credit_share=round(timer.seconds / wall, 4),
                  credit_calls=timer.calls,
                  peak_rss_mb=round(peak_rss_mb(), 1),
                  rss_before_mb=round(rss0, 1))
        with open(Path(log_dir) / f"runtime_{r['label']}.json", "w") as f:
            json.dump(rt, f, indent=2)
        print(f"  wall {wall:.0f}s | credit share "
              f"{100 * rt['credit_share']:.1f}% | peak RSS "
              f"{rt['peak_rss_mb']:.0f} MB")


# =============================================================================
# Aggregation
# =============================================================================

def aggregate(log_dir=LOG_DIR, window=10):
    # performance
    perf = {}
    for p in sorted(Path(log_dir).glob("*seed*.json")):
        if p.name.startswith("runtime_"):
            continue
        with open(p) as f:
            d = json.load(f)
        c = d["config"]
        ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
        if len(ev) >= window:
            perf.setdefault((c["k"], c["method"]), []).append(
                float(np.mean(ev[-window:])))

    # runtime
    runt = {}
    for p in sorted(Path(log_dir).glob("runtime_*.json")):
        with open(p) as f:
            d = json.load(f)
        runt.setdefault((d["k"], d["method"]), []).append(d)

    if not perf and not runt:
        print(f"Nothing to aggregate in {log_dir}.")
        return

    ks = sorted({k for k, _ in list(perf) + list(runt)})
    for k in ks:
        print(f"\n=== k = {k} ===")
        print(f"{'method':<10} {'final F':>16} {'s/episode':>10} "
              f"{'credit %':>9} {'peak MB':>8}")
        for m in METHODS + ["QMIX"]:
            pv = perf.get((k, m))
            rv = runt.get((k, m))
            if not pv and not rv:
                continue
            fstr = "--"
            if pv:
                a = np.array(pv)
                sd = a.std(ddof=1) if len(a) > 1 else 0.0
                fstr = f"{a.mean():7.2f} +- {sd:5.2f}"
            spe = f"{np.mean([r['sec_per_episode'] for r in rv]):.3f}" \
                if rv else "--"
            csh = f"{100 * np.mean([r['credit_share'] for r in rv]):.1f}" \
                if rv else "--"
            mem = f"{np.mean([r['peak_rss_mb'] for r in rv]):.0f}" \
                if rv else "--"
            print(f"{m:<10} {fstr:>16} {spe:>10} {csh:>9} {mem:>8}")
    print("\nNote: peak RSS is process-wide and monotone within a session; "
          "the per-method figure is only meaningful when each run is "
          "launched in a fresh process (run one method per invocation for "
          "clean memory numbers).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 10])
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()

    if args.aggregate:
        aggregate()
        sys.exit(0)

    cfg = SMOKE_CFG if args.smoke else TRAIN_CFG
    plan = list(build_plan(args.seeds, args.ks, args.methods,
                           smoke=args.smoke))
    if args.dry_run:
        for r in plan:
            print(r["label"])
        print(f"\n{len(plan)} runs.")
        sys.exit(0)

    if not M.TORCH_OK:
        print("torch not available; cannot train.")
        sys.exit(1)

    execute(plan, cfg)
    aggregate()
