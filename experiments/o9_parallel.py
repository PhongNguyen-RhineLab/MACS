"""
O9 at scale: 25 seeds x 2 arms, run in parallel.

The augmented-state ablation with enough seeds to make a paired claim.
A paired sign-flip test has smallest attainable two-sided p = 2/2^n, so 5
seeds can never reach 0.05 and 6 is the minimum; 25 gives room to report an
effect size rather than a bare threshold.

Each run gets a fresh process (spawn + maxtasksperchild=1) so torch state
and the replay buffer are released between runs rather than accumulating.

The environment factory seeds per CALL, not per run: the trainer calls
env_factory() once for training and again for EVERY evaluation episode, so
a fixed seed would hand all n_eval episodes the same map and collapse a
greedy evaluation to one deterministic rollout. The factory draws a fresh
child seed each call from a generator seeded off the run seed --
reproducible across relaunches, distinct across evaluation episodes.

demand_seed is FIXED across runs on purpose: F must be the same problem in
every run, only the seeds vary.

Usage:
  python -m experiments.o9_parallel --dry-run
  python -m experiments.o9_parallel --workers 8
  python -m experiments.o9_parallel --smoke
  python -m experiments.o9_parallel --aggregate
"""

import argparse
import itertools
import multiprocessing as mp
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from macs import runlog, sweep

LOG_DIR = "logs/macs_o9"
CONSOLE_DIR = "logs/macs_o9/console"

SEEDS = list(range(25))
ARMS = ["shared", "private"]
LRS = [5e-4]

ENV_CFG = dict(size=16, k=6, horizon=60, rho=4.0, patch=0,
               demand="uniform", demand_seed=0)
TRAIN_CFG = dict(n_episodes=4000, eval_every=100, log_every=100, n_eval=8)
SMOKE_CFG = dict(n_episodes=40, warmup=200, eval_every=10, n_eval=2,
                 log_every=10, target_every=10)

COMPARISONS = [("MACS/shared/none", "MACS/private/none",
                "value of the shared augmented state")]


def run_label(k, arm, lr, seed):
    return f"Facility_k{k}_MACS_{arm}_lr{lr:g}_seed{seed}"


def build_plan(seeds, arms, lrs, k):
    return [dict(arm=arm, lr=lr, seed=seed, k=k,
                 label=run_label(k, arm, lr, seed))
            for arm, lr, seed in itertools.product(arms, lrs, seeds)]


def make_factory(seed, env_cfg):
    """Environment factory with per-CALL seeding; see the module docstring."""
    from macs.envs import FacilityCoverageEnv
    src = np.random.default_rng(0xC0FFEE + seed)
    cfg = dict(env_cfg)

    def factory():
        env = FacilityCoverageEnv(seed=int(src.integers(0, 2 ** 31 - 1)),
                                  **cfg)
        env.rho_hint = cfg["rho"]
        return env
    return factory


def _worker(job):
    r, env_cfg, train_cfg, log_dir, console_dir = job

    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    from macs.train import train

    Path(console_dir).mkdir(parents=True, exist_ok=True)
    out = Path(console_dir) / f"{r['label']}.txt"

    t0 = time.perf_counter()
    try:
        with open(out, "w") as fh, redirect_stdout(fh), redirect_stderr(fh):
            train(make_factory(r["seed"], env_cfg),
                  env_name=f"Facility {env_cfg['size']}x{env_cfg['size']}",
                  mode="MACS", obs_mode=r["arm"],
                  clip="none",       # O9 isolates observation, not the clip
                  lr=r["lr"], seed=r["seed"], label=r["label"],
                  log_dir=log_dir, **train_cfg)
        return (r["label"], True, time.perf_counter() - t0, "")
    except Exception as e:                                   # noqa: BLE001
        return (r["label"], False, time.perf_counter() - t0,
                f"{type(e).__name__}: {e}")


def buffer_mb(env_cfg, buffer_cap=40_000):
    """Replay footprint per process: four int8 (k,H,W) maps dominate."""
    k, s = env_cfg["k"], env_cfg["size"]
    per_transition = 4 * k * s * s + 4 * k * 2 * 4 + k * 8 + k * 4 + 4 * 4
    return per_transition * buffer_cap / 1e6


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--arms", nargs="+", default=ARMS,
                    choices=["shared", "private"])
    ap.add_argument("--lrs", type=float, nargs="+", default=LRS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--log-dir", default=LOG_DIR)
    args = ap.parse_args(argv)

    if args.inspect:
        sweep.inspect(args.log_dir)
        return 0
    if args.aggregate:
        sweep.aggregate(COMPARISONS, args.log_dir, window=args.window,
                        paired=True,
                        out_path=Path(args.log_dir) / "o9_summary.txt")
        return 0

    env_cfg, train_cfg = dict(ENV_CFG), dict(TRAIN_CFG)
    log_dir, console_dir = args.log_dir, CONSOLE_DIR
    if args.smoke:
        env_cfg.update(size=10, k=3, horizon=15)
        train_cfg = dict(SMOKE_CFG)
        log_dir, console_dir = log_dir + "_smoke", console_dir + "_smoke"

    plan = build_plan(args.seeds, args.arms, args.lrs, env_cfg["k"])
    todo = [r for r in plan
            if not runlog.is_complete(r["label"], log_dir,
                                      train_cfg["n_episodes"],
                                      train_cfg["log_every"])]

    mb = buffer_mb(env_cfg)
    print(f"Plan     : {len(plan)} runs ({len(args.arms)} arms x "
          f"{len(args.lrs)} lrs x {len(args.seeds)} seeds)")
    print(f"Complete : {len(plan) - len(todo)}   To run: {len(todo)}")
    print(f"Workers  : {args.workers} (torch pinned to 1 thread each)")
    print(f"RAM est. : ~{mb:.0f} MB replay per worker -> "
          f"~{mb * args.workers / 1000:.1f} GB at {args.workers} workers, "
          f"plus ~200 MB/proc torch overhead")
    if args.dry_run:
        for r in todo:
            print("  ", r["label"])
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    jobs = [(r, env_cfg, train_cfg, log_dir, console_dir) for r in todo]
    t0 = time.perf_counter()
    done = failed = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
        for label, ok, dt, err in pool.imap_unordered(_worker, jobs):
            done += 1
            if ok:
                rate = (time.perf_counter() - t0) / done
                eta = rate * (len(todo) - done) / 3600
                print(f"[{done}/{len(todo)}] {label}  {dt / 60:.1f} min "
                      f"| ETA {eta:.1f} h")
            else:
                failed += 1
                print(f"[{done}/{len(todo)}] FAILED {label}: {err}")

    print(f"\nWall clock: {(time.perf_counter() - t0) / 3600:.2f} h  "
          f"| {done - failed} ok, {failed} failed")
    print(f"Logs: {log_dir}/   Console: {console_dir}/")
    if failed:
        print("Rerun the same command to retry only the failed runs.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
