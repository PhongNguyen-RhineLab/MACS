"""
run_o9.py -- parallel runner for the O9 augmented-state ablation.

O9 is an existence result: with obs_mode="shared" the agents converge; with
obs_mode="private" they do not.  Both arms keep memory (a private agent still
sees its own trail), so the only difference is SHARING.  Reward and credits
always come from the shared environment.

Why this file exists: macs_ablation.train_ablation is a single-run trainer with
no sweep driver, and its default label omits the seed --

    label = f"{env_name}_k{k}_{mode}_{obs_mode}_clip-{clip}"

so N concurrent runs would all write logs/<label>.json and clobber each other.
run_label() below appends arm, lr and seed, which makes the runs disjoint on
disk and makes is_complete() resume work after a crash.

Parallelism is at the PROCESS level, one run per process, torch pinned to a
single thread inside each.  These nets are small (16/32/32 conv on a 16x16 map,
batch 64) so intra-run threading scales badly, while 50 independent seeds
scale linearly until you run out of cores or RAM.  Single-threaded workers are
also more reproducible than multi-threaded ones: CPU reduction order can vary
with thread count.

The run itself is NOT changed relative to a serial launch.  Same trainer, same
hyperparameters, same update-to-data ratio.  Only the scheduling differs.

Usage:
    python run_o9.py --dry-run              print the plan and the RAM estimate
    python run_o9.py --workers 6            run the sweep, 6 at a time
    python run_o9.py --smoke --workers 2    tiny end-to-end check (~2 min)
    python run_o9.py --workers 6            rerun: skips completed runs
    python run_o9.py --lrs 5e-4 2e-4 1e-4   widen to the lr sweep
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

LOG_DIR = "logs/macs_o9"
CONSOLE_DIR = "logs/macs_o9/console"

# ---------------------------------------------------------------------------
# Sweep definition.  25 seeds x 2 arms = 50 runs.
# The lr sweep {5e-4, 2e-4, 1e-4} was already run as a smaller confirmation;
# pass --lrs to widen this back out (it multiplies the run count).
# ---------------------------------------------------------------------------
SEEDS = list(range(25))
ARMS = ["shared", "private"]
LRS = [5e-4]

# Facility-location config.  demand_seed is FIXED across runs on purpose: the
# objective F must be the same problem in every run, only the seeds vary.
ENV_CFG = dict(size=16, k=6, horizon=60, rho=4.0, patch=0,
               demand="uniform", demand_seed=0)

TRAIN_CFG = dict(n_episodes=4000, eval_every=100, log_every=100, n_eval=8)
SMOKE_CFG = dict(n_episodes=40, warmup=200, eval_every=10, n_eval=2,
                 log_every=10, target_every=10)


# =============================================================================
# Plan
# =============================================================================

def run_label(k, arm, lr, seed):
    return f"Facility_k{k}_MACS_{arm}_lr{lr:g}_seed{seed}"


def build_plan(seeds, arms, lrs, k):
    for arm, lr, seed in itertools.product(arms, lrs, seeds):
        yield dict(arm=arm, lr=lr, seed=seed, k=k,
                   label=run_label(k, arm, lr, seed))


def is_complete(label, n_episodes, log_every, log_dir=LOG_DIR):
    """A run counts as complete if its log holds every logging block."""
    p = Path(log_dir) / f"{label}.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        return len(d["per_block"]["episode"]) >= n_episodes // log_every
    except (json.JSONDecodeError, KeyError):
        return False


def make_factory(seed, env_cfg):
    """
    Environment factory with per-call seeding.

    train_ablation calls env_factory() once for training and again for EVERY
    evaluation episode.  A factory that returns a fixed seed would hand all
    n_eval episodes the same map, and since evaluation is greedy that would
    collapse the eval to a single deterministic rollout.  So the factory draws
    a fresh child seed each call from a generator seeded off the run seed:
    reproducible across relaunches, distinct across eval episodes.
    """
    from macs_fl import FacilityCoverageEnv
    src = np.random.default_rng(0xC0FFEE + seed)
    cfg = dict(env_cfg)

    def factory():
        return FacilityCoverageEnv(seed=int(src.integers(0, 2**31 - 1)), **cfg)
    return factory


# =============================================================================
# Worker
# =============================================================================

def _worker(job):
    """
    Runs in a fresh process (spawn + maxtasksperchild=1) so torch state and the
    ~250 MB replay buffer are released between runs rather than accumulating.
    """
    r, env_cfg, train_cfg, log_dir, console_dir = job

    import torch
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import macs_ablation as A
    A.install_facility_credits()

    Path(console_dir).mkdir(parents=True, exist_ok=True)
    out = Path(console_dir) / f"{r['label']}.txt"

    t0 = time.perf_counter()
    try:
        with open(out, "w") as fh, redirect_stdout(fh), redirect_stderr(fh):
            A.train_ablation(
                make_factory(r["seed"], env_cfg),
                env_name=f"Facility {env_cfg['size']}x{env_cfg['size']}",
                mode="MACS",
                obs_mode=r["arm"],
                clip="none",          # O9 isolates observation, not the clip
                lr=r["lr"], seed=r["seed"], label=r["label"],
                log_dir=log_dir, **train_cfg)
        return (r["label"], True, time.perf_counter() - t0, "")
    except Exception as e:                                   # noqa: BLE001
        return (r["label"], False, time.perf_counter() - t0,
                f"{type(e).__name__}: {e}")


# =============================================================================
# Resource estimate
# =============================================================================

def buffer_mb(env_cfg, buffer_cap=40_000):
    """Replay footprint per process: 4 bool (k,H,W) maps dominate."""
    k, s = env_cfg["k"], env_cfg["size"]
    per_transition = 4 * k * s * s + 4 * k * 2 * 4 + k * 8 + k * 4 + 4 * 4
    return per_transition * buffer_cap / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=["shared", "private"])
    ap.add_argument("--lrs", type=float, nargs="+", default=LRS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--log-dir", default=LOG_DIR)
    args = ap.parse_args()

    env_cfg = dict(ENV_CFG)
    train_cfg = dict(TRAIN_CFG)
    log_dir, console_dir = args.log_dir, CONSOLE_DIR
    if args.smoke:
        env_cfg.update(size=10, k=3, horizon=15)
        train_cfg = dict(SMOKE_CFG)
        log_dir, console_dir = log_dir + "_smoke", console_dir + "_smoke"

    plan = list(build_plan(args.seeds, args.arms, args.lrs, env_cfg["k"]))
    todo = [r for r in plan
            if not is_complete(r["label"], train_cfg["n_episodes"],
                               train_cfg["log_every"], log_dir)]

    mb = buffer_mb(env_cfg)
    print(f"Plan     : {len(plan)} runs "
          f"({len(args.arms)} arms x {len(args.lrs)} lrs x {len(args.seeds)} seeds)")
    print(f"Complete : {len(plan) - len(todo)}   To run: {len(todo)}")
    print(f"Workers  : {args.workers} (torch pinned to 1 thread each)")
    print(f"RAM est. : ~{mb:.0f} MB replay per worker "
          f"-> ~{mb * args.workers / 1000:.1f} GB at {args.workers} workers, "
          f"plus ~200 MB/proc torch overhead")
    if args.dry_run:
        for r in todo:
            print("  ", r["label"])
        return

    if not todo:
        print("Nothing to do.")
        return

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
                print(f"[{done}/{len(todo)}] {label}  {dt/60:.1f} min "
                      f"| ETA {eta:.1f} h")
            else:
                failed += 1
                print(f"[{done}/{len(todo)}] FAILED {label}: {err}")

    print(f"\nWall clock: {(time.perf_counter() - t0)/3600:.2f} h  "
          f"| {done - failed} ok, {failed} failed")
    print(f"Logs: {log_dir}/   Console: {console_dir}/")
    if failed:
        print("Rerun the same command to retry only the failed runs.")
        sys.exit(1)


if __name__ == "__main__":
    main()