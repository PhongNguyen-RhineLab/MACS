"""
Run the ICML 2027 revision batches (notes/RUNPLAN_ICML2027.md) under a
wall-clock budget, on one or more GPUs. Built for Kaggle and Colab sessions,
which are killed at a time limit: no run is started that is not expected to
finish before the deadline, and every finished run is skipped on relaunch.
A run interrupted half way restarts from scratch, so the budget matters.

  python -m scripts.run_batch --batches crowded_core int_islands \\
      --log-root /kaggle/working/MACS_runs --hours 11 --workers 4

  --dry-run   list pending runs per batch and exit
  --smoke     tiny config (30 episodes), into <log-root>/smoke, to check the
              pipeline and time a run before committing a session

Workers are spread over the visible GPUs round-robin (Kaggle T4 x2 gets both).

--fused auto (default) trains the k agent networks as one batched ensemble
(macs.ensemble) whenever the worker sees a GPU, and with the per-agent loop on
CPU, where batching is slower. Same update either way (tests/test_ensemble.py).
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

SEEDS_1_6 = [1, 2, 3, 4, 5, 6]
SEEDS_7_12 = [7, 8, 9, 10, 11, 12]
CREDIT_LR = 2e-5

# name -> (exp, ks, seeds, {mode: lr or None}, log subdir, episodes)
BATCHES = {
    # Priority 1
    "crowded_core": ("crowded", [16, 12, 8], SEEDS_1_6,
                     {"MACS": CREDIT_LR, "INT": CREDIT_LR, "DR": CREDIT_LR,
                      "LOCAL": CREDIT_LR}, "final_crowded", 8000),
    "crowded_base": ("crowded", [16, 12, 8], SEEDS_1_6,
                     {"SHARED": CREDIT_LR, "VDN": 1e-5},
                     "final_crowded", 8000),
    "int_islands": ("islands", [6, 4, 3], SEEDS_1_6, {"INT": CREDIT_LR},
                    "final_islands", 8000),
    "int_decoy": ("decoy", [6], SEEDS_1_6 + SEEDS_7_12, {"INT": CREDIT_LR},
                  "final_decoy", 8000),
    "h1_seeds": ("islands", [6, 4, 3], SEEDS_7_12,
                 {"MACS": CREDIT_LR, "SHARED": CREDIT_LR, "VDN": 1e-5},
                 "final_islands", 8000),
    # Priority 2
    "baselinelr": ("baselinelr", [6], [0], {"VDN": None, "QMIX": None},
                   "final_baselinelr", 8000),
    "vdn16k": ("islands", [6], SEEDS_1_6, {"VDN": 1e-5}, "final_vdn16k",
               16000),
    # Optional: does this trainer reproduce the final runs? Compare with
    # Table 2 (MACS, islands k = 4: 83.0 +- 0.8 over seeds 1-6).
    "pipeline_check": ("islands", [4], [1, 2, 3], {"MACS": CREDIT_LR},
                       "check_pipeline", 8000),
}
PRIORITY = ["crowded_core", "int_islands", "int_decoy", "h1_seeds",
            "crowded_base", "baselinelr", "vdn16k"]


def _gpu_worker(job):
    # must run before torch is imported in this (spawned) process
    if job.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    fused = job.get("fused", "auto")
    if fused == "auto":
        import torch
        fused = torch.cuda.is_available()
    job = dict(job, cfg=dict(job["cfg"], fused=bool(fused)))
    from experiments.parallel import _worker
    return _worker(job)


def _fake_worker(job):
    """Stand-in for tests: sleeps job['fake_s'] seconds, trains nothing."""
    time.sleep(job.get("fake_s", 0.05))
    if job.get("fake_fail"):
        raise RuntimeError("boom")
    return job["label"], job.get("fake_s", 0.05)


def build_jobs(names, log_root, smoke=False, shapley="auto", mc_m=64):
    from experiments.parallel import _runs
    from macs import config as CFG
    from macs import runlog
    jobs, seen = [], set()
    for name in names:
        exp, ks, seeds, lr_map, sub, episodes = BATCHES[name]
        cfg = dict(CFG.SMOKE_CFG if smoke else CFG.TRAIN_CFG)
        if not smoke:
            cfg["n_episodes"] = episodes
        log_dir = str(Path(log_root) / ("smoke" if smoke else sub))
        for mode, lr in lr_map.items():
            for k in ks:
                for s in (seeds[:1] if smoke else seeds):
                    for r in _runs(exp, [k], [s], mode, lr, True):
                        key = (log_dir, r.label)
                        if key in seen:
                            continue
                        seen.add(key)
                        done = runlog.is_complete(r.label, log_dir,
                                                  cfg["n_episodes"],
                                                  cfg["log_every"])
                        jobs.append(dict(batch=name, exp=exp, k=k, seed=s,
                                         mode=mode, lr=lr, time_obs=True,
                                         label=r.label, cfg=cfg,
                                         log_dir=log_dir, backend=shapley,
                                         mc_m=mc_m, done=done))
    return jobs


def summarize(jobs):
    by = {}
    for j in jobs:
        d = by.setdefault(j["batch"], [0, 0])
        d[0] += j["done"]
        d[1] += 1
    for name, (d, n) in by.items():
        print(f"  {name:<15} {d:>4}/{n:<4} complete")


def run(names, log_root, hours, workers, gpus, smoke=False, dry_run=False,
        first_run_hours=None, worker_fn=None, fused="auto"):
    jobs = build_jobs(names, log_root, smoke=smoke)
    print("Status before this session:")
    summarize(jobs)
    todo = [j for j in jobs if not j["done"]]
    # keep the requested batch order, largest teams first within a batch
    order = {n: i for i, n in enumerate(names)}
    todo.sort(key=lambda j: (order[j["batch"]], -j["k"]))
    print(f"{len(todo)} run(s) pending, {workers} worker(s), "
          f"GPUs {gpus if gpus else 'none'}, budget {hours:.1f} h")
    if dry_run:
        for j in todo:
            print(f"  would run [{j['batch']}]: {j['label']}  (lr={j['lr']})")
        return 0
    if not todo:
        return 0

    worker_fn = worker_fn or _gpu_worker
    t0 = time.time()
    deadline = t0 + hours * 3600
    durations = {}                     # k -> list of minutes
    default_h = first_run_hours if first_run_hours is not None else hours / 4

    def est_hours(k):
        # slowest run seen at this team size, else at any larger one, else
        # the default; runs at larger k are never faster
        seen = [max(v) for kk, v in durations.items() if kk >= k]
        return (min(seen) if seen else default_h * 60) / 60

    ctx = mp.get_context("spawn")
    pending = list(todo)
    in_flight = {}
    finished = skipped = 0
    slot = 0
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        while pending or in_flight:
            # submit while a slot is free and the run should fit
            while pending and len(in_flight) < workers:
                j = pending[0]
                left_h = (deadline - time.time()) / 3600
                if est_hours(j["k"]) > left_h:
                    break
                pending.pop(0)
                j = dict(j, gpu=(gpus[slot % len(gpus)] if gpus else None),
                         fused=fused)
                slot += 1
                in_flight[j["label"]] = (pool.apply_async(worker_fn, (j,)),
                                         j, time.time())
                print(f"  start [{j['batch']}] {j['label']} "
                      f"(gpu {j['gpu']})", flush=True)
            if pending and not in_flight:
                skipped = len(pending)
                print(f"Budget reached: {skipped} run(s) left for the next "
                      "session (est. "
                      f"{est_hours(pending[0]['k']):.2f} h each > "
                      f"{max(0, (deadline - time.time()) / 3600):.2f} h "
                      "left).")
                break
            time.sleep(2 if worker_fn is _gpu_worker else 0.02)
            for label in list(in_flight):
                res, j, ts = in_flight[label]
                if res.ready():
                    del in_flight[label]
                    try:
                        res.get()
                        mins = (time.time() - ts) / 60
                        durations.setdefault(j["k"], []).append(mins)
                        finished += 1
                        el = (time.time() - t0) / 3600
                        print(f"[{finished}] done [{j['batch']}] {label}  "
                              f"{mins:.1f} min  (elapsed {el:.2f} h)",
                              flush=True)
                    except Exception as exc:          # keep the others going
                        print(f"  FAILED {label}: {exc!r}  (see "
                              f"{j['log_dir']}/console/{label}.txt)",
                              flush=True)
    print(f"\nSession: {finished} run(s) finished, {skipped} deferred, "
          f"{(time.time() - t0) / 3600:.2f} h.")
    print("Status after this session:")
    summarize(build_jobs(names, log_root, smoke=smoke))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("--batches", nargs="+", default=PRIORITY,
                    choices=sorted(BATCHES))
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", type=int, nargs="*", default=None,
                    help="GPU indices; default: all visible")
    ap.add_argument("--first-run-hours", type=float, default=None,
                    help="time estimate before any run has finished "
                         "(default: a quarter of the budget)")
    ap.add_argument("--fused", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    gpus = args.gpus
    if gpus is None:
        try:
            import torch
            gpus = list(range(torch.cuda.device_count()))
        except ImportError:
            gpus = []
    return run(args.batches, args.log_root, args.hours, args.workers, gpus,
               smoke=args.smoke, dry_run=args.dry_run,
               first_run_hours=args.first_run_hours,
               fused={"auto": "auto", "on": True, "off": False}[args.fused])


if __name__ == "__main__":
    sys.exit(main())
