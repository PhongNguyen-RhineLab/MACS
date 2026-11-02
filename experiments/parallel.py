"""
Run an experiments.facility plan as many processes on one GPU.

A single MACS run uses a small fraction of a GPU: the networks are tiny and
the environment, credits and replay batching are CPU work. Throughput comes
from running several independent runs at once. Each worker is a fresh process
(spawn, one task each) with one CPU thread, so torch state and replay buffers
are released between runs. Logs, labels and the resume check are exactly those
of experiments.facility: interrupt, relaunch, and finished runs are skipped.

Per-method learning rates in one call:

  python -m experiments.parallel --exp islands --time-obs --ks 3 4 6 \\
      --seeds 1 2 3 4 5 6 --episodes 8000 --workers 4 \\
      --lr-map MACS=2e-5 MACS-CLIP=2e-5 LOCAL=2e-5 DR=2e-5 SHARED=2e-5 \\
               VDN=1e-5 QMIX=5e-4 \\
      --log-dir logs/final_islands

  --dry-run      list the runs that would execute, then exit
  --smoke        tiny config, one seed, to check the pipeline end to end

Console output of each run goes to <log-dir>/console/<label>.txt.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


def _parse_lr_map(items):
    out = {}
    for it in items or []:
        mode, val = it.split("=")
        out[mode] = float(val)
    return out


def _cfg(args):
    from macs import config as CFG
    cfg = dict(CFG.SMOKE_CFG if args.smoke else CFG.TRAIN_CFG)
    if args.episodes and not args.smoke:
        cfg["n_episodes"] = args.episodes
    return cfg


def _runs(exp, ks, seeds, mode, lr, time_obs):
    from experiments.facility import plan
    return plan(exp, ks, seeds, lr=lr, modes=[mode], time_obs=time_obs)


def _worker(job):
    # one CPU thread per process: with many processes, thread oversubscription
    # costs more than it gains
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import torch
    torch.set_num_threads(1)
    from macs import sweep

    label = job["label"]
    console = Path(job["log_dir"]) / "console"
    console.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(console / f"{label}.txt", "w") as fh, \
            redirect_stdout(fh), redirect_stderr(fh):
        runs = [r for r in _runs(job["exp"], [job["k"]], [job["seed"]],
                                 job["mode"], job["lr"], job["time_obs"])
                if r.label == label]
        assert len(runs) == 1, f"could not rebuild run {label}"
        sweep.execute(runs, job["cfg"], job["log_dir"],
                      backend=job["backend"], mc_m=job["mc_m"])
    return label, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--ks", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=None,
                    help="methods to run; default: every mode in --lr-map")
    ap.add_argument("--lr", type=float, default=None,
                    help="one rate for every mode (overridden by --lr-map)")
    ap.add_argument("--lr-map", nargs="+", default=None,
                    help="per-mode rates, e.g. MACS=2e-5 VDN=1e-5")
    ap.add_argument("--time-obs", action="store_true")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--shapley", default="auto")
    ap.add_argument("--mc-m", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from macs import runlog
    lr_map = _parse_lr_map(args.lr_map)
    modes = args.modes or sorted(lr_map)
    if not modes:
        ap.error("give --modes or --lr-map")
    seeds = args.seeds[:1] if args.smoke else args.seeds
    cfg = _cfg(args)

    jobs, seen = [], set()
    for mode in modes:
        lr = lr_map.get(mode, args.lr)
        for k in args.ks:
            for s in seeds:
                for r in _runs(args.exp, [k], [s], mode, lr, args.time_obs):
                    if r.label in seen:
                        continue
                    seen.add(r.label)
                    if runlog.is_complete(r.label, args.log_dir,
                                          cfg["n_episodes"], cfg["log_every"]):
                        print(f"  skip (complete): {r.label}")
                        continue
                    jobs.append(dict(exp=args.exp, k=k, seed=s, mode=mode,
                                     lr=lr, time_obs=args.time_obs,
                                     label=r.label, cfg=cfg,
                                     log_dir=args.log_dir,
                                     backend=args.shapley, mc_m=args.mc_m))
    # largest teams first, so the slowest runs do not straggle at the end
    jobs.sort(key=lambda j: -j["k"])
    print(f"{len(jobs)} run(s) to execute, {args.workers} worker(s), "
          f"{cfg['n_episodes']} episodes, log_dir={args.log_dir}")
    if args.dry_run:
        for j in jobs:
            print(f"  would run: {j['label']}  (lr={j['lr']})")
        return 0
    if not jobs:
        return 0

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, maxtasksperchild=1) as pool:
        for n, (label, dt) in enumerate(
                pool.imap_unordered(_worker, jobs), 1):
            el = time.time() - t0
            eta = el / n * (len(jobs) - n)
            print(f"[{n}/{len(jobs)}] {label}  {dt / 60:.1f} min  "
                  f"(elapsed {el / 3600:.2f} h, eta {eta / 3600:.2f} h)",
                  flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
