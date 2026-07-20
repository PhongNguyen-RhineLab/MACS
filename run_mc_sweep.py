"""
run_mc_sweep.py — Monte Carlo Shapley ablation over m in {8, 16, 32, 64, 128}.

Two independent measurements, cheap one first:

  PART 1 (--offline, seconds, numpy only). Credit ERROR vs m, no training.
  Samples random (S_prev, patches) states on the 12x12 saturating F with
  k = 4, where exact Shapley (24 permutations) is ground truth, and
  measures max_i |phi_hat_i - phi_i| as a function of m. This is the
  empirical counterpart of Prop. mc-error and directly shows how loose the
  Hoeffding bound is in practice. Also verifies sum_i phi_hat_i = r exactly
  at every m (Rem. efficiency-exact).

  PART 2 (--train, GPU-days). Final PERFORMANCE vs m. Trains MACS-MC at
  k = 4 with each m over multiple seeds, using the same protocol as
  run_multiseed_sat.py. Logs land in logs/macs_v3_mcsweep with labels
  "Saturating_12x12_k4_MACS-MC-m<m>_seed<s>"; the aggregator prints
  seed-level mean +- std per m next to the exact-MACS reference if
  logs/macs_v3_multiseed contains it.

Usage:
  python run_mc_sweep.py --offline                # figure + table, no torch
  python run_mc_sweep.py --train --seeds 0 1 2 3 4
  python run_mc_sweep.py --aggregate
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import macs_v3 as V3
import macs_main as M
from macs_v3 import SaturatingCoverageEnv, SaturatingF
from macs_main import shapley_exact, shapley_mc

LOG_DIR = "logs/macs_v3_mcsweep"
MS = [8, 16, 32, 64, 128]
K = 4
SIZE, HORIZON = 12, 40
TRAIN_CFG = dict(n_episodes=4000, eval_every=100, log_every=100)


# =============================================================================
# Part 1 — offline credit error vs m (no training)
# =============================================================================

def sample_state(rng, size=SIZE, k=K, cover_p=None):
    """Random coverage state + k random 3x3 patches, as in training."""
    if cover_p is None:
        cover_p = rng.uniform(0.1, 0.7)      # sweep over training phases
    S_prev = rng.random((size, size)) < cover_p
    patches = []
    for _ in range(k):
        pos = rng.integers(1, size - 1, size=2)
        m = np.zeros((size, size), dtype=bool)
        m[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2] = True
        patches.append(m)
    return S_prev, patches


def offline_error(n_states=500, seed=0, out_dir="plots/mc_sweep"):
    rng = np.random.default_rng(seed)
    Fsat = SaturatingF(SIZE, region=4, cap_frac=0.5)

    max_err = {m: [] for m in MS}       # max_i |phi_hat - phi| per state
    eff_err = {m: 0.0 for m in MS}      # worst |sum phi_hat - r|
    t_exact, t_mc = 0.0, {m: 0.0 for m in MS}

    for _ in range(n_states):
        S_prev, patches = sample_state(rng)
        u = S_prev.copy()
        for pm in patches:
            u |= pm
        r = Fsat(u) - Fsat(S_prev)

        t0 = time.perf_counter()
        exact = shapley_exact(S_prev, patches, Fsat)
        t_exact += time.perf_counter() - t0

        for m in MS:
            t0 = time.perf_counter()
            approx = shapley_mc(S_prev, patches, Fsat, m, rng)
            t_mc[m] += time.perf_counter() - t0
            max_err[m].append(np.max(np.abs(approx - exact)))
            eff_err[m] = max(eff_err[m], abs(approx.sum() - r))

    print(f"\nOffline MC ablation: k={K}, saturating F on {SIZE}x{SIZE}, "
          f"{n_states} random states")
    print(f"{'m':>5} {'mean max err':>13} {'p95 max err':>12} "
          f"{'worst':>7} {'eff err':>9} {'time vs exact':>14}")
    rows = []
    for m in MS:
        e = np.array(max_err[m])
        rows.append((m, e.mean(), np.percentile(e, 95), e.max(),
                     eff_err[m], t_mc[m] / t_exact))
        print(f"{m:>5} {e.mean():>13.4f} {np.percentile(e, 95):>12.4f} "
              f"{e.max():>7.3f} {eff_err[m]:>9.2e} "
              f"{t_mc[m] / t_exact:>13.2f}x")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "offline_error.json", "w") as f:
        json.dump({"k": K, "n_states": n_states,
                   "rows": [dict(zip(("m", "mean", "p95", "max", "eff_err",
                                      "rel_time"), r)) for r in rows]},
                  f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        means = [np.mean(max_err[m]) for m in MS]
        p95s = [np.percentile(max_err[m], 95) for m in MS]
        ax.plot(MS, means, "o-", color="#2ecc71", label="mean")
        ax.plot(MS, p95s, "s--", color="#e67e22", label="95th percentile")
        # Hoeffding reference: with prob 0.95, err <= Fmax*sqrt(log(40)/2m)
        Fmax = SaturatingF(SIZE, region=4, cap_frac=0.5).F_max
        hoeff = [Fmax * np.sqrt(np.log(40) / (2 * m)) for m in MS]
        ax.plot(MS, hoeff, ":", color="gray",
                label="Hoeffding bound (delta=0.05)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(MS)
        ax.set_xticklabels(MS)
        ax.set_xlabel("Sampled permutations m")
        ax.set_ylabel("max_i |phi_hat_i - phi_i|")
        ax.set_title(f"MC Shapley credit error vs m (k={K}, saturating F)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, which="both")
        fp = Path(out_dir) / "mc_error_vs_m.png"
        plt.tight_layout()
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"Saved: {fp}")
    except ImportError:
        print("matplotlib not installed; skipped the figure.")


# =============================================================================
# Part 2 — training sweep over m, multi-seed
# =============================================================================

def bind_mc_m(m):
    """
    Re-point macs_main.compute_credits so MACS-MC uses `m` samples.
    Note: the default argument of compute_credits_v3 was bound at import
    time, so we must pass mc_m explicitly rather than mutate V3.MC_M.
    """
    def cc(mode, r_team, info, F, k, mc_m=m, rng=None):
        return V3.compute_credits_v3(mode, r_team, info, F, k,
                                     mc_m=m, rng=rng)
    M.compute_credits = cc


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


def train_sweep(seeds, cfg=TRAIN_CFG, log_dir=LOG_DIR):
    factory = lambda: SaturatingCoverageEnv(
        size=SIZE, horizon=HORIZON, k=K, patch=1, region=4, cap_frac=0.5)
    plan = [(m, s) for m in MS for s in seeds]
    todo = []
    for m, s in plan:
        label = f"Saturating_{SIZE}x{SIZE}_k{K}_MACS-MC-m{m}_seed{s}"
        if not is_complete(label, cfg, log_dir):
            todo.append((m, s, label))
    print(f"Sweep plan: {len(plan)} runs, {len(plan) - len(todo)} complete, "
          f"{len(todo)} to run.")
    for i, (m, s, label) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {label}")
        bind_mc_m(m)
        M.train_macs(factory, env_name=f"Saturating {SIZE}x{SIZE}",
                     mode="MACS-MC", seed=s, label=label,
                     log_dir=log_dir, **cfg)
    # restore the default binding for anything imported after us
    M.compute_credits = V3.compute_credits_v3


def aggregate(log_dir=LOG_DIR, window=10,
              exact_ref_dir="logs/macs_v3_multiseed"):
    by_m = {}
    for p in sorted(Path(log_dir).glob("*MACS-MC-m*_seed*.json")):
        with open(p) as f:
            d = json.load(f)
        label = d["label"]
        m = int(label.split("MACS-MC-m")[1].split("_")[0])
        ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
        if len(ev) >= window:
            by_m.setdefault(m, []).append(float(np.mean(ev[-window:])))
    if not by_m:
        print(f"No sweep logs in {log_dir}.")
        return
    print(f"\nMC sweep, k={K}, final F(S_T) (seed-level mean +- std):")
    for m in sorted(by_m):
        arr = np.array(by_m[m])
        print(f"  m={m:>4}: {arr.mean():6.2f} +- "
              f"{arr.std(ddof=1) if len(arr) > 1 else 0.0:5.2f} "
              f"(n={len(arr)})")
    # exact-MACS reference from the main multi-seed suite, if present
    ref = []
    for p in Path(exact_ref_dir).glob(f"*k{K}_MACS_seed*.json"):
        with open(p) as f:
            d = json.load(f)
        ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
        if len(ev) >= window:
            ref.append(float(np.mean(ev[-window:])))
    if ref:
        arr = np.array(ref)
        print(f"  exact MACS reference: {arr.mean():6.2f} +- "
              f"{arr.std(ddof=1) if len(arr) > 1 else 0.0:5.2f} "
              f"(n={len(arr)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-states", type=int, default=500)
    args = ap.parse_args()

    if not (args.offline or args.train or args.aggregate):
        args.offline = True                 # cheap default

    if args.offline:
        offline_error(n_states=args.n_states)
    if args.train:
        if not M.TORCH_OK:
            print("torch not available; cannot train.")
            sys.exit(1)
        train_sweep(args.seeds)
        aggregate()
    if args.aggregate:
        aggregate()