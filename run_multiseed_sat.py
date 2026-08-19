"""
run_multiseed_sat.py — Multi-seed replication of the SATURATING suite only.
Addresses Open Item O6 of main_macs.tex (draft v0.8):

  "Rerun the saturating suite with >= 5 seeds; report seed-level mean +- std
   and a significance test for MACS-Clip vs. LOCAL at k in {4, 6}."

Configurations (same as Table `tab:configs`, saturating rows):

  12x12, k=2, horizon 40, F(V)=72,  exact Shapley (2 perms)
  12x12, k=3, horizon 40, F(V)=72,  exact Shapley (6 perms)
  12x12, k=4, horizon 40, F(V)=72,  exact Shapley (24 perms)  [+ MACS-MC ablation]
  16x16, k=6, horizon 60, F(V)=128, MC Shapley m=64

Per-seed metric (matches the paper's protocol): mean of the LAST 10
greedy-evaluation checkpoints of eval_coverage. The aggregator then reports,
per (config, method), the mean +- std ACROSS SEEDS — which is the quantity
the single-run tables could not provide — and an exact permutation test on
the difference of seed-level means for MACS-CLIP vs LOCAL at k in {4, 6}.
With 5 seeds per arm the permutation test enumerates all C(10,5) = 252
splits exactly, so no scipy and no normality assumption is needed.

Usage:
  python run_multiseed_sat.py                  run all seeds x configs x methods
  python run_multiseed_sat.py --seeds 0 1 2    subset of seeds
  python run_multiseed_sat.py --ks 4 6         subset of team sizes
  python run_multiseed_sat.py --methods MACS MACS-CLIP LOCAL
  python run_multiseed_sat.py --smoke          tiny end-to-end sanity run
  python run_multiseed_sat.py --aggregate      skip training, only build the
                                               summary table + tests from logs
  python run_multiseed_sat.py --dry-run        print the run plan and exit

Resume behavior: a run is skipped if its JSON log exists and contains the
full number of logging blocks, so the script can be interrupted and
relaunched. Every run gets label "<env>_k<k>_<mode>_seed<seed>" inside
LOG_DIR, which plot_macs.py can also consume with --log-dir.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from decimal import Decimal, ROUND_HALF_UP

# macs_v3 import has two side effects we rely on:
#   1. it re-points macs_main.compute_credits to the v3 dispatcher
#      (correct mc_m plumbing for k > 4 and the MACS-MC mode)
#   2. it runs nothing at import time (guarded by __main__), so importing
#      is safe.
import macs_v3 as V3
import macs_main as M
from macs_v3 import SaturatingCoverageEnv, MODES

LOG_DIR = "logs/macs_v3_multiseed"

DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# (k, grid size, horizon, extra modes beyond the standard seven)
CONFIGS = [
    dict(k=2, size=12, horizon=40, extra_modes=[]),
    dict(k=3, size=12, horizon=40, extra_modes=[]),
    dict(k=4, size=12, horizon=40, extra_modes=["MACS-MC"]),
    dict(k=6, size=16, horizon=60, extra_modes=[]),
]

TRAIN_CFG = dict(n_episodes=4000, eval_every=100, log_every=100)
SMOKE_CFG = dict(n_episodes=30, warmup=200, eval_every=10, n_eval=2,
                 log_every=10, target_every=10)

# Number of final eval checkpoints averaged into the per-seed metric.
# 4000 episodes / eval_every 100 -> checkpoints at 100..4000; the last 10
# span episodes 3100..4000, matching the paper's reporting window.
FINAL_WINDOW = 10


# =============================================================================
# Run plan
# =============================================================================

def make_factory(size, horizon, k):
    return lambda: SaturatingCoverageEnv(
        size=size, horizon=horizon, k=k, patch=1, region=4, cap_frac=0.5)


def run_label(env_name, k, mode, seed):
    return f"{env_name.replace(' ', '_')}_k{k}_{mode}_seed{seed}"


def build_plan(seeds, ks, methods, smoke=False):
    """Yield dicts describing every (config, mode, seed) run."""
    for cfg in CONFIGS:
        if cfg["k"] not in ks:
            continue
        size = 8 if smoke else cfg["size"]
        horizon = 15 if smoke else cfg["horizon"]
        env_name = f"Saturating {size}x{size}"
        modes = [m for m in MODES + cfg["extra_modes"] if m in methods]
        for mode, seed in itertools.product(modes, seeds):
            yield dict(env_name=env_name, k=cfg["k"], size=size,
                       horizon=horizon, mode=mode, seed=seed,
                       label=run_label(env_name, cfg["k"], mode, seed))


def is_complete(label, n_episodes, log_every, log_dir=LOG_DIR):
    """A run counts as complete if its log has all logging blocks."""
    p = Path(log_dir) / f"{label}.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
        expected = n_episodes // log_every
        return len(d["per_block"]["episode"]) >= expected
    except (json.JSONDecodeError, KeyError):
        return False


def execute(plan, cfg, log_dir=LOG_DIR):
    plan = list(plan)
    n_ep, log_ev = cfg["n_episodes"], cfg["log_every"]
    todo = [r for r in plan if not is_complete(r["label"], n_ep, log_ev,
                                               log_dir)]
    print(f"Plan: {len(plan)} runs total, {len(plan) - len(todo)} already "
          f"complete, {len(todo)} to run.")
    for i, r in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {r['label']}")
        factory = make_factory(r["size"], r["horizon"], r["k"])
        M.train_macs(factory, env_name=r["env_name"], mode=r["mode"],
                     seed=r["seed"], label=r["label"], log_dir=log_dir,
                     **cfg)


# =============================================================================
# Aggregation: seed-level statistics + exact permutation test
# =============================================================================

def final_metric(log_path, window=FINAL_WINDOW):
    """Per-seed metric: mean eval_coverage over the last `window` checkpoints."""
    with open(log_path) as f:
        d = json.load(f)
    ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
    if len(ev) < window:
        return None
    return float(np.mean(ev[-window:]))


def collect(log_dir=LOG_DIR):
    """
    Returns {(env_name, k): {mode: {seed: metric}}} from all seed-labelled
    logs in log_dir.
    """
    out = {}
    for p in sorted(Path(log_dir).glob("*seed*.json")):
        with open(p) as f:
            d = json.load(f)
        c = d["config"]
        # seed is encoded in the label suffix "_seed<n>"
        try:
            seed = int(d["label"].rsplit("_seed", 1)[1])
        except (IndexError, ValueError):
            continue
        m = final_metric(p)
        if m is None:
            continue
        key = (c["env_name"], c["k"])
        out.setdefault(key, {}).setdefault(c["method"], {})[seed] = m
    return out


def exact_perm_test(a, b, n_max_exact=2 ** 16):
    """
    Two-sided exact permutation test on the difference of means.
    Enumerates all ways to split a+b into groups of the original sizes
    (252 splits for 5 vs 5). Falls back to 20,000 Monte Carlo splits if
    the exact enumeration would exceed n_max_exact.
    Returns (observed difference mean(a) - mean(b), p-value).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n, na = len(pooled), len(a)
    obs = a.mean() - b.mean()

    from math import comb
    if comb(n, na) <= n_max_exact:
        diffs = []
        for idx in itertools.combinations(range(n), na):
            mask = np.zeros(n, dtype=bool)
            mask[list(idx)] = True
            diffs.append(pooled[mask].mean() - pooled[~mask].mean())
        diffs = np.array(diffs)
    else:
        rng = np.random.default_rng(0)
        diffs = np.array([
            (lambda perm: pooled[perm[:na]].mean() - pooled[perm[na:]].mean())
            (rng.permutation(n)) for _ in range(20_000)])

    p = float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))
    return float(obs), p


def r1(x):
    """Round half AWAY FROM ZERO to one decimal.

    Python's format() and round() use banker's rounding, so 52.75 -> 52.7
    and 125.65 -> 125.6, which disagrees with the hand-typed paper table in
    five cells. The paper convention is half-up; this makes the emitted
    LaTeX match it.
    """
    return float(Decimal(repr(float(x))).quantize(Decimal("0.1"),
                                                  rounding=ROUND_HALF_UP))


def f1(x):
    return f"{r1(x):.1f}"


# Pairwise seed-level comparisons reported in the paper. Both rows of the
# bottom block of the results table are produced from this list, at every
# configuration -- earlier versions hardcoded k in (4, 6) for the first
# comparison and never emitted the second at all.
COMPARISONS = [
    ("MACS-CLIP", "LOCAL", "clip vs. strongest baseline"),
    ("MACS-CLIP", "MACS", "clip ablation, identical credits"),
]


def run_comparisons(by_mode):
    """Returns [(hi, lo, note, diff, p)] for every comparison available."""
    out = []
    for hi, lo, note in COMPARISONS:
        if hi not in by_mode or lo not in by_mode:
            continue
        seeds = sorted(set(by_mode[hi]) & set(by_mode[lo]))
        if len(seeds) < 2:
            continue
        a = [by_mode[hi][s] for s in seeds]
        b = [by_mode[lo][s] for s in seeds]
        diff, p = exact_perm_test(a, b)
        out.append((hi, lo, note, diff, p, len(seeds)))
    return out


def aggregate(log_dir=LOG_DIR, out_path=None):
    data = collect(log_dir)
    if not data:
        print(f"No seed-labelled logs found in {log_dir}.")
        return

    lines = []
    order = MODES + ["MACS-MC"]

    for (env_name, k), by_mode in sorted(data.items(), key=lambda x: x[0][1]):
        lines.append(f"\n=== {env_name}, k={k} ===")
        lines.append(f"{'method':<10} {'n_seeds':>7} {'mean':>8} {'std':>7}"
                     f"  seeds -> metric")
        for mode in order:
            if mode not in by_mode:
                continue
            vals = by_mode[mode]
            arr = np.array([vals[s] for s in sorted(vals)])
            per_seed = ", ".join(f"{s}:{vals[s]:.1f}" for s in sorted(vals))
            lines.append(f"{mode:<10} {len(arr):>7} {arr.mean():>8.2f} "
                         f"{arr.std(ddof=1):>7.2f}  [{per_seed}]")

        # Significance tests: every comparison in COMPARISONS, every config.
        for hi, lo, note, diff, p, n in run_comparisons(by_mode):
            lines.append(f"  {hi} vs {lo}: diff of seed means = {diff:+.2f}, "
                         f"exact permutation p = {p:.4f} (two-sided, "
                         f"n={n}+{n})   [{note}]")

    report = "\n".join(lines)
    print(report)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(report + "\n")
            f.write("\n\n% LaTeX rows (mean \\pm std over seeds, "
                    "half-up rounding):\n")
            ks = []
            for (env_name, k), by_mode in sorted(data.items(),
                                                 key=lambda x: x[0][1]):
                ks.append(k)
                row = [f"${env_name.split()[-1]}$, $k{{=}}{k}$"]
                for mode in MODES:
                    if mode in by_mode:
                        arr = np.array(list(by_mode[mode].values()))
                        row.append(f"${f1(arr.mean())}{{\\pm}}"
                                   f"{f1(arr.std(ddof=1))}$")
                    else:
                        row.append("--")
                f.write(" & ".join(row) + r" \\" + "\n")

            # bottom block of the results table: one row per comparison
            f.write("\n% p-value rows (exact two-sided permutation):\n")
            for hi, lo, note in COMPARISONS:
                cells = []
                for (env_name, k), by_mode in sorted(data.items(),
                                                     key=lambda x: x[0][1]):
                    got = {(h, l): (d, p) for h, l, _, d, p, _
                           in run_comparisons(by_mode)}
                    if (hi, lo) in got:
                        cells.append(f"$k{{=}}{k}$: ${got[(hi, lo)][1]:.3f}$")
                if not cells:
                    continue
                label = ("Clip vs.\\ LOCAL" if lo == "LOCAL"
                         else "Clip vs.\\ MACS")
                f.write(f"{label}, $p$ & \\multicolumn{{7}}{{c}}{{%\n"
                        + " \\quad ".join(cells) + "} \\\\\n")

            # machine-readable summary, small enough to commit alongside code
            summ = {}
            for (env_name, k), by_mode in sorted(data.items(),
                                                 key=lambda x: x[0][1]):
                key = f"{env_name}|k={k}"
                summ[key] = {
                    "methods": {m: {"seeds": {str(s): v for s, v in
                                              sorted(vals.items())},
                                    "mean": float(np.mean(list(vals.values()))),
                                    "std": float(np.std(list(vals.values()),
                                                        ddof=1))}
                                for m, vals in sorted(by_mode.items())},
                    "tests": [{"hi": h, "lo": l, "note": nt, "diff": d,
                               "p": p, "n_seeds": n}
                              for h, l, nt, d, p, n
                              in run_comparisons(by_mode)]}
            jp = Path(out_path).with_suffix(".json")
            with open(jp, "w") as jf:
                json.dump(summ, jf, indent=2)
            print(f"Written: {jp}")
        print(f"\nWritten: {out_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 6])
    ap.add_argument("--methods", nargs="+", default=MODES + ["MACS-MC"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--log-dir", default=LOG_DIR)
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.log_dir,
                  out_path=Path(args.log_dir) / "multiseed_summary.txt")
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

    execute(plan, cfg, log_dir=args.log_dir)
    print("\nAll runs complete. Aggregating...")
    aggregate(args.log_dir,
              out_path=Path(args.log_dir) / "multiseed_summary.txt")