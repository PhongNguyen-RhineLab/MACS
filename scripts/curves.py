"""
Learning curves and sample-efficiency tests from existing run logs.

The final score (mean of the last 10 evaluation checkpoints) answers "where
did each arm end up". The paper's open question is different: whether exact
efficiency changes the LEARNING dynamics. This script reads the same logs and
reports, per (suite, k) and arm pair, paired over seeds:

  auc       mean evaluation coverage over the whole run, % of F(V)
            (normalized area under the learning curve)
  auc_half  the same over the first half of training only
  t90       first episode at which the curve reaches 90% of that run's own
            final score (smaller = faster)

each with the exact paired sign-flip p and a percentile bootstrap 95% CI of
the mean paired difference. Nothing is retrained.

Usage:
  python -m scripts.curves --log-dir logs/final_islands \\
      --pairs MACS:DR MACS:LOCAL INT:DR MACS:SHARED MACS:VDN \\
      --plot figures/curves.pdf
  python -m scripts.curves --log-dir logs/final_decoy --pairs MACS:DR MACS:LOCAL

Arms are matched by mode name; obs/clip/lr/tau suffixes are taken from the
first run of that mode found for the (suite, k), so pass a log directory
that holds one configuration per mode (as the final runs do).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from macs import runlog
from macs.stats import paired_sign_flip_test


def load_curves(log_dir):
    """{(env_name, k): {mode: {seed: (episodes, eval % of F(V))}}}"""
    out = {}
    for p in sorted(Path(log_dir).glob("*seed*.json")):
        try:
            d = runlog.load(p)
        except (json.JSONDecodeError, OSError):
            continue
        c, pb = d["config"], d["per_block"]
        F_V = float(c.get("F_total") or c.get("max_cells") or 1.0)
        ep = np.asarray(pb["episode"], dtype=float)
        ev = np.asarray([np.nan if x is None else x
                         for x in pb["eval_coverage"]], dtype=float)
        ok = ~np.isnan(ev)
        if ok.sum() < 2:
            continue
        key = (c["env_name"], int(c["k"]))
        mode = c.get("method") or c.get("mode")
        if c.get("clip", "none") not in ("none", None):
            mode = f"{mode}[{c['clip']}]"
        out.setdefault(key, {}).setdefault(mode, {})[int(c.get("seed", 0))] = \
            (ep[ok], 100.0 * ev[ok] / F_V)
    return out


def metrics(ep, y, final_window=10):
    final = float(np.mean(y[-final_window:]))
    half = ep <= ep[-1] / 2
    t90 = ep[np.argmax(y >= 0.9 * final)] if np.any(y >= 0.9 * final) \
        else ep[-1]
    return dict(final=final, auc=float(np.mean(y)),
                auc_half=float(np.mean(y[half])), t90=float(t90))


def boot_ci(d, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def compare(curves, a, b):
    seeds = sorted(set(curves[a]) & set(curves[b]))
    if len(seeds) < 2:
        return None
    ma = [metrics(*curves[a][s]) for s in seeds]
    mb = [metrics(*curves[b][s]) for s in seeds]
    row = dict(n=len(seeds))
    for key in ("final", "auc", "auc_half", "t90"):
        x = [m[key] for m in ma]
        y = [m[key] for m in mb]
        diff, p = paired_sign_flip_test(x, y)
        lo, hi = boot_ci(np.asarray(x) - np.asarray(y))
        wins = int(np.sum(np.asarray(x) > np.asarray(y)))
        row[key] = dict(a=float(np.mean(x)), b=float(np.mean(y)), diff=diff,
                        p=p, ci=(lo, hi), wins=wins)
    return row


def plot(curves_by_key, path, modes=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted(curves_by_key)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.6),
                             squeeze=False)
    for ax, key in zip(axes[0], keys):
        cur = curves_by_key[key]
        for mode in (modes or sorted(cur)):
            if mode not in cur:
                continue
            runs = list(cur[mode].values())
            n = min(len(r[0]) for r in runs)
            ep = runs[0][0][:n]
            Y = np.stack([r[1][:n] for r in runs])
            m, se = Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))
            ax.plot(ep, m, label=mode, lw=1.2)
            ax.fill_between(ep, m - se, m + se, alpha=0.2, lw=0)
        ax.set_title(f"{key[0].split()[0]} k={key[1]}", fontsize=9)
        ax.set_xlabel("episode", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("greedy eval, % of F(V)", fontsize=8)
    axes[0][-1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", required=True, nargs="+")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="A:B mode pairs, e.g. MACS:DR")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--plot-modes", nargs="*", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    curves = {}
    for d in args.log_dir:
        for key, arms in load_curves(d).items():
            curves.setdefault(key, {}).update(arms)
    if not curves:
        print("no logs found")
        return 1

    rows = []
    for key in sorted(curves):
        print(f"\n== {key[0]}  k={key[1]}  arms: {sorted(curves[key])}")
        print(f"{'pair':>14} {'n':>3} | {'metric':>8} {'A':>7} {'B':>7} "
              f"{'A-B':>7} {'95% CI':>15} {'wins':>5} {'p':>7}")
        for pr in args.pairs:
            a, b = pr.split(":")
            if a not in curves[key] or b not in curves[key]:
                continue
            row = compare(curves[key], a, b)
            if row is None:
                continue
            for metric in ("final", "auc", "auc_half", "t90"):
                m = row[metric]
                print(f"{pr:>14} {row['n']:>3} | {metric:>8} {m['a']:>7.2f} "
                      f"{m['b']:>7.2f} {m['diff']:>+7.2f} "
                      f"[{m['ci'][0]:>+6.2f},{m['ci'][1]:>+6.2f}] "
                      f"{m['wins']:>2}/{row['n']:<2} {m['p']:>7.4f}")
            rows.append(dict(env=key[0], k=key[1], pair=pr, **row))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    if args.plot:
        plot(curves, args.plot, args.plot_modes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
