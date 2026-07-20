"""
plot_multiseed.py — figures from multi-seed logs (run_multiseed_sat.py etc.)

Covers the "Bat buoc" checklist items 2 and 3:
  * learning curves: per (config, method), mean over seeds with a +-1 std
    band, aligned by evaluation checkpoint
  * final bar chart: seed-level mean with std error bars per method, one
    panel per configuration, annotated with the exact permutation p-value
    for MACS-CLIP vs LOCAL
  * clip-fraction diagnostic: mean +- std of clip_frac over training for
    MACS-CLIP (the mechanism plot backing hypothesis H4)

Expects logs whose labels end in "_seed<n>" (the convention of
run_multiseed_sat.py, run_mc_sweep.py, run_scalability.py).

Usage:
  python plot_multiseed.py --log-dir logs/macs_v3_multiseed
  python plot_multiseed.py --log-dir logs/macs_v3_multiseed --window 10
"""

import argparse
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np

COLOR = {"SHARED": "#e67e22", "LOCAL": "#e74c3c", "DR": "#9b59b6",
         "VDN": "#3498db", "QMIX": "#1abc9c",
         "MACS": "#2ecc71", "MACS-CLIP": "#27ae60", "MACS-MC": "#95a5a6"}
ORDER = ["SHARED", "LOCAL", "DR", "VDN", "QMIX", "MACS", "MACS-CLIP",
         "MACS-MC"]
FINAL_WINDOW = 10


# =============================================================================
# Loading
# =============================================================================

def load_runs(log_dir):
    """
    Returns {(env_name, k): {method: {seed: log_dict}}} for all
    seed-labelled logs.
    """
    out = {}
    for p in sorted(Path(log_dir).glob("*seed*.json")):
        with open(p) as f:
            d = json.load(f)
        try:
            seed = int(d["label"].rsplit("_seed", 1)[1])
        except (IndexError, ValueError):
            continue
        c = d["config"]
        out.setdefault((c["env_name"], c["k"]), {}) \
           .setdefault(c["method"], {})[seed] = d
    return out


def stack_metric(runs_by_seed, key):
    """
    Align one per-block metric across seeds. Truncates to the shortest run
    so partially finished seeds do not crash plotting. Returns
    (episodes, matrix of shape (n_seeds, n_blocks)) or (None, None).
    """
    series, episodes = [], None
    for seed in sorted(runs_by_seed):
        pb = runs_by_seed[seed]["per_block"]
        vals = pb.get(key)
        if not vals:
            continue
        series.append([v for v in vals])
        episodes = pb["episode"]
    if not series:
        return None, None
    n = min(len(s) for s in series)
    mat = np.array([s[:n] for s in series], dtype=float)
    return np.array(episodes[:n]), mat


def final_metrics(runs_by_seed, window=FINAL_WINDOW):
    """Per-seed scalar: mean eval_coverage over the last `window` blocks."""
    vals = {}
    for seed, d in runs_by_seed.items():
        ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
        if len(ev) >= window:
            vals[seed] = float(np.mean(ev[-window:]))
    return vals


def exact_perm_p(a, b):
    """Two-sided exact permutation test on difference of means."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n, na = len(pooled), len(a)
    obs = abs(a.mean() - b.mean())
    if comb(n, na) > 2 ** 16:
        rng = np.random.default_rng(0)
        diffs = [abs(pooled[p[:na]].mean() - pooled[p[na:]].mean())
                 for p in (rng.permutation(n) for _ in range(20_000))]
    else:
        diffs = []
        for idx in itertools.combinations(range(n), na):
            m = np.zeros(n, bool)
            m[list(idx)] = True
            diffs.append(abs(pooled[m].mean() - pooled[~m].mean()))
    return float(np.mean(np.array(diffs) >= obs - 1e-12))


# =============================================================================
# Figures
# =============================================================================

def fig_learning_curves(data, out_dir, window):
    import matplotlib.pyplot as plt
    for (env_name, k), methods in sorted(data.items(), key=lambda x: x[0]):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fmax = None
        for m in ORDER:
            if m not in methods:
                continue
            ep, mat = stack_metric(methods[m], "eval_coverage")
            if ep is None:
                continue
            mu, sd = mat.mean(0), mat.std(0, ddof=1) if len(mat) > 1 \
                else (mat.mean(0), np.zeros(mat.shape[1]))
            n_seeds = mat.shape[0]
            ax.plot(ep, mu, color=COLOR[m], lw=1.8,
                    label=f"{m} (n={n_seeds})")
            ax.fill_between(ep, mu - sd, mu + sd, color=COLOR[m], alpha=0.15)
            fmax = methods[m][sorted(methods[m])[0]]["config"].get("max_cells")
        if fmax:
            ax.axhline(fmax, color="k", ls=":", lw=0.8, alpha=0.5)
            ax.text(ax.get_xlim()[0], fmax, f" F(V)={fmax}", va="bottom",
                    fontsize=8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Eval objective F(S_T)")
        ax.set_title(f"{env_name}, k={k}  (mean +- std over seeds)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2)
        fp = Path(out_dir) / f"curves_{env_name.replace(' ','_')}_k{k}.png"
        plt.tight_layout()
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fp}")


def fig_final_bars(data, out_dir, window):
    import matplotlib.pyplot as plt
    configs = sorted(data.keys(), key=lambda x: (x[1], x[0]))
    fig, axes = plt.subplots(1, len(configs),
                             figsize=(3.2 * len(configs), 4.0), sharey=False)
    if len(configs) == 1:
        axes = [axes]
    for ax, key in zip(axes, configs):
        env_name, k = key
        methods = data[key]
        names, means, stds = [], [], []
        finals = {}
        for m in ORDER:
            if m not in methods:
                continue
            fv = final_metrics(methods[m], window)
            if not fv:
                continue
            arr = np.array(list(fv.values()))
            finals[m] = list(fv.values())
            names.append(m)
            means.append(arr.mean())
            stds.append(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, capsize=3,
               color=[COLOR[n] for n in names])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        title = f"{env_name}, k={k}"
        if "MACS-CLIP" in finals and "LOCAL" in finals:
            p = exact_perm_p(finals["MACS-CLIP"], finals["LOCAL"])
            title += f"\nCLIP vs LOCAL: p={p:.3f}"
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", alpha=0.2)
    axes[0].set_ylabel(f"Final F(S_T), last {window} checkpoints")
    fp = Path(out_dir) / "final_bars.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")


def fig_clip_fraction(data, out_dir):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for (env_name, k), methods in sorted(data.items(), key=lambda x: x[0]):
        if "MACS-CLIP" not in methods:
            continue
        ep, mat = stack_metric(methods["MACS-CLIP"], "clip_frac")
        if ep is None or np.all(np.isnan(mat)):
            continue
        mu = np.nanmean(mat, 0)
        sd = np.nanstd(mat, 0, ddof=1) if len(mat) > 1 else np.zeros_like(mu)
        line, = ax.plot(ep, 100 * mu, lw=1.8, label=f"{env_name}, k={k}")
        ax.fill_between(ep, 100 * (mu - sd), 100 * (mu + sd),
                        color=line.get_color(), alpha=0.15)
        plotted = True
    if not plotted:
        print("No clip_frac data found; skipping clip figure.")
        return
    ax.set_xlabel("Episode")
    ax.set_ylabel("Bootstrap targets clipped (%)")
    ax.set_title("MACS-Clip activation over training (mean +- std over seeds)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    fp = Path(out_dir) / "clip_fraction.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/macs_v3_multiseed")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--window", type=int, default=FINAL_WINDOW)
    args = ap.parse_args()

    out_dir = args.out_dir or str(Path("plots") /
                                  Path(args.log_dir).name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data = load_runs(args.log_dir)
    if not data:
        print(f"No seed-labelled logs in {args.log_dir}. "
              "Run run_multiseed_sat.py first.")
        raise SystemExit(1)

    fig_learning_curves(data, out_dir, args.window)
    fig_final_bars(data, out_dir, args.window)
    fig_clip_fraction(data, out_dir)