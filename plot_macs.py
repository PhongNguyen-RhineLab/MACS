"""
plot_macs.py — figures for the MACS experiments (v2 and v3 logs).

Produces, for a given log directory:

  1. One learning-curve figure per (environment, k) configuration:
     greedy evaluation coverage vs episode, all methods overlaid, with
     the shaded raw training coverage behind each eval curve.
     -> plots/<dir>/<env>_k<k>_curves.png

  2. One grouped summary bar chart across ALL configurations:
     final performance = mean of the last 5 evaluation points,
     normalized to percent of the achievable objective (max_cells,
     which the v3 environments set to F(V)).
     -> plots/<dir>/summary_bar.png

  3. A console table with the same numbers (paste-ready for the paper).

Usage:
  python plot_macs.py                              # logs/macs_v2
  python plot_macs.py --log-dir logs/macs_v3
  python plot_macs.py --log-dir logs/macs_v2 --out-dir plots/macs_v2
"""

import argparse
import json
from pathlib import Path

import numpy as np

COLOR = {
    "SHARED":    "#e67e22",
    "LOCAL":     "#e74c3c",
    "DR":        "#9b59b6",
    "VDN":       "#3498db",
    "QMIX":      "#34495e",
    "MACS":      "#2ecc71",
    "MACS-CLIP": "#16a085",
    "MACS-MC":   "#f1c40f",
}
ORDER = ["SHARED", "LOCAL", "DR", "VDN", "QMIX", "MACS-MC", "MACS", "MACS-CLIP"]


def load_logs(log_dir):
    """Group logs by (env_name, k) -> {method: log_dict}."""
    groups = {}
    for p in sorted(Path(log_dir).glob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
            key = (d["config"]["env_name"], d["config"]["k"])
            groups.setdefault(key, {})[d["config"]["method"]] = d
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    return groups


def last5_eval(d):
    ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
    return float(np.mean(ev[-5:])) if ev else float("nan")


def plot_curves(groups, out_dir):
    import matplotlib.pyplot as plt
    for (env_name, k), methods in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for m in ORDER:
            if m not in methods:
                continue
            d = methods[m]
            pb = d["per_block"]
            ep = pb["episode"]
            ax.plot(ep, pb["eval_coverage"], label=m,
                    color=COLOR.get(m, "gray"), lw=2.0)
            ax.plot(ep, pb["coverage_mean"],
                    color=COLOR.get(m, "gray"), alpha=0.15, lw=0.8)
        mc = methods[next(iter(methods))]["config"].get("max_cells")
        if mc:
            ax.axhline(mc, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Eval coverage (objective value)")
        ax.set_title(f"{env_name}, k={k}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fp = Path(out_dir) / f"{env_name.replace(' ', '_')}_k{k}_curves.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"Saved: {fp}")
        plt.close()


def plot_summary_bar(groups, out_dir):
    import matplotlib.pyplot as plt
    keys = sorted(groups.keys())
    labels = [f"{env}\nk={k}" for env, k in keys]
    present = [m for m in ORDER if any(m in groups[key] for key in keys)]
    x = np.arange(len(keys))
    width = 0.9 / max(len(present), 1)

    fig, ax = plt.subplots(figsize=(max(10, 1.8 * len(keys)), 5))
    for i, m in enumerate(present):
        vals = []
        for key in keys:
            d = groups[key].get(m)
            if d is None:
                vals.append(0.0)
                continue
            mc = d["config"].get("max_cells", 1) or 1
            vals.append(100.0 * last5_eval(d) / mc)
        ax.bar(x + i * width, vals, width, label=m,
               color=COLOR.get(m, "gray"), alpha=0.9)
    ax.set_xticks(x + width * (len(present) - 1) / 2)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Final eval, % of achievable F(V) (mean of last 5 evals)")
    ax.set_ylim(0, 105)
    ax.set_title("MACS comparison — final greedy evaluation performance")
    ax.legend(fontsize=9, ncol=min(len(present), 4))
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    fp = Path(out_dir) / "summary_bar.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    print(f"Saved: {fp}")
    plt.close()


def console_table(groups):
    keys = sorted(groups.keys())
    present = [m for m in ORDER if any(m in groups[key] for key in keys)]
    head = f"  {'Configuration':<26}" + "".join(f"{m:>11}" for m in present)
    print("\n" + "-" * len(head))
    print(head)
    print("-" * len(head))
    for key in keys:
        env, k = key
        row = f"  {env + f', k={k}':<26}"
        for m in present:
            d = groups[key].get(m)
            if d is None:
                row += f"{'---':>11}"
            else:
                mc = d["config"].get("max_cells", 1) or 1
                v = last5_eval(d)
                row += f"{v:>6.1f}/{100 * v / mc:>3.0f}%"
        print(row)
    print("-" * len(head))
    print("  value = mean of last 5 greedy evals | % of achievable F(V)\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/macs_v2")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.log_dir.replace("logs", "plots")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("pip install matplotlib")
        raise SystemExit(1)

    groups = load_logs(args.log_dir)
    if not groups:
        print(f"No logs found in {args.log_dir}")
        raise SystemExit(1)
    plot_curves(groups, out_dir)
    plot_summary_bar(groups, out_dir)
    console_table(groups)