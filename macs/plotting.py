"""
Figures from run logs.

One module replaces plot_macs.py and plot_multiseed.py, which were the
single-seed and multi-seed versions of the same three figures. The seed
count is data, not a reason for a second script: a directory with one seed
per arm plots a line with no band, a directory with five plots a band.

Arms are keyed the way macs.runlog keys them, so observation modes, clip
modes and learning-rate sweeps each get their own line instead of being
silently averaged together under their method name.

Figures:
  curves_<env>_k<k>.png   evaluation score against episode, mean +- sd
  final_bars.png          final score per arm, with the paired test on the
                          headline comparison where both arms are present
  clip_fraction.png       what fraction of bootstrap targets the ceiling
                          actually cut. Worth plotting because on islands
                          B(S') sits well above the bootstrap values, the
                          ceiling never binds, and the clipped arm is then
                          running the same backup as the unclipped one --
                          which has to be shown, not inferred.
"""

from pathlib import Path

import numpy as np

from macs import runlog
from macs.stats import paired_sign_flip_test

COLOR = {"SHARED": "#e67e22", "LOCAL": "#e74c3c", "DR": "#9b59b6",
         "VDN": "#3498db", "QMIX": "#1abc9c", "MACS": "#2ecc71",
         "MACS-CLIP": "#27ae60", "MACS-MC": "#95a5a6"}
ORDER = ["SHARED", "LOCAL", "DR", "VDN", "QMIX", "MACS", "MACS-CLIP",
         "MACS-MC"]
FINAL_WINDOW = 10


def _color(arm):
    return COLOR.get(arm.split("/")[0], "#7f8c8d")


def _rank(arm):
    head = arm.split("/")[0]
    return (ORDER.index(head) if head in ORDER else len(ORDER), arm)


def load_runs(log_dir):
    """{(env_name, k): {arm: {seed: log}}} over every seed-labelled log."""
    out = {}
    for p in sorted(Path(log_dir).glob("*.json")):
        if p.name.startswith("runtime_") or "_seed" not in p.name:
            continue
        try:
            d = runlog.load(p)
        except (ValueError, OSError):
            continue
        c = d["config"]
        seed = c.get("seed")
        if seed is None:
            try:
                seed = int(d["label"].rsplit("_seed", 1)[1])
            except (IndexError, ValueError):
                continue
        out.setdefault((c["env_name"], c["k"]), {}) \
           .setdefault(runlog.arm_key(c), {})[seed] = d
    return out


def stack_metric(by_seed, key):
    """(episodes, (n_seeds, n_blocks)) truncated to the shortest run."""
    series, episodes = [], None
    for seed in sorted(by_seed):
        pb = by_seed[seed]["per_block"]
        vals = [np.nan if v is None else float(v) for v in pb.get(key, [])]
        if not vals:
            continue
        series.append(vals)
        ep = pb["episode"]
        episodes = ep if episodes is None else episodes
    if not series:
        return None, None
    n = min(len(s) for s in series)
    return np.asarray(episodes[:n]), np.asarray([s[:n] for s in series])


def final_metrics(by_seed, window=FINAL_WINDOW):
    out = {}
    for seed, d in by_seed.items():
        ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
        if ev:
            out[seed] = float(np.mean(ev[-window:]))
    return out


def _band(mat):
    mu = np.nanmean(mat, 0)
    sd = (np.nanstd(mat, 0, ddof=1) if mat.shape[0] > 1
          else np.zeros_like(mu))
    return mu, sd


def fig_learning_curves(data, out_dir):
    import matplotlib.pyplot as plt
    for (env_name, k), arms in sorted(data.items()):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fmax = None
        for arm in sorted(arms, key=_rank):
            ep, mat = stack_metric(arms[arm], "eval_coverage")
            if ep is None:
                continue
            mu, sd = _band(mat)
            ax.plot(ep, mu, color=_color(arm), lw=1.8,
                    label=f"{arm} (n={mat.shape[0]})")
            ax.fill_between(ep, mu - sd, mu + sd, color=_color(arm),
                            alpha=0.15)
            first = arms[arm][sorted(arms[arm])[0]]
            fmax = first["config"].get("max_cells", fmax)
        if fmax:
            ax.axhline(fmax, color="k", ls=":", lw=0.8, alpha=0.5)
            ax.text(ax.get_xlim()[0], fmax, f" F(V)={fmax:g}",
                    va="bottom", fontsize=8)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Eval objective F(S_T)")
        ax.set_title(f"{env_name}, k={k}  (mean +- sd over seeds)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2)
        fp = Path(out_dir) / f"curves_{env_name.replace(' ', '_')}_k{k}.png"
        plt.tight_layout()
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fp}")


def fig_final_bars(data, out_dir, window=FINAL_WINDOW, headline=None):
    import matplotlib.pyplot as plt
    configs = sorted(data, key=lambda x: (x[1], x[0]))
    fig, axes = plt.subplots(1, len(configs),
                             figsize=(3.4 * len(configs), 4.2))
    axes = np.atleast_1d(axes)
    for ax, key in zip(axes, configs):
        env_name, k = key
        names, means, stds, finals = [], [], [], {}
        for arm in sorted(data[key], key=_rank):
            fv = final_metrics(data[key][arm], window)
            if not fv:
                continue
            finals[arm] = fv
            arr = np.array(list(fv.values()))
            names.append(arm)
            means.append(arr.mean())
            stds.append(arr.std(ddof=1) if len(arr) > 1 else 0.0)
        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, capsize=3,
               color=[_color(n) for n in names])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        title = f"{env_name}, k={k}"
        if headline:
            hi = runlog.resolve_arm(headline[0], finals)
            lo = runlog.resolve_arm(headline[1], finals)
            if hi and lo:
                seeds = sorted(set(finals[hi]) & set(finals[lo]))
                if len(seeds) >= 2:
                    _, p = paired_sign_flip_test(
                        [finals[hi][s] for s in seeds],
                        [finals[lo][s] for s in seeds])
                    title += (f"\n{hi.split('/')[0]} vs {lo.split('/')[0]}: "
                              f"p={p:.3f} (n={len(seeds)})")
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
    for (env_name, k), arms in sorted(data.items()):
        for arm in sorted(arms, key=_rank):
            if "clip-none" in arm or "/none" in arm:
                continue
            ep, mat = stack_metric(arms[arm], "clip_frac")
            if ep is None or np.all(np.isnan(mat)) or np.nanmax(mat) <= 0:
                continue
            mu, sd = _band(mat)
            line, = ax.plot(ep, 100 * mu, lw=1.8,
                            label=f"{arm}  {env_name}, k={k}")
            ax.fill_between(ep, 100 * (mu - sd), 100 * (mu + sd),
                            color=line.get_color(), alpha=0.15)
            plotted = True
    if not plotted:
        print("No nonzero clip_frac data found; skipping clip figure. "
              "A flat zero means the ceiling never bound, so the clipped "
              "arm ran the same backup as the unclipped one.")
        plt.close()
        return
    ax.set_xlabel("Episode")
    ax.set_ylabel("Bootstrap targets clipped (%)")
    ax.set_title("Clip activation over training (mean +- sd over seeds)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    fp = Path(out_dir) / "clip_fraction.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")
