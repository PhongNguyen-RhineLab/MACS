"""
The run log: one JSON file per arm, per seed.

Schema (unchanged from the pre-refactor logs, so every existing log in
logs/ is still readable and every existing plot script still works):

    {
      "label":  str,
      "config": {method, env_name, k, map_size, horizon, max_cells,
                 n_episodes, obs_mode, clip, seed, lr, gamma, batch_size,
                 F_total, shapley_backend, mc_m, ...},
      "per_block": {episode, coverage_mean, eval_coverage, loss, epsilon,
                    credit_sum_err, clip_frac},
      "summary": {last5_train, last10_eval, final_eval}
    }

Two conventions worth keeping straight:

  credit_sum_err is None, not 0.0, when the arm has no credit rule. An
  empty list means the mode learns from the team reward directly, which is
  not the same as a credit rule with zero efficiency error. Readers must
  render it as n/a.

  The old two trainers wrote different summary keys -- last5_train in one,
  last10_eval in the other -- so aggregators had to know which trainer
  produced a file. Both are written now.
"""

import json
from pathlib import Path

import numpy as np

BLOCK_KEYS = ("episode", "coverage_mean", "eval_coverage", "loss",
              "epsilon", "credit_sum_err", "clip_frac")


class RunLog:
    """Accumulates per-block metrics and writes them atomically."""

    def __init__(self, label, config, log_dir):
        self.label = label
        self.log_dir = Path(log_dir)
        self.log = {"label": label,
                    "config": dict(config),
                    "per_block": {key: [] for key in BLOCK_KEYS},
                    "summary": {}}

    @property
    def config(self):
        return self.log["config"]

    @property
    def path(self):
        return self.log_dir / f"{self.label}.json"

    def append_block(self, *, episode, block, eval_coverage, epsilon):
        pb = self.log["per_block"]
        pb["episode"].append(episode)
        pb["coverage_mean"].append(round(float(np.mean(block["cov"])), 3)
                                   if block["cov"] else None)
        pb["eval_coverage"].append(round(float(eval_coverage), 3))
        pb["loss"].append(round(float(np.mean(block["loss"])), 5)
                          if block["loss"] else None)
        pb["epsilon"].append(round(float(epsilon), 3))
        pb["credit_sum_err"].append(round(float(np.max(block["err"])), 8)
                                    if block["err"] else None)
        pb["clip_frac"].append(round(float(np.mean(block["clip"])), 5)
                               if block["clip"] else 0.0)

    def finish(self):
        pb = self.log["per_block"]
        train = [x for x in pb["coverage_mean"] if x is not None]
        ev = [x for x in pb["eval_coverage"] if x is not None]
        self.log["summary"] = {
            "last5_train": round(float(np.mean(train[-5:])), 3) if train
            else None,
            "last10_eval": round(float(np.mean(ev[-10:])), 3) if ev else None,
            "final_eval": round(float(ev[-1]), 3) if ev else None,
        }

    def save(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.log, indent=2))
        tmp.replace(self.path)          # atomic: no half-written log on kill


# =============================================================================
# Reading
# =============================================================================

def load(path):
    with open(path) as f:
        return json.load(f)


def is_complete(label, log_dir, n_episodes, log_every):
    """True when the log already holds every block the run would write."""
    p = Path(log_dir) / f"{label}.json"
    if not p.exists():
        return False
    try:
        d = load(p)
    except (json.JSONDecodeError, KeyError, OSError):
        return False
    blocks = d.get("per_block", {}).get("episode", [])
    return len(blocks) >= n_episodes // log_every


def final_metric(path, window=10):
    """Mean evaluation score over the last `window` logged blocks."""
    d = load(path)
    ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
    if not ev:
        return None
    return float(np.mean(ev[-window:]))


def still_rising(path, window=10, tol=1e-9):
    """
    True when the last `window` blocks are still improving on the previous
    `window`. A run that is still rising at the cutoff reports its training
    budget position, not converged performance.
    """
    d = load(path)
    ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
    if len(ev) < 2 * window:
        return None
    return float(np.mean(ev[-window:])) > float(
        np.mean(ev[-2 * window:-window])) + tol


def collect(log_dir, window=10):
    """
    Group every log under `log_dir` by (env_name, k) then by arm.

    The arm key is method plus whichever axes vary, so o9 files
    (MACS/shared vs MACS/private), o10 files (clip-budget vs clip-const) and
    LR sweeps do not collide.
    """
    out = {}
    for p in sorted(Path(log_dir).glob("*.json")):
        try:
            d = load(p)
        except (json.JSONDecodeError, OSError):
            continue
        cfg = d["config"]
        key = (cfg["env_name"], cfg["k"])
        arm = arm_key(cfg)
        seed = cfg.get("seed", 0)
        out.setdefault(key, {}).setdefault(arm, {})[seed] = {
            "path": p,
            "final": final_metric(p, window),
            "clip_frac": _mean_tail(d["per_block"]["clip_frac"], window),
            "credit_sum_err": _max_tail(d["per_block"]["credit_sum_err"],
                                        window),
            "rising": still_rising(p, window),
        }
    return out


def arm_key(cfg):
    """
    "method/obs_mode/clip" plus "/lr{rate}" when the config records one.

    Unchanged from the pre-refactor format so the COMPARISONS tables in
    macs.sweep keep resolving against old logs.
    """
    arm = (f"{cfg['method']}/{cfg.get('obs_mode', 'shared')}"
           f"/{cfg.get('clip', 'none')}")
    if cfg.get("lr") is not None:
        arm += f"/lr{cfg['lr']:g}"
    return arm


def resolve_arm(key, by_arm):
    """
    Map a COMPARISONS key onto an arm actually present in `by_arm`.

    arm_key() appends "/lr{rate}" to every arm whose config records one, so
    the bare keys never match. With per-method learning rates the two sides
    of a comparison carry different suffixes, so no single bare key could
    match both. Resolve by prefix instead.

    Returns the arm name, or None when absent or ambiguous. Ambiguity is
    real information -- several rates for one method means the sweep has
    not been reduced to a headline -- so the caller reports it rather than
    guessing.
    """
    if key in by_arm:
        return key
    hits = [a for a in by_arm if a == key or a.startswith(key + "/lr")]
    return hits[0] if len(hits) == 1 else None


def _mean_tail(xs, window):
    vals = [x for x in xs if x is not None]
    return float(np.mean(vals[-window:])) if vals else None


def _max_tail(xs, window):
    vals = [x for x in xs if x is not None]
    return float(np.max(vals[-window:])) if vals else None
