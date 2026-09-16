"""
The sweep harness: plan -> execute -> inspect -> aggregate.

Five runner scripts each carried their own copy of this scaffolding
(run_facility, run_mc_sweep, run_multiseed_sat, run_o9, run_scalability),
and the copies had drifted: only two had the resume check, only one had the
paired test, one was an outdated fork of another down to its docstring.

A Run is the unit of work. Every field in it appears in the label, so two
runs differ in their label exactly when they differ in their configuration,
and the resume check is then just "does this label already have a complete
log".
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from macs import runlog
from macs.stats import exact_perm_test, f1, min_attainable_p, \
    paired_sign_flip_test


@dataclass(frozen=True)
class Run:
    env_name: str
    k: int
    mode: str
    seed: int
    obs_mode: str = "shared"
    clip: str = "none"
    lr: Optional[float] = None
    factory: Optional[Callable] = field(default=None, compare=False,
                                        repr=False)
    #: emit the historic short label "{env}_k{k}_{mode}_seed{s}" instead of
    #: the full one. The saturating multiseed sweep predates observation and
    #: clip modes, and its logs on disk carry the short form; changing the
    #: label would make every completed run look incomplete and rerun it.
    compact: bool = False

    @property
    def label(self):
        stem = f"{self.env_name.replace(' ', '_')}_k{self.k}_{self.mode}"
        if self.compact:
            return f"{stem}_seed{self.seed}"
        base = f"{stem}_{self.obs_mode}_clip-{self.clip}"
        if self.lr is not None:
            base += f"_lr{self.lr:g}"
        return f"{base}_seed{self.seed}"

    @property
    def arm(self):
        arm = f"{self.mode}/{self.obs_mode}/{self.clip}"
        if self.lr is not None:
            arm += f"/lr{self.lr:g}"
        return arm


def execute(runs, cfg, log_dir, dry_run=False, **train_kw):
    """
    Train every run that does not already have a complete log.

    Resumable: interrupt and relaunch and it picks up where it stopped.
    """
    from macs.train import train
    n_ep, log_ev = cfg["n_episodes"], cfg["log_every"]
    done = skipped = 0
    for r in runs:
        if runlog.is_complete(r.label, log_dir, n_ep, log_ev):
            print(f"  skip (complete): {r.label}")
            skipped += 1
            continue
        if dry_run:
            print(f"  would run: {r.label}")
            continue
        kw = dict(cfg)
        if r.lr is not None:
            kw["lr"] = r.lr
        train(r.factory, env_name=r.env_name, mode=r.mode,
              obs_mode=r.obs_mode, clip=r.clip, seed=r.seed,
              label=r.label, log_dir=log_dir, **kw, **train_kw)
        done += 1
    if not dry_run:
        print(f"\n{done} run(s) completed, {skipped} skipped.")
    return done, skipped


# =============================================================================
# Inspection
# =============================================================================

def inspect(log_dir):
    """
    Per-run health check. Four things worth looking at before trusting a
    sweep:

      credit_err  max |sum_i phi_i - r_t|. Zero to machine precision for the
                  permutation, subset and facility backends alike. A nonzero
                  value on a MACS arm means the exact backend is NOT in use.
                  Expected to be nonzero for SHARED, LOCAL and DR, and n/a
                  for VDN and QMIX, which have no credit rule.
      rising      whether the last evaluation block is still above the one
                  before. If every arm is still rising, the run is
                  undertrained and a null result is uninterpretable.
      diverging   TD loss still climbing against the MIDPOINT of the run and
                  evaluation below its own peak. Compared against the
                  midpoint rather than the previous block because a
                  diverging run climbs steadily: the k=6 private arm went
                  0.11 -> 3.3 -> 18.3 and a consecutive-thirds test missed
                  it.
      clip_frac   fraction of bootstrap targets the ceiling actually cut.
    """
    rows = sorted(Path(log_dir).glob("*seed*.json"))
    if not rows:
        print(f"No logs in {log_dir}.")
        return
    print(f"\n{'run':<58} {'blocks':>6} {'last10':>8} {'credit_err':>11} "
          f"{'rising':>7} {'clip':>6}")
    print("-" * 100)
    for p in rows:
        d = runlog.load(p)
        pb = d["per_block"]
        ev = [x for x in pb["eval_coverage"] if x is not None]
        errs = [x for x in (pb.get("credit_sum_err") or []) if x is not None]
        err = max(errs) if errs else float("nan")
        cf = pb.get("clip_frac") or [0.0]
        last10 = np.mean(ev[-10:]) if ev else float("nan")
        rising = (len(ev) >= 4 and
                  np.mean(ev[-2:]) > np.mean(ev[-4:-2]) + 1e-9)
        flag = ("  <-- credit error nonzero"
                if d["config"].get("method") in ("MACS", "MACS-CLIP")
                and err > 1e-6 else "")
        ls = [x for x in (pb.get("loss") or []) if x is not None]
        mid = len(ls) // 2
        diverging = (len(ls) >= 8 and len(ev) >= 8
                     and np.mean(ls[-3:]) > 1.5 * np.mean(ls[mid - 1:mid + 2])
                     and last10 < 0.9 * max(ev))
        if diverging:
            flag += "  <-- DIVERGING (loss climbing, eval past peak)"
        es = "n/a" if err != err else f"{err:.2e}"      # nan -> no credit rule
        print(f"{p.stem[:58]:<58} {len(ev):>6} {last10:>8.2f} "
              f"{es:>11} {str(rising):>7} {np.mean(cf):>6.3f}{flag}")
    print("\nrising=True on every arm means the sweep is undertrained; a null "
          "difference there is not evidence of no difference.")
    print("DIVERGING means the run did not converge, so its score is not an "
          "information-limited optimum and should not be read as one.")


# =============================================================================
# Aggregation
# =============================================================================

def collect(log_dir, window=10):
    """
    {(env_name, k): {arm: {seed: metric}}} plus a note about short logs.

    Also returns mean bootstrap-clip activation per arm. On islands B(S') =
    F(V) - F(S') stays near 310 while bootstrap values sit near 250, so the
    ceiling never binds and the clipped arm runs the same backup as the
    unclipped one. That has to be printed, not inferred.
    """
    out, clip, short, n_files = {}, {}, [], 0
    for p in sorted(Path(log_dir).glob("*seed*.json")):
        try:
            d = runlog.load(p)
        except (json.JSONDecodeError, OSError):
            continue
        c = d["config"]
        n_files += 1
        ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
        if len(ev) < window:
            short.append((p.name, len(ev)))
            continue
        arm = runlog.arm_key(c)
        key = (c["env_name"], c["k"])
        out.setdefault(key, {}).setdefault(arm, {})[c.get("seed", 0)] = \
            float(np.mean(ev[-window:]))
        cf = [x for x in d["per_block"].get("clip_frac", []) if x is not None]
        clip.setdefault(key, {}).setdefault(arm, []).append(
            float(np.mean(cf)) if cf else 0.0)
    if short:
        print(f"note: {len(short)} of {n_files} log(s) have fewer than "
              f"{window} evaluation checkpoints and were skipped "
              f"(e.g. {short[0][0]} has {short[0][1]}). This is expected for "
              f"--smoke runs; pass --window to lower the requirement.")
    return out, clip


def aggregate(comparisons, log_dir, window=10, floors=None, paired=True,
              out_path=None):
    """
    Print a per-configuration summary and run every available comparison.

    comparisons : [(arm_hi, arm_lo, note)] using bare arm keys; learning-rate
                  suffixes are resolved by prefix
    floors      : optional callable(k) -> {name: value in units of F(V)},
                  printed alongside the learned arms
    paired      : sign-flip test on per-seed differences when the two arms
                  share seeds; falls back to the unpaired permutation test
    """
    data, clip = collect(log_dir, window)
    if not data:
        print(f"No seed-labelled logs in {log_dir}.")
        return []
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    for (env_name, k), by_arm in sorted(data.items(), key=lambda x: x[0][1]):
        emit(f"\n=== {env_name}, k={k} ===")
        if floors is not None:
            fl = floors(k)
            emit("  reference policies (no learning):")
            for name, v in fl.items():
                emit(f"    {name:<12} {f1(v):>10}")
        emit(f"  {'arm':<34} {'n':>3} {'mean':>9} {'sd':>8} {'clip':>7}")
        for arm in sorted(by_arm):
            vals = np.array(list(by_arm[arm].values()), dtype=float)
            cf = np.mean(clip.get((env_name, k), {}).get(arm, [0.0]))
            emit(f"  {arm:<34} {len(vals):>3} {f1(vals.mean()):>9} "
                 f"{f1(vals.std(ddof=1) if len(vals) > 1 else 0.0):>8} "
                 f"{cf:>7.3f}")

        for hi_key, lo_key, note in comparisons:
            hi = runlog.resolve_arm(hi_key, by_arm)
            lo = runlog.resolve_arm(lo_key, by_arm)
            if hi is None or lo is None:
                continue
            seeds = sorted(set(by_arm[hi]) & set(by_arm[lo]))
            if len(seeds) < 2:
                emit(f"  {hi} vs {lo}: only {len(seeds)} shared seed(s), "
                     f"no test")
                continue
            a = np.array([by_arm[hi][s] for s in seeds])
            b = np.array([by_arm[lo][s] for s in seeds])
            if paired:
                diff, p = paired_sign_flip_test(a, b)
                floor = min_attainable_p(len(seeds))
                bound = ("" if floor <= 0.05 else
                         f"  (n={len(seeds)}: smallest attainable p is "
                         f"{floor:.3f}, so 0.05 is unreachable)")
            else:
                diff, p = exact_perm_test(a, b)
                bound = ""
            emit(f"  {note}: {hi} - {lo} = {f1(diff)}  p={p:.4f}"
                 f"  (n={len(seeds)}){bound}")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n".join(lines))
        print(f"\nWrote {out_path}")
    return lines


# =============================================================================
# Cached reference floors
# =============================================================================

_FLOOR_CACHE = {}


def cached_floors(key, compute, cache_path=None):
    """
    Reference floors are deterministic given their seed and take seconds per
    configuration, so they are memoised in process and optionally on disk.
    """
    if key in _FLOOR_CACHE:
        return _FLOOR_CACHE[key]
    if cache_path and Path(cache_path).exists():
        disk = json.loads(Path(cache_path).read_text())
        if str(key) in disk:
            _FLOOR_CACHE[key] = disk[str(key)]
            return disk[str(key)]
    out = compute()
    _FLOOR_CACHE[key] = out
    if cache_path:
        disk = {}
        if Path(cache_path).exists():
            disk = json.loads(Path(cache_path).read_text())
        disk[str(key)] = out
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(disk, indent=2))
    return out
