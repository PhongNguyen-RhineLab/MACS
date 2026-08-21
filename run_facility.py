"""
run_facility.py -- the two experiments the paper still needs.

O9  Augmented-state ablation on the facility objective.
    MACS with obs_mode="shared" vs obs_mode="private", k in {2,3,4,6,8},
    five seeds. Turns the S-value ladder of the policy-level probe (8-21
    points, measured with greedy oracles) into a statement about learned
    agents, which is what Theorem `sufficient-statistic` actually needs.
    Neither arm clips, so the observation is the only difference.

O9-LR  Learning-rate robustness for the private arm.
    The k=6 pilot showed the private arm DIVERGING rather than converging
    to an information-limited optimum: its TD loss climbs monotonically
    (0.11 -> 18.3 over 2500 episodes) and its evaluation score peaks at
    episode 200 and decays to 34.7% of F(V) -- roughly half what a
    non-learning greedy blind policy scores on the same configuration
    (67.7%). That is consistent with Theorem `sufficient-statistic`:
    without S_t the per-agent process is not Markov, the credit phi_i for
    a given private state varies with unobserved teammate history, and
    the contraction argument of Lemma `contraction` does not apply.

    But it also means the pilot's 197-point gap confounds two things: the
    information deficit (bounded by the policy-level probe at ~122 points)
    and a learnability failure (the rest). Before reporting either, sweep
    the private arm over learning rates. If it diverges at every rate, the
    divergence is intrinsic to the observation and can be claimed as such;
    if a lower rate converges near the blind-greedy floor, report the
    information gap and the instability separately.

O10 Isolating the budget clip on the saturating suite.
    MACS-CLIP (state-dependent ceiling B(S')) vs clip="const" (a
    state-INDEPENDENT ceiling matched to the mean of B(S')), k in
    {2,3,4,6}, five seeds. If the constant reproduces the effect, the
    submodular structure is not what is doing the work.

Both sweeps resume: a run is skipped when its log already has the full
number of blocks, so the script can be interrupted and relaunched.

Usage:
  python run_facility.py --dry-run                  print the plan, train nothing
  python run_facility.py --pilot                    2 runs, 1 seed, k=6 (do this first)
  python run_facility.py --exp o9lr                 3 private runs at 3 learning rates
  python run_facility.py --exp o9 --lr 2e-4         override the learning rate
  python run_facility.py --exp o9                   full O9 sweep
  python run_facility.py --exp o10                  full O10 sweep
  python run_facility.py --exp o9 --seeds 0 1 2     subset of seeds
  python run_facility.py --exp o9 --ks 4 6          subset of team sizes
  python run_facility.py --aggregate --exp o9       summary + permutation tests
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import macs_main as M
import macs_v3  # noqa: F401   rebinds compute_credits to the v3 dispatcher
from macs_fl import FacilityCoverageEnv
from macs_v3 import SaturatingCoverageEnv
from macs_ablation import install_facility_credits, run_checks
from run_multiseed_sat import exact_perm_test, r1

LOG_DIR = {"o9": "logs/macs_facility", "o10": "logs/macs_clipcontrol",
           "o9lr": "logs/macs_facility_lr"}

# Learning rates for the O9-LR robustness arm. 5e-4 is the published
# setting and the one that diverged in the pilot.
LR_GRID = [5e-4, 2e-4, 1e-4]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

TRAIN_CFG = dict(n_episodes=4000, eval_every=100, log_every=100)
# 24x24 is four times the state space of the 12x12 saturating grid the
# published runs used, so a short pilot risks measuring undertraining rather
# than the ablation. 2500 episodes is the compromise; `inspect` reports
# whether both arms are still rising at the end, which is the check that
# matters more than the episode count itself.
PILOT_CFG = dict(n_episodes=2500, eval_every=100, log_every=100, warmup=1500)
SMOKE_CFG = dict(n_episodes=30, warmup=200, eval_every=10, n_eval=2,
                 log_every=10, target_every=10)

# Facility geometry. rho=4 with single-cell footprints and a 24x24 grid is
# the configuration where the probe measured the largest S-value ladder
# (7.9 / 13.5 / 17.9 / 21.2 / 19.6 at k = 2,3,4,6,8).
FACILITY = dict(size=24, horizon=60, rho=4.0, patch=0, demand="uniform")

# O10 reuses the exact saturating configurations of the paper's Table 2.
SATURATING = {2: dict(size=12, horizon=40), 3: dict(size=12, horizon=40),
              4: dict(size=12, horizon=40), 6: dict(size=16, horizon=60)}


def facility_factory(k):
    return lambda: FacilityCoverageEnv(k=k, **FACILITY)


def saturating_factory(k):
    c = SATURATING[k]
    return lambda: SaturatingCoverageEnv(k=k, patch=1, region=4,
                                         cap_frac=0.5, **c)


def plan(exp, ks, seeds, lr=None):
    """[(env_name, k, mode, obs_mode, clip, seed, lr, factory)]"""
    out = []
    if exp == "o9lr":
        # Both arms at every rate: the shared arm is the control that shows
        # whether a rate change moves the converging arm too.
        name = f"Facility {FACILITY['size']}x{FACILITY['size']}"
        for k in ks:
            for rate in LR_GRID:
                for obs in ("private", "shared"):
                    for s in seeds:
                        out.append((name, k, "MACS", obs, "none", s, rate,
                                    facility_factory(k)))
    elif exp == "o9":
        for k in ks:
            name = f"Facility {FACILITY['size']}x{FACILITY['size']}"
            for obs in ("shared", "private"):
                for s in seeds:
                    out.append((name, k, "MACS", obs, "none", s, lr,
                                facility_factory(k)))
    else:
        for k in ks:
            c = SATURATING[k]
            name = f"Saturating {c['size']}x{c['size']}"
            for mode, clip in (("MACS-CLIP", "budget"),
                               ("MACS-CLIP", "const"),
                               ("MACS", "none")):
                for s in seeds:
                    out.append((name, k, mode, "shared", clip, s, lr,
                                saturating_factory(k)))
    return out


def label_of(env_name, k, mode, obs, clip, seed, lr=None):
    base = (f"{env_name.replace(' ', '_')}_k{k}_{mode}"
            f"_{obs}_clip-{clip}")
    if lr is not None:
        base += f"_lr{lr:g}"
    return f"{base}_seed{seed}"


def is_complete(label, log_dir, n_episodes, log_every):
    p = Path(log_dir) / f"{label}.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            d = json.load(f)
    except (json.JSONDecodeError, KeyError):
        return False
    blocks = d.get("per_block", {}).get("episode", [])
    return len(blocks) >= n_episodes // log_every


def execute(exp, rows, cfg, log_dir):
    from macs_ablation import train_ablation
    n_ep, log_ev = cfg["n_episodes"], cfg["log_every"]
    done = skipped = 0
    for env_name, k, mode, obs, clip, seed, lr, factory in rows:
        lab = label_of(env_name, k, mode, obs, clip, seed, lr)
        if is_complete(lab, log_dir, n_ep, log_ev):
            print(f"  skip (complete): {lab}")
            skipped += 1
            continue
        kw = dict(cfg)
        if lr is not None:
            kw["lr"] = lr
        train_ablation(factory, env_name=env_name, mode=mode, obs_mode=obs,
                       clip=clip, seed=seed, label=lab, log_dir=log_dir, **kw)
        done += 1
    print(f"\n{done} run(s) completed, {skipped} skipped.")


# =============================================================================
# Aggregation
# =============================================================================

def final_metric(path, window=10):
    with open(path) as f:
        d = json.load(f)
    ev = [x for x in d["per_block"]["eval_coverage"] if x is not None]
    if len(ev) < window:
        return None, len(ev)
    return float(np.mean(ev[-window:])), len(ev)


def collect(log_dir, window=10):
    """{(env_name, k): {arm: {seed: metric}}}, plus a skip report."""
    out, short, n_files = {}, [], 0
    for p in sorted(Path(log_dir).glob("*seed*.json")):
        with open(p) as f:
            d = json.load(f)
        c = d["config"]
        n_files += 1
        m, n_blocks = final_metric(p, window)
        if m is None:
            short.append((p.name, n_blocks))
            continue
        arm = f"{c['method']}/{c.get('obs_mode','shared')}/{c.get('clip','none')}"
        if c.get("lr") is not None:
            arm += f"/lr{c['lr']:g}"
        out.setdefault((c["env_name"], c["k"]), {}) \
           .setdefault(arm, {})[c.get("seed", 0)] = m
    if short:
        print(f"note: {len(short)} of {n_files} log(s) have fewer than "
              f"{window} evaluation checkpoints and were skipped "
              f"(e.g. {short[0][0]} has {short[0][1]}). This is expected for "
              f"--smoke runs; pass --window to lower the requirement.")
    return out


def inspect(log_dir):
    """Per-run health check. Three things worth looking at before trusting
    a sweep:

      credit_err  max |sum_i phi_i - r_t|. Remark `efficiency-exact` says
                  this is zero to machine precision for the permutation and
                  subset estimators alike. A nonzero value means the
                  facility closed form is NOT being used and the routing
                  patch did not take.
      rising      whether the last evaluation block is still above the one
                  before it. If both arms are still rising, the run is
                  undertrained and a null result is uninterpretable.
      clip_frac   fraction of bootstrap targets cut, for the O10 arms.
    """
    rows = sorted(Path(log_dir).glob("*seed*.json"))
    if not rows:
        print(f"No logs in {log_dir}.")
        return
    print(f"\n{'run':<58} {'blocks':>6} {'last10':>8} {'credit_err':>11} "
          f"{'rising':>7} {'clip':>6}")
    print("-" * 100)
    for p in rows:
        with open(p) as f:
            d = json.load(f)
        pb = d["per_block"]
        ev = [x for x in pb["eval_coverage"] if x is not None]
        err = max(pb.get("credit_sum_err") or [0.0])
        cf = pb.get("clip_frac") or [0.0]
        last10 = np.mean(ev[-10:]) if ev else float("nan")
        rising = (len(ev) >= 4 and
                  np.mean(ev[-2:]) > np.mean(ev[-4:-2]) + 1e-9)
        flag = "  <-- credit error nonzero" if err > 1e-6 else ""
        # divergence: TD loss still climbing over the last third AND
        # evaluation below its own peak. Both together distinguish a
        # diverging run from one that is merely still learning.
        ls = [x for x in (pb.get("loss") or []) if x is not None]
        # Compare the tail against the MIDPOINT, not against the previous
        # block. A diverging run climbs steadily rather than accelerating,
        # so consecutive thirds can look flat while the run has grown by an
        # order of magnitude overall: the k=6 pilot's private arm went
        # 0.11 -> 3.3 -> 18.3, and a consecutive-thirds test missed it.
        mid = len(ls) // 2
        diverging = (len(ls) >= 8 and len(ev) >= 8
                     and np.mean(ls[-3:]) > 1.5 * np.mean(ls[mid-1:mid+2])
                     and last10 < 0.9 * max(ev))
        if diverging:
            flag += "  <-- DIVERGING (loss climbing, eval past peak)"
        print(f"{p.stem[:58]:<58} {len(ev):>6} {last10:>8.2f} "
              f"{err:>11.2e} {str(rising):>7} {np.mean(cf):>6.3f}{flag}")
    print("\nrising=True on every arm means the sweep is undertrained; a "
          "null difference there is not evidence of no difference.")
    print("DIVERGING means the run did not converge, so its score is not "
          "an information-limited optimum and should not be read as one.")


COMPARISONS = {
    "o9": [("MACS/shared/none", "MACS/private/none",
            "value of the shared augmented state")],
    "o9lr": [(f"MACS/shared/none/lr{r:g}", f"MACS/private/none/lr{r:g}",
              f"shared vs private at lr={r:g}") for r in LR_GRID],
    "o10": [("MACS-CLIP/shared/budget", "MACS-CLIP/shared/const",
             "state-dependent vs constant ceiling"),
            ("MACS-CLIP/shared/budget", "MACS/shared/none",
             "clip vs no clip")],
}


def aggregate(exp, log_dir, out_path=None, window=10):
    data = collect(log_dir, window)
    if not data:
        print(f"No seed-labelled logs in {log_dir}.")
        return
    lines = []
    for (env_name, k), by_arm in sorted(data.items(), key=lambda x: x[0][1]):
        lines.append(f"\n=== {env_name}, k={k} ===")
        lines.append(f"{'arm':<32} {'n':>3} {'mean':>9} {'std':>7}  seeds")
        for arm in sorted(by_arm):
            v = by_arm[arm]
            a = np.array([v[s] for s in sorted(v)])
            per = ", ".join(f"{s}:{v[s]:.1f}" for s in sorted(v))
            lines.append(f"{arm:<32} {len(a):>3} {a.mean():>9.2f} "
                         f"{a.std(ddof=1) if len(a) > 1 else 0.0:>7.2f}  [{per}]")
        for hi, lo, note in COMPARISONS[exp]:
            if hi not in by_arm or lo not in by_arm:
                continue
            seeds = sorted(set(by_arm[hi]) & set(by_arm[lo]))
            if len(seeds) < 2:
                continue
            diff, p = exact_perm_test([by_arm[hi][s] for s in seeds],
                                      [by_arm[lo][s] for s in seeds])
            lines.append(f"  {hi} vs {lo}: diff = {diff:+.2f}, "
                         f"exact permutation p = {p:.4f} "
                         f"(two-sided, n={len(seeds)}+{len(seeds)})  [{note}]")
    report = "\n".join(lines)
    print(report)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(report + "\n")
        summ = {}
        for (env_name, k), by_arm in sorted(data.items(),
                                            key=lambda x: x[0][1]):
            key = f"{env_name}|k={k}"
            summ[key] = {"arms": {a: {"seeds": {str(s): v for s, v
                                                in sorted(vals.items())},
                                      "mean": float(np.mean(
                                          list(vals.values()))),
                                      "std": float(np.std(
                                          list(vals.values()), ddof=1))
                                      if len(vals) > 1 else 0.0}
                                  for a, vals in sorted(by_arm.items())}}
        jp = Path(out_path).with_suffix(".json")
        jp.write_text(json.dumps(summ, indent=2))
        print(f"\nWritten: {out_path}\nWritten: {jp}")


# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["o9", "o10", "o9lr"], default="o9")
    ap.add_argument("--lr", type=float, default=None,
                    help="override the learning rate (default 5e-4)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--ks", type=int, nargs="+", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--window", type=int, default=10,
                    help="evaluation checkpoints averaged per seed")
    ap.add_argument("--inspect", action="store_true",
                    help="print per-run diagnostics and exit")
    args = ap.parse_args()

    exp = args.exp
    log_dir = args.log_dir or LOG_DIR[exp]
    ks = args.ks or ({"o9": [2, 3, 4, 6, 8], "o9lr": [6]}.get(
        exp, [2, 3, 4, 6]))
    seeds = args.seeds

    if args.inspect:
        inspect(log_dir)
        sys.exit(0)

    if args.aggregate:
        aggregate(exp, log_dir, window=args.window,
                  out_path=Path(log_dir) / f"{exp}_summary.txt")
        sys.exit(0)

    run_checks(verbose=False)
    install_facility_credits()
    print("wiring checks passed; facility credits routed to the closed form")

    if args.pilot:
        exp, ks, seeds = "o9", [6], [0]
        cfg = PILOT_CFG
        log_dir = args.log_dir or (LOG_DIR["o9"] + "_pilot")
        print("\nPILOT: k=6, one seed, shared vs private, 1200 episodes.\n"
              "Checks the S-value survives in learned agents before "
              "committing the full sweep.")
    elif args.smoke:
        cfg = SMOKE_CFG
        ks, seeds = ks[:1], seeds[:1]
        log_dir = args.log_dir or (LOG_DIR[exp] + "_smoke")
    else:
        cfg = TRAIN_CFG

    if exp == "o9lr":
        seeds = seeds[:1]
        print(f"\nO9-LR: private and shared arms at lr in "
              f"{[f'{r:g}' for r in LR_GRID]}, k={ks}, one seed.\n"
              f"Checks whether the private arm's divergence in the pilot is "
              f"intrinsic to the observation or a tuning artifact.")
    rows = plan(exp, ks, seeds, lr=args.lr)
    print(f"\nPlan ({exp}): {len(rows)} runs into {log_dir}")
    for env_name, k, mode, obs, clip, seed, lr, _ in rows:
        print(f"  {label_of(env_name, k, mode, obs, clip, seed, lr)}")
    if args.dry_run:
        sys.exit(0)
    if not M.TORCH_OK:
        print("\ntorch unavailable; cannot train here.")
        sys.exit(1)
    execute(exp, rows, cfg, log_dir)
    inspect(log_dir)
    aggregate(exp, log_dir,
              window=min(args.window, cfg["n_episodes"] // cfg["log_every"]),
              out_path=Path(log_dir) / f"{exp}_summary.txt")