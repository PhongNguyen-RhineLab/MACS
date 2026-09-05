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

ISLANDS  Credit assignment benchmark on sparse-island demand (Sept 2026).
    Uniform and clustered facility maps have coord - dec ~ 0: once S_t is
    shared, myopic greedy is near-optimal and no credit rule can win.
    probe_islands.py found a facility configuration with both planning
    headroom (plan - dec ~ 45 points) and assignment headroom
    (plan - plan_self ~ 11-24 points, growing with k). SHARED, VDN, QMIX,
    MACS and MACS-CLIP at k in {3,4,6}, against oracle rows random / sweep
    / dec / plan_self / plan. The claim to test: MACS closes more of the
    plan - dec gap than the baselines, with the margin largest at k=6.

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
  python run_facility.py --exp islands --pilot      k=6, seed 0, all five learners
  python run_facility.py --exp islands --seeds 0 1 2 --ks 3 4 6
  python run_facility.py --aggregate --exp islands
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import macs_main as M
import macs_v3  # noqa: F401   rebinds compute_credits to the v3 dispatcher
from macs_fl import FacilityCoverageEnv, IslandEnv
from macs_v3 import SaturatingCoverageEnv
from macs_ablation import install_facility_credits, run_checks
from run_multiseed_sat import exact_perm_test, r1

LOG_DIR = {"o9": "logs/macs_facility", "o10": "logs/macs_clipcontrol",
           "o9lr": "logs/macs_facility_lr", "islands": "logs/macs_islands",
           "islandslr": "logs/macs_islands_lr"}

# Learning rates for the O9-LR robustness arm. 5e-4 is the published
# setting and the one that diverged in the pilot.
LR_GRID = [5e-4, 2e-4, 1e-4, 5e-5]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# 4000 episodes matches the published saturating runs, so the two suites
# share a budget and "you gave them different amounts of training" is not
# available as an objection. The k=6 pilot was still rising at 2500.
TRAIN_CFG = dict(n_episodes=4000, eval_every=100, log_every=100)
# 24x24 is four times the state space of the 12x12 saturating grid the
# published runs used, so a short pilot risks measuring undertraining rather
# than the ablation. 2500 episodes is the compromise; `inspect` reports
# whether both arms are still rising at the end, which is the check that
# matters more than the episode count itself.
ISLAND_PILOT_CFG = dict(n_episodes=8000, eval_every=200, log_every=200,
                        warmup=1500)
PILOT_CFG = dict(n_episodes=2500, eval_every=100, log_every=100, warmup=1500)
SMOKE_CFG = dict(n_episodes=30, warmup=200, eval_every=10, n_eval=2,
                 log_every=10, target_every=10)

# Facility geometry. rho=4 with single-cell footprints and a 24x24 grid is
# the configuration where the probe measured the largest S-value ladder
# (7.9 / 13.5 / 17.9 / 21.2 / 19.6 at k = 2,3,4,6,8).
FACILITY = dict(size=24, horizon=60, rho=4.0, patch=0, demand="uniform")

# Islands: probe_islands.py canonical config. H=30 is load-bearing (at H=60
# the assignment headroom collapses to ~2 points and the sweep recovers).
# layout_seed=0 fixes the map, so no demand input channel is needed.
ISLANDS = dict(size=24, horizon=30, rho=4.0, n_islands=8, radius=1,
               min_sep=8, eps=0.0, layout_seed=0, depot=True)
ISLAND_MODES = ("SHARED", "LOCAL", "DR", "VDN", "QMIX",
                "MACS", "MACS-CLIP")
# The k=6 pilot flagged VDN and QMIX as diverging at the published 5e-4.
# A reviewer will read an untuned baseline as a stacked comparison, so
# `islandslr` reruns the three baselines across the same grid the private
# arm got in O9-LR. Whatever rate is best per baseline is what the headline
# table should report.
ISLANDLR_MODES = ("SHARED", "LOCAL", "DR", "VDN", "QMIX", "MACS")
# O10 reuses the exact saturating configurations of the paper's Table 2.
SATURATING = {2: dict(size=12, horizon=40), 3: dict(size=12, horizon=40),
              4: dict(size=12, horizon=40), 6: dict(size=16, horizon=60)}


# -----------------------------------------------------------------------
# Reference policies as reported floors
# -----------------------------------------------------------------------
# A learned score is hard to read on its own. Four references bracket it:
#
#   random   what no policy at all achieves
#   blind    greedy against the agent's OWN visited set, no learning.
#            This is the one that matters: the k=6 pilot's LEARNED private
#            agent scored 27% of F(V) against this policy's 67.7%, i.e.
#            learning without S_t is worse than not learning. Reporting the
#            floor is what makes that legible rather than looking like the
#            private arm is merely weaker.
#   dec      greedy against the SHARED set, no learning. The natural
#            ceiling for the shared arm.
#   coord    one-step sequential greedy oracle.
#
# Cached to disk, since they are deterministic given the seed and take a
# few seconds per configuration.

_ORACLE_CACHE = {}


def oracle_floors(k, episodes=8, seed=0, cache_path=None):
    """{'random':…, 'blind':…, 'dec':…, 'coord':…} in units of F(V)."""
    key = (k, episodes, seed)
    if key in _ORACLE_CACHE:
        return _ORACLE_CACHE[key]
    if cache_path and Path(cache_path).exists():
        disk = json.loads(Path(cache_path).read_text())
        if str(key) in disk:
            _ORACLE_CACHE[key] = disk[str(key)]
            return disk[str(key)]
    from probe_fl import pol_random, pol_dec, pol_coord
    from probe_blind import make_blind, rollout
    cfg = dict(k=k, **FACILITY)
    out = {}
    for name, pol in (("random", pol_random), ("dec", pol_dec),
                      ("coord", pol_coord), ("blind", make_blind)):
        out[name] = rollout(FacilityCoverageEnv, cfg, pol, episodes, seed)
    _ORACLE_CACHE[key] = out
    if cache_path:
        disk = {}
        if Path(cache_path).exists():
            disk = json.loads(Path(cache_path).read_text())
        disk[str(key)] = out
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(disk, indent=2))
    return out


def facility_factory(k):
    return lambda: FacilityCoverageEnv(k=k, **FACILITY)


def island_factory(k):
    return lambda: IslandEnv(k=k, **ISLANDS)


def island_floors(k, episodes=10, seed=0, cache_path=None):
    """{'random','sweep','dec','plan_self','plan'} in units of F(V).
    plan - dec is the planning headroom, plan - plan_self the assignment
    headroom; these are the two numbers a learned arm is read against."""
    key = ("islands", k, episodes, seed)
    if cache_path and Path(cache_path).exists():
        disk = json.loads(Path(cache_path).read_text())
        if str(key) in disk:
            return disk[str(key)]
    from probe_islands import rollout, build_policies
    cfg = {kk: v for kk, v in dict(k=k, **ISLANDS).items()
           if kk not in ("layout_seed", "depot")}
    rand_layout = ISLANDS["layout_seed"] is None
    F_V = IslandEnv(k=k, **ISLANDS).max_cells
    out = {}
    for name, pol in build_policies():
        if name in ("random", "sweep", "dec", "plan_self", "plan"):
            m, _ = rollout(cfg, pol, episodes, seed, rand_layout)
            out[name] = m * F_V / 100.0
    if cache_path:
        disk = {}
        if Path(cache_path).exists():
            disk = json.loads(Path(cache_path).read_text())
        disk[str(key)] = out
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(disk, indent=2))
    return out


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
    elif exp == "islandslr":
        name = f"Islands {ISLANDS['size']}x{ISLANDS['size']}"
        for k in ks:
            for mode in ISLANDLR_MODES:
                for rate in LR_GRID:
                    for s in seeds:
                        out.append((name, k, mode, "shared", "none", s, rate,
                                    island_factory(k)))
    elif exp == "islands":
        name = f"Islands {ISLANDS['size']}x{ISLANDS['size']}"
        for k in ks:
            for mode in ISLAND_MODES:
                clip = "budget" if mode == "MACS-CLIP" else "none"
                for s in seeds:
                    out.append((name, k, mode, "shared", clip, s, lr,
                                island_factory(k)))
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
        if mode in ("VDN", "QMIX"):
            # mixing baselines live in macs_main.train_macs. Its log config
            # lacks seed/obs_mode/clip, which collect() needs, so patch them
            # in after the run.
            M.train_macs(factory, env_name=env_name, mode=mode, seed=seed,
                         label=lab, log_dir=log_dir, **kw)
            p = Path(log_dir) / f"{lab}.json"
            d = json.loads(p.read_text())
            d["config"].update(seed=seed, obs_mode=obs, clip=clip,
                               lr=kw.get("lr", 5e-4))
            p.write_text(json.dumps(d, indent=2))
        else:
            train_ablation(factory, env_name=env_name, mode=mode,
                           obs_mode=obs, clip=clip, seed=seed, label=lab,
                           log_dir=log_dir, **kw)
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


CLIP = {}   # {(env_name, k): {arm: [mean clip_frac per seed]}}


def collect(log_dir, window=10):
    """{(env_name, k): {arm: {seed: metric}}}, plus a skip report.

    Also fills CLIP with mean bootstrap-clip activation per arm. On islands
    B(S') = F(V) - F(S') stays near 310 while bootstrap values sit near
    250, so the ceiling never binds and the clipped arm is the same backup
    as the unclipped one. That has to be printed, not inferred.
    """
    out, short, n_files = {}, [], 0
    CLIP.clear()
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
        cf = [x for x in d["per_block"].get("clip_frac", []) if x is not None]
        CLIP.setdefault((c["env_name"], c["k"]), {}) \
            .setdefault(arm, []).append(float(np.mean(cf)) if cf else 0.0)
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


def resolve_arm(key, by_arm):
    """Map a COMPARISONS key onto an arm actually present in `by_arm`.

    collect() appends "/lr{rate}" to every arm whose config records one, so
    the bare keys below never match. With per-method learning rates the two
    sides of a comparison carry different suffixes, so no single bare key
    could match both. Resolve by prefix instead.

    Returns the arm name, or None when absent or ambiguous. Ambiguity is
    real information -- several rates for one method means the sweep has
    not been reduced to a headline -- so the caller reports it rather than
    guessing.
    """
    if key in by_arm:
        return key
    hits = [a for a in by_arm if a == key or a.startswith(key + "/lr")]
    return hits[0] if len(hits) == 1 else None


COMPARISONS = {
    "o9": [("MACS/shared/none", "MACS/private/none",
            "value of the shared augmented state")],
    "o9lr": [(f"MACS/shared/none/lr{r:g}", f"MACS/private/none/lr{r:g}",
              f"shared vs private at lr={r:g}") for r in LR_GRID],
    "islandslr": [(f"MACS/shared/none/lr{r:g}", f"SHARED/shared/none/lr{r:g}",
                   f"MACS vs team reward at lr={r:g}") for r in LR_GRID],
    "islands": [("MACS/shared/none", "SHARED/shared/none",
                 "Shapley credit vs team reward"),
                ("MACS/shared/none", "LOCAL/shared/none",
                 "Shapley credit vs first-arrival marginal"),
                ("MACS/shared/none", "DR/shared/none",
                 "Shapley credit vs last-arrival marginal"),
                ("MACS/shared/none", "VDN/shared/none",
                 "Shapley credit vs additive mixing"),
                ("MACS/shared/none", "QMIX/shared/none",
                 "Shapley credit vs monotone mixing"),
                ("MACS-CLIP/shared/budget", "MACS/shared/none",
                 "budget clip on top of Shapley")],
    "o10": [("MACS-CLIP/shared/budget", "MACS-CLIP/shared/const",
             "state-dependent vs constant ceiling"),
            ("MACS-CLIP/shared/budget", "MACS/shared/none",
             "clip vs no clip")],
}


def aggregate(exp, log_dir, out_path=None, window=10, show_floors=True):
    data = collect(log_dir, window)
    if not data:
        print(f"No seed-labelled logs in {log_dir}.")
        return
    lines = []
    for (env_name, k), by_arm in sorted(data.items(), key=lambda x: x[0][1]):
        lines.append(f"\n=== {env_name}, k={k} ===")
        if env_name.startswith("Facility"):
            F_V = FacilityCoverageEnv(k=k, **FACILITY).max_cells
        elif env_name.startswith("Islands"):
            F_V = IslandEnv(k=k, **ISLANDS).max_cells
        else:
            F_V = None
        lines.append(f"{'arm':<32} {'n':>3} {'mean':>9} {'std':>7} "
                     f"{'%F(V)':>7} {'clip':>6}  seeds")
        for arm in sorted(by_arm):
            v = by_arm[arm]
            a = np.array([v[s] for s in sorted(v)])
            per = ", ".join(f"{s}:{v[s]:.1f}" for s in sorted(v))
            pct = f"{100*a.mean()/F_V:>6.1f}%" if F_V else " " * 7
            cfs = CLIP.get((env_name, k), {}).get(arm, [])
            cf = f"{np.mean(cfs):>6.3f}" if cfs else " " * 6
            lines.append(f"{arm:<32} {len(a):>3} {a.mean():>9.2f} "
                         f"{a.std(ddof=1) if len(a) > 1 else 0.0:>7.2f} "
                         f"{pct} {cf}  [{per}]")
        if env_name.startswith("Islands") and show_floors:
            fl = island_floors(k, cache_path=Path(log_dir) / "oracles.json")
            lines.append("  -- non-learning reference policies --")
            for nm, note in (("random", "no policy"),
                             ("sweep", "F-free serpentine, no state"),
                             ("dec", "myopic greedy on SHARED set"),
                             ("plan_self", "nearest island, ignores teammates"),
                             ("plan", "nearest UNCLAIMED island (oracle)")):
                lines.append(f"  {nm:<30} {'':>3} {fl[nm]:>9.2f} {'':>7} "
                             f"{100*fl[nm]/F_V:>6.1f}%  ({note})")
            for arm in sorted(by_arm):
                if not arm.startswith("MACS-CLIP"):
                    continue
                cfs = CLIP.get((env_name, k), {}).get(arm, [])
                if cfs and max(cfs) < 1e-9:
                    lines.append(
                        "  NOTE: clip activation is identically zero for "
                        f"{arm}. B(S') never binds on this\n        "
                        "configuration, so the clipped arm is the SAME "
                        "backup as MACS and any difference\n        between "
                        "them is seed noise, not an effect of the clip.")
            lines.append(f"  planning headroom plan-dec = "
                         f"{100*(fl['plan']-fl['dec'])/F_V:.1f} pts, "
                         f"assignment headroom plan-plan_self = "
                         f"{100*(fl['plan']-fl['plan_self'])/F_V:.1f} pts")
            # Anchor on random -> plan, NOT dec -> plan. `dec`, `plan_self`
            # and `plan` all call env.F, i.e. they read the demand field
            # exactly; a model-free learner has to discover it. Anchoring a
            # learner on dec asks it to beat privileged information and
            # reports every arm as negative even when the arms separate
            # cleanly from each other. random -> plan is the span a learner
            # can actually traverse; dec stays in the table as a reference.
            span = max(fl["plan"] - fl["random"], 1e-9)
            # Denominator is the best of ALL baselines. Restricting it
            # to {SHARED, VDN, QMIX} excluded LOCAL and DR, the two arms
            # most likely to be competitive: in the k=6 smoke DR closed
            # 33.3% against MACS's 35.8%, so the honest multiplier was
            # 1.08x while the old denominator printed 1.55x.
            BASELINES = ("SHARED", "LOCAL", "DR", "VDN", "QMIX")
            cand = {a: np.mean(list(v.values())) for a, v in by_arm.items()
                    if a.startswith(BASELINES)}
            base_arm = max(cand, key=cand.get) if cand else None
            base = cand[base_arm] if base_arm else None
            lines.append("  -- learned arms, share of the random -> plan "
                         "span closed --")
            for arm in sorted(by_arm):
                a = np.mean(list(by_arm[arm].values()))
                extra = ""
                if base is not None and base > fl["random"]:
                    extra = (f"   ({(a-fl['random'])/(base-fl['random']):.2f}x "
                             f"{base_arm.split('/')[0]}, best baseline)")
                lines.append(f"  {arm:<30} "
                             f"{100*(a-fl['random'])/span:>6.1f}%{extra}")
        elif F_V is not None and show_floors:
            fl = oracle_floors(k, cache_path=Path(log_dir) / "oracles.json")
            lines.append("  -- non-learning reference policies --")
            for nm, note in (("random", "no policy"),
                             ("blind", "greedy on OWN set, no learning"),
                             ("dec", "greedy on SHARED set, no learning"),
                             ("coord", "sequential greedy oracle")):
                lines.append(f"  {nm:<30} {'':>3} {fl[nm]:>9.2f} {'':>7} "
                             f"{100*fl[nm]/F_V:>6.1f}%  ({note})")
            lines.append(
                "  NOTE: a learned arm scoring below `blind` has not found "
                "an information-limited\n        optimum -- it is doing "
                "worse than a policy that does no learning at all.")
        for hi_key, lo_key, note in COMPARISONS[exp]:
            hi, lo = resolve_arm(hi_key, by_arm), resolve_arm(lo_key, by_arm)
            if hi is None or lo is None:
                for key, got in ((hi_key, hi), (lo_key, lo)):
                    if got is None and any(a.startswith(key + "/lr")
                                           for a in by_arm):
                        lines.append(
                            f"  SKIPPED {hi_key} vs {lo_key}: several "
                            f"learning rates present for {key}; aggregate a "
                            f"log dir holding one rate per method, or name "
                            f"the rate in COMPARISONS.")
                continue
            seeds = sorted(set(by_arm[hi]) & set(by_arm[lo]))
            if len(seeds) < 2:
                lines.append(f"  SKIPPED {hi} vs {lo}: {len(seeds)} shared "
                             f"seed(s), need 2+ for a permutation test.")
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
    ap.add_argument("--exp", choices=["o9", "o10", "o9lr", "islands", "islandslr"], default="o9")
    ap.add_argument("--lr", type=float, default=None,
                    help="override the learning rate (default 5e-4)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--ks", type=int, nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", default=None,
                    help="restrict the plan to these methods, so a "
                         "per-method learning rate can be run from "
                         "one invocation")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--window", type=int, default=10,
                    help="evaluation checkpoints averaged per seed")
    ap.add_argument("--inspect", action="store_true",
                    help="print per-run diagnostics and exit")
    ap.add_argument("--no-floors", action="store_true",
                    help="skip the non-learning reference policies")
    ap.add_argument("--floors-only", action="store_true",
                    help="compute and print the reference policies, then exit")
    args = ap.parse_args()

    exp = args.exp
    log_dir = args.log_dir or LOG_DIR[exp]
    ks = args.ks or ({"o9": [2, 3, 4, 6, 8], "o9lr": [6],
                      "islands": [3, 4, 6],
                      "islandslr": [6]}.get(exp, [2, 3, 4, 6]))
    seeds = args.seeds

    if args.inspect:
        inspect(log_dir)
        sys.exit(0)

    if args.floors_only and exp.startswith("islands"):
        print(f"{'k':>3} {'random':>9} {'sweep':>9} {'dec':>9} "
              f"{'plan_self':>9} {'plan':>9} | {'plan-dec':>9} {'plan-self':>9}")
        for k in ks:
            F_V = IslandEnv(k=k, **ISLANDS).max_cells
            fl = island_floors(k, cache_path=Path(log_dir) / "oracles.json")
            pc = lambda x: 100 * x / F_V
            print(f"{k:>3} {pc(fl['random']):>8.1f}% {pc(fl['sweep']):>8.1f}% "
                  f"{pc(fl['dec']):>8.1f}% {pc(fl['plan_self']):>8.1f}% "
                  f"{pc(fl['plan']):>8.1f}% | "
                  f"{pc(fl['plan']-fl['dec']):>8.1f} "
                  f"{pc(fl['plan']-fl['plan_self']):>8.1f}")
        sys.exit(0)

    if args.floors_only:
        print(f"{'k':>3} {'random':>9} {'blind':>9} {'dec':>9} {'coord':>9} "
              f"| {'blind%':>7} {'dec%':>7}")
        for k in ks:
            F_V = FacilityCoverageEnv(k=k, **FACILITY).max_cells
            fl = oracle_floors(k, cache_path=Path(log_dir) / "oracles.json")
            print(f"{k:>3} {fl['random']:>9.1f} {fl['blind']:>9.1f} "
                  f"{fl['dec']:>9.1f} {fl['coord']:>9.1f} | "
                  f"{100*fl['blind']/F_V:>6.1f}% {100*fl['dec']/F_V:>6.1f}%")
        sys.exit(0)

    if args.aggregate:
        aggregate(exp, log_dir, window=args.window,
                  show_floors=not args.no_floors,
                  out_path=Path(log_dir) / f"{exp}_summary.txt")
        sys.exit(0)

    run_checks(verbose=False)
    install_facility_credits()
    print("wiring checks passed; facility credits routed to the closed form")

    if args.pilot and exp.startswith("islands"):
        ks, seeds = [6], [0]
        cfg = ISLAND_PILOT_CFG
        log_dir = args.log_dir or (LOG_DIR["islands"] + "_pilot")
        print("\nPILOT (islands): k=6, one seed, all five learners, "
              f"{cfg['n_episodes']} episodes.\n"
              "Read each arm against the plan-dec gap. If every arm sits "
              "near `random`, eps=0 is too sparse: retry with eps=0.02 "
              "(assignment headroom survives that, planning headroom "
              "does not).")
    elif args.pilot:
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
    if args.modes:
        want = set(args.modes)
        unknown = want - {r[2] for r in rows}
        if unknown:
            sys.exit(f"--modes: no planned run uses {sorted(unknown)}; "
                     f"available here: {sorted({r[2] for r in rows})}")
        rows = [r for r in rows if r[2] in want]
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
              show_floors=not args.no_floors,
              out_path=Path(log_dir) / f"{exp}_summary.txt")