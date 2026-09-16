"""
The training sweeps: O9, O9-LR, O10, ISLANDS, ISLANDS-LR.

O9  Augmented-state ablation on the facility objective.
    MACS with obs_mode="shared" vs obs_mode="private", five seeds. Turns the
    S-value ladder of the policy-level probe into a statement about learned
    agents, which is what Theorem `sufficient-statistic` needs. Neither arm
    clips, so the observation is the only difference.

O9-LR  Learning-rate robustness for the private arm.
    The k=6 pilot showed the private arm DIVERGING rather than converging to
    an information-limited optimum: TD loss climbing monotonically and
    evaluation peaking early then decaying below a non-learning blind greedy
    policy. That is consistent with Theorem `sufficient-statistic` -- without
    S_t the per-agent process is not Markov -- but it means the gap confounds
    an information deficit with a learnability failure. Sweep before
    reporting either.

O10 Isolating the budget clip on the saturating suite.
    MACS-CLIP (state-dependent ceiling B(S')) vs clip="const" (a
    state-INDEPENDENT ceiling matched to the mean of B(S')). If the constant
    reproduces the effect, the submodular structure is not doing the work.

ISLANDS  The credit-assignment benchmark.
    Uniform and clustered facility maps have coord - dec ~ 0: once S_t is
    shared, myopic greedy is near-optimal and no credit rule can win. Islands
    have both planning headroom (plan - dec) and assignment headroom
    (plan - plan_self, growing with k). The claim to test is that MACS closes
    more of the plan - dec gap than the baselines, with the margin largest at
    k=6.

    MACS vs SHARED is the WRONG headline: LOCAL and DR are per-agent credit
    rules too, and the k=6 sweep put them within 2 and 19 points of MACS
    while MACS beat SHARED by ~165. MACS vs LOCAL and MACS vs DR are the
    tests that decide the paper, so they run at every rate.

Every sweep resumes: a run whose log already has the full number of blocks
is skipped, so the script can be interrupted and relaunched.

Usage:
  python -m experiments.facility --dry-run
  python -m experiments.facility --pilot
  python -m experiments.facility --exp islands --seeds 0 1 2 --ks 3 4 6
  python -m experiments.facility --exp islandslr --ks 6
  python -m experiments.facility --aggregate --exp islands
  python -m experiments.facility --inspect --exp islands
  python -m experiments.facility --floors-only --exp islands
"""

import argparse
import sys
from pathlib import Path

from macs import config as CFG
from macs import policies as P
from macs import sweep
from macs.envs import FacilityCoverageEnv, IslandEnv, SaturatingCoverageEnv
from macs.rollout import mean_coverage
from macs.stats import f1
from macs.sweep import Run, cached_floors

EXPERIMENTS = ("o9", "o9lr", "o10", "islands", "islandslr")

DEFAULT_KS = {"o9": [2, 3, 4, 6, 8], "o9lr": [6],
              "islands": [3, 4, 6], "islandslr": [6], "o10": [2, 3, 4, 6]}


# =============================================================================
# Environment factories
# =============================================================================

def facility_factory(k):
    def make():
        env = FacilityCoverageEnv(k=k, **CFG.FACILITY)
        env.rho_hint = CFG.FACILITY["rho"]
        return env
    return make


def island_factory(k):
    def make():
        env = IslandEnv(k=k, **CFG.ISLANDS)
        env.rho_hint = CFG.ISLANDS["rho"]
        return env
    return make


def saturating_factory(k):
    return lambda: SaturatingCoverageEnv(**CFG.saturating_cfg(k))


# =============================================================================
# Reference floors
# =============================================================================
#
# A learned score is hard to read on its own. On the facility objective four
# references bracket it; `blind` is the one that matters, because a learned
# private agent scoring below a non-learning blind greedy policy means
# learning without S_t is worse than not learning, and that only reads as a
# finding if the floor is printed.
#
# On islands the references are the headroom decomposition itself:
# plan - dec is the planning headroom, plan - plan_self the assignment
# headroom, and those two numbers are what a learned arm is read against.

def facility_floors(k, episodes=8, seed=0, cache_path=None):
    def compute():
        make = lambda e, s: FacilityCoverageEnv(k=k, **CFG.FACILITY, seed=s)
        return {name: mean_coverage(make, pol, episodes, seed)
                for name, pol in P.FACILITY_FLOORS}
    return cached_floors(("facility", k, episodes, seed), compute, cache_path)


def island_floors(k, episodes=10, seed=0, cache_path=None):
    def compute():
        def make(e, s):
            env = IslandEnv(k=k, **CFG.ISLANDS, seed=s)
            env.rho_hint = CFG.ISLANDS["rho"]
            return env
        return {name: mean_coverage(make, pol, episodes, seed)
                for name, pol in P.ISLAND_FLOORS}
    return cached_floors(("islands", k, episodes, seed), compute, cache_path)


# =============================================================================
# Plans
# =============================================================================

def plan(exp, ks, seeds, lr=None, modes=None):
    runs = []

    def keep(m):
        return modes is None or m in modes

    if exp == "o9lr":
        # Both arms at every rate: the shared arm is the control that shows
        # whether a rate change moves the converging arm too.
        name = CFG.env_name("facility")
        for k in ks:
            for rate in CFG.LR_GRID:
                for obs in ("private", "shared"):
                    for s in seeds:
                        runs.append(Run(name, k, "MACS", s, obs, "none", rate,
                                        facility_factory(k)))
    elif exp == "o9":
        name = CFG.env_name("facility")
        for k in ks:
            for obs in ("shared", "private"):
                for s in seeds:
                    runs.append(Run(name, k, "MACS", s, obs, "none", lr,
                                    facility_factory(k)))
    elif exp == "islandslr":
        name = CFG.env_name("islands")
        for k in ks:
            for mode in filter(keep, CFG.ISLANDLR_MODES):
                for rate in CFG.ISLANDLR_GRID:
                    for s in seeds:
                        runs.append(Run(name, k, mode, s, "shared", "none",
                                        rate, island_factory(k)))
    elif exp == "islands":
        name = CFG.env_name("islands")
        for k in ks:
            for mode in filter(keep, CFG.ISLAND_MODES):
                clip = "budget" if mode == "MACS-CLIP" else "none"
                for s in seeds:
                    runs.append(Run(name, k, mode, s, "shared", clip, lr,
                                    island_factory(k)))
    elif exp == "o10":
        for k in ks:
            name = CFG.env_name("saturating", k)
            for mode, clip in (("MACS-CLIP", "budget"),
                               ("MACS-CLIP", "const"),
                               ("MACS", "none")):
                if not keep(mode):
                    continue
                for s in seeds:
                    runs.append(Run(name, k, mode, s, "shared", clip, lr,
                                    saturating_factory(k)))
    else:
        raise ValueError(f"unknown experiment {exp!r}")
    return runs


COMPARISONS = {
    "o9": [("MACS/shared/none", "MACS/private/none",
            "value of the shared augmented state")],
    "o9lr": [(f"MACS/shared/none/lr{r:g}", f"MACS/private/none/lr{r:g}",
              f"shared vs private at lr={r:g}") for r in CFG.LR_GRID],
    "islandslr": (
        [(f"MACS/shared/none/lr{r:g}", f"LOCAL/shared/none/lr{r:g}",
          f"Shapley vs first-arrival marginal at lr={r:g}")
         for r in CFG.ISLANDLR_GRID]
        + [(f"MACS/shared/none/lr{r:g}", f"DR/shared/none/lr{r:g}",
            f"Shapley vs last-arrival marginal at lr={r:g}")
           for r in CFG.ISLANDLR_GRID]
        + [(f"MACS/shared/none/lr{r:g}", f"SHARED/shared/none/lr{r:g}",
            f"MACS vs team reward at lr={r:g}")
           for r in CFG.ISLANDLR_GRID]),
    "islands": [("MACS/shared/none", "LOCAL/shared/none",
                 "Shapley credit vs first-arrival marginal"),
                ("MACS/shared/none", "DR/shared/none",
                 "Shapley credit vs last-arrival marginal"),
                ("MACS/shared/none", "SHARED/shared/none",
                 "Shapley credit vs team reward"),
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", choices=EXPERIMENTS, default="o9")
    ap.add_argument("--lr", type=float, default=None,
                    help="override the learning rate (default 5e-4)")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=CFG.DEFAULT_SEEDS)
    ap.add_argument("--ks", type=int, nargs="+", default=None)
    ap.add_argument("--modes", nargs="+", default=None,
                    help="restrict the plan to these methods, so a "
                         "per-method learning rate can be run from one "
                         "invocation")
    ap.add_argument("--shapley", default="auto",
                    choices=("auto", "facility", "permutation", "subset",
                             "mc"),
                    help="Shapley backend. auto routes facility-location F "
                         "to the closed form, k<=4 to permutations and "
                         "larger k to Monte Carlo. 'subset' is the exact "
                         "backend for k>4 on a non-facility objective.")
    ap.add_argument("--mc-m", type=int, default=64,
                    help="sampled permutations when the backend is mc")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--inspect", action="store_true",
                    help="print per-run diagnostics and exit")
    ap.add_argument("--floors-only", action="store_true",
                    help="compute and print the reference policies, exit")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--window", type=int, default=10,
                    help="evaluation checkpoints averaged per seed")
    ap.add_argument("--unpaired", action="store_true",
                    help="use the unpaired permutation test instead of the "
                         "paired sign-flip test")
    args = ap.parse_args(argv)

    exp = args.exp
    log_dir = args.log_dir or CFG.LOG_DIR[exp]
    ks = args.ks or DEFAULT_KS[exp]
    seeds = args.seeds

    if args.inspect:
        sweep.inspect(log_dir)
        return 0

    if args.floors_only:
        cache = Path(log_dir) / "oracles.json"
        if exp.startswith("islands"):
            print(f"{'k':>3} {'random':>9} {'sweep':>9} {'dec':>9} "
                  f"{'plan_self':>10} {'plan':>9} | {'plan-dec':>9} "
                  f"{'plan-self':>10}")
            for k in ks:
                F_V = IslandEnv(k=k, **CFG.ISLANDS).max_cells
                fl = island_floors(k, cache_path=cache)
                def pc(x):
                    return 100 * x / F_V
                print(f"{k:>3} {pc(fl['random']):>8.1f}% "
                      f"{pc(fl['sweep']):>8.1f}% {pc(fl['dec']):>8.1f}% "
                      f"{pc(fl['plan_self']):>9.1f}% {pc(fl['plan']):>8.1f}% "
                      f"| {pc(fl['plan'] - fl['dec']):>9.1f} "
                      f"{pc(fl['plan'] - fl['plan_self']):>10.1f}")
            print("\nplan-dec is the planning headroom and plan-plan_self "
                  "the assignment headroom.\nA credit rule can only be "
                  "tested where both are clearly positive.")
        else:
            print(f"{'k':>3} {'random':>9} {'blind':>9} {'dec':>9} "
                  f"{'coord':>9} | {'blind%':>7} {'dec%':>7}")
            for k in ks:
                F_V = FacilityCoverageEnv(k=k, **CFG.FACILITY).max_cells
                fl = facility_floors(k, cache_path=cache)
                print(f"{k:>3} {fl['random']:>9.1f} {fl['blind']:>9.1f} "
                      f"{fl['dec']:>9.1f} {fl['coord']:>9.1f} | "
                      f"{100 * fl['blind'] / F_V:>6.1f}% "
                      f"{100 * fl['dec'] / F_V:>6.1f}%")
        return 0

    if args.aggregate:
        floors = None
        if exp.startswith("islands"):
            floors = lambda k: island_floors(
                k, cache_path=Path(log_dir) / "oracles.json")
        elif exp in ("o9", "o9lr"):
            floors = lambda k: facility_floors(
                k, cache_path=Path(log_dir) / "oracles.json")
        sweep.aggregate(COMPARISONS[exp], log_dir, window=args.window,
                        floors=floors, paired=not args.unpaired,
                        out_path=Path(log_dir) / f"{exp}_summary.txt")
        return 0

    # -- training ----------------------------------------------------------
    cfg = dict(CFG.TRAIN_CFG)
    if args.pilot and exp.startswith("islands"):
        ks, seeds, cfg = [6], [0], dict(CFG.ISLAND_PILOT_CFG)
        log_dir = args.log_dir or (CFG.LOG_DIR["islands"] + "_pilot")
        print(f"\nPILOT (islands): k=6, one seed, every learner, "
              f"{cfg['n_episodes']} episodes.\n"
              "Read each arm against the plan-dec gap. If every arm sits "
              "near `random`, eps=0 is too sparse: retry with eps=0.02 "
              "(assignment headroom survives that, planning headroom "
              "does not).")
    elif args.pilot:
        exp, ks, seeds, cfg = "o9", [6], [0], dict(CFG.PILOT_CFG)
        log_dir = args.log_dir or (CFG.LOG_DIR["o9"] + "_pilot")
    if args.smoke:
        cfg = dict(CFG.SMOKE_CFG)
        seeds = seeds[:1]
        log_dir = args.log_dir or (log_dir + "_smoke")

    runs = plan(exp, ks, seeds, lr=args.lr, modes=args.modes)
    print(f"\n{exp}: {len(runs)} run(s), ks={ks}, seeds={seeds}, "
          f"log_dir={log_dir}")
    sweep.execute(runs, cfg, log_dir, dry_run=args.dry_run,
                  backend=args.shapley, mc_m=args.mc_m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
