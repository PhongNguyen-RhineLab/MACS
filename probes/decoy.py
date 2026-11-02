"""
Decoy-island probe. numpy only; seconds per configuration.

The equilibrium argument says per-agent credit rules differ in WHICH joint
behaviours they make stable. The herding construction: one island worth
more than any other, reachable by the whole team. Every agent that ignores
its teammates heads there.

    LOCAL pays each co-arriving agent its first-arrival marginal, so an
    agent loses nothing by joining the herd as long as it arrives no later
    than the others -- herding can be stable.
    Shapley splits a simultaneous arrival, so a herd member earns ~W/m and
    deviates whenever an alternative is worth more than that.
    DR pays a redundant herd member ~0, so it deviates too.

This probe measures, per configuration (decoy weight x depot placement):

  floors     random / dec / plan_self_v / plan_v (value-aware planners),
             as % of F(V)
  headroom   planning   plan_v - dec
             assignment plan_v - plan_self_v   (what herding costs the team)
  kappa      coupling mass sum(phi_LOC - phi) / sum(r) on plan_self_v and
             plan_v trajectories: how much LOCAL over-pays on each
  deviation  for each credit rule, the gain to ONE agent from leaving the
             herd for its best alternative island, all others staying:
                 U_i(deviate) - U_i(herd),
             discounted credit streams, averaged over spawns and over which
             agent deviates. Negative => the herd is stable against that
             agent under that rule. The theory predicts LOCAL >= Shapley >= DR
             in stability of the herd; the variant is worth training only if
             LOCAL's herd is stable (gain < 0) and Shapley's is not (gain > 0).
             Scripted strategies: walk to the assigned island until it is
             finished, then plan_v. That is a restricted strategy space, so a
             positive gain proves the herd is NOT an equilibrium, while a
             negative gain is evidence it is, not proof.

Usage:
  python -m probes.decoy                       # default grid
  python -m probes.decoy --mults 1 2 4 --k 6 --episodes 12
  python -m probes.decoy --depots none 12,5 5,8
"""

import argparse

import numpy as np

from macs import config as CFG
from macs import policies as P
from macs import shapley
from macs.envs import IslandEnv
from macs.envs.facility import decoy_index
from macs.rollout import factory, rollout

GAMMA = 0.99
HORIZON = None          # None -> CFG.ISLANDS["horizon"]; set by --horizon


# -- credit rules on one step --------------------------------------------------

def step_credits(env, info):
    S, patches = info["S_prev"], info["patches"]
    f0 = env.F(S)
    loc = np.array([env.F(S | pm) - f0 for pm in patches])
    union = S.copy()
    for pm in patches:
        union |= pm
    f_all = env.F(union)
    dr = np.zeros(env.k)
    for i in range(env.k):
        others = S.copy()
        for j, pm in enumerate(patches):
            if j != i:
                others |= pm
        dr[i] = f_all - env.F(others)
    shp = shapley.compute(S, patches, env.F, backend="auto")
    return {"LOCAL": loc, "MACS": shp, "DR": dr}, f_all - f0


def episode_streams(env, policy, rng):
    """Discounted per-agent credit streams under every rule, plus totals."""
    env.reset()
    pol = policy(env)
    U = {r: np.zeros(env.k) for r in ("LOCAL", "MACS", "DR")}
    over = team = 0.0
    for t in range(env.horizon):
        _, _, _, info = env.step(pol(env, rng))
        cr, r = step_credits(env, info)
        for rule in U:
            U[rule] += GAMMA ** t * cr[rule]
        over += float(cr["LOCAL"].sum() - cr["MACS"].sum())
        team += r
    return U, over, team, env.total_coverage


# -- the configuration ----------------------------------------------------------

def island_cfg(mult, depot_at):
    cfg = dict(CFG.ISLANDS)
    cfg.update(decoy_mult=mult, depot_at=depot_at)
    if HORIZON is not None:
        cfg["horizon"] = HORIZON
    return cfg


def best_alternative(env, i, decoy):
    """Agent i's best non-decoy island by the value planners' own score."""
    vals = P.island_values(env)
    d = np.abs(env.Ffl.centres - env.pos[i]).sum(1)
    score = GAMMA ** d * vals
    score[decoy] = -np.inf
    return int(np.argmax(score))


def deviation_gains(k, mult, depot_at, episodes, seed):
    """
    Mean U_i(deviate) - U_i(herd) per rule. Episode e: deviator i = e % k,
    herd and deviation rolled out from the SAME spawn and the same rng.
    """
    gains = {r: [] for r in ("LOCAL", "MACS", "DR")}
    team = {"herd": [], "deviate": []}
    cfg = island_cfg(mult, depot_at)
    for e in range(episodes):
        env_seed = seed * 1000 + e
        probe = IslandEnv(k=k, **cfg, seed=env_seed)
        probe.reset()
        decoy = decoy_index(probe.Ffl.centres, probe.size)
        i = e % k
        alt = best_alternative(probe, i, decoy)
        herd = [decoy] * k
        dev = list(herd)
        dev[i] = alt
        res = {}
        for name, assign in (("herd", herd), ("deviate", dev)):
            env = IslandEnv(k=k, **cfg, seed=env_seed)
            rng = np.random.default_rng(seed * 7919 + e)
            U, _, _, cov = episode_streams(
                env, P.value_planner(assigned=assign), rng)
            res[name] = U
            team[name].append(100 * cov / env.max_cells)
        for r in gains:
            gains[r].append(res["deviate"][r][i] - res["herd"][r][i])
    return ({r: (float(np.mean(v)), float(np.std(v))) for r, v in
             gains.items()},
            {n: float(np.mean(v)) for n, v in team.items()})


def kappa(k, mult, depot_at, policy, episodes, seed):
    cfg = island_cfg(mult, depot_at)
    over = team = 0.0
    for e in range(episodes):
        env = IslandEnv(k=k, **cfg, seed=seed * 1000 + e)
        rng = np.random.default_rng(seed * 7919 + e)
        _, o, t, _ = episode_streams(env, policy, rng)
        over += o
        team += t
    return 100 * over / max(team, 1e-9)


def floors(k, mult, depot_at, episodes, seed):
    make = factory(IslandEnv, dict(k=k, **island_cfg(mult, depot_at)))
    return {name: float(rollout(make, pol, episodes, seed,
                                normalize=True)["coverage"].mean())
            for name, pol in P.DECOY_FLOORS}


def spawn_coverage_of_decoy(k, mult, depot_at):
    """Fraction of the decoy's value already served at spawn (paid for by
    nobody). A depot on top of the decoy defeats the construction."""
    env = IslandEnv(k=k, **island_cfg(mult, depot_at), seed=0)
    env.reset()
    dcy = decoy_index(env.Ffl.centres, env.size)
    cx, cy = env.Ffl.centres[dcy]
    xs, ys = np.meshgrid(np.arange(env.size), np.arange(env.size),
                         indexing="ij")
    cells = (xs - cx) ** 2 + (ys - cy) ** 2 <= env.Ffl.radius ** 2
    best = env.Ffl.utility_map(env.covered)
    return float(best[cells].mean()), int(abs(np.array(env.depot_at or
                                                       (-99, -99))
                                              - (cx, cy)).sum())


def parse_depot(s):
    if s.lower() == "none":
        return None
    x, y = s.split(",")
    return (int(x), int(y))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.
                                 RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--mults", type=float, nargs="+",
                    default=[1.0, 2.0, 3.0, 4.0, 6.0])
    ap.add_argument("--depots", nargs="+", default=["none"],
                    help="'none' for the random depot, or x,y")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=None,
                    help="override the episode horizon (default: the "
                         "benchmark's 30). Shorter horizons leave less slack "
                         "to recover from a herd, so J herd - J dev grows.")
    args = ap.parse_args(argv)
    global HORIZON
    HORIZON = args.horizon

    env0 = IslandEnv(k=args.k, **island_cfg(1.0, None), seed=0)
    dcy = decoy_index(env0.Ffl.centres, env0.size)
    print(f"layout_seed={CFG.ISLANDS['layout_seed']}  islands="
          f"{env0.Ffl.centres.tolist()}  decoy=#{dcy} at "
          f"{env0.Ffl.centres[dcy].tolist()}")
    print(f"k={args.k}  episodes={args.episodes}  gamma={GAMMA}  "
          f"horizon={HORIZON or CFG.ISLANDS['horizon']}\n")

    hdr = (f"{'depot':>8} {'mult':>5} | {'dec':>5} {'pself':>6} {'plan':>5}"
           f" | {'plan-dec':>8} {'plan-self':>9} | {'k_self':>6} "
           f"{'k_plan':>6} | {'dev LOCAL':>10} {'dev MACS':>9} "
           f"{'dev DR':>8} | {'J herd':>6} {'J dev':>6}")
    print(hdr)
    print("-" * len(hdr))
    for dep_s in args.depots:
        dep = parse_depot(dep_s)
        if dep is not None:
            served, dist = spawn_coverage_of_decoy(args.k, 2.0, dep)
            if served > 0:
                print(f"  note: depot {dep} serves the decoy at "
                      f"{served:.2f} at spawn (L1 distance {dist})")
        for m in args.mults:
            fl = floors(args.k, m, dep, args.episodes, args.seed)
            ks = kappa(args.k, m, dep, P.plan_self_v, args.episodes,
                       args.seed)
            kp = kappa(args.k, m, dep, P.plan_v, args.episodes, args.seed)
            g, J = deviation_gains(args.k, m, dep, args.episodes, args.seed)
            print(f"{dep_s:>8} {m:>5.1f} | {fl['dec']:>5.1f} "
                  f"{fl['plan_self_v']:>6.1f} {fl['plan_v']:>5.1f} | "
                  f"{fl['plan_v'] - fl['dec']:>8.1f} "
                  f"{fl['plan_v'] - fl['plan_self_v']:>9.1f} | "
                  f"{ks:>5.1f}% {kp:>5.1f}% | "
                  f"{g['LOCAL'][0]:>+10.2f} {g['MACS'][0]:>+9.2f} "
                  f"{g['DR'][0]:>+8.2f} | {J['herd']:>6.1f} "
                  f"{J['deviate']:>6.1f}")
    print("\nfloors and J in % of F(V); deviation gains in raw discounted "
          "credit (F(V) = 576).\nThe variant does its job where dev LOCAL "
          "< 0 < dev MACS and plan - plan_self is large.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
