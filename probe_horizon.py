"""
probe_horizon.py -- does ANY (size, horizon) setting give the learner a job?

Two headrooms are measured on the same rollouts:

  plan  = coord - dec       what one-step lookahead buys over myopic greedy.
                            At size=24, H=60 this is ~0 at every k, which is
                            why a myopic policy reaches 92% of F(V) and why
                            there is nothing for RL to win.
  est   = dec  - dec_est    what knowing the demand field buys over the
                            uninformed prior (see probe_est.py).

Hypothesis under test: `plan` is ~0 because the horizon is long relative to
the grid, so an agent has time to recover from any myopic step. Shortening H
or enlarging the grid should punish myopia -- but only when the demand field
is non-uniform, since under uniform demand every direction is equally good
and greedy is already near-optimal by construction.

All figures are percentages of F(V) so that size=24 and size=40 compare.
"""

import numpy as np

from macs_fl import FacilityCoverageEnv
from probe_fl import pol_random, pol_dec, pol_coord
from probe_blind import make_blind
from probe_est import make_dec_est, POLICIES, _PER_ENV

EPS = 8


def rollout(cfg, mk, eps, seed, rand_field):
    out, sensed = [], []
    for e in range(eps):
        rng = np.random.default_rng(seed * 7919 + e)
        c = dict(cfg)
        if rand_field:
            c["demand_seed"] = 10_000 + e
        env = FacilityCoverageEnv(**c, seed=seed * 1000 + e)
        env.reset()
        pol = mk(env) if mk in _PER_ENV else mk
        for _ in range(env.horizon):
            env.step(pol(env, rng))
        out.append(env.total_coverage)
        sensed.append(env.covered.mean())
    return float(np.mean(out)), float(np.mean(sensed))


def row(size, H, demand, k, rand_field):
    cfg = dict(size=size, horizon=H, rho=4.0, patch=0, k=k, demand=demand)
    if demand == "clusters":
        cfg.update(n_clusters=6, cluster_sigma=2.5)
    F_V = FacilityCoverageEnv(**cfg, seed=0).max_cells
    vals, sens = {}, 0.0
    for n, p in POLICIES:
        m, sn = rollout(cfg, p, EPS, 0, rand_field)
        vals[n] = 100 * m / F_V
        if n == "dec":
            sens = sn
    plan = vals["coord"] - vals["dec"]
    est = vals["dec"] - vals["dec_est"]
    print(f"{size:>5} {H:>4} {100*sens:>6.1f}% " +
          " ".join(f"{vals[n]:>7.1f}" for n, _ in POLICIES) +
          f" | {plan:>6.1f} {est:>6.1f}")
    return plan, est


HEAD = (f"{'size':>5} {'H':>4} {'sensed':>7} " +
        " ".join(f"{n:>7}" for n, _ in POLICIES) +
        f" | {'plan':>6} {'est':>6}")

if __name__ == "__main__":
    print("all policy columns are % of F(V);  k = 6 throughout")
    print("plan = coord - dec   (lookahead headroom)")
    print("est  = dec - dec_est (demand-estimation headroom)")

    print("\n=== clustered demand, new field every episode ===")
    print(HEAD)
    print("-" * 86)
    best = []
    for size in (24, 40):
        for H in (10, 20, 40, 60):
            p, e = row(size, H, "clusters", 6, True)
            best.append((p, size, H))
        print()

    print("=== uniform demand (control: greedy should already be near-optimal) ===")
    print(HEAD)
    print("-" * 86)
    for H in (10, 20, 40, 60):
        row(24, H, "uniform", 6, False)

    b = max(best)
    print(f"\nlargest lookahead headroom found: {b[0]:.1f}% of F(V) "
          f"at size={b[1]}, H={b[2]}")
