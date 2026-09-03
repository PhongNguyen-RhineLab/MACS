"""
run_diag.py -- the O9 collapse diagnostic: one private seed, one shared seed.

The shared arm is not optional. `q_spread_mean` has no absolute scale, so a
small number on the private arm is uninterpretable without the converging arm
measured the same way on the same day with the same code.

Everything matches the O9 sweep exactly (24x24 facility location, uniform
demand, H=60, 4000 episodes, lr 5e-4, clip="none"), so the resulting
`objective` should land near the published means, 419.12 shared and 152.58
private at k=6. If it does not, the diagnostic is measuring a different run
and the numbers below mean nothing.

    python run_diag.py --smoke        60 episodes, wiring check, ~1 min
    python run_diag.py                full pair, k=6, 4000 episodes each
    python run_diag.py --k 8 --seed 2

Interpretation, k=6 (probe_cells.py, probe_ceiling.py):

    cells/agent      ceiling 61   blind 61.0   random 25.5   predicted ~5.8
    |S_H| union      blind 159.2  random 136.0             predicted ~35
    objective F      blind 390.0  random 283.3   private arm 152.58

  * cells/agent ~ 6 and stay-dominated histogram -> self-locking collapse,
    the predicted mechanism.
  * cells/agent ~ 12 with one direction dominant -> constant-argmax collapse
    into a wall (that policy scores 228.5, above the private arm, so this
    would leave the score unexplained).
  * cells/agent ~ 25 with high entropy -> no collapse; the policy is
    diffusing and the deficit is elsewhere.
  * q_spread_mean small on private AND on shared -> the measurement is
    uninformative, not the arm.
"""

import argparse
import json
from pathlib import Path

from macs_fl import FacilityCoverageEnv
from macs_ablation import install_facility_credits, train_ablation
import macs_v3  # noqa: F401   rebinds compute_credits to the v3 dispatcher

FACILITY = dict(size=24, horizon=60, rho=4.0, patch=0, demand="uniform")
LOG_DIR = "logs/macs_o9_diag"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--diag-episodes", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    install_facility_credits()

    cfg = dict(n_episodes=4000, eval_every=100, log_every=100)
    if args.smoke:
        cfg = dict(n_episodes=60, warmup=200, eval_every=20, n_eval=2,
                   log_every=20, target_every=10)
        args.diag_episodes = 4

    def factory():
        return FacilityCoverageEnv(k=args.k, **FACILITY)

    name = f"Facility {FACILITY['size']}x{FACILITY['size']}"
    out = {}
    for obs_mode in ("shared", "private"):
        log = train_ablation(
            factory, name, mode="MACS", obs_mode=obs_mode, clip="none",
            lr=args.lr, seed=args.seed, log_dir=LOG_DIR,
            label=f"diag_k{args.k}_{obs_mode}_seed{args.seed}",
            diag=True, diag_episodes=args.diag_episodes, **cfg)
        out[obs_mode] = log["diag"]

    print("\n" + "=" * 62)
    print(f"  O9 diagnostic, k={args.k}, seed={args.seed}")
    print("=" * 62)
    print(f"{'quantity':<22} {'shared':>12} {'private':>12} {'ratio':>8}")
    print("-" * 58)
    for key in ("cells_per_agent", "cells_union", "objective",
                "q_spread_mean", "action_entropy_norm"):
        s, p = out["shared"][key], out["private"][key]
        r = f"{p/s:.3f}" if s else "n/a"
        print(f"{key:<22} {s:>12.3f} {p:>12.3f} {r:>8}")
    print(f"{'action_top':<22} {out['shared']['action_top']:>12} "
          f"{out['private']['action_top']:>12}")

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    dest = Path(LOG_DIR) / f"diag_summary_k{args.k}_seed{args.seed}.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n-> {dest}")


if __name__ == "__main__":
    main()