"""
Experiment configuration.

Every geometry, grid and training budget the paper reports, in one file.
These were previously spread across run_facility.py, run_mc_sweep.py (a
stale copy of it), run_multiseed_sat.py, run_o9.py and run_scalability.py,
with the copies already out of sync.
"""

# =============================================================================
# Environment geometry
# =============================================================================

# Facility geometry. rho=4 with single-cell footprints and a 24x24 grid is
# the configuration where the probe measured the largest S-value ladder
# (7.9 / 13.5 / 17.9 / 21.2 / 19.6 at k = 2,3,4,6,8).
FACILITY = dict(size=24, horizon=60, rho=4.0, patch=0, demand="uniform")

# Islands: the canonical probe config. H=30 is load-bearing -- at H=60 the
# assignment headroom collapses to ~2 points and a stateless sweep recovers.
# layout_seed=0 fixes the map, so no demand input channel is needed.
ISLANDS = dict(size=24, horizon=30, rho=4.0, n_islands=8, radius=1,
               min_sep=8, eps=0.0, layout_seed=0, depot=True)

# Decoy islands: the islands layout with one island (nearest the grid centre)
# carrying 3x the demand of each other one, and a shorter horizon. Chosen by
# probes.decoy (k=6, 12 episodes, random depot), which measured, for one agent
# leaving the herd on the decoy for its best alternative island:
#
#     horizon  mult | dev LOCAL  dev Shapley  dev DR | J(dev) - J(herd)
#       30      3   |   -25.4      -10.5       -0.4  |    0.6
#       24      2   |    -3.4       +7.7      +15.4  |    1.7
#       20      3   |    -7.7       +5.6      +14.0  |    5.8   <- this
#       16      3   |    -7.2       +6.2      +14.7  |    6.8
#
# i.e. at H=20, 3x the herd is stable under LOCAL and not under Shapley or
# DR, and herding costs the team ~6 points of F(V). At H=30 there is enough
# slack that a herd costs < 1 point, so no rule can lose much to it. H=16
# gives a similar separation but plan_v only reaches ~70% of F(V).
DECOY = dict(ISLANDS, horizon=20, decoy_mult=3.0, depot_at=None)

# The saturating suite, reusing the exact configurations of Table 2.
SATURATING = {2: dict(size=12, horizon=40),
              3: dict(size=12, horizon=40),
              4: dict(size=12, horizon=40),
              6: dict(size=16, horizon=60)}
SATURATING_KW = dict(patch=1, region=4, cap_frac=0.5)


def saturating_cfg(k):
    """Full kwargs for SaturatingCoverageEnv at team size k."""
    if k not in SATURATING:
        raise KeyError(
            f"no saturating geometry for k={k}; known: {sorted(SATURATING)}")
    return dict(k=k, **SATURATING[k], **SATURATING_KW)


def env_name(kind, k=None):
    """The env_name string written into every log, so labels stay stable."""
    if kind == "facility":
        return f"Facility {FACILITY['size']}x{FACILITY['size']}"
    if kind == "islands":
        return f"Islands {ISLANDS['size']}x{ISLANDS['size']}"
    if kind == "decoy":
        return (f"Decoy {DECOY['size']}x{DECOY['size']} "
                f"x{DECOY['decoy_mult']:g} H{DECOY['horizon']}")
    if kind == "saturating":
        c = SATURATING[k]
        return f"Saturating {c['size']}x{c['size']}"
    raise ValueError(kind)


# =============================================================================
# Training budgets
# =============================================================================

# 4000 episodes matches the published saturating runs, so the two suites
# share a budget and "you gave them different amounts of training" is not
# available as an objection.
TRAIN_CFG = dict(n_episodes=4000, eval_every=100, log_every=100)

# 24x24 is four times the state space of the 12x12 saturating grid, so a
# short pilot risks measuring undertraining rather than the ablation.
PILOT_CFG = dict(n_episodes=2500, eval_every=100, log_every=100, warmup=1500)
ISLAND_PILOT_CFG = dict(n_episodes=8000, eval_every=200, log_every=200,
                        warmup=1500)
SMOKE_CFG = dict(n_episodes=30, warmup=200, eval_every=10, n_eval=2,
                 log_every=10, target_every=10)

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


# =============================================================================
# Sweeps
# =============================================================================

# Learning rates for the O9-LR robustness arm. 5e-4 is the published setting
# and the one that diverged in the pilot.
LR_GRID = [5e-4, 2e-4, 1e-4, 5e-5]

# The islands sweep found 5e-5 best for all six modes, i.e. optimal at the
# grid EDGE, for the second round running. Extended downward so the selected
# rate is interior; kept separate from LR_GRID so the o9lr plan is unchanged.
ISLANDLR_GRID = [5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]

ISLAND_MODES = ("SHARED", "LOCAL", "DR", "VDN", "QMIX", "MACS", "MACS-CLIP")

# The k=6 pilot flagged VDN and QMIX as diverging at the published 5e-4. A
# reviewer will read an untuned baseline as a stacked comparison, so
# islandslr reruns the baselines across the same grid the private arm got.
ISLANDLR_MODES = ("SHARED", "LOCAL", "DR", "VDN", "QMIX", "MACS")

# The decoy experiment exists to decide MACS vs LOCAL and vs DR; the mixing
# baselines add cost and answer a different question.
DECOY_MODES = ("LOCAL", "DR", "MACS")

SATURATING_MODES = ("SHARED", "LOCAL", "DR", "VDN", "QMIX",
                    "MACS", "MACS-CLIP")

LOG_DIR = {
    "o9": "logs/macs_facility",
    "o9lr": "logs/macs_facility_lr",
    "o10": "logs/macs_clipcontrol",
    "islands": "logs/macs_islands",
    "islandslr": "logs/macs_islands_lr",
    "decoy": "logs/macs_decoy",
    "saturating": "logs/macs_saturating",
    "scalability": "logs/macs_scalability",
}
