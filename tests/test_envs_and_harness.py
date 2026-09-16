"""
Environments, policies, rollout, logs and statistics.
"""

import json

import numpy as np
import pytest

from macs import config as CFG
from macs import policies as P
from macs import runlog
from macs.envs import (FacilityCoverageEnv, IslandEnv, MultiAgentCoverageEnv,
                       SaturatingCoverageEnv, TwoRoomCoverageEnv,
                       compute_frontier)
from macs.obs import ObsTracker
from macs.rollout import factory, ladder, mean_coverage, rollout
from macs.stats import (exact_perm_test, f1, min_attainable_p,
                        paired_sign_flip_test, r1)

ENVS = {
    "base": (MultiAgentCoverageEnv, dict(size=12, horizon=15, k=3, patch=1)),
    "tworoom": (TwoRoomCoverageEnv, dict(size=12, horizon=15, k=3, patch=1)),
    "saturating": (SaturatingCoverageEnv,
                   dict(size=12, horizon=15, k=3, patch=1, region=4,
                        cap_frac=0.5)),
    "facility": (FacilityCoverageEnv,
                 dict(size=16, horizon=15, k=3, patch=0, rho=3.0)),
    "islands": (IslandEnv,
                dict(size=24, horizon=15, k=3, rho=4.0, n_islands=8,
                     radius=1, min_sep=8, eps=0.0, layout_seed=0)),
}


def build(name, seed=0):
    cls, cfg = ENVS[name]
    return cls(**cfg, seed=seed)


# =============================================================================
# Environments
# =============================================================================

@pytest.mark.parametrize("name", sorted(ENVS))
def test_same_seed_same_trajectory(name):
    a, b = build(name, 3), build(name, 3)
    a.reset()
    b.reset()
    rng = np.random.default_rng(0)
    for _ in range(10):
        acts = rng.integers(0, a.n_actions, size=a.k)
        (pa, za), ra, da, ia = a.step(acts)
        (pb, zb), rb, db, ib = b.step(acts)
        assert np.array_equal(pa, pb)
        assert np.array_equal(za, zb)
        assert ra == rb and da == db
        assert np.array_equal(ia["S_prev"], ib["S_prev"])


@pytest.mark.parametrize("name", sorted(ENVS))
def test_reward_is_the_marginal_of_the_step(name):
    env = build(name)
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(10):
        f_before = env.F(env.covered)
        _, r, _, info = env.step(rng.integers(0, env.n_actions, size=env.k))
        assert env.F(info["S_prev"]) == pytest.approx(f_before, abs=1e-9)
        assert r == pytest.approx(env.F(env.covered) - f_before, abs=1e-9)
        assert r >= -1e-9                    # F is monotone


@pytest.mark.parametrize("name", sorted(ENVS))
def test_coverage_is_monotone_and_bounded(name):
    env = build(name)
    env.reset()
    rng = np.random.default_rng(2)
    prev = env.total_coverage
    for _ in range(10):
        env.step(rng.integers(0, env.n_actions, size=env.k))
        assert env.total_coverage >= prev - 1e-9
        assert env.total_coverage <= env.max_cells + 1e-9
        prev = env.total_coverage


def test_initial_coverage_earns_no_reward():
    """S_0 = P(s_0) is covered at reset WITHOUT reward, so the very first
    step cannot be paid for footprints the agents already occupy."""
    env = build("base")
    env.reset()
    assert env.covered.any()
    _, r, _, info = env.step([4] * env.k)     # everyone stays put
    assert r == pytest.approx(0.0, abs=1e-9)


def test_two_room_walls_block_movement():
    env = TwoRoomCoverageEnv(size=12, horizon=30, k=2, patch=1, seed=0)
    env.reset()
    assert not env.passable[:, 6].all()       # there is a wall
    rng = np.random.default_rng(0)
    for _ in range(30):
        env.step(rng.integers(0, env.n_actions, size=env.k))
        for i in range(env.k):
            assert env.passable[tuple(env.pos[i])]
    assert not env.covered[~env.passable].any()


def test_next_pos_does_not_move_the_agent():
    env = build("base")
    env.reset()
    before = env.pos.copy()
    for a in range(env.n_actions):
        env.next_pos(0, a)
    assert np.array_equal(env.pos, before)


def test_frontier_is_uncovered_and_adjacent():
    z = np.zeros((8, 8), dtype=np.float32)
    z[3:5, 3:5] = 1.0
    fr = compute_frontier(z) > 0.5
    assert not (fr & (z == 1.0)).any()        # never a covered cell
    assert fr[2, 3] and fr[5, 4] and fr[3, 2]
    assert not fr[0, 0]                       # not adjacent to anything


# =============================================================================
# Observation modes
# =============================================================================

def test_shared_observation_is_the_same_for_every_agent():
    env = build("base")
    (_, z) = env.reset()
    zs, frs = ObsTracker(env, "shared").channels(z)
    assert zs.shape == (env.k, env.size, env.size)
    for i in range(1, env.k):
        assert np.array_equal(zs[0], zs[i])
        assert np.array_equal(frs[0], frs[i])


def test_private_observation_is_a_subset_of_the_shared_one():
    env = build("base")
    (_, z) = env.reset()
    tracker = ObsTracker(env, "private")
    rng = np.random.default_rng(0)
    for _ in range(8):
        (_, z), _, _, _ = env.step(rng.integers(0, env.n_actions, size=env.k))
        tracker.update()
    zs, _ = tracker.channels(z)
    shared = env.covered
    for i in range(env.k):
        assert not (zs[i].astype(bool) & ~shared).any()
    assert zs.sum() < env.k * shared.sum()    # strictly less, not a copy


def test_shared_observation_preserves_the_wall_channel():
    """
    Regression: booleanizing the coverage map erases the two-room wall
    encoding (-1.0), which the Q network reads as "uncovered and reachable".
    """
    env = TwoRoomCoverageEnv(size=12, horizon=20, k=2, patch=1, seed=0)
    (_, z) = env.reset()
    zs, _ = ObsTracker(env, "shared").channels(z)
    assert (zs[0] == -1).any()
    assert np.array_equal(zs[0].astype(np.float32), z)


def test_obs_tracker_rejects_unknown_mode():
    env = build("base")
    with pytest.raises(ValueError, match="obs_mode"):
        ObsTracker(env, "telepathic")


# =============================================================================
# Policies and rollout
# =============================================================================

@pytest.mark.parametrize("name,pol", P.LADDER)
def test_ladder_policies_return_valid_actions(name, pol):
    env = build("facility")
    env.rho_hint = 3.0
    env.reset()
    rng = np.random.default_rng(0)
    fn = pol(env)
    for _ in range(5):
        acts = np.asarray(fn(env, rng))
        assert acts.shape == (env.k,)
        assert np.all((acts >= 0) & (acts < env.n_actions))
        env.step(acts)


def test_every_policy_is_a_factory():
    """
    The convention the four ad-hoc `mk in _PER_ENV` checks existed to
    replace: calling the policy with an env always yields the callable.
    """
    rng = np.random.default_rng(0)
    for env_name, pols in (("facility", P.LADDER + (("front", P.front),)),
                           ("islands", P.ISLAND_LADDER)):
        env = build(env_name)
        env.rho_hint = env.Ffl.rho
        env.reset()
        for name, pol in pols:
            fn = pol(env)
            assert callable(fn), name
            assert np.asarray(fn(env, rng)).shape == (env.k,), name


def test_island_planner_rejects_a_non_island_environment():
    """A clear message beats an AttributeError three frames deep."""
    env = build("facility")
    env.reset()
    with pytest.raises(TypeError, match="island centres"):
        P.plan(env)


def test_rollout_is_reproducible():
    cfg = dict(size=16, horizon=10, k=3, patch=0, rho=3.0)
    fac = factory(FacilityCoverageEnv, cfg)
    a = rollout(fac, P.dec, episodes=3, seed=0)["coverage"]
    b = rollout(fac, P.dec, episodes=3, seed=0)["coverage"]
    assert np.array_equal(a, b)
    c = rollout(fac, P.dec, episodes=3, seed=1)["coverage"]
    assert not np.array_equal(a, c)


def test_stateful_policy_state_does_not_leak_between_episodes():
    """
    A blind agent's private coverage set must be rebuilt each episode. The
    factory is called per episode precisely so this cannot go wrong.
    """
    cfg = dict(size=16, horizon=10, k=3, patch=0, rho=3.0)
    fac = factory(FacilityCoverageEnv, cfg)
    one = rollout(fac, P.blind, episodes=1, seed=0)["coverage"][0]
    many = rollout(fac, P.blind, episodes=3, seed=0)["coverage"][0]
    assert one == pytest.approx(many, abs=1e-9)


def test_islands_have_assignment_headroom_and_it_grows_with_k():
    """
    The benchmark's reason to exist. plan - plan_self is what a credit rule
    could plausibly improve, and the predicted pattern is that it grows with
    the team size. If this ever fails, the islands results are not testing
    credit assignment.
    """
    gaps = {}
    for k in (3, 6):
        def make(e, s, k=k):
            env = IslandEnv(k=k, **CFG.ISLANDS, seed=s)
            env.rho_hint = CFG.ISLANDS["rho"]
            return env
        res = ladder(make, (("dec", P.dec), ("plan_self", P.plan_self),
                            ("plan", P.plan)), episodes=3, seed=0)
        assert res["plan"][0] - res["dec"][0] > 10.0        # planning
        gaps[k] = res["plan"][0] - res["plan_self"][0]      # assignment
    assert gaps[3] > 0.0
    assert gaps[6] > gaps[3]


def test_mean_coverage_matches_rollout():
    cfg = dict(size=16, horizon=10, k=2, patch=0, rho=3.0)
    fac = factory(FacilityCoverageEnv, cfg)
    assert mean_coverage(fac, P.dec, 3, 0) == pytest.approx(
        rollout(fac, P.dec, 3, 0)["coverage"].mean(), abs=1e-12)


# =============================================================================
# Logs
# =============================================================================

def write_log(tmp_path, label, config, n_blocks=12, eval_base=10.0):
    log = {"label": label, "config": config,
           "per_block": {k: [] for k in runlog.BLOCK_KEYS}, "summary": {}}
    pb = log["per_block"]
    for i in range(n_blocks):
        pb["episode"].append(100 * (i + 1))
        pb["coverage_mean"].append(eval_base + i)
        pb["eval_coverage"].append(eval_base + i)
        pb["loss"].append(1.0)
        pb["epsilon"].append(0.5)
        pb["credit_sum_err"].append(None if config["method"] in ("VDN",)
                                    else 0.0)
        pb["clip_frac"].append(0.0)
    p = tmp_path / f"{label}.json"
    p.write_text(json.dumps(log))
    return p


def test_is_complete_counts_blocks(tmp_path):
    cfg = dict(method="MACS", env_name="E", k=2, seed=0)
    write_log(tmp_path, "run_seed0", cfg, n_blocks=10)
    assert runlog.is_complete("run_seed0", tmp_path, 1000, 100)
    assert not runlog.is_complete("run_seed0", tmp_path, 2000, 100)
    assert not runlog.is_complete("absent", tmp_path, 1000, 100)


def test_is_complete_survives_a_truncated_log(tmp_path):
    (tmp_path / "broken_seed0.json").write_text('{"per_block": {"epis')
    assert not runlog.is_complete("broken_seed0", tmp_path, 100, 10)


def test_arm_key_separates_the_axes():
    base = dict(method="MACS", obs_mode="shared", clip="none")
    assert runlog.arm_key(base) == "MACS/shared/none"
    assert runlog.arm_key(dict(base, obs_mode="private")) == \
        "MACS/private/none"
    assert runlog.arm_key(dict(base, clip="budget")) == "MACS/shared/budget"
    assert runlog.arm_key(dict(base, lr=5e-4)) == "MACS/shared/none/lr0.0005"


def test_resolve_arm_by_prefix_and_refuses_ambiguity():
    arms = {"MACS/shared/none/lr0.0005": {}, "LOCAL/shared/none/lr0.0001": {}}
    assert runlog.resolve_arm("MACS/shared/none", arms) == \
        "MACS/shared/none/lr0.0005"
    arms["MACS/shared/none/lr0.0001"] = {}
    assert runlog.resolve_arm("MACS/shared/none", arms) is None


def test_collect_groups_by_config_and_arm(tmp_path):
    for seed in (0, 1):
        write_log(tmp_path, f"E_k2_MACS_shared_clip-none_seed{seed}",
                  dict(method="MACS", env_name="E", k=2, seed=seed,
                       obs_mode="shared", clip="none"))
        write_log(tmp_path, f"E_k2_MACS_private_clip-none_seed{seed}",
                  dict(method="MACS", env_name="E", k=2, seed=seed,
                       obs_mode="private", clip="none"), eval_base=5.0)
    out = runlog.collect(tmp_path)
    assert set(out[("E", 2)]) == {"MACS/shared/none", "MACS/private/none"}
    assert set(out[("E", 2)]["MACS/shared/none"]) == {0, 1}


def test_still_rising_detects_an_undertrained_run(tmp_path):
    p = write_log(tmp_path, "rising_seed0",
                  dict(method="MACS", env_name="E", k=2, seed=0),
                  n_blocks=30)
    assert runlog.still_rising(p, window=5) is True


# =============================================================================
# Statistics
# =============================================================================

def test_paired_test_is_symmetric_and_bounded():
    a = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    b = [9.0, 10.5, 11.0, 12.5, 13.0, 14.5]
    d1, p1 = paired_sign_flip_test(a, b)
    d2, p2 = paired_sign_flip_test(b, a)
    assert d1 == pytest.approx(-d2, abs=1e-12)
    assert p1 == pytest.approx(p2, abs=1e-12)
    assert 0.0 <= p1 <= 1.0


def test_paired_test_p_floor():
    """
    2/2^n is the smallest two-sided p the paired design can produce, so five
    seeds can never reach 0.05 and six is the minimum for a paired claim at
    that level. Reporting p=0.0625 as "not significant" without this is
    misleading.
    """
    assert min_attainable_p(5) == pytest.approx(0.0625)
    assert min_attainable_p(6) == pytest.approx(0.03125)
    a = [10.0, 11.0, 12.0, 13.0, 14.0]
    b = [x - 1.0 for x in a]            # every difference the same sign
    _, p = paired_sign_flip_test(a, b)
    assert p == pytest.approx(0.0625)   # the floor, not 0.0
    a6, b6 = a + [15.0], b + [14.0]
    _, p6 = paired_sign_flip_test(a6, b6)
    assert p6 < 0.05


def test_identical_arms_give_p_one():
    a = [1.0, 2.0, 3.0, 4.0]
    _, p = paired_sign_flip_test(a, a)
    assert p == pytest.approx(1.0)


def test_unpaired_test_runs_and_is_bounded():
    obs, p = exact_perm_test([5, 6, 7, 8, 9], [1, 2, 3, 4, 5])
    assert obs == pytest.approx(4.0)
    assert 0.0 <= p <= 1.0


def test_rounding_is_half_away_from_zero():
    """
    Python rounds half to even, so round(52.75, 1) gives 52.7 and the
    emitted table disagrees with the hand-typed one in the paper.
    """
    assert r1(52.75) == 52.8
    assert r1(125.65) == 125.7
    assert r1(-52.75) == -52.8
    assert f1(None) == "n/a"
    assert f1(52.75) == "52.8"
