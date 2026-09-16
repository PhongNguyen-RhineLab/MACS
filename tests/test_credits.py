"""
Credit rule tests: the ordering claim, the efficiency accounting, and the
dispatch that used to depend on import order.
"""

import numpy as np
import pytest

from macs import credits as C
from macs import shapley
from macs.envs import (FacilityLocationF, IslandF, SaturatingCoverageEnv,
                       SaturatingF, F_cardinality)

OBJECTIVES = {
    "cardinality": (F_cardinality, 10),
    "saturating": (SaturatingF(12, region=4, cap_frac=0.5), 12),
    "facility": (FacilityLocationF(12, rho=3.0), 12),
    "islands": (IslandF(12, rho=3.0, n_islands=4, radius=1, min_sep=4,
                        eps=0.0, layout_seed=0), 12),
}


@pytest.fixture
def rng():
    return np.random.default_rng(13)


def case(rng, k, size):
    S_prev = rng.random((size, size)) < 0.35
    patches = []
    for _ in range(k):
        pos = rng.integers(1, size - 1, size=2)
        m = np.zeros((size, size), dtype=bool)
        m[pos[0] - 1:pos[0] + 2, pos[1] - 1:pos[1] + 2] = True
        patches.append(m)
    return {"S_prev": S_prev, "patches": patches}


def reward(F, info):
    return F(info["S_prev"] | np.logical_or.reduce(info["patches"])) \
        - F(info["S_prev"])


# =============================================================================
# The ordering remark
# =============================================================================

@pytest.mark.parametrize("name", sorted(OBJECTIVES))
@pytest.mark.parametrize("k", [2, 3, 4])
def test_dr_le_shapley_le_local(rng, name, k):
    """
    Remark `credit-order`: for submodular F, per agent and per step,

        DR <= Shapley <= LOCAL

    since the last-arrival marginal is the smallest, the first-arrival
    marginal the largest, and Shapley averages marginals over every arrival
    position.
    """
    F, size = OBJECTIVES[name]
    for _ in range(5):
        info = case(rng, k, size)
        r = reward(F, info)
        dr = C.compute_credits("DR", r, info, F, k)
        loc = C.compute_credits("LOCAL", r, info, F, k)
        phi = C.compute_credits("MACS", r, info, F, k)
        assert np.all(dr <= phi + 1e-9)
        assert np.all(phi <= loc + 1e-9)


@pytest.mark.parametrize("name", sorted(OBJECTIVES))
def test_local_over_counts_dr_under_counts(rng, name):
    """
    LOCAL sums to at least r and DR to at most r, both strictly on overlap.
    This is what makes them the two natural failure modes that Shapley sits
    between, and why an efficiency-error column is informative rather than
    decorative.
    """
    F, size = OBJECTIVES[name]
    k = 3
    strict = 0
    for _ in range(8):
        info = case(rng, k, size)
        r = reward(F, info)
        loc = C.compute_credits("LOCAL", r, info, F, k).sum()
        dr = C.compute_credits("DR", r, info, F, k).sum()
        assert loc >= r - 1e-9
        assert dr <= r + 1e-9
        if loc > r + 1e-6:
            strict += 1
    assert strict > 0, "no overlapping case generated; test is vacuous"


# =============================================================================
# Efficiency accounting
# =============================================================================

@pytest.mark.parametrize("mode", C.CREDIT_MODES)
def test_efficiency_error_is_defined_for_every_credit_rule(rng, mode):
    """
    Regression: credit_sum_err was logged only on the MACS branch, so every
    other arm's column read as an exact zero rather than as the
    over-attribution or under-attribution it actually was.
    """
    F, size = OBJECTIVES["saturating"]
    k = 3
    info = case(rng, k, size)
    r = reward(F, info)
    phi = C.compute_credits(mode, r, info, F, k, rng=rng)
    err = C.efficiency_error(phi, r)
    if mode in ("MACS", "MACS-CLIP", "MACS-MC"):
        assert err < 1e-9                         # exact by construction
    elif mode == "SHARED":
        assert err == pytest.approx((k - 1) * r, abs=1e-9)
    else:
        assert err >= 0.0


def test_mixing_modes_have_no_credit_rule():
    """VDN and QMIX learn from r_team; they are not credit rules."""
    for mode in C.MIXING_MODES:
        assert mode not in C.CREDIT_MODES


def test_unknown_mode_raises(rng):
    F, size = OBJECTIVES["cardinality"]
    with pytest.raises(ValueError, match="unknown mode"):
        C.compute_credits("NOPE", 1.0, case(rng, 2, size), F, 2)


# =============================================================================
# Dispatch is a function of arguments, not of import order
# =============================================================================

@pytest.mark.parametrize("name", ["facility", "islands"])
def test_facility_credits_are_exact_without_any_install_call(rng, name):
    """
    The routing that used to require calling install_facility_credits() in
    the right order now follows from the type of F. Nothing is patched, so
    nothing can be patched in the wrong order.
    """
    F, size = OBJECTIVES[name]
    for k in (2, 4, 6):
        info = case(rng, k, size)
        r = reward(F, info)
        phi = C.compute_credits("MACS", r, info, F, k)
        assert phi.sum() == pytest.approx(r, abs=1e-9)
        assert phi == pytest.approx(
            shapley.shapley_facility(info["S_prev"], info["patches"], F),
            abs=1e-12)


def test_large_k_on_non_facility_objective_never_returns_nan(rng):
    """
    Regression for the live bug: at k>4 on a saturating objective the old
    dispatcher took a Monte Carlo path whose sample count defaulted to zero,
    and `run_facility.py --exp o10` (default ks include 6) trained on NaN
    credits. Both the auto path and the explicit exact path must be finite.
    """
    F, size = OBJECTIVES["saturating"]
    k = 6
    info = case(rng, k, size)
    r = reward(F, info)
    for backend in ("auto", "subset"):
        phi = C.compute_credits("MACS", r, info, F, k, backend=backend,
                                rng=np.random.default_rng(0))
        assert np.all(np.isfinite(phi)), backend
        assert phi.sum() == pytest.approx(r, abs=1e-6)


def test_macs_mc_forces_monte_carlo_even_where_exact_is_cheap(rng):
    """
    The MACS-MC ablation isolates approximation error, so it must not be
    quietly upgraded to the exact backend at small k or on facility F.
    """
    F, size = OBJECTIVES["facility"]
    k = 3
    info = case(rng, k, size)
    r = reward(F, info)
    exact = C.compute_credits("MACS", r, info, F, k)
    mc = C.compute_credits("MACS-MC", r, info, F, k, mc_m=8,
                           rng=np.random.default_rng(0))
    assert mc.sum() == pytest.approx(r, abs=1e-9)    # efficiency survives
    assert not np.allclose(mc, exact, atol=1e-9)     # but the split differs


# =============================================================================
# The budget bound
# =============================================================================

def test_budget_bound_holds_along_a_trajectory():
    """
    Lemma `budget`: the remaining discounted credit an agent can still earn
    is bounded by B(S_t) = F(V) - F(S_t). This is what licenses clipping the
    bootstrap at B(S').
    """
    env = SaturatingCoverageEnv(size=12, horizon=40, k=3, patch=1, seed=0)
    env.reset()
    F_V = env.F_total()
    rng = np.random.default_rng(0)
    rewards = []
    budgets = [F_V - env.F(env.covered)]
    done = False
    while not done:
        _, r, done, _ = env.step(rng.integers(0, env.n_actions, size=env.k))
        rewards.append(r)
        budgets.append(F_V - env.F(env.covered))
    for t in range(len(rewards)):
        assert sum(rewards[t:]) <= budgets[t] + 1e-9


def test_reward_telescopes():
    """sum_t r_t = F(S_T) - F(S_0): no value is created or lost by the
    per-step accounting."""
    env = SaturatingCoverageEnv(size=12, horizon=40, k=3, patch=1, seed=1)
    env.reset()
    f0 = env.F(env.covered)
    rng = np.random.default_rng(2)
    total = 0.0
    done = False
    while not done:
        _, r, done, _ = env.step(rng.integers(0, env.n_actions, size=env.k))
        total += r
    assert total == pytest.approx(env.F(env.covered) - f0, abs=1e-9)
