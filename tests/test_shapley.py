"""
Shapley backend tests.

These were `macs_main.run_unit_tests`, `macs_fl.run_unit_tests` and
`macs_v3.run_unit_tests` -- three functions guarded by `if __name__ ==
"__main__"` and a `--test` flag, so nothing ran them unless a human
remembered to. They are the theory-side claims of the draft, which is
exactly the kind of thing that should fail a commit rather than a reviewer.
"""

import numpy as np
import pytest

from macs import shapley
from macs.envs import (F_cardinality, FacilityLocationF, IslandF,
                       SaturatingF, make_F_weighted)

SIZE = 10
EXACT = (shapley.shapley_permutation, shapley.shapley_subset)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def random_state(rng, k, size=SIZE, density=0.4):
    S_prev = rng.random((size, size)) < density
    patches = []
    for _ in range(k):
        pos = rng.integers(1, size - 1, size=2)
        m = np.zeros((size, size), dtype=bool)
        m[pos[0] - 1:pos[0] + 2, pos[1] - 1:pos[1] + 2] = True
        patches.append(m)
    return S_prev, patches


def team_reward(F, S_prev, patches):
    return F(S_prev | np.logical_or.reduce(patches)) - F(S_prev)


OBJECTIVES = {
    "cardinality": F_cardinality,
    "saturating": SaturatingF(12, region=4, cap_frac=0.5),
    "facility": FacilityLocationF(12, rho=3.0),
    "islands": IslandF(12, rho=3.0, n_islands=4, radius=1, min_sep=4,
                       eps=0.0, layout_seed=0),
}


def objective(name):
    """SaturatingF and the facility objectives are sized 12, not SIZE."""
    return OBJECTIVES[name], (SIZE if name == "cardinality" else 12)


# =============================================================================
# The Shapley axioms
# =============================================================================

@pytest.mark.parametrize("name", sorted(OBJECTIVES))
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("backend", EXACT)
def test_efficiency(rng, name, k, backend):
    """Axiom (i): sum_i phi_i = v(N) = r_t, to machine precision."""
    F, size = objective(name)
    for _ in range(4):
        S_prev, patches = random_state(rng, k, size)
        phi = backend(S_prev, patches, F)
        assert phi.sum() == pytest.approx(team_reward(F, S_prev, patches),
                                          abs=1e-9)


@pytest.mark.parametrize("backend", EXACT)
def test_null_agent(rng, backend):
    """Axiom (ii): an agent whose footprint adds nothing is paid nothing."""
    F, size = objective("cardinality")
    S_prev, patches = random_state(rng, 3, size)
    patches[1] = S_prev.copy()                  # already covered: adds zero
    phi = backend(S_prev, patches, F)
    assert phi[1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("backend", EXACT)
def test_symmetry(rng, backend):
    """Axiom (iii): interchangeable agents are paid the same."""
    F, size = objective("cardinality")
    S_prev, patches = random_state(rng, 3, size)
    patches[1] = patches[0].copy()              # identical footprints
    phi = backend(S_prev, patches, F)
    assert phi[0] == pytest.approx(phi[1], abs=1e-9)


# =============================================================================
# The backends agree
# =============================================================================

@pytest.mark.parametrize("name", sorted(OBJECTIVES))
@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_subset_matches_permutation(rng, name, k):
    """
    The subset form costs 2^k evaluations shared across agents, the
    permutation form k!. They are the same number.
    """
    F, size = objective(name)
    S_prev, patches = random_state(rng, k, size)
    assert shapley.shapley_subset(S_prev, patches, F) == pytest.approx(
        shapley.shapley_permutation(S_prev, patches, F), abs=1e-9)


@pytest.mark.parametrize("name", ["facility", "islands"])
@pytest.mark.parametrize("k", [2, 3, 4, 6])
def test_facility_closed_form(rng, name, k):
    """
    Prop. `facility`: the O(|D| k log k) closed form equals the
    enumeration. This is what makes exact MACS credits affordable at any k
    on the islands benchmark.
    """
    F, size = objective(name)
    S_prev, patches = random_state(rng, k, size)
    assert shapley.shapley_facility(S_prev, patches, F) == pytest.approx(
        shapley.shapley_subset(S_prev, patches, F), abs=1e-9)


@pytest.mark.parametrize("weighted", [False, True])
def test_modular_closed_form(rng, weighted):
    """
    Prop. `modular`: for a MODULAR F each new cell's weight is split equally
    among the agents covering it. Only valid for modular F, so a saturating
    objective is deliberately not tested here.
    """
    size = SIZE
    weights = rng.random((size, size)) if weighted else None
    F = make_F_weighted(weights) if weighted else F_cardinality
    S_prev, patches = random_state(rng, 3, size)
    closed = shapley.shapley_modular_closed_form(S_prev, patches, weights)
    assert closed == pytest.approx(
        shapley.shapley_permutation(S_prev, patches, F), abs=1e-9)


def test_mc_is_near_exact(rng):
    """
    Prop. `mc-error`: the permutation-sampling estimator is unbiased, and
    its marginals telescope within each permutation, so efficiency holds
    EXACTLY at any sample count -- only the split between agents is
    approximate.
    """
    F, size = objective("saturating")
    S_prev, patches = random_state(rng, 4, size)
    exact = shapley.shapley_permutation(S_prev, patches, F)
    mc = shapley.shapley_mc(S_prev, patches, F, 4000,
                            np.random.default_rng(0))
    assert mc.sum() == pytest.approx(exact.sum(), abs=1e-9)   # exact
    assert mc == pytest.approx(exact, abs=0.5)                # approximate


def test_mc_rejects_zero_samples(rng):
    """
    Regression: the old dispatcher defaulted mc_m to 0 and train_macs never
    passed it, so MACS at k>4 silently trained on NaN credits whenever the
    module that fixed the default had not been imported first.
    """
    F, size = objective("cardinality")
    S_prev, patches = random_state(rng, 6, size)
    with pytest.raises(ValueError, match="m >= 1"):
        shapley.shapley_mc(S_prev, patches, F, 0, rng)


# =============================================================================
# Backend resolution
# =============================================================================

def test_resolve_routes_facility_to_closed_form():
    """Including IslandF, which subclasses FacilityLocationF."""
    for name in ("facility", "islands"):
        F, _ = objective(name)
        for k in (2, 4, 6, 10):
            assert shapley.resolve(F, k) == "facility"


def test_resolve_falls_back_by_team_size():
    F, _ = objective("saturating")
    assert shapley.resolve(F, 4) == "permutation"
    assert shapley.resolve(F, 6) == "mc"
    assert shapley.resolve(F, 6, "subset") == "subset"


def test_resolve_rejects_unknown_backend():
    F, _ = objective("cardinality")
    with pytest.raises(ValueError, match="unknown Shapley backend"):
        shapley.resolve(F, 3, "magic")


def test_subset_cost_is_shared_across_agents(rng):
    """
    The cost correction that inverted the paper's justification for Monte
    Carlo: the subset form evaluates F once per coalition and shares the
    table across agents, so it is 2^k evaluations total, not k * 2^(k-1).
    """
    F, size = objective("saturating")
    k = 6
    S_prev, patches = random_state(rng, k, size)
    _, n = shapley.count_evals(shapley.shapley_subset, S_prev, patches, F)
    assert n == 2 ** k                       # 64, not 6 * 32 = 192
    assert n < 64 * k                        # cheaper than MC with m=64
