"""
The interpolated efficient credit INT and the bracket it lives in.

Every learning and equilibrium result of the paper uses three properties of
a credit rule: efficiency, phi_i >= DR_i, and phi_i <= LOC_i. INT is the
cheapest rule with all three; these tests check them on the paper's own
objectives, plus null agent, symmetry, and equality with Shapley at k = 2.
"""

import numpy as np
import pytest

from macs import credits as C
from macs import shapley
from macs.envs import IslandEnv, SaturatingCoverageEnv
from macs.envs.facility import FacilityLocationF


def _random_step(F, size, k, rng, radius=1, n_cov=20):
    S = np.zeros((size, size), bool)
    idx = rng.choice(size * size, n_cov, replace=False)
    S.flat[idx] = True
    patches = []
    for _ in range(k):
        x, y = rng.integers(0, size, 2)
        m = np.zeros((size, size), bool)
        m[max(0, x - radius):x + radius + 1, max(0, y - radius):y + radius + 1] = True
        patches.append(m)
    return {"S_prev": S, "patches": patches}


def _objectives():
    yield "facility", FacilityLocationF(12, rho=3.0), 12
    sat = SaturatingCoverageEnv(k=3, size=12, horizon=10, patch=1, region=4,
                                cap_frac=0.5, seed=0)
    yield "saturating", sat.F, 12
    yield "modular", (lambda m: float(m.sum())), 12


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_efficient_and_bracketed(k):
    rng = np.random.default_rng(k)
    for name, F, size in _objectives():
        for _ in range(40):
            info = _random_step(F, size, k, rng)
            loc, dr, r = C.marginal_bounds(info, F)
            phi = C.compute_credits("INT", r, info, F, k)
            assert abs(phi.sum() - r) < 1e-8, name
            assert np.all(phi >= dr - 1e-9), name
            assert np.all(phi <= loc + 1e-9), name
            assert C.efficiency_error(phi, r) < 1e-8


def test_equals_shapley_at_k2():
    rng = np.random.default_rng(7)
    for name, F, size in _objectives():
        for _ in range(40):
            info = _random_step(F, size, 2, rng)
            loc, dr, r = C.marginal_bounds(info, F)
            phi = C.interpolated_credit(loc, dr, r)
            shp = shapley.shapley_permutation(info["S_prev"], info["patches"], F)
            assert np.allclose(phi, shp, atol=1e-9), name


def test_differs_from_shapley_at_k3():
    # modular F, footprints {a,b}, {a,b}, {b,c}: Shapley (5/6, 5/6, 4/3),
    # INT (0.8, 0.8, 1.4) -- the worked example in the paper
    F = lambda m: float(m.sum())
    a, b, c = (0, 0), (0, 1), (0, 2)
    def mask(*cells):
        m = np.zeros((1, 3), bool)
        for cell in cells:
            m[cell] = True
        return m
    info = {"S_prev": mask(), "patches": [mask(a, b), mask(a, b), mask(b, c)]}
    loc, dr, r = C.marginal_bounds(info, F)
    assert r == 3 and np.allclose(loc, [2, 2, 2]) and np.allclose(dr, [0, 0, 1])
    phi = C.interpolated_credit(loc, dr, r)
    assert np.allclose(phi, [0.8, 0.8, 1.4])
    shp = shapley.shapley_permutation(info["S_prev"], info["patches"], F)
    assert np.allclose(shp, [5 / 6, 5 / 6, 4 / 3])


def test_null_agent_and_symmetry():
    rng = np.random.default_rng(3)
    F = FacilityLocationF(12, rho=3.0)
    for _ in range(30):
        info = _random_step(F, 12, 3, rng)
        # agent 2 lands inside already-covered ground -> null
        info["patches"][2] = info["S_prev"] & np.zeros_like(info["S_prev"])
        loc, dr, r = C.marginal_bounds(info, F)
        phi = C.interpolated_credit(loc, dr, r)
        assert abs(phi[2]) < 1e-12
        # identical footprints -> identical credit
        info["patches"][1] = info["patches"][0].copy()
        loc, dr, r = C.marginal_bounds(info, F)
        phi = C.interpolated_credit(loc, dr, r)
        assert abs(phi[0] - phi[1]) < 1e-12


def test_on_island_trajectories():
    env = IslandEnv(k=12, seed=0)
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(env.horizon):
        _, r, done, info = env.step(rng.integers(0, 5, env.k))
        phi = C.compute_credits("INT", r, info, env.F, env.k)
        loc, dr, _ = C.marginal_bounds(info, env.F)
        assert abs(phi.sum() - r) < 1e-8
        assert np.all(phi >= dr - 1e-9) and np.all(phi <= loc + 1e-9)
        if done:
            break
