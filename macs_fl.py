"""
macs_fl.py -- Facility-location objective for MACS (O8, option 1).

Motivation
----------
The saturating block-cap objective F(S) = sum_R min(|S n R|, c_R) is the only
non-modular structure in the current suite, and it is a structure that gets
EXHAUSTED: once a region hits its cap, the marginal value of every further
cell in it is exactly zero.  The probe in the previous session showed that
caps bind only when coverage is locally dense, while coordination headroom
requires it to be globally sparse, so `gap` (coord% - dec%) and `cap%` are
anti-correlated and no configuration has both.

Facility location does not exhaust:

    F(S) = sum_{j in D} d_j * max_{v in S} u(v, j),      u(v,j) = kernel(dist)

Adding a facility near an already-served demand point still pays -- just
less.  The marginal decays smoothly instead of dropping to zero, which is
what "diminishing returns" is supposed to mean.  Two further properties
matter for MACS:

  1. Overlap is graded and acts AT A DISTANCE.  Two agents 3 cells apart
     share no covered cell, but their utility maps overlap heavily, so
     LOCAL / Shapley / DR disagree in exactly the states a good policy
     visits -- not only in the piled-up states a good policy avoids.
  2. F is strictly submodular but the induced per-step game decomposes
     over demand points, which yields a CLOSED-FORM exact Shapley in
     O(|D| k log k) -- see shapley_facility() below.  This is a strict
     generalization of Prop. `modular` (modular F is the special case where
     each demand point is reachable by exactly one candidate facility).

Ground set / dynamics are unchanged from MultiAgentCoverageEnv: S_t is the
cumulative set of visited cells, agents move on the 5-action grid, initial
footprints are covered without reward.  Only F changes.

Usage:
    python macs_fl.py --test     unit tests (numpy only, no torch)
"""

import sys
import numpy as np

import macs_main as M
from macs_main import MultiAgentCoverageEnv, shapley_exact, F_cardinality


# =============================================================================
# Kernel and objective
# =============================================================================

def _shift(mask, dx, dy):
    """mask shifted so that a facility at v serves demand at v + (dx,dy)."""
    out = np.zeros_like(mask)
    H, W = mask.shape
    xs0, xs1 = max(0, dx), min(H, H + dx)
    ys0, ys1 = max(0, dy), min(W, W + dy)
    if xs0 >= xs1 or ys0 >= ys1:
        return out
    out[xs0:xs1, ys0:ys1] = mask[xs0 - dx:xs1 - dx, ys0 - dy:ys1 - dy]
    return out


def linear_kernel(rho):
    """u(d) = 1 - d/rho for Euclidean d < rho, else 0. u(0) = 1."""
    offs = []
    r = int(np.ceil(rho))
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            d = np.hypot(dx, dy)
            if d < rho:
                offs.append((dx, dy, 1.0 - d / rho))
    return offs


class FacilityLocationF:
    """
    F(S) = sum_j d_j * max_{v in S} u(v, j),  F(empty) = 0.

    Monotone (adding facilities can only raise a max) and submodular (the
    marginal of a new facility at v is sum_j d_j * max(0, u(v,j) - best(j)),
    which is non-increasing in S).  NOT modular whenever two candidate
    facilities can serve a common demand point, i.e. always for rho > 1.

    demand : "uniform"  -> d_j = 1 for every cell,  F(V) = size^2
             "clusters" -> Gaussian blobs, rescaled so sum_j d_j = size^2
                           (so F(V) is the same number in both modes and the
                           two are directly comparable)
    """

    def __init__(self, size, rho=4.0, demand="uniform", n_clusters=6,
                 cluster_sigma=2.5, demand_seed=0):
        self.size = size
        self.rho = float(rho)
        self.offs = linear_kernel(rho)
        # group offsets by weight, descending
        ws = sorted({w for _, _, w in self.offs}, reverse=True)
        self.groups = [(w, [(dx, dy) for dx, dy, w2 in self.offs if w2 == w])
                       for w in ws]
        self.demand_mode = demand
        if demand == "uniform":
            self.d = np.ones((size, size), dtype=np.float64)
        elif demand == "clusters":
            rng = np.random.default_rng(demand_seed)
            xs, ys = np.meshgrid(np.arange(size), np.arange(size),
                                 indexing="ij")
            self.d = np.zeros((size, size), dtype=np.float64)
            margin = max(1, int(0.15 * size))
            for _ in range(n_clusters):
                cx, cy = rng.integers(margin, size - margin, size=2)
                self.d += np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2)
                                 / (2 * cluster_sigma ** 2))
            self.d += 0.05                       # floor, keeps F monotone-useful
            self.d *= (size * size) / self.d.sum()
        else:
            raise ValueError(demand)
        self._Fmax = float(self.d.sum())         # every cell is its own facility

    # -- objective ---------------------------------------------------------
    def utility_map(self, mask):
        """best(j) = max_{v in mask} u(v,j), as an (H,W) float array."""
        best = np.zeros((self.size, self.size), dtype=np.float64)
        unassigned = np.ones((self.size, self.size), dtype=bool)
        for w, offs in self.groups:              # descending w
            reach = np.zeros((self.size, self.size), dtype=bool)
            for dx, dy in offs:
                reach |= _shift(mask, dx, dy)
            newly = reach & unassigned
            if newly.any():
                best[newly] = w
                unassigned &= ~newly
            if not unassigned.any():
                break
        return best

    def __call__(self, mask):
        return float((self.d * self.utility_map(mask)).sum())

    @property
    def F_max(self):
        return self._Fmax


class FacilityCoverageEnv(MultiAgentCoverageEnv):
    """
    Same dynamics as MultiAgentCoverageEnv; F is facility location.

    patch=0 by default: an agent standing at v opens a facility at v and the
    kernel does the spreading, which is the faithful facility-location model
    and keeps the augmented state S_t = set of visited cells.
    """

    def __init__(self, size=16, horizon=60, k=2, patch=0, rho=4.0,
                 demand="uniform", n_clusters=6, cluster_sigma=2.5,
                 demand_seed=0, seed=None):
        self.Ffl = FacilityLocationF(size, rho=rho, demand=demand,
                                     n_clusters=n_clusters,
                                     cluster_sigma=cluster_sigma,
                                     demand_seed=demand_seed)
        super().__init__(size=size, horizon=horizon, k=k, patch=patch,
                         seed=seed)
        self.F = self.Ffl

    @property
    def total_coverage(self):
        return float(self.Ffl(self.covered))

    @property
    def max_cells(self):
        return self.Ffl.F_max

    def F_total(self):
        return self.Ffl.F_max


# =============================================================================
# Sparse-island demand (September 2026)
# =============================================================================
#
# probe_islands.py: on uniform and clustered demand, coord - dec ~ 0 because
# the shared S_t already coordinates a myopic greedy team, so no credit rule
# can win asymptotically. Islands change the structure:
#
#   * demand is zero between islands, so the one-step marginal is zero
#     there and a myopic policy stalls        -> planning headroom
#   * all agents spawn in one depot, so their nearest islands coincide and
#     someone has to yield                    -> assignment headroom
#   * the horizon is too short to tile the map, so a stateless sweep loses
#
# Canonical config: size=24, horizon=30, rho=4, 8 islands of radius 1,
# min_sep=8, eps=0, depot spawn. See PROBES_2026-09.md for the numbers.

def island_demand(size, n_islands, radius, min_sep, eps, rng):
    """Disjoint discs of demand 1, `eps` elsewhere, rescaled to sum size^2
    so F(V) is comparable with the uniform and clustered modes."""
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    best = []
    for _ in range(500):                     # restart until all fit
        centres, tries = [], 0
        while len(centres) < n_islands and tries < 200:
            tries += 1
            c = rng.integers(radius + 1, size - radius - 1, size=2)
            if all(np.hypot(*(c - o)) >= min_sep for o in centres):
                centres.append(c)
        if len(centres) > len(best):
            best = centres
        if len(best) == n_islands:
            break
    centres = best
    d = np.full((size, size), eps, dtype=np.float64)
    for cx, cy in centres:
        d[(xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2] = 1.0
    d *= (size * size) / d.sum()
    return d, np.array(centres)


class IslandF(FacilityLocationF):
    """Facility location with sparse island demand. Still monotone
    submodular; shapley_facility() applies unchanged."""

    def __init__(self, size, rho, n_islands, radius, min_sep, eps,
                 layout_seed):
        super().__init__(size, rho=rho, demand="uniform")
        self.demand_mode = "islands"
        rng = np.random.default_rng(layout_seed)
        self.d, self.centres = island_demand(size, n_islands, radius,
                                             min_sep, eps, rng)
        self._Fmax = float(self.d.sum())
        self.radius = radius


class IslandEnv(FacilityCoverageEnv):
    """
    Facility dynamics with island demand and a depot spawn.

    layout_seed  fixes the island layout (None -> a new layout every reset,
                 which a learner can only handle with a demand input channel;
                 use an int for the fixed-layout benchmark).
    depot        all k agents start inside a 3x3 block at a random location.
    """

    def __init__(self, size=24, horizon=30, k=2, rho=4.0, n_islands=8,
                 radius=1, min_sep=8, eps=0.0, layout_seed=0, depot=True,
                 seed=None):
        self.rho_hint = float(rho)
        self.depot = depot
        self.layout_seed = layout_seed
        self._isl = dict(rho=rho, n_islands=n_islands, radius=radius,
                         min_sep=min_sep, eps=eps)
        self._layout_rng = np.random.default_rng(seed)
        super().__init__(size=size, horizon=horizon, k=k, patch=0, rho=rho,
                         demand="uniform", seed=seed)

    def _new_layout(self):
        ls = (self.layout_seed if self.layout_seed is not None
              else int(self._layout_rng.integers(2 ** 31)))
        self.Ffl = IslandF(self.size, layout_seed=ls, **self._isl)
        self.F = self.Ffl

    def reset(self):
        self._new_layout()
        out = super().reset()
        if self.depot:
            c = self.rng.integers(2, self.size - 2, size=2)
            self.pos = np.clip(c + self.rng.integers(-1, 2, size=(self.k, 2)),
                               0, self.size - 1)
            self.covered[:] = False
            for i in range(self.k):
                self.covered |= self._patch_mask(self.pos[i])
            out = self._obs()
        return out


# =============================================================================
# Closed-form exact Shapley for facility location
# =============================================================================

def shapley_facility(S_prev, patches, Ffl):
    """
    Exact Shapley credits for the per-step game

        v(C) = F(S u U_{i in C} P(s^i)) - F(S)

    when F is facility location, computed in O(|D| k log k) with NO subset
    or permutation enumeration.

    Derivation.  Write a_ij = max_{v in P(s^i)} u(v,j) and b_j = best_S(j).
    Then v(C) = sum_j d_j [ max(b_j, max_{i in C} a_ij) - b_j ], i.e. a sum
    of independent per-demand MAX games, so by additivity (Shapley axiom iv)
    the credit is the sum of the per-demand Shapley values.

    Fix a demand point j and sort a_(1) >= ... >= a_(k).  Put
    c_l = max(a_(l), b) for l <= k and c_{k+1} = b.  Then

        v(C) = sum_{l=1}^{k} (c_l - c_{l+1}) * 1[ C n T_l != empty ],
        T_l = the top-l agents,

    a non-negative combination of OR-games.  The Shapley value of an OR-game
    on T_l with weight w is w/l for each member (null agent + symmetry +
    efficiency), so the agent of rank r receives

        phi_r = sum_{l=r}^{k} (c_l - c_{l+1}) / l.

    Reduces to Prop. `modular` when every demand point is reachable by
    exactly one candidate facility.
    """
    k = len(patches)
    A = np.stack([Ffl.utility_map(pm) for pm in patches])      # (k,H,W)
    B = Ffl.utility_map(S_prev)                                # (H,W)

    order = np.argsort(-A, axis=0, kind="stable")              # (k,H,W)
    As = np.take_along_axis(A, order, axis=0)                  # sorted desc
    C = np.maximum(As, B[None])                                # c_1..c_k
    C = np.concatenate([C, B[None]], axis=0)                   # append c_{k+1}
    w = C[:-1] - C[1:]                                         # (k,H,W) >= 0
    lvl = np.arange(1, k + 1, dtype=np.float64)[:, None, None]
    contrib = w / lvl
    # phi at rank r = suffix sum of contrib from r..k
    suffix = np.cumsum(contrib[::-1], axis=0)[::-1]            # (k,H,W)

    phi_sorted = suffix * Ffl.d[None]
    phi = np.zeros_like(phi_sorted)
    np.put_along_axis(phi, order, phi_sorted, axis=0)
    return phi.sum(axis=(1, 2))


def install():
    """Point macs_main.shapley_exact at the closed form when F is facility
    location, falling back to enumeration otherwise."""
    _orig = M.shapley_exact

    def _dispatch(S_prev, patches, F):
        if isinstance(F, FacilityLocationF):
            return shapley_facility(S_prev, patches, F)
        return _orig(S_prev, patches, F)

    M.shapley_exact = _dispatch
    return _dispatch


# =============================================================================
# Unit tests
# =============================================================================

def _rand_mask(size, p, rng):
    return rng.random((size, size)) < p


def _rand_patches(env, k, rng):
    pos = rng.integers(0, env.size, size=(k, 2))
    return [env._patch_mask(pos[i]) for i in range(k)]


def run_unit_tests(verbose=True):
    rng = np.random.default_rng(11)
    size = 12

    for demand in ("uniform", "clusters"):
        F = FacilityLocationF(size, rho=3.0, demand=demand, demand_seed=1)

        # 1. F(empty) = 0 and F(V) = sum of demand
        assert abs(F(np.zeros((size, size), bool))) < 1e-12
        assert abs(F(np.ones((size, size), bool)) - F.F_max) < 1e-9

        # 2. monotone
        for _ in range(40):
            A = _rand_mask(size, 0.15, rng)
            B = A | _rand_mask(size, 0.15, rng)
            assert F(B) >= F(A) - 1e-12, "monotonicity violated"

        # 3. submodular: marginal of v is non-increasing in S
        for _ in range(60):
            A = _rand_mask(size, 0.08, rng)
            B = A | _rand_mask(size, 0.20, rng)
            v = np.zeros((size, size), bool)
            p = rng.integers(0, size, size=2)
            v[p[0], p[1]] = True
            mA = F(A | v) - F(A)
            mB = F(B | v) - F(B)
            assert mA >= mB - 1e-9, "diminishing returns violated"

        # 4. NOT modular: exhibit a strict witness
        strict = False
        for _ in range(60):
            A = _rand_mask(size, 0.05, rng)
            u = np.zeros((size, size), bool); u[3, 3] = True
            w = np.zeros((size, size), bool); w[3, 4] = True
            if (F(A | u) - F(A)) + (F(A | w) - F(A)) - (F(A | u | w) - F(A)) > 1e-6:
                strict = True; break
        assert strict, "F should be strictly submodular"

    # 5. closed form == permutation enumeration, and Shapley axioms
    for demand in ("uniform", "clusters"):
        for rho in (2.0, 3.5):
            env = FacilityCoverageEnv(size=size, k=3, rho=rho, demand=demand,
                                      patch=0, seed=5)
            F = env.Ffl
            for _ in range(15):
                for k in (2, 3, 4):
                    S = _rand_mask(size, 0.10, rng)
                    P = _rand_patches(env, k, rng)
                    a = shapley_exact(S, P, F)
                    b = shapley_facility(S, P, F)
                    assert np.allclose(a, b, atol=1e-8), \
                        f"closed form mismatch {a} vs {b}"
                    # efficiency
                    U = S.copy()
                    for pm in P: U |= pm
                    r = F(U) - F(S)
                    assert abs(b.sum() - r) < 1e-8, "efficiency violated"
                    # non-negativity
                    assert (b >= -1e-9).all(), "negative credit"

    # 6. null agent and symmetry
    env = FacilityCoverageEnv(size=size, k=2, rho=3.0, patch=0, seed=1)
    F = env.Ffl
    p0 = np.zeros((size, size), bool); p0[6, 6] = True
    S = p0.copy()
    phi = shapley_facility(S, [p0.copy(), p0.copy()], F)
    assert np.allclose(phi, 0.0, atol=1e-9), "null agents should get 0"
    S2 = np.zeros((size, size), bool)
    phi = shapley_facility(S2, [p0.copy(), p0.copy()], F)
    assert abs(phi[0] - phi[1]) < 1e-9, "symmetry violated"

    # 7. ordering theorem DR <= Shapley <= LOCAL, per agent
    rng2 = np.random.default_rng(3)
    for demand in ("uniform", "clusters"):
        env = FacilityCoverageEnv(size=size, k=4, rho=3.5, demand=demand,
                                  patch=0, seed=2)
        F = env.Ffl
        for _ in range(40):
            k = 4
            S = _rand_mask(size, 0.10, rng2)
            P = _rand_patches(env, k, rng2)
            f0 = F(S)
            U = S.copy()
            for pm in P: U |= pm
            fa = F(U)
            loc = np.array([F(S | pm) - f0 for pm in P])
            dr = np.array([fa - F(np.logical_or.reduce(
                [S] + [P[j] for j in range(k) if j != i])) for i in range(k)])
            sh = shapley_facility(S, P, F)
            assert (dr <= sh + 1e-8).all(), "DR <= Shapley violated"
            assert (sh <= loc + 1e-8).all(), "Shapley <= LOCAL violated"

    # 8. env reward telescopes
    env = FacilityCoverageEnv(size=10, k=3, horizon=20, rho=3.0, patch=0,
                              seed=4)
    env.reset(); c0 = env.total_coverage; tot = 0.0; done = False
    while not done:
        a = rng.integers(0, 5, size=3)
        _, r, done, info = env.step(a)
        phi = shapley_facility(info["S_prev"], info["patches"], env.Ffl)
        assert abs(phi.sum() - r) < 1e-8
        tot += r
    assert abs((env.total_coverage - c0) - tot) < 1e-7, "telescoping failed"

    if verbose:
        print("macs_fl unit tests passed:")
        print("  F(empty)=0, F(V)=sum d, monotone, diminishing returns")
        print("  strictly submodular (non-modular witness found)")
        print("  closed-form Shapley == permutation enumeration (k=2,3,4)")
        print("  efficiency, non-negativity, null agent, symmetry")
        print("  ordering DR <= Shapley <= LOCAL holds per agent")
        print("  env reward telescopes to F(S_T) - F(S_0)")


if __name__ == "__main__":
    run_unit_tests()
    if "--test" in sys.argv:
        sys.exit(0)