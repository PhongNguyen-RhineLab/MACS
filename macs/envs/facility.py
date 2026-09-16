"""
Facility location objective and environments, including the sparse-islands
credit-assignment benchmark.

    F(S) = sum_j d_j * max_{v in S} u(v, j),   F(empty) = 0

Monotone (adding facilities can only raise a max) and submodular (the
marginal of a new facility at v is sum_j d_j * max(0, u(v,j) - best(j)),
non-increasing in S). NOT modular whenever two candidate facilities can
serve a common demand point, i.e. always for rho > 1.

Exact Shapley credits for this F have a closed form -- see
macs.shapley.shapley_facility -- so `shapley.resolve` routes any
FacilityLocationF (IslandF included) to it automatically.
"""

import numpy as np

from macs.envs.base import MultiAgentCoverageEnv


# =============================================================================
# Kernel
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


# =============================================================================
# Objective
# =============================================================================

class FacilityLocationF:
    """
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
            self.d += 0.05                    # floor, keeps F monotone-useful
            self.d *= (size * size) / self.d.sum()
        else:
            raise ValueError(demand)
        self._Fmax = float(self.d.sum())      # every cell is its own facility

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
# Sparse-island demand
# =============================================================================
#
# On uniform and clustered demand, coord - dec ~ 0 because the shared S_t
# already coordinates a myopic greedy team, so no credit rule can win
# asymptotically. Islands change the structure:
#
#   * demand is zero between islands, so the one-step marginal is zero
#     there and a myopic policy stalls        -> planning headroom
#   * all agents spawn in one depot, so their nearest islands coincide and
#     someone has to yield                    -> assignment headroom
#   * the horizon is too short to tile the map, so a stateless sweep loses
#
# Canonical config lives in macs.config.ISLANDS.

def island_demand(size, n_islands, radius, min_sep, eps, rng):
    """
    Disjoint discs of demand 1, `eps` elsewhere, rescaled to sum size^2 so
    F(V) is comparable with the uniform and clustered modes.
    """
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
    """
    Facility location with sparse island demand. Still monotone submodular,
    and still a FacilityLocationF, so the closed-form Shapley backend
    applies unchanged.
    """

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
