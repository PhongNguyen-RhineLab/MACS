"""
Saturating regional coverage: a strictly submodular, non-modular objective.

    F(S) = sum_R w_R * min(|S n R|, c_R)

over a block partition of the grid. Monotone and submodular; strictly
submodular across cells within a region once |S n R| can exceed c_R (the
marginal value of a cell drops from w_R to 0 when its region saturates).

Example (region = 4, cap_frac = 0.5 on a 12x12 grid): 9 regions of 16
cells, each capped at 8. Covering the 9th cell of a region is worth 0, so
an agent filling an almost-full region earns less Shapley credit than one
opening a fresh region -- the diminishing-returns signal a modular F
cannot express. Because F is not modular, the modular closed form does not
apply and Shapley credits need an enumeration or MC backend.
"""

import numpy as np

from macs.envs.base import MultiAgentCoverageEnv


class SaturatingF:
    """F(S) = sum_R w_R * min(|S n R|, c_R) over a block partition."""

    def __init__(self, size, region=4, cap_frac=0.5, weights=None,
                 dtype=np.int64):
        assert size % region == 0, "region must divide the grid size"
        ids = np.arange(size // region)
        gx, gy = np.meshgrid(ids, ids, indexing="ij")
        blk = gx * (size // region) + gy                       # block index
        self.region_id = np.repeat(np.repeat(blk, region, 0), region, 1)
        self.n_regions = (size // region) ** 2
        self.caps = np.full(self.n_regions, int(region * region * cap_frac),
                            dtype=dtype)
        self.w = (np.ones(self.n_regions)
                  if weights is None else np.asarray(weights, dtype=float))
        self.size = size
        self.region = region

    def __call__(self, mask):
        counts = np.bincount(self.region_id[mask], minlength=self.n_regions)
        return float((self.w * np.minimum(counts, self.caps)).sum())

    @property
    def F_max(self):
        return float((self.w * self.caps).sum())


class SaturatingCoverageEnv(MultiAgentCoverageEnv):
    """
    Same dynamics as MultiAgentCoverageEnv; only the objective changes.

    total_coverage / max_cells report the OBJECTIVE VALUE F(S_t) / F(V)
    rather than raw cell counts, so logging and plotting normalize
    correctly without a special case.
    """

    def __init__(self, size=12, horizon=40, k=2, patch=1,
                 region=4, cap_frac=0.5, seed=None):
        self.Fsat = SaturatingF(size, region=region, cap_frac=cap_frac)
        super().__init__(size=size, horizon=horizon, k=k, patch=patch,
                         seed=seed)
        self.F = self.Fsat                      # replace cardinality

    @property
    def total_coverage(self):
        return float(self.Fsat(self.covered))

    @property
    def max_cells(self):
        return self.Fsat.F_max

    def F_total(self):
        return self.Fsat.F_max                  # = F(V), used for B(S)
