"""Environments. All share the dynamics of MultiAgentCoverageEnv and differ
only in the objective F and the spawn rule."""

from macs.envs.base import (
    MultiAgentCoverageEnv,
    TwoRoomCoverageEnv,
    F_cardinality,
    make_F_weighted,
    compute_frontier,
)
from macs.envs.saturating import SaturatingF, SaturatingCoverageEnv
from macs.envs.facility import (
    FacilityLocationF,
    FacilityCoverageEnv,
    IslandF,
    IslandEnv,
    island_demand,
    linear_kernel,
)

__all__ = [
    "MultiAgentCoverageEnv", "TwoRoomCoverageEnv",
    "F_cardinality", "make_F_weighted", "compute_frontier",
    "SaturatingF", "SaturatingCoverageEnv",
    "FacilityLocationF", "FacilityCoverageEnv",
    "IslandF", "IslandEnv", "island_demand", "linear_kernel",
]
