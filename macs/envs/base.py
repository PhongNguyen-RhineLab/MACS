"""
Base multi-agent coverage environment and the modular objectives.

Index convention (unchanged from every earlier version):

  reset(): initial footprints are covered WITHOUT reward, S_0 = P(s_0).
  step t : agents move to s_{t+1},
           M = ( u_i P(s_{t+1}^i) ) minus S_t
           r = F(S_t u M) - F(S_t)
           S_{t+1} = S_t u M
  Shapley is computed over the per-step coverage game
           v(C) = F(S_t u  u_{i in C} P(s_{t+1}^i)) - F(S_t).

Subclasses override the objective (`self.F`) and the reporting properties
`total_coverage` / `max_cells` / `F_total`; the dynamics stay put.
"""

import numpy as np


def F_cardinality(mask):
    """F(S) = |S|. mask: boolean (H,W)."""
    return float(mask.sum())


def make_F_weighted(weights):
    """F(S) = sum of cell weights, still modular hence submodular."""
    def F(mask):
        return float((mask * weights).sum())
    return F


def compute_frontier(z_map):
    """
    Uncovered cells 4-adjacent to covered cells.

    Keys on == 1.0 / == 0.0 so that the two-room convention (walls at -1.0)
    is handled without a special case.
    """
    covered = (z_map == 1.0)
    uncovered = (z_map == 0.0)
    frontier = np.zeros_like(covered, dtype=bool)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        s = np.zeros_like(covered)
        if dx == -1:
            s[1:, :] = covered[:-1, :]
        elif dx == 1:
            s[:-1, :] = covered[1:, :]
        elif dy == -1:
            s[:, 1:] = covered[:, :-1]
        elif dy == 1:
            s[:, :-1] = covered[:, 1:]
        frontier |= s
    return (frontier & uncovered).astype(np.float32)


class MultiAgentCoverageEnv:
    """
    Open-grid coverage with k agents.

    step(joint_action) returns:
      (positions (k,2) normalized, z_map copy), r_team, done, info
    info carries what credit assignment needs:
      info["S_prev"]  boolean (H,W)  coverage before the step
      info["patches"] list of k boolean (H,W)  each agent's footprint AFTER
                      moving
    """

    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def __init__(self, size=12, horizon=40, k=2, patch=1, weights=None,
                 seed=None):
        self.size = size
        self.horizon = horizon
        self.k = k
        self.patch = patch
        self.n_actions = 5
        self.weights = weights            # None -> cardinality
        self.F = F_cardinality if weights is None else make_F_weighted(weights)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- helpers -----------------------------------------------------------
    def _patch_mask(self, pos):
        m = np.zeros((self.size, self.size), dtype=bool)
        p = self.patch
        x0, x1 = max(0, pos[0] - p), min(self.size, pos[0] + p + 1)
        y0, y1 = max(0, pos[1] - p), min(self.size, pos[1] + p + 1)
        m[x0:x1, y0:y1] = True
        return m

    def _obs(self):
        return (self.pos.astype(np.float32) / self.size,
                self.covered.astype(np.float32))

    def next_pos(self, i, a):
        """Position agent i would occupy after action a. No side effects."""
        dx, dy = self.ACTIONS[a]
        return (int(np.clip(self.pos[i, 0] + dx, 0, self.size - 1)),
                int(np.clip(self.pos[i, 1] + dy, 0, self.size - 1)))

    # -- API ---------------------------------------------------------------
    def reset(self):
        margin = self.patch + 1
        self.pos = self.rng.integers(margin, self.size - margin,
                                     size=(self.k, 2))
        self.covered = np.zeros((self.size, self.size), dtype=bool)
        for i in range(self.k):                      # S_0, no reward
            self.covered |= self._patch_mask(self.pos[i])
        self.steps = 0
        return self._obs()

    def step(self, joint_action):
        S_prev = self.covered.copy()
        for i, a in enumerate(joint_action):
            dx, dy = self.ACTIONS[a]
            self.pos[i, 0] = np.clip(self.pos[i, 0] + dx, 0, self.size - 1)
            self.pos[i, 1] = np.clip(self.pos[i, 1] + dy, 0, self.size - 1)
        patches = [self._patch_mask(self.pos[i]) for i in range(self.k)]
        f_prev = self.F(S_prev)
        for pm in patches:
            self.covered |= pm
        r_team = self.F(self.covered) - f_prev
        self.steps += 1
        done = self.steps >= self.horizon
        info = {"S_prev": S_prev, "patches": patches}
        return self._obs(), float(r_team), done, info

    # -- reporting ---------------------------------------------------------
    @property
    def total_coverage(self):
        return int(self.covered.sum())

    @property
    def max_cells(self):
        return self.size * self.size

    def F_total(self):
        """F(V): value of the fully covered ground set. Used for B(S)."""
        return self.F(np.ones((self.size, self.size), dtype=bool))


class TwoRoomCoverageEnv(MultiAgentCoverageEnv):
    """
    Vertical wall at column size//2 with a single door cell at the middle
    row. Differences from the open grid:

      - Moves into wall cells are blocked (position unchanged).
      - Footprints are intersected with the passable mask, so walls are
        never covered and F counts passable cells only.
      - The observed coverage map uses the UMaze convention:
        covered -> 1.0, uncovered passable -> 0.0, wall -> -1.0.

    Retained because the paper's earlier drafts used it; it is not part of
    the current benchmark suite.
    """

    def __init__(self, size=12, horizon=60, k=2, patch=1, seed=None):
        S = size
        self.passable = np.ones((S, S), dtype=bool)
        self.passable[:, S // 2] = False
        self.passable[S // 2, S // 2] = True          # the door
        super().__init__(size=size, horizon=horizon, k=k, patch=patch,
                         seed=seed)

    def _patch_mask(self, pos):
        return super()._patch_mask(pos) & self.passable

    def _obs(self):
        z = np.full((self.size, self.size), -1.0, dtype=np.float32)
        z[self.passable] = 0.0
        z[self.covered] = 1.0
        return self.pos.astype(np.float32) / self.size, z

    def reset(self):
        cands = np.argwhere(self.passable)
        idx = self.rng.integers(0, len(cands), size=self.k)
        self.pos = cands[idx].copy()
        self.covered = np.zeros((self.size, self.size), dtype=bool)
        for i in range(self.k):                       # S_0, no reward
            self.covered |= self._patch_mask(self.pos[i])
        self.steps = 0
        return self._obs()

    def step(self, joint_action):
        S_prev = self.covered.copy()
        for i, a in enumerate(joint_action):
            dx, dy = self.ACTIONS[a]
            nx = int(np.clip(self.pos[i, 0] + dx, 0, self.size - 1))
            ny = int(np.clip(self.pos[i, 1] + dy, 0, self.size - 1))
            if self.passable[nx, ny]:                 # walls block movement
                self.pos[i] = (nx, ny)
        patches = [self._patch_mask(self.pos[i]) for i in range(self.k)]
        f_prev = self.F(S_prev)
        for pm in patches:
            self.covered |= pm
        r_team = self.F(self.covered) - f_prev
        self.steps += 1
        done = self.steps >= self.horizon
        return self._obs(), float(r_team), done, \
            {"S_prev": S_prev, "patches": patches}

    @property
    def max_cells(self):
        return int(self.passable.sum())

    def F_total(self):
        return self.F(self.passable)                  # F over reachable cells
