"""
Budgets that bound a per-agent credit stream. Used as bootstrap ceilings.

Two bounds, both valid for EVERY joint policy, so both leave the fixed point
of the per-agent backup unchanged (the clip is inactive at every policy value
function):

team budget      B(S) = F(V) - F(S)
                 The whole team's remaining value. Ignores the horizon and
                 ignores that agent i can only reach part of the map.

reach budget     B_i(s^i, S, tau) = sum_{j=1}^{tau} gamma^{j-1} [H(j) - H(j-1)]
                 H(j) = F(S u R_i(j)) - F(S)
                 R_i(j) = union of footprints of every cell agent i can reach
                          in at most j steps from s^i

Why the reach budget bounds agent i's discounted Shapley stream
(Lemma `reach-budget`):

  1. phi_{i,u} <= phi^LOC_{i,u} = F(S_{u-1} u c_u) - F(S_{u-1})    (Prop. 1)
  2. S_{u-1} contains S u C_{<u}, agent i's own earlier covers, so by
     submodularity phi^LOC_{i,u} <= F(S u C_{<u} u c_u) - F(S u C_{<u})
  3. summing telescopes: the undiscounted partial sum up to step t+j is at
     most F(S u C_{<=t+j}) - F(S) <= H(j), since C_{<=t+j} is inside R_i(j)
  4. Abel summation with the non-increasing weights gamma^{j-1} turns the
     partial-sum bound into the discounted bound above

The bound holds pathwise whatever the teammates do, depends only on
(s^i, S, tau) -- the decentralized signature plus time-to-go -- and satisfies
B_i <= H(tau) <= B(S), so it is never looser than the team budget.
"""

from collections import deque

import numpy as np


def passable_mask(env):
    """Cells an agent can stand on. Open grids: everything."""
    p = getattr(env, "passable", None)
    if p is None:
        return np.ones((env.size, env.size), dtype=bool)
    return p


def step_distance(env, pos):
    """
    BFS step distance from `pos` to every cell under the environment's move
    set (four moves plus stay, walls block). Unreachable cells get a large
    sentinel. Matches MultiAgentCoverageEnv / TwoRoomCoverageEnv dynamics:
    clipping at the border only produces a stay, which BFS already allows.
    """
    size = env.size
    if getattr(env, "passable", None) is None:
        # open grid: BFS distance is the Manhattan distance
        xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
        return (np.abs(xs - int(pos[0])) + np.abs(ys - int(pos[1]))
                ).astype(np.int32)
    passable = passable_mask(env)
    INF = np.iinfo(np.int32).max // 2
    dist = np.full((size, size), INF, dtype=np.int32)
    x0, y0 = int(pos[0]), int(pos[1])
    dist[x0, y0] = 0
    q = deque([(x0, y0)])
    while q:
        x, y = q.popleft()
        d = dist[x, y] + 1
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if (0 <= nx < size and 0 <= ny < size and passable[nx, ny]
                    and dist[nx, ny] > d):
                dist[nx, ny] = d
                q.append((nx, ny))
    return dist


def dilate(mask, p):
    """Square dilation by radius p, clipped at the border. p=0 is identity.
    Equals the union of env._patch_mask over the cells of `mask` on an open
    grid; intersect with passable_mask for walled maps."""
    if p == 0:
        return mask.copy()
    H, W = mask.shape
    out = np.zeros_like(mask)
    for dx in range(-p, p + 1):
        for dy in range(-p, p + 1):
            xs0, xs1 = max(0, dx), min(H, H + dx)
            ys0, ys1 = max(0, dy), min(W, W + dy)
            if xs0 < xs1 and ys0 < ys1:
                out[xs0:xs1, ys0:ys1] |= mask[xs0 - dx:xs1 - dx,
                                              ys0 - dy:ys1 - dy]
    return out


def reach_cover(env, dist, j):
    """R_i(j): union of the footprints of every cell within j steps."""
    return dilate(dist <= j, env.patch) & passable_mask(env)


# -- fast path: facility location with single-cell footprints ----------------
#
# F(S u R) = sum_c d_c max(best_S(c), best_R(c)), and for R = {v: dist(v) <= j}
# best_R(c) = max over kernel groups w of  w * [D_w(c) <= j],
# D_w(c) = min step distance to any facility position that serves c at
# weight w. D_w is one shifted-min per kernel offset, computed once, after
# which H(1..tau) is a vectorized pass over j. Same numbers as the generic
# path (tests/test_budget.py checks this), roughly an order of magnitude
# cheaper at k=6 on the islands grid.

def _facility_fast_ok(env):
    from macs.envs.facility import FacilityLocationF
    return (isinstance(env.F, FacilityLocationF) and env.patch == 0
            and getattr(env, "passable", None) is None)


def _shift_min(dist, dx, dy, fill):
    """out[c] = dist[c - (dx,dy)], `fill` where that falls off the grid."""
    out = np.full_like(dist, fill)
    H, W = dist.shape
    xs0, xs1 = max(0, dx), min(H, H + dx)
    ys0, ys1 = max(0, dy), min(W, W + dy)
    if xs0 < xs1 and ys0 < ys1:
        out[xs0:xs1, ys0:ys1] = dist[xs0 - dx:xs1 - dx, ys0 - dy:ys1 - dy]
    return out


def _facility_H(env, S, dist, tau, f_S):
    Ffl = env.F
    INF = np.iinfo(np.int32).max // 2
    ws, Dg = [], []
    for w, offs in Ffl.groups:
        D = np.full_like(dist, INF)
        for dx, dy in offs:
            np.minimum(D, _shift_min(dist, dx, dy, INF), out=D)
        ws.append(w)
        Dg.append(D)
    ws = np.asarray(ws)
    Dg = np.stack(Dg)                                 # (G, H, W)
    best_S = Ffl.utility_map(S)
    Hs = np.zeros(tau + 1)
    for j in range(1, tau + 1):
        ok = Dg <= j
        best_R = np.where(ok.any(0), ws[ok.argmax(0)], 0.0)  # groups desc
        Hs[j] = float((Ffl.d * np.maximum(best_S, best_R)).sum()) - f_S
    return Hs


def reach_budget(env, S, pos, tau, gamma, discounted=True, f_S=None,
                 team_budget=None):
    """
    B_i(pos, S, tau). `discounted=False` returns H(tau), one F evaluation,
    looser by at most a factor gamma^{-(tau-1)}.

    Early exits keep the cost at a handful of F calls in practice:
      * once R_i(j) stops growing (the reachable region is exhausted) every
        later increment is zero
      * once H(j) reaches the team budget F(V) - F(S) nothing is left to add
    """
    if tau <= 0:
        return 0.0
    f_S = env.F(S) if f_S is None else f_S
    dist = step_distance(env, pos)
    if _facility_fast_ok(env):
        Hs = _facility_H(env, S, dist, tau, f_S)
        if not discounted:
            return float(Hs[tau])
        return float(sum(gamma ** (j - 1) * (Hs[j] - Hs[j - 1])
                         for j in range(1, tau + 1)))
    if not discounted:
        return float(env.F(S | reach_cover(env, dist, tau)) - f_S)
    if team_budget is None:
        team_budget = env.F_total() - f_S
    total, H_prev, R_prev = 0.0, 0.0, None
    max_d = int(dist[dist < np.iinfo(np.int32).max // 2].max())
    for j in range(1, tau + 1):
        if j > max_d + 1:
            break                                  # region exhausted
        R = reach_cover(env, dist, j)
        if R_prev is not None and np.array_equal(R, R_prev):
            continue
        H = float(env.F(S | R) - f_S)
        total += gamma ** (j - 1) * (H - H_prev)
        H_prev, R_prev = H, R
        if H >= team_budget - 1e-9:
            break                                  # nothing left anywhere
    return total


def reach_budgets(env, gamma, discounted=True):
    """
    B_i at the environment's CURRENT state for every agent, with
    tau = horizon - steps. Call right after env.step() to get the ceiling for
    the bootstrap at the successor state.
    """
    tau = env.horizon - env.steps
    out = np.zeros(env.k, dtype=np.float32)
    if tau <= 0:
        return out
    S = env.covered
    f_S = env.F(S)
    tb = env.F_total() - f_S
    for i in range(env.k):
        out[i] = reach_budget(env, S, env.pos[i], tau, gamma,
                              discounted=discounted, f_S=f_S,
                              team_budget=tb)
    return out
