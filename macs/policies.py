"""
Reference policies, and one convention for what a policy is.

Before this refactor a "policy" was either a bare function `pol(env, rng)`
or a factory `make(env) -> pol`, and callers told them apart with checks
like `mk is make_blind`, `mk in _PER_ENV`, `mk in _PER_ENV_LOCAL` or
`getattr(mk, "_per_env", False)` -- four conventions across five files, each
with its own copy of the list. Adding a stateful policy meant editing every
probe that might use it.

Here every policy is a factory: `make(env) -> callable(env, rng) -> actions`.
Stateless policies wrap through `stateless()`. `macs.rollout` calls the
factory once per episode, so per-episode state (a blind agent's private
coverage set, a sweep's waypoint index) has an obvious place to live and
nothing has to guess.

The policy bodies are unchanged from the probes they came from, including
tie-breaking, so every floor these produce reproduces the numbers already
reported.

Reference rows, weakest to strongest:

  random      what no policy at all achieves
  sweep       serpentine lawnmower within a per-agent stripe; no sensing
  blind       greedy against the agent's OWN visited set, no learning and
              no sharing. The floor that matters for O9: a learned private
              agent scoring below this is worse than not learning.
  dec         greedy against the SHARED set, no learning. The natural
              ceiling for a myopic shared-observation arm.
  dec_est     greedy against the shared set but an ESTIMATED demand field
  front       frontier-seeking via a distance transform (needs scipy)
  coord       one-step sequential greedy oracle (agent i conditions on
              1..i-1 already placed)
  plan_self   island planner where each agent targets its own nearest
              unfinished island, ignoring teammates
  plan        island planner with a shared claim list -- the assignment
              oracle
  plan+g      plan, but switching to a one-step greedy move once inside an
              island

plan - dec is the planning headroom and plan - plan_self the assignment
headroom; a credit rule can only be tested where both are positive.
"""

import copy

import numpy as np


# =============================================================================
# The protocol
# =============================================================================

def stateless(fn):
    """Turn `pol(env, rng)` into a factory, so every policy has one shape."""
    def make(env):
        return fn
    make.__name__ = getattr(fn, "__name__", "policy")
    make.__doc__ = fn.__doc__
    return make


def greedy_action(env, i, base, rng, F=None):
    """
    Action for agent i maximizing F(base | footprint) - F(base), ties broken
    uniformly at random. `base` is whatever set the agent is greedy against:
    the shared cover, its own private cover, or a running union.
    """
    F = F or env.F
    f0 = F(base)
    best, ties = -np.inf, []
    for a in range(env.n_actions):
        pm = env._patch_mask(env.next_pos(i, a))
        g = F(base | pm) - f0
        if g > best + 1e-12:
            best, ties = g, [a]
        elif abs(g - best) <= 1e-12:
            ties.append(a)
    return ties[rng.integers(len(ties))]


def step_toward(env, i, tgt):
    """
    Action minimizing L1 distance to tgt, first-best wins so the choice is
    deterministic and consumes no randomness. Action 4 (stay) when already
    there and no move improves.
    """
    best, ba = None, 4
    for a in range(env.n_actions):
        p = env.next_pos(i, a)
        d = abs(p[0] - tgt[0]) + abs(p[1] - tgt[1])
        if best is None or d < best:
            best, ba = d, a
    return ba


# =============================================================================
# Stateless policies
# =============================================================================

def _random(env, rng):
    return rng.integers(0, env.n_actions, size=env.k)


def _dec(env, rng):
    """Each agent maximizes its OWN marginal on the shared set, ignoring
    teammates."""
    S = env.covered
    f0 = env.F(S)
    acts = np.zeros(env.k, dtype=np.int64)
    for i in range(env.k):
        best, ties = -np.inf, []
        for a in range(env.n_actions):
            pm = env._patch_mask(env.next_pos(i, a))
            g = env.F(S | pm) - f0
            if g > best + 1e-12:
                best, ties = g, [a]
            elif abs(g - best) <= 1e-12:
                ties.append(a)
        acts[i] = ties[rng.integers(len(ties))]
    return acts


def _coord(env, rng):
    """Sequential greedy: agent i conditions on agents 1..i-1 already
    placed."""
    run = env.covered.copy()
    acts = np.zeros(env.k, dtype=np.int64)
    for i in range(env.k):
        f0 = env.F(run)
        best, ties = -np.inf, []
        for a in range(env.n_actions):
            pm = env._patch_mask(env.next_pos(i, a))
            g = env.F(run | pm) - f0
            if g > best + 1e-12:
                best, ties = g, [a]
            elif abs(g - best) <= 1e-12:
                ties.append(a)
        acts[i] = ties[rng.integers(len(ties))]
        run |= env._patch_mask(env.next_pos(i, acts[i]))
    return acts


def _front(env, rng):
    """
    Move to the cell furthest from anything already worked, by Euclidean
    distance transform. The "spread out without sensing demand" reference.
    """
    from scipy.ndimage import distance_transform_edt
    work = env.covered.copy()
    acts = np.zeros(env.k, dtype=np.int64)
    for i in range(env.k):
        D = distance_transform_edt(~work)
        best, ties = -np.inf, []
        for a in range(env.n_actions):
            p = env.next_pos(i, a)
            v = D[p[0], p[1]]
            if v > best + 1e-12:
                best, ties = v, [a]
            elif abs(v - best) <= 1e-12:
                ties.append(a)
        acts[i] = ties[rng.integers(len(ties))]
        work[env.next_pos(i, acts[i])] = True
    return acts


random_policy = stateless(_random)
dec = stateless(_dec)
coord = stateless(_coord)
front = stateless(_front)


# =============================================================================
# Stateful policies
# =============================================================================

def blind(env):
    """Per-agent private coverage set; no shared map, no learning."""
    own = [env._patch_mask(env.pos[i]).copy() for i in range(env.k)]

    def pol(env, rng):
        acts = np.zeros(env.k, dtype=np.int64)
        for i in range(env.k):
            f0 = env.F(own[i])
            best, ties = -np.inf, []
            for a in range(env.n_actions):
                pm = env._patch_mask(env.next_pos(i, a))
                g = env.F(own[i] | pm) - f0
                if g > best + 1e-12:
                    best, ties = g, [a]
                elif abs(g - best) <= 1e-12:
                    ties.append(a)
            acts[i] = ties[rng.integers(len(ties))]
        for i in range(env.k):
            own[i] |= env._patch_mask(env.next_pos(i, acts[i]))
        return acts
    return pol


def dec_est(env):
    """
    Greedy on the shared set against an ESTIMATED demand field: unsensed
    cells are assumed to carry unit demand. Isolates how much of `dec`'s
    advantage comes from knowing the true field rather than from sharing.
    Facility environments only -- it reads env.Ffl.
    """
    def pol(env, rng):
        Fhat = copy.copy(env.Ffl)
        Fhat.d = np.where(env.covered, env.Ffl.d, 1.0)
        S = env.covered
        f0 = Fhat(S)
        acts = np.zeros(env.k, dtype=np.int64)
        for i in range(env.k):
            best, ties = -np.inf, []
            for a in range(env.n_actions):
                pm = env._patch_mask(env.next_pos(i, a))
                g = Fhat(S | pm) - f0
                if g > best + 1e-12:
                    best, ties = g, [a]
                elif abs(g - best) <= 1e-12:
                    ties.append(a)
            acts[i] = ties[rng.integers(len(ties))]
        return acts
    return pol


def sweep(env):
    """
    Serpentine within a per-agent stripe; rows spaced by rho, so the team
    tiles the grid without sensing anything. Needs env.rho_hint, which the
    facility and island environments set.
    """
    n, k = env.size, env.k
    step = max(1, int(round(getattr(env, "rho_hint", 1.0))))
    wpts = []
    for i in range(k):
        x0 = int(round(i * n / k))
        x1 = int(round((i + 1) * n / k)) - 1
        cols = list(range(x0 + step // 2, x1 + 1, step)) or [x0]
        path, up = [], True
        for c in cols:
            ys = list(range(step // 2, n, step))
            path += [(c, y) for y in (ys if up else ys[::-1])]
            up = not up
        wpts.append(path)
    idx = [0] * k

    def pol(env, rng):
        acts = np.zeros(env.k, dtype=np.int64)
        for i in range(env.k):
            path = wpts[i]
            if idx[i] >= len(path):
                idx[i] = 0
            tx, ty = path[idx[i]]
            cx, cy = env.pos[i]
            if (cx, cy) == (tx, ty):
                idx[i] += 1
                tx, ty = path[min(idx[i], len(path) - 1)]
            acts[i] = step_toward(env, i, (tx, ty))
        return acts
    return pol


# =============================================================================
# Island planners
# =============================================================================

def island_done(env, ci):
    """
    Island finished when every island cell is served at weight >= 0.7
    (standing on the centre of a radius-1 island at rho=4 serves every
    island cell at 0.75, so this means someone has stood on it).
    """
    cx, cy = env.Ffl.centres[ci]
    xs, ys = np.meshgrid(np.arange(env.size), np.arange(env.size),
                         indexing="ij")
    cells = (xs - cx) ** 2 + (ys - cy) ** 2 <= env.Ffl.radius ** 2
    best = env.Ffl.utility_map(env.covered)
    return bool((best[cells] >= 0.7).all())


def island_planner(coord_claims=True, local_greedy=False):
    """
    Each agent heads for its nearest unfinished island, by L1 distance.

    coord_claims=True   agents write their target to a shared claim list, so
                        two agents never take the same island -- the
                        assignment oracle, `plan`
    coord_claims=False  each agent picks its own nearest island regardless of
                        teammates -- `plan_self`, which measures what is lost
                        with no assignment coordination
    local_greedy=True   once within radius+1 of the target, switch to a
                        one-step greedy move rather than walking to the
                        centre -- `plan+g`
    """
    def make(env):
        if not hasattr(getattr(env, "Ffl", None), "centres"):
            raise TypeError(
                "the island planners need an environment whose objective "
                "carries island centres (macs.envs.IslandEnv); got "
                f"{type(env).__name__}")

        def pol(env, rng):
            n_isl = len(env.Ffl.centres)
            done = np.array([island_done(env, c) for c in range(n_isl)])
            claimed = np.zeros(n_isl, dtype=bool)
            acts = np.full(env.k, 4, dtype=np.int64)
            for i in range(env.k):
                avail = ~done & (~claimed if coord_claims else True)
                if not avail.any():
                    avail = ~done if (~done).any() else np.ones(n_isl, bool)
                dists = np.abs(env.Ffl.centres - env.pos[i]).sum(1)
                dists = np.where(avail, dists, 10 ** 9)
                c = int(np.argmin(dists))
                claimed[c] = True
                tgt = env.Ffl.centres[c]
                near = np.hypot(*(env.pos[i] - tgt)) <= env.Ffl.radius + 1
                if local_greedy and near:
                    acts[i] = greedy_action(env, i, env.covered, rng)
                else:
                    acts[i] = step_toward(env, i, tgt)
            return acts
        return pol
    return make


plan = island_planner(coord_claims=True)
plan_self = island_planner(coord_claims=False)
plan_greedy = island_planner(coord_claims=True, local_greedy=True)


# =============================================================================
# Value-aware island planners (for unequal island weights, e.g. the decoy)
# =============================================================================
#
# The planners above rank islands by distance alone, which is right when
# every island carries the same demand and wrong when one carries more. These
# rank by discounted remaining value
#
#     score_i(c) = gamma^{L1(s^i, centre_c)} * [F(S u {centre_c}) - F(S)],
#
# the value agent i would collect by walking to island c and standing on it,
# if nobody else did. With equal weights and fresh islands this is the
# nearest-island rule again.
#
# Targets are COMMITTED: an agent keeps its island until that island is
# finished. Without commitment the planner dithers -- at gamma = 0.99 a 1%
# drop in an island's value, from a trail passing within kernel range,
# outweighs a step of distance and flips the argmax, and the team wanders
# (measured: 57.9% of F(V) against 88.8% for the distance planner on the
# unweighted benchmark). The distance planner commits implicitly, because
# nearest-island is stable as an agent approaches.

def island_values(env):
    """Remaining one-shot value of standing on each island centre."""
    S = env.covered
    f0 = env.F(S)
    vals = np.zeros(len(env.Ffl.centres))
    for c, (cx, cy) in enumerate(env.Ffl.centres):
        vals[c] = env.F(S | env._patch_mask((int(cx), int(cy)))) - f0
    return vals


def _value_target(env, i, vals, avail, gamma):
    d = np.abs(env.Ffl.centres - env.pos[i]).sum(1)
    score = np.where(avail, gamma ** d * vals, -np.inf)
    return int(np.argmax(score))


def value_planner(coord_claims=True, gamma=0.99, assigned=None):
    """
    coord_claims=True   shared claim list: `plan_v`, the assignment oracle
    coord_claims=False  every agent takes its own best island: `plan_self_v`
    assigned            optional list, agent i -> island index (or None).
                        An assigned agent walks to its island until that
                        island is finished, then falls back to the planner.
                        Scripted strategies for the equilibrium probe
                        (probes/decoy.py); None everywhere is plain planning.
    """
    def make(env):
        if not hasattr(getattr(env, "Ffl", None), "centres"):
            raise TypeError("value planners need an IslandEnv")
        # live[i]: agent i's committed island, or None. Scripted
        # assignments are just pre-seeded commitments.
        live = list(assigned) if assigned is not None else [None] * env.k

        def pol(env, rng):
            n_isl = len(env.Ffl.centres)
            done = np.array([island_done(env, c) for c in range(n_isl)])
            for i in range(env.k):
                if live[i] is not None and done[live[i]]:
                    live[i] = None                     # job done, re-plan
            claimed = np.zeros(n_isl, dtype=bool)
            if coord_claims:
                for c in live:
                    if c is not None:
                        claimed[c] = True
            vals = None
            acts = np.full(env.k, 4, dtype=np.int64)
            for i in range(env.k):
                if live[i] is None:
                    if vals is None:
                        vals = island_values(env)
                    avail = ~done & (~claimed if coord_claims else True)
                    if not avail.any():
                        avail = ~done if (~done).any() else np.ones(n_isl,
                                                                     bool)
                    live[i] = _value_target(env, i, vals, avail, gamma)
                    if coord_claims:
                        claimed[live[i]] = True
                acts[i] = step_toward(env, i, env.Ffl.centres[live[i]])
            return acts
        return pol
    return make


plan_v = value_planner(coord_claims=True)
plan_self_v = value_planner(coord_claims=False)


# =============================================================================
# Named sets
# =============================================================================

#: floors reported alongside learned scores on the facility objective
FACILITY_FLOORS = (("random", random_policy), ("dec", dec),
                   ("coord", coord), ("blind", blind))

#: floors reported alongside learned scores on the islands benchmark
ISLAND_FLOORS = (("random", random_policy), ("sweep", sweep), ("dec", dec),
                 ("plan_self", plan_self), ("plan", plan))

#: the full ladder used by the policy-level probes
LADDER = (("random", random_policy), ("blind", blind), ("dec_est", dec_est),
          ("dec", dec), ("coord", coord))

#: floors for weighted islands (the decoy variant): the value-aware planners
#: replace the distance-only ones, which ignore the decoy's weight
DECOY_FLOORS = (("random", random_policy), ("sweep", sweep), ("dec", dec),
                ("plan_self_v", plan_self_v), ("plan_v", plan_v))

#: every island policy, including the ones that are not reported as floors
ISLAND_LADDER = (("random", random_policy), ("sweep", sweep), ("dec", dec),
                 ("coord", coord), ("plan_self", plan_self), ("plan", plan),
                 ("plan+g", plan_greedy), ("plan_self_v", plan_self_v),
                 ("plan_v", plan_v))

BY_NAME = dict((*LADDER, *ISLAND_LADDER, ("front", front)))
