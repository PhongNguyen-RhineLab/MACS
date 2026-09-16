"""
Shapley value of the per-step coverage game.

For a step with coverage set S before the move and arrival footprints
P_1..P_k, the game is

    v(C) = F(S u U_{i in C} P_i) - F(S),      v(empty) = 0.

Four backends compute the same quantity (up to Monte Carlo error), at
different cost:

    permutation   k! evaluations       exact, the literal Alg. 1 form
    subset        2^k evaluations      exact, shared across agents
    facility      O(|D| k log k)       exact, closed form for facility
                                       location F (Prop. `facility`)
    mc            m*k evaluations      unbiased, sum_i phi_i = r_t exactly

`resolve` picks one from the objective and the team size; see credits.py
for how it is wired into training. Nothing in this module mutates global
state -- the previous `install()` monkey-patches are gone.
"""

from itertools import permutations

import numpy as np

MC_M = 64          # default sampled permutations for the "mc" backend
EXACT_K = 4        # "auto" uses an exact enumeration backend up to this k


# =============================================================================
# Enumeration backends
# =============================================================================

def shapley_permutation(S_prev, patches, F):
    """
    Exact Shapley via all k! permutations (Alg. 1, mode=EXACT).

    S_prev : boolean (H,W) coverage before the step
    patches: list of k boolean (H,W) masks, P(s^i) for each agent
    F      : set function on boolean masks

    Returns phi (k,) with sum_i phi_i = F(S_prev u all patches) - F(S_prev).
    Cost k! * k evaluations; use `shapley_subset` beyond k = 4.
    """
    k = len(patches)
    phi = np.zeros(k)
    perms = list(permutations(range(k)))
    for sigma in perms:
        run = S_prev.copy()
        f_run = F(run)
        for j in sigma:
            run = run | patches[j]
            f_new = F(run)
            phi[j] += f_new - f_run
            f_run = f_new
    return phi / len(perms)


def shapley_subset(S_prev, patches, F):
    """
    Exact Shapley via the subset form

        phi_i = sum_{C subset N\\{i}} |C|!(k-|C|-1)!/k! [v(C u i) - v(C)].

    Every coalition value v(C) is computed once and SHARED across the k
    agents, so the cost is 2^k evaluations of F, not k * 2^(k-1): the
    memoised table is the same table for every agent. That makes this the
    cheapest exact backend for 5 <= k <= ~20, and at k = 6 it is roughly
    6.6x cheaper than MC with m = 64.

    Agrees with `shapley_permutation` to floating point.
    """
    k = len(patches)
    n_sub = 1 << k

    # v(C) for every coalition, built by Gray-code-free incremental union:
    # mask m is built from the union of its members. Each union is a cheap
    # boolean OR; the expensive part is the F call, done once per subset.
    unions = [None] * n_sub
    unions[0] = S_prev
    vals = np.zeros(n_sub)
    f0 = F(S_prev)
    for m in range(1, n_sub):
        low = m & (-m)                     # lowest set bit
        i = low.bit_length() - 1
        unions[m] = unions[m ^ low] | patches[i]
        vals[m] = F(unions[m]) - f0

    # Shapley weights by coalition size: |C|! (k-|C|-1)! / k!
    fact = np.array([float(_factorial(j)) for j in range(k + 1)])
    weight = np.array([fact[c] * fact[k - c - 1] / fact[k]
                       for c in range(k)])
    popcount = np.array([bin(m).count("1") for m in range(n_sub)])

    phi = np.zeros(k)
    for i in range(k):
        bit = 1 << i
        acc = 0.0
        for m in range(n_sub):
            if m & bit:
                continue
            acc += weight[popcount[m]] * (vals[m | bit] - vals[m])
        phi[i] = acc
    return phi


def _factorial(n):
    out = 1
    for j in range(2, n + 1):
        out *= j
    return out


def shapley_mc(S_prev, patches, F, m, rng):
    """
    Monte Carlo Shapley (Alg. 1, mode=MC), m sampled permutations.

    Marginals telescope within each permutation, so sum_i phi_i = r_t
    holds EXACTLY at any sample count; only the split between agents is
    approximate.

    Raises on m <= 0 rather than returning NaN, which is what the previous
    default of mc_m=0 did silently whenever a caller forgot to pass it.
    """
    if m is None or m <= 0:
        raise ValueError(
            f"shapley_mc needs m >= 1 sampled permutations, got {m!r}. "
            "Pass mc_m explicitly or use an exact backend.")
    k = len(patches)
    phi = np.zeros(k)
    for _ in range(m):
        sigma = rng.permutation(k)
        run = S_prev.copy()
        f_run = F(run)
        for j in sigma:
            run = run | patches[j]
            f_new = F(run)
            phi[j] += f_new - f_run
            f_run = f_new
    return phi / m


def shapley_modular_closed_form(S_prev, patches, weights=None):
    """
    Closed form valid only when F is MODULAR (cardinality or weighted
    coverage): each newly covered cell's weight is split equally among the
    agents whose footprint covers it. Kept as an independent check against
    the enumeration backends, not used in training.
    """
    stack = np.stack(patches)                    # (k,H,W)
    new = stack & ~S_prev[None]                  # new cells per agent
    counts = new.sum(0).astype(np.float64)       # multiplicity per cell
    w = np.ones_like(counts) if weights is None else weights.astype(np.float64)
    share = np.divide(w, counts, out=np.zeros_like(w), where=counts > 0)
    return (new * share[None]).sum(axis=(1, 2))


# =============================================================================
# Facility-location closed form
# =============================================================================

def shapley_facility(S_prev, patches, Ffl):
    """
    Exact Shapley credits when F is facility location, in O(|D| k log k)
    with NO subset or permutation enumeration.

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

    Reduces to the modular closed form when every demand point is reachable
    by exactly one candidate facility.
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
    suffix = np.cumsum(contrib[::-1], axis=0)[::-1]            # rank-r suffix

    phi_sorted = suffix * Ffl.d[None]
    phi = np.zeros_like(phi_sorted)
    np.put_along_axis(phi, order, phi_sorted, axis=0)
    return phi.sum(axis=(1, 2))


# =============================================================================
# Cost accounting
# =============================================================================

def count_evals(backend_fn, S_prev, patches, F):
    """Return (phi, n_distinct_F_evaluations) for cost reporting."""
    class _Counter:
        def __init__(self, f):
            self.f, self.n = f, 0

        def __call__(self, m):
            self.n += 1
            return self.f(m)

        def __getattr__(self, name):          # proxy utility_map, d, ...
            return getattr(self.f, name)

    c = _Counter(F)
    return backend_fn(S_prev, patches, c), c.n


# =============================================================================
# Backend resolution
# =============================================================================

BACKENDS = ("auto", "facility", "permutation", "subset", "mc")


def resolve(F, k, backend="auto"):
    """
    Return the name of the backend to use for objective `F` at team size `k`.

    "auto" reproduces what the pre-refactor code did once every install()
    hook had been applied in the right order:

        facility-location F (incl. islands)  -> "facility"   exact, cheap
        k <= EXACT_K                         -> "permutation" exact
        otherwise                            -> "mc"          approximate

    The important change is that this is now a pure function of (F, k) and
    the caller's choice, rather than of module import order. "subset" is
    the exact backend to ask for at k > 4 when the objective is not
    facility location.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown Shapley backend {backend!r}; "
                         f"expected one of {BACKENDS}")
    if backend != "auto":
        return backend
    # Imported here to keep this module free of environment dependencies.
    from macs.envs.facility import FacilityLocationF
    if isinstance(F, FacilityLocationF):
        return "facility"
    return "permutation" if k <= EXACT_K else "mc"


def compute(S_prev, patches, F, *, backend="auto", mc_m=MC_M, rng=None):
    """Compute Shapley credits with the resolved backend."""
    k = len(patches)
    name = resolve(F, k, backend)
    if name == "facility":
        return shapley_facility(S_prev, patches, F)
    if name == "permutation":
        return shapley_permutation(S_prev, patches, F)
    if name == "subset":
        return shapley_subset(S_prev, patches, F)
    if name == "mc":
        if rng is None:
            raise ValueError("the 'mc' backend needs an rng")
        return shapley_mc(S_prev, patches, F, mc_m, rng)
    raise ValueError(name)
