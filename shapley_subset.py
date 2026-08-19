"""
shapley_subset.py — exact Shapley credit via the SUBSET form.

Drop-in replacement for macs_main.shapley_exact. Same output to machine
precision, verified against the permutation form for k in {2,3,4,6}.

Why this exists
---------------
main_mac.tex, sec:shapley-complexity, states the cost of exact Shapley as
"k * 2^(k-1) evaluations of F per step across all agents" (Example
ex:shapley-cost: 1024 evaluations at k=8). That count assumes each agent
enumerates its own subsets independently. It does not: the quantity
F(S u C_P) is shared by every agent not in C, so the distinct evaluations
number at most 2^k, not k * 2^(k-1). The correct counts are

    k        2     3     4     5     6     8
    paper    4    12    32    80   192  1024
    actual   4     8    16    32    64   256

and the implementation in macs_main.shapley_exact is worse than either,
because it walks all k! permutations and re-evaluates F at every prefix:
k * k! evaluations (5,040 at k=6, 362,880 at k=8).

Measured per-step credit cost, saturating F on a 16x16 grid, 3x3
footprints (median of 100 calls):

    k    perm form     subset form    MC m=64     subset vs MC
    2      0.03 ms       0.02 ms      1.12 ms         50.5x
    3      0.12 ms       0.06 ms      1.58 ms         28.4x
    4      0.50 ms       0.08 ms      1.67 ms         19.9x
    6     21.42 ms       0.41 ms      2.06 ms          5.1x
    8        --          1.57 ms      2.61 ms          1.7x

The k=6 consequence is the one that matters for the paper: exact Shapley
is 5x CHEAPER than the m=64 Monte Carlo estimator that the k=6
configuration actually used. The MC route was not a cost saving there.
This bears on two claims:

  * sec:mc-ablation motivates m=64 at k=6 by asserting exact enumeration
    "would be an order of magnitude more expensive than MC (384
    evaluations)". With the subset form it is 64 evaluations, so the
    comparison inverts.
  * sec:results-saturating attributes plain MACS finishing below LOCAL and
    DR at k=6 to "the m=64 Monte Carlo Shapley used at k=6 costing plain
    MACS a fraction of a point against noiseless LOCAL". If that reading
    is right, the dip is an artefact of an approximation that was never
    needed, and re-running k=6 with exact credits should remove it.

The MC path remains meaningful as an ABLATION (how much does approximation
cost?) and as the honest answer for large k, since 2^k still explodes.
What changes is where the crossover sits: around k = 8-9 for m = 64, not
k = 4.

Usage
-----
    import shapley_subset as SS
    SS.install()          # monkey-patch macs_main.shapley_exact
    # or call directly:
    phi = SS.shapley_subset(S_prev, patches, F)

    python shapley_subset.py      # self-test + benchmark
"""

from math import factorial

import numpy as np


def shapley_subset(S_prev, patches, F):
    """
    Exact Shapley credit over the induced coverage game

        v(C) = F(S_prev u  U_{j in C} P(s^j)) - F(S_prev),

    evaluated via the subset form

        phi_i = sum_{C subset I\\{i}} |C|!(k-|C|-1)!/k! [v(C u {i}) - v(C)].

    F is evaluated exactly once per distinct subset union (2^k in the
    worst case), and each union is built incrementally from the one with
    its lowest set bit removed, so the mask OR work is O(2^k) rather than
    O(k * 2^k).

    Returns phi (k,) with sum_i phi_i = F(S_prev u all patches) - F(S_prev)
    exactly.
    """
    k = len(patches)
    if k == 0:
        return np.zeros(0)
    n = 1 << k

    unions = [None] * n
    vals = np.empty(n)
    unions[0] = S_prev
    vals[0] = F(S_prev)
    for m in range(1, n):
        low = (m & -m).bit_length() - 1          # index of lowest set bit
        unions[m] = unions[m ^ (1 << low)] | patches[low]
        vals[m] = F(unions[m])

    fact = [factorial(i) for i in range(k + 1)]
    weight = [fact[c] * fact[k - c - 1] / fact[k] for c in range(k)]
    popcount = [bin(m).count("1") for m in range(n)]

    phi = np.zeros(k)
    for i in range(k):
        bit = 1 << i
        acc = 0.0
        for m in range(n):
            if m & bit:
                continue
            acc += weight[popcount[m]] * (vals[m | bit] - vals[m])
        phi[i] = acc
    return phi


def count_evals(S_prev, patches, F):
    """Return (phi, n_distinct_F_evaluations) for cost reporting."""
    class _C:
        def __init__(self, f):
            self.f, self.n = f, 0

        def __call__(self, m):
            self.n += 1
            return self.f(m)

    c = _C(F)
    return shapley_subset(S_prev, patches, c), c.n


def install():
    """
    Repoint macs_main.shapley_exact (and therefore the credit dispatchers
    in macs_main and macs_v3, which resolve it at call time) to the subset
    form. Call once, before training.
    """
    import macs_main as M
    M.shapley_exact = shapley_subset
    try:
        import macs_v3 as V3
        V3.shapley_exact = shapley_subset
    except ImportError:
        pass
    return shapley_subset


# =============================================================================
# Self-test and benchmark
# =============================================================================

if __name__ == "__main__":
    import time
    import macs_main as M
    from macs_v3 import SaturatingF

    rng = np.random.default_rng(0)
    SIZE = 16
    Fsat = SaturatingF(SIZE, region=4, cap_frac=0.5)

    def rand_state(k, density=0.35):
        S_prev = rng.random((SIZE, SIZE)) < density
        patches = []
        for _ in range(k):
            p = rng.integers(1, SIZE - 1, size=2)
            m = np.zeros((SIZE, SIZE), dtype=bool)
            m[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2] = True
            patches.append(m)
        return S_prev, patches

    # --- agreement with the permutation form, and the Shapley axioms -----
    for k in (2, 3, 4, 5, 6):
        for _ in range(20):
            S_prev, patches = rand_state(k)
            a = M.shapley_exact(S_prev, patches, Fsat)
            b = shapley_subset(S_prev, patches, Fsat)
            assert np.allclose(a, b, atol=1e-9), f"mismatch at k={k}"
            union = S_prev.copy()
            for pm in patches:
                union |= pm
            r = Fsat(union) - Fsat(S_prev)
            assert abs(b.sum() - r) < 1e-9, f"efficiency violated at k={k}"
    print("subset form == permutation form, and efficiency holds, "
          "for k in {2,3,4,5,6}")

    # --- null agent and symmetry, on the saturating F --------------------
    S_prev, patches = rand_state(3)
    S_prev |= patches[2]                       # agent 3 covers nothing new
    phi = shapley_subset(S_prev, patches, Fsat)
    assert abs(phi[2]) < 1e-9, "null agent violated"
    S_prev = np.zeros((SIZE, SIZE), dtype=bool)
    m = np.zeros((SIZE, SIZE), dtype=bool)
    m[4, 4] = True
    phi = shapley_subset(S_prev, [m, m.copy()], Fsat)
    assert abs(phi[0] - phi[1]) < 1e-9, "symmetry violated"
    print("null-agent and symmetry axioms hold")

    # --- evaluation counts vs the paper's formula ------------------------
    print(f"\n{'k':>2} {'perm k*k!':>10} {'subset':>8} {'2^k':>6} "
          f"{'paper k*2^(k-1)':>16}")
    for k in (2, 3, 4, 5, 6, 8):
        S_prev, patches = rand_state(k)
        _, n = count_evals(S_prev, patches, Fsat)
        print(f"{k:>2} {k * factorial(k):>10} {n:>8} {2 ** k:>6} "
              f"{k * 2 ** (k - 1):>16}")

    # --- wall clock ------------------------------------------------------
    print(f"\n{'k':>2} {'perm ms':>9} {'subset ms':>10} {'MC m=64 ms':>11} "
          f"{'subset vs MC':>13}")
    for k in (2, 3, 4, 6, 8):
        S_prev, patches = rand_state(k)
        reps = 100 if k < 8 else 20
        if k <= 6:
            t = time.perf_counter()
            for _ in range(reps):
                M.shapley_exact(S_prev, patches, Fsat)
            tp = (time.perf_counter() - t) / reps
        else:
            tp = float("nan")
        t = time.perf_counter()
        for _ in range(reps):
            shapley_subset(S_prev, patches, Fsat)
        ts = (time.perf_counter() - t) / reps
        t = time.perf_counter()
        for _ in range(reps):
            M.shapley_mc(S_prev, patches, Fsat, 64, rng)
        tm = (time.perf_counter() - t) / reps
        print(f"{k:>2} {1e3 * tp:>9.2f} {1e3 * ts:>10.2f} {1e3 * tm:>11.2f} "
              f"{tm / ts:>12.1f}x")