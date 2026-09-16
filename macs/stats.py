"""
The two exact tests the aggregators use, plus the rounding helpers.

Both were duplicated across run_multiseed_sat.py, run_facility.py and
run_mc_sweep.py, and the two copies of the sign-flip test had already
drifted apart.

Which test to use:

  exact_perm_test    unpaired. Seeds are independent between arms -- use it
                     when the two arms were not run on matched seeds.
  paired_sign_flip   paired. The two arms share seeds, so each seed
                     contributes a difference d_i and the null is that each
                     d_i is equally likely to have either sign. Most of the
                     between-seed spread (layout, spawn, network init) is
                     shared by the two arms, which is what the paired design
                     buys.

A hard limit worth remembering: the sign-flip test has smallest attainable
two-sided p = 2/2^n, so n=5 seeds can never reach 0.05 and n>=6 is required
for a paired claim at that level.
"""

import itertools
from decimal import Decimal, ROUND_HALF_UP

import numpy as np


def exact_perm_test(a, b, n_max_exact=2 ** 16):
    """
    Two-sided exact permutation test on the difference in means. Enumerates
    every split when the number of arrangements is small enough, otherwise
    samples 20000 of them.

    Returns (observed difference, p).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    obs = float(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    n, m = len(a), len(pool)
    from math import comb
    if comb(m, n) <= n_max_exact:
        diffs = []
        for idx in itertools.combinations(range(m), n):
            mask = np.zeros(m, dtype=bool)
            mask[list(idx)] = True
            diffs.append(pool[mask].mean() - pool[~mask].mean())
        diffs = np.asarray(diffs)
    else:
        rng = np.random.default_rng(0)
        diffs = np.empty(20_000)
        for i in range(20_000):
            perm = rng.permutation(m)
            diffs[i] = pool[perm[:n]].mean() - pool[perm[n:]].mean()
    p = float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))
    return obs, p


def paired_sign_flip_test(a, b, n_max_exact=2 ** 20):
    """
    Two-sided exact paired randomization test on the per-seed differences,
    which is what the paired design licenses: both arms see the same layout,
    the same spawn and the same network init at a given seed, so most of the
    between-seed spread is shared.

    Returns (observed mean difference, p).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    obs = float(d.mean())
    if 2 ** n <= n_max_exact:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        means = (signs * d).mean(axis=1)
    else:
        rng = np.random.default_rng(0)
        means = (rng.choice([1.0, -1.0], size=(20_000, n)) * d).mean(axis=1)
    p = float(np.mean(np.abs(means) >= abs(obs) - 1e-12))
    return obs, p


def min_attainable_p(n):
    """Smallest two-sided p a paired sign-flip test on n seeds can produce."""
    return 2.0 / (2 ** n)


# =============================================================================
# Rounding
# =============================================================================
#
# Python rounds half to even, so round(0.5) == 0 and a table built with the
# builtin disagrees with the same table built by hand. These round half away
# from zero, which is what a reader expects.

def r1(x):
    """Round to one decimal, half away from zero."""
    if x is None:
        return None
    return float(Decimal(repr(float(x))).quantize(Decimal("0.1"),
                                                  rounding=ROUND_HALF_UP))


def f1(x):
    """Format to one decimal, half away from zero."""
    return "n/a" if x is None else f"{r1(x):.1f}"
