"""
Credit assignment: the per-agent signal phi_i that replaces the team
reward in the per-agent TD target.

    MACS       phi_i = Shapley credit of the per-step coverage game
    MACS-CLIP  MACS credits + clipped bootstrap (the credits are identical;
               only the backup differs, see macs.train)
    MACS-MC    Shapley forced through Monte Carlo even where an exact
               backend exists -- an ablation on the approximation error
    SHARED     phi_i = r_team                (every agent gets everything)
    LOCAL      phi_i = F(S u P_i) - F(S)     first-arrival marginal
    DR         phi_i = F(S u all) - F(S u others)   last-arrival marginal
    INT        phi_i = DR_i + lam (LOC_i - DR_i), one lam per step chosen so
               that sum_i phi_i = r_t: the cheapest efficient rule inside
               the [DR, LOCAL] bracket (2k + 2 evaluations of F). Equal to
               Shapley at k = 2; differs at k >= 3.
    VDN, QMIX  not credit rules; both learn from r_team directly and are
               handled in the loss. compute_credits returns r_team for them
               so the logging path has something to store.

For submodular F these obey, per agent and per step,

    DR <= Shapley <= LOCAL

since the last marginal is the smallest, the first is the largest, and
Shapley averages marginals over all arrival positions.

This module replaces four import-order-dependent monkey-patch layers
(macs_v3's import side effect, macs_ablation.install_facility_credits,
macs_fl.install, shapley_subset.install). The Shapley backend is now an
argument: see macs.shapley.resolve.
"""

import numpy as np

from macs import shapley

# Modes whose loss consumes a per-agent credit phi_i. Efficiency
# (sum_i phi_i == r_team) is only DEFINED for these; VDN and QMIX learn
# from r_team directly and have no credit rule to measure.
CREDIT_MODES = ("MACS", "MACS-CLIP", "MACS-MC", "SHARED", "LOCAL", "DR",
                "INT")

# Modes that learn a joint value by architecture rather than by credit.
MIXING_MODES = ("VDN", "QMIX")

ALL_MODES = CREDIT_MODES + MIXING_MODES

# Modes whose credits are Shapley values.
SHAPLEY_MODES = ("MACS", "MACS-CLIP", "MACS-MC")

# Efficient rules inside the [DR, LOCAL] bracket: every result of the paper's
# Sections 5-7 holds for these (Lemma 1, Thms 2-3, Lemma 2, Prop 5).
BRACKETED_MODES = SHAPLEY_MODES + ("INT",)


def marginal_bounds(info, F):
    """
    (loc, dr, r) for one step: first-arrival marginals, last-arrival
    marginals, and the team reward. For submodular F,
    0 <= dr_i <= loc_i and sum(dr) <= r <= sum(loc).
    Cost: 2k + 2 evaluations of F.
    """
    S_prev, patches = info["S_prev"], info["patches"]
    k = len(patches)
    f0 = F(S_prev)
    loc = np.array([F(S_prev | pm) - f0 for pm in patches])
    union = S_prev.copy()
    for pm in patches:
        union |= pm
    f_all = F(union)
    dr = np.zeros(k)
    for i in range(k):
        others = S_prev.copy()
        for j, pm in enumerate(patches):
            if j != i:
                others |= pm
        dr[i] = f_all - F(others)
    return loc, dr, f_all - f0


def interpolated_credit(loc, dr, r, tol=1e-12):
    """
    phi_i = dr_i + lam (loc_i - dr_i), lam = (r - sum dr) / sum (loc - dr).

    lam lies in [0, 1] because sum(dr) <= r <= sum(loc), so the rule is
    efficient, bracketed, pays a null agent (loc_i = 0) nothing and treats
    symmetric agents symmetrically. When sum(loc - dr) = 0 every agent has
    loc_i = dr_i, r = sum(dr), and phi = dr.
    """
    gap = float(np.sum(loc - dr))
    if gap <= tol:
        return dr.astype(np.float64).copy()
    lam = (float(r) - float(np.sum(dr))) / gap
    lam = min(max(lam, 0.0), 1.0)       # guards float error only
    return dr + lam * (loc - dr)


def compute_credits(mode, r_team, info, F, k, *,
                    backend="auto", mc_m=shapley.MC_M, rng=None):
    """
    Returns phi (k,).

    `backend` is passed through to macs.shapley.resolve for the Shapley
    modes and ignored otherwise. "MACS-MC" pins the backend to "mc"
    regardless, which is the point of that ablation.
    """
    if mode == "SHARED":
        return np.full(k, r_team)

    if mode == "LOCAL":
        S_prev = info["S_prev"]
        f0 = F(S_prev)
        return np.array([F(S_prev | pm) - f0 for pm in info["patches"]])

    if mode == "DR":
        S_prev = info["S_prev"]
        patches = info["patches"]
        union = S_prev.copy()
        for pm in patches:
            union |= pm
        f_all = F(union)
        phi = np.zeros(k)
        for i in range(k):
            others = S_prev.copy()
            for j, pm in enumerate(patches):
                if j != i:
                    others |= pm
            phi[i] = f_all - F(others)
        return phi

    if mode == "INT":
        loc, dr, r = marginal_bounds(info, F)
        return interpolated_credit(loc, dr, r)

    if mode in SHAPLEY_MODES:
        be = "mc" if mode == "MACS-MC" else backend
        return shapley.compute(info["S_prev"], info["patches"], F,
                               backend=be, mc_m=mc_m, rng=rng)

    if mode in MIXING_MODES:
        return np.full(k, r_team)   # stored for logging; loss uses r_team

    raise ValueError(f"unknown mode {mode!r}; expected one of {ALL_MODES}")


def efficiency_error(phi, r_team):
    """
    |sum_i phi_i - r_t|.

    Zero for Shapley by construction, (k-1)*r_t for SHARED, and
    footprint-overlap dependent for LOCAL (over-attributes) and DR
    (under-attributes). Logging it only for MACS made the column read as if
    every rule were efficiency-exact, which is the bug this function exists
    to prevent: call it for every mode in CREDIT_MODES.
    """
    return float(abs(float(np.sum(phi)) - float(r_team)))
