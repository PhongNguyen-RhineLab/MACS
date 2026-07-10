"""
MACS v3 — experiment extensions on top of macs_main.py (the 7-method v2 code).

Adds, in priority order:

  A. STRICTLY SUBMODULAR reward (most important)
     Saturating regional coverage F(S) = sum_R min(|S n R|, c_R) over a
     block partition of the grid. Modular closed form is INVALID here, so
     MACS must use the permutation form (Alg. 1, mode=EXACT). This is the
     regime where LOCAL over-counts systematically even without physical
     footprint overlap (two agents filling the same near-capped region),
     so the Shapley-vs-LOCAL comparison finally has teeth.

  B. SCALABILITY: k = 4 (exact, 24 permutations/step) and k = 6
     (Monte Carlo Shapley, m = 64 sampled permutations/step, per
     Prop. mc-error of the draft). Plus a MACS-MC ablation at k = 4:
     forced MC where exact ground truth is available, measuring the
     empirical cost of the approximation (Prop. approx-convergence).

  C. OBSTACLE MAP: two-room grid with a one-cell doorway (same topology
     as TwoRoomEnv in sbrl_v16). Walls are encoded as -1 in the coverage
     map channel; coverage and F are defined over passable cells only.

  D. LARGER GRID: 16x16, horizon 60 (flag --large; k=6 runs there by
     default since a 12x12 grid is crowded for six 3x3 footprints).

Everything reuses train_macs() from macs_main unchanged; the ONLY code
this file touches in macs_main is the compute_credits binding, which is
re-pointed to a wrapper that (a) supplies the MC sample count m for k > 4
(the v2 call site never passed mc_m, so k > 4 would divide by zero) and
(b) adds the forced-MC mode "MACS-MC". If you prefer an upstream fix
instead, add `mc_m=64` to the train_macs signature and pass it through at
the compute_credits call site.

Usage:
  python macs_v3.py --test                numpy-only unit tests, no torch
  python macs_v3.py --smoke               tiny end-to-end run of each suite
  python macs_v3.py --sat                 suite A only (saturating F)
  python macs_v3.py --scale               suite B only (k=4, k=6, MACS-MC)
  python macs_v3.py --obstacle            suite C only (two-room)
  python macs_v3.py --large               suite D (16x16, k in {2,3})
  python macs_v3.py                       suites A + B + C
  python plot_macs.py --log-dir logs/macs_v3     figures
"""

import sys
import numpy as np

import macs_main as M
from macs_main import (
    F_cardinality, shapley_exact, shapley_mc, shapley_coverage_closed_form,
    MultiAgentCoverageEnv, compute_frontier, TORCH_OK, DEVICE,
)

LOG_DIR = "logs/macs_v3"
MC_M    = 64          # sampled permutations for MC Shapley (k > 4 or MACS-MC)


# =============================================================================
# A. Strictly submodular F — saturating regional coverage
# =============================================================================

class SaturatingF:
    """
    F(S) = sum_R w_R * min(|S n R|, c_R) over a block partition of the grid.

    Monotone and submodular; STRICTLY submodular across cells within a
    region once |S n R| can exceed c_R (the marginal value of a cell drops
    from w_R to 0 when its region saturates). It is NOT modular, so the
    closed form of Prop. modular does not apply and Shapley credits must be
    computed by permutation enumeration (k <= 4) or Monte Carlo (k > 4).

    Example (region = 4, cap_frac = 0.5 on a 12x12 grid): 9 regions of 16
    cells, each capped at 8. Covering the 9th cell of a region is worth 0,
    so an agent filling an almost-full region earns less Shapley credit
    than one opening a fresh region -- exactly the diminishing-returns
    signal that modular F cannot express.
    """

    def __init__(self, size, region=4, cap_frac=0.5, weights=None, dtype=np.int64):
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
        self.size = size; self.region = region

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
    rather than raw cell counts, so all existing logging and plotting
    normalizes correctly without modification.
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


# =============================================================================
# C. Obstacle map — two-room grid with a one-cell doorway
# =============================================================================

class TwoRoomCoverageEnv(MultiAgentCoverageEnv):
    """
    Vertical wall at column size//2 with a single door cell at the middle
    row, matching TwoRoomEnv from sbrl_v16. Differences from the open grid:

      - Moves into wall cells are blocked (position unchanged).
      - Footprints are intersected with the passable mask, so walls are
        never covered and F counts passable cells only.
      - The observed coverage map uses the sbrl UMaze convention:
        covered -> 1.0, uncovered passable -> 0.0, wall -> -1.0.
        compute_frontier() already respects this (it keys on ==1.0/==0.0),
        and the -1 channel tells the Q-network where the walls are.
    """

    def __init__(self, size=12, horizon=60, k=2, patch=1, seed=None):
        S = size
        self.passable = np.ones((S, S), dtype=bool)
        self.passable[:, S // 2] = False
        self.passable[S // 2, S // 2] = True          # the door
        super().__init__(size=size, horizon=horizon, k=k, patch=patch,
                         seed=seed)

    # -- masks and observation -------------------------------------------
    def _patch_mask(self, pos):
        return super()._patch_mask(pos) & self.passable

    def _obs(self):
        z = np.full((self.size, self.size), -1.0, dtype=np.float32)
        z[self.passable] = 0.0
        z[self.covered] = 1.0
        return self.pos.astype(np.float32) / self.size, z

    # -- dynamics ----------------------------------------------------------
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


# =============================================================================
# B. MC Shapley plumbing + forced-MC ablation mode
# =============================================================================

_orig_compute_credits = M.compute_credits

def compute_credits_v3(mode, r_team, info, F, k, mc_m=MC_M, rng=None):
    """
    Same dispatcher as macs_main.compute_credits, with two changes:
      1. mc_m defaults to MC_M so MACS / MACS-CLIP at k > 4 use MC Shapley
         with a valid sample count (the v2 call site never passed mc_m).
      2. New mode "MACS-MC": FORCE Monte Carlo even for k <= 4, where the
         exact value is available. Same credits pipeline, same TD update,
         no clip -- an ablation isolating the approximation error of
         Prop. mc-error against exact ground truth.
    """
    if mode == "MACS-MC":
        return shapley_mc(info["S_prev"], info["patches"], F, mc_m, rng)
    return _orig_compute_credits(mode, r_team, info, F, k, mc_m=mc_m, rng=rng)

M.compute_credits = compute_credits_v3       # train_macs resolves it here


# =============================================================================
# Unit tests (numpy only) — theory-side claims for the new pieces
# =============================================================================

def run_unit_tests(verbose=True):
    rng = np.random.default_rng(11)
    S = 12
    Fsat = SaturatingF(S, region=4, cap_frac=0.5)

    def rand_mask(p):
        return rng.random((S, S)) < p

    def rand_patches(k):
        out = []
        for _ in range(k):
            pos = rng.integers(1, S - 1, size=2)
            m = np.zeros((S, S), dtype=bool)
            m[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2] = True
            out.append(m)
        return out

    # 1. Saturating F: monotone, and submodular (diminishing returns)
    #    F(A u {v}) - F(A) >= F(B u {v}) - F(B) whenever A subset of B.
    for _ in range(300):
        A = rand_mask(0.35)
        B = A | rand_mask(0.35)
        assert Fsat(B) >= Fsat(A) - 1e-12, "monotonicity violated"
        v = np.zeros((S, S), dtype=bool)
        v[tuple(rng.integers(0, S, size=2))] = True
        gA = Fsat(A | v) - Fsat(A)
        gB = Fsat(B | v) - Fsat(B)
        assert gA >= gB - 1e-12, "submodularity violated"

    # 2. Saturating F is NOT modular: the modular closed form must disagree
    #    with the permutation form on a cap-binding state. Construct one:
    #    empty S, two disjoint 3x3 patches inside ONE region (16 cells,
    #    cap 8). Union covers 15 cells but F caps at 8; per-cell equal
    #    split (closed form) reports 9 + 6 = 15.
    S0 = np.zeros((S, S), dtype=bool)
    p1 = np.zeros((S, S), dtype=bool); p1[0:3, 0:3] = True
    p2 = np.zeros((S, S), dtype=bool); p2[3:4, 0:3] = True; p2[0:3, 3:4] = True
    p2 &= (Fsat.region_id == Fsat.region_id[0, 0])   # keep inside region 0
    phi_perm = shapley_exact(S0, [p1, p2], Fsat)
    phi_cf   = shapley_coverage_closed_form(S0, [p1, p2])
    r = Fsat(p1 | p2) - Fsat(S0)
    assert abs(phi_perm.sum() - r) < 1e-9, "efficiency (saturating) violated"
    assert not np.allclose(phi_perm, phi_cf), \
        "closed form should NOT match permutation form for saturating F"

    # 3. Efficiency of exact Shapley under saturating F, random states
    for k in (2, 3, 4):
        for _ in range(40):
            Sp = rand_mask(0.4); patches = rand_patches(k)
            phi = shapley_exact(Sp, patches, Fsat)
            u = Sp.copy()
            for pm in patches: u |= pm
            assert abs(phi.sum() - (Fsat(u) - Fsat(Sp))) < 1e-9

    # 4. Credit ordering DR <= Shapley <= LOCAL still holds (it relies only
    #    on submodularity, so it must transfer from the modular case)
    for k in (2, 3):
        for _ in range(40):
            Sp = rand_mask(0.4); patches = rand_patches(k)
            info = {"S_prev": Sp, "patches": patches}
            u = Sp.copy()
            for pm in patches: u |= pm
            r = Fsat(u) - Fsat(Sp)
            sh  = shapley_exact(Sp, patches, Fsat)
            loc = compute_credits_v3("LOCAL", r, info, Fsat, k)
            dr  = compute_credits_v3("DR",    r, info, Fsat, k)
            assert np.all(dr <= sh + 1e-9) and np.all(sh <= loc + 1e-9)

    # 5. MC Shapley (m = 2000) concentrates near exact for saturating F,
    #    k = 4; and the m = MC_M estimator has bounded efficiency error
    Sp = rand_mask(0.3); patches = rand_patches(4)
    exact = shapley_exact(Sp, patches, Fsat)
    mc    = shapley_mc(Sp, patches, Fsat, m=2000, rng=rng)
    assert np.max(np.abs(exact - mc)) < 0.5, "MC too far from exact"
    mc64 = compute_credits_v3("MACS-MC", 0.0,
                              {"S_prev": Sp, "patches": patches}, Fsat, 4,
                              rng=rng)
    u = Sp.copy()
    for pm in patches: u |= pm
    r = Fsat(u) - Fsat(Sp)
    # every sampled permutation telescopes, so MC is efficient EXACTLY
    assert abs(mc64.sum() - r) < 1e-9, "MC efficiency (telescoping) violated"

    # 6. k = 6 path: dispatcher must route MACS to MC without crashing
    patches6 = rand_patches(6)
    phi6 = compute_credits_v3("MACS", 0.0,
                              {"S_prev": Sp, "patches": patches6}, Fsat, 6,
                              rng=rng)
    assert phi6.shape == (6,) and np.all(phi6 >= -1e-12)

    # 7. Two-room env: walls never covered, moves into walls blocked,
    #    reward telescopes F(S_T) - F(S_0) = sum_t r_t, door is reachable
    env = TwoRoomCoverageEnv(size=12, horizon=200, k=2, seed=4)
    env.reset()
    f0 = env.F(env.covered); tot = 0.0; done = False
    crossed = False
    rng2 = np.random.default_rng(2)
    while not done:
        (_, z), r, done, info = env.step(rng2.integers(0, 5, size=2))
        tot += r
        assert not env.covered[~env.passable].any(), "wall covered"
        assert all(env.passable[tuple(p)] for p in env.pos), "agent in wall"
        if (env.pos[:, 1] > env.size // 2).any():
            crossed = True
        assert set(np.unique(z)) <= {-1.0, 0.0, 1.0}
    assert abs((env.F(env.covered) - f0) - tot) < 1e-9, "telescoping failed"
    assert crossed, "random walk never crossed the door in 200 steps " \
                    "(unlucky seed; rerun) "

    # 8. Budget bound for saturating F (what MACS-CLIP clips against):
    #    remaining discounted credit <= B(S_t) = F(V) - F(S_t) per agent
    env = SaturatingCoverageEnv(size=12, horizon=15, k=3, seed=6)
    F_total = env.F_total(); gamma = 0.99
    for _ in range(5):
        env.reset()
        S_list = [env.covered.copy()]; phis = []
        done = False
        while not done:
            _, r, done, info = env.step(rng.integers(0, 5, size=3))
            phis.append(shapley_exact(info["S_prev"], info["patches"], env.F))
            S_list.append(env.covered.copy())
        phis = np.array(phis); T = len(phis)
        for t in range(T):
            disc = gamma ** np.arange(T - t)
            remaining = (phis[t:] * disc[:, None]).sum(axis=0)
            B = F_total - env.F(S_list[t])
            assert np.all(remaining <= B + 1e-9), "budget bound violated"

    if verbose:
        print("All v3 unit tests passed:")
        print("  saturating F: monotone, submodular, strictly non-modular")
        print("  exact Shapley efficient under saturating F")
        print("  DR <= Shapley <= LOCAL transfers to saturating F")
        print("  MC Shapley: concentrates near exact; exactly efficient")
        print("  k=6 dispatch routes to MC with m=%d" % MC_M)
        print("  two-room env: walls, door, telescoping all consistent")
        print("  budget bound holds for saturating F")


# =============================================================================
# Experiment suites
# =============================================================================

MODES = ["SHARED", "LOCAL", "DR", "VDN", "QMIX", "MACS", "MACS-CLIP"]


def suite_saturating(cfg, ks=(2, 3), size=12, horizon=40):
    """Suite A (most important): strictly submodular saturating coverage."""
    for k in ks:
        env_name = f"Saturating {size}x{size}"
        factory = (lambda k=k: SaturatingCoverageEnv(
            size=size, horizon=horizon, k=k, patch=1, region=4, cap_frac=0.5))
        for mode in MODES:
            M.train_macs(factory, env_name=env_name, mode=mode,
                         log_dir=LOG_DIR, **cfg)


def suite_scale(cfg, size12=12, size16=16):
    """Suite B: k=4 exact (+ MACS-MC ablation), k=6 Monte Carlo on 16x16."""
    # k = 4, exact Shapley (24 permutations per step), saturating F so the
    # exact-vs-MC comparison is on the non-modular objective where it matters
    factory4 = (lambda: SaturatingCoverageEnv(
        size=size12, horizon=40, k=4, patch=1, region=4, cap_frac=0.5))
    for mode in MODES + ["MACS-MC"]:
        M.train_macs(factory4, env_name=f"Saturating {size12}x{size12}",
                     mode=mode, log_dir=LOG_DIR, **cfg)

    # k = 6 forces MC Shapley (6! = 720 permutations would be exact;
    # MC uses m = 64 -> 6*64 = 384 F evaluations per env step).
    # 16x16 grid: six 3x3 footprints on 12x12 would saturate in a few steps.
    factory6 = (lambda: SaturatingCoverageEnv(
        size=size16, horizon=60, k=6, patch=1, region=4, cap_frac=0.5))
    for mode in MODES:
        M.train_macs(factory6, env_name=f"Saturating {size16}x{size16}",
                     mode=mode, log_dir=LOG_DIR, **cfg)


def suite_obstacle(cfg, ks=(2, 3), size=12, horizon=60):
    """Suite C: two-room map with a one-cell doorway, cardinality F."""
    for k in ks:
        env_name = f"Two-room {size}x{size}"
        factory = (lambda k=k: TwoRoomCoverageEnv(
            size=size, horizon=horizon, k=k, patch=1))
        for mode in MODES:
            M.train_macs(factory, env_name=env_name, mode=mode,
                         log_dir=LOG_DIR, **cfg)


def suite_large(cfg, ks=(2, 3), size=16, horizon=60):
    """Suite D (optional): larger open grid, cardinality F."""
    for k in ks:
        env_name = f"MA coverage {size}x{size}"
        factory = (lambda k=k: MultiAgentCoverageEnv(
            size=size, horizon=horizon, k=k, patch=1))
        for mode in MODES:
            M.train_macs(factory, env_name=env_name, mode=mode,
                         log_dir=LOG_DIR, **cfg)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_unit_tests()
    if "--test" in sys.argv:
        sys.exit(0)

    if not TORCH_OK:
        print("\ntorch not available: unit tests passed, training skipped.")
        sys.exit(0)

    import torch
    np.random.seed(42); torch.manual_seed(42)
    print(f"Device: {DEVICE}")

    SMOKE = "--smoke" in sys.argv
    if SMOKE:
        CFG = dict(n_episodes=30, warmup=200, eval_every=10,
                   n_eval=2, log_every=10, target_every=10)
        suite_saturating(CFG, ks=(2,), size=8, horizon=15)
        suite_obstacle(CFG, ks=(2,), size=8, horizon=15)
        # scale smoke: just verify k=6 MC path end to end
        f6 = (lambda: SaturatingCoverageEnv(size=8, horizon=15, k=6,
                                            region=4, cap_frac=0.5))
        M.train_macs(f6, env_name="Saturating 8x8", mode="MACS",
                     log_dir=LOG_DIR, **CFG)
        sys.exit(0)

    CFG = dict(n_episodes=4000, eval_every=100, log_every=100)

    only = {f for f in ("--sat", "--scale", "--obstacle", "--large")
            if f in sys.argv}
    run_all = not only

    if run_all or "--sat" in only:
        suite_saturating(CFG)
    if run_all or "--scale" in only:
        suite_scale(CFG)
    if run_all or "--obstacle" in only:
        suite_obstacle(CFG)
    if "--large" in only:                 # opt-in, not part of the default
        suite_large(CFG)

    print(f"\nDone. Logs in {LOG_DIR}/. "
          f"Run 'python plot_macs.py --log-dir {LOG_DIR}' for figures.")

