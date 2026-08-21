"""
macs_ablation.py -- trainer for the two controls the paper still needs.

O9  Augmented-state ablation.  Theorem `sufficient-statistic` says
    (s_t, S_t) is sufficient for the history.  Nothing in the trained
    results tests this: on the saturating suite blind agents reach 99.9%
    of the reachable optimum at k=6, so the shared set buys nothing there.
    On the facility objective the policy-level probe measures it at 8-21
    points and growing with k, so that is where to run it.

      obs_mode="shared"   agent i observes the SHARED cumulative set S_t
                          (the setting of Prop. `decentralized`)
      obs_mode="private"  agent i observes only its OWN visited set S_t^i

    Both arms keep memory -- the private agent still sees its own trail --
    so the comparison isolates SHARING, not memory.  Reward and credits
    always come from the shared environment: only the observation changes.

O10 Isolating the budget clip.  Lemma `budget` is a STATE-DEPENDENT bound,
    so the sharp control is a clip at a state-INDEPENDENT constant matched
    to the mean of B(S').  If that reproduces the effect, the submodular
    structure is not what is doing the work.

      clip="none"     plain MACS backup
      clip="budget"   y_i = phi_i + gamma*min(boot, B(S'))     (MACS-Clip)
      clip="const"    y_i = phi_i + gamma*min(boot, c),
                      c = mean of B(S') over the warmup window, then frozen

Why a separate trainer rather than an edit to macs_main.train_macs: the
published results come out of that function, and a private-observation
path changes the shape of the coverage tensor everywhere it is touched.
Keeping this separate means the numbers in the paper cannot move. The
cost is that VDN and QMIX are not supported here -- they are not needed
for either control.

The log schema is identical to train_macs, so plot_multiseed.py and the
permutation tests in run_multiseed_sat.py consume these logs unchanged.

Usage:
    python macs_ablation.py --check     numpy-only wiring checks, no torch
"""

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

import macs_main as M
from macs_main import compute_frontier, TORCH_OK, DEVICE
from macs_fl import FacilityLocationF, shapley_facility

if TORCH_OK:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from macs_main import AgentQNet, _pos_maps


# =============================================================================
# Credit routing for facility location
# =============================================================================

def install_facility_credits():
    """
    macs_main.compute_credits sends MACS credits to shapley_mc whenever
    k > 4, which is right for a general submodular F and wrong for facility
    location, where Prop. `facility` gives the exact credits in
    O(|D| k log k) at any k.  This re-points the MACS branch at the closed
    form when F is a FacilityLocationF, and leaves every other mode and
    every other F untouched.

    Idempotent: calling twice does not stack wrappers.
    """
    if getattr(M.compute_credits, "_facility_aware", False):
        return M.compute_credits
    prev = M.compute_credits

    def dispatch(mode, r_team, info, F, k, mc_m=0, rng=None):
        if isinstance(F, FacilityLocationF) and mode in ("MACS", "MACS-CLIP"):
            return shapley_facility(info["S_prev"], info["patches"], F)
        return prev(mode, r_team, info, F, k, mc_m=mc_m, rng=rng)

    dispatch._facility_aware = True
    M.compute_credits = dispatch
    return dispatch


# =============================================================================
# Observation bookkeeping
# =============================================================================

class ObsTracker:
    """
    Produces the per-agent (coverage, frontier) channels for either mode.

    shared : every agent gets the environment's cumulative set S_t
    private: agent i gets S_t^i, the union of its own footprints so far

    Returned arrays are (k, H, W) bool in both modes, so the replay buffer
    and the batch path have one shape to deal with. Bool rather than float
    keeps the buffer affordable: at k=6 on a 24x24 grid a transition costs
    ~7 KB instead of ~28 KB.
    """

    def __init__(self, env, mode):
        assert mode in ("shared", "private")
        self.mode = mode
        self.env = env
        self.reset()

    def reset(self):
        env = self.env
        self.own = [env._patch_mask(env.pos[i]).copy() for i in range(env.k)]

    def update(self):
        """Call after env.step(): fold each agent's new footprint into its
        own set. The shared set is maintained by the environment itself."""
        env = self.env
        for i in range(env.k):
            self.own[i] |= env._patch_mask(env.pos[i])

    def channels(self, z_shared):
        env = self.env
        if self.mode == "shared":
            zb = np.asarray(z_shared) > 0.5
            fr = compute_frontier(np.asarray(z_shared, dtype=np.float32)) > 0.5
            return (np.repeat(zb[None], env.k, 0),
                    np.repeat(fr[None], env.k, 0))
        zs = np.stack(self.own)
        frs = np.stack([compute_frontier(o.astype(np.float32)) > 0.5
                        for o in self.own])
        return zs, frs


# =============================================================================
# Trainer
# =============================================================================

if TORCH_OK:

    class Replay:
        def __init__(self, cap): self.buf = deque(maxlen=cap)
        def push(self, tr):      self.buf.append(tr)
        def __len__(self):       return len(self.buf)
        def sample(self, bs, rng):
            idx = rng.integers(0, len(self.buf), size=bs)
            return [self.buf[i] for i in idx]

    def train_ablation(
        env_factory, env_name, mode="MACS",
        obs_mode="shared", clip="none",
        n_episodes=4000, gamma=0.99, lr=5e-4,
        batch_size=64, buffer_cap=40_000, warmup=1_500,
        eps_start=1.0, eps_end=0.05, eps_frac=0.6,
        target_every=50, eval_every=100, n_eval=8,
        label=None, log_dir="logs/macs_ablation", log_every=100, seed=0,
    ):
        assert obs_mode in ("shared", "private")
        assert clip in ("none", "budget", "const")
        assert mode in ("MACS", "MACS-CLIP", "LOCAL", "DR", "SHARED"), \
            "VDN/QMIX are not supported here; use macs_main.train_macs"
        if clip == "budget" and obs_mode == "private":
            raise ValueError(
                "clip='budget' needs B(S') from the shared set, which a "
                "private-observation agent cannot see. Use clip='none' for "
                "the O9 ablation so the observation is the only difference.")

        env = env_factory()
        k, size, H = env.k, env.size, env.horizon
        if label is None:
            label = (f"{env_name.replace(' ', '_')}_k{k}_{mode}"
                     f"_{obs_mode}_clip-{clip}")
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        F_total = env.F_total()

        nets = [AgentQNet(size).to(DEVICE) for _ in range(k)]
        targets = [AgentQNet(size).to(DEVICE) for _ in range(k)]
        for q, qt in zip(nets, targets):
            qt.load_state_dict(q.state_dict())
            for p in qt.parameters():
                p.requires_grad_(False)
        opts = [optim.Adam(q.parameters(), lr=lr, eps=1e-5) for q in nets]
        buf = Replay(buffer_cap)

        # constant clip level, estimated over the warmup window then frozen
        const_c, const_acc = None, []

        print(f"\n{'='*66}\n  {label}\n  k={k} size={size} H={H} mode={mode} "
              f"obs={obs_mode} clip={clip}\n{'='*66}")

        log = {"label": label,
               "config": dict(method=mode, env_name=env_name, k=k,
                              map_size=size, horizon=H,
                              max_cells=env.max_cells, n_episodes=n_episodes,
                              obs_mode=obs_mode, clip=clip, seed=seed,
                              lr=lr, gamma=gamma, batch_size=batch_size),
               "per_block": {key: [] for key in
                             ("episode", "coverage_mean", "eval_coverage",
                              "loss", "epsilon", "credit_sum_err",
                              "clip_frac")},
               "summary": {}}
        block_cov, block_loss, block_err, block_clip = [], [], [], []
        eval_cov_last = 0.0

        def greedy_actions(pos_n, zs, frs, epsilon, nets_=None):
            nets_ = nets_ or nets
            acts = np.zeros(k, dtype=np.int64)
            pt = torch.tensor(pos_n, dtype=torch.float32, device=DEVICE)
            zt = torch.tensor(zs, dtype=torch.float32, device=DEVICE)
            ft = torch.tensor(frs, dtype=torch.float32, device=DEVICE)
            pm = _pos_maps(pt, size, DEVICE)
            with torch.no_grad():
                for i in range(k):
                    if rng.random() < epsilon:
                        acts[i] = rng.integers(env.n_actions)
                    else:
                        q = nets_[i](pt[i:i+1], zt[i:i+1], ft[i:i+1],
                                     pm[i:i+1])
                        acts[i] = int(q.argmax())
            return acts

        def run_eval():
            covs = []
            for _ in range(n_eval):
                e = env_factory()
                (pn, z) = e.reset()
                tr = ObsTracker(e, obs_mode)
                zs, frs = tr.channels(z)
                done = False
                while not done:
                    a = greedy_actions(pn, zs, frs, 0.0)
                    (pn, z), _, done, _ = e.step(a)
                    tr.update()
                    zs, frs = tr.channels(z)
                covs.append(e.total_coverage)
            return float(np.mean(covs))

        for ep_i in range(1, n_episodes + 1):
            t = min(1.0, (ep_i - 1) / max(1, int(n_episodes * eps_frac)))
            epsilon = eps_start + t * (eps_end - eps_start)

            (pos_n, z) = env.reset()
            tracker = ObsTracker(env, obs_mode)
            zs, frs = tracker.channels(z)
            done = False
            while not done:
                acts = greedy_actions(pos_n, zs, frs, epsilon)
                (pos_n2, z2), r_team, done, info = env.step(acts)
                tracker.update()
                zs2, frs2 = tracker.channels(z2)

                phi = M.compute_credits(mode, r_team, info, env.F, k, rng=rng)
                if mode in ("MACS", "MACS-CLIP"):
                    block_err.append(abs(phi.sum() - r_team))

                bud2 = np.float32(F_total - env.F(env.covered))
                if clip == "const" and const_c is None:
                    const_acc.append(float(bud2))

                buf.push((pos_n.copy(), zs, frs, acts.copy(),
                          phi.astype(np.float32), np.float32(r_team),
                          pos_n2.copy(), zs2, frs2, np.float32(done), bud2))
                pos_n, zs, frs = pos_n2, zs2, frs2

                if len(buf) >= warmup:
                    if clip == "const" and const_c is None:
                        const_c = float(np.mean(const_acc))
                        print(f"  [{label}] constant clip level c = "
                              f"{const_c:.3f} (mean B(S') over warmup, "
                              f"n={len(const_acc)})")
                        log["config"]["const_clip_level"] = const_c

                    batch = buf.sample(batch_size, rng)
                    T = lambda j, dt=torch.float32: torch.tensor(
                        np.stack([b[j] for b in batch]), dtype=dt,
                        device=DEVICE)
                    bp, bz, bf = T(0), T(1), T(2)            # (B,k,2),(B,k,H,W)
                    ba = T(3, torch.long)
                    bphi, bd = T(4), T(9)
                    bp2, bz2, bf2 = T(6), T(7), T(8)
                    bbud = T(10)

                    losses = []
                    for i in range(k):
                        pmi = _pos_maps(bp[:, i], size, DEVICE)
                        pmi2 = _pos_maps(bp2[:, i], size, DEVICE)
                        qi = nets[i](bp[:, i], bz[:, i], bf[:, i], pmi
                                     ).gather(1, ba[:, i:i+1]).squeeze(1)
                        with torch.no_grad():
                            boot = targets[i](bp2[:, i], bz2[:, i], bf2[:, i],
                                              pmi2).max(1).values
                            if clip != "none":
                                ceil = (bbud if clip == "budget"
                                        else torch.full_like(boot, const_c))
                                block_clip.append(
                                    (boot > ceil).float().mean().item())
                                boot = torch.minimum(boot, ceil)
                            yi = bphi[:, i] + gamma * (1 - bd) * boot
                        li = nn.functional.mse_loss(qi, yi)
                        opts[i].zero_grad()
                        li.backward()
                        nn.utils.clip_grad_norm_(nets[i].parameters(), 5.0)
                        opts[i].step()
                        losses.append(li.item())
                    block_loss.append(float(np.mean(losses)))

            block_cov.append(env.total_coverage)

            if ep_i % target_every == 0:
                for q, qt in zip(nets, targets):
                    qt.load_state_dict(q.state_dict())
            if ep_i % eval_every == 0:
                eval_cov_last = run_eval()

            if ep_i % log_every == 0:
                pb = log["per_block"]
                pb["episode"].append(ep_i)
                pb["coverage_mean"].append(round(float(np.mean(block_cov)), 3))
                pb["eval_coverage"].append(round(eval_cov_last, 3))
                pb["loss"].append(round(float(np.mean(block_loss)), 5)
                                  if block_loss else None)
                pb["epsilon"].append(round(epsilon, 3))
                pb["credit_sum_err"].append(
                    round(float(np.max(block_err)), 8) if block_err else 0.0)
                pb["clip_frac"].append(round(float(np.mean(block_clip)), 5)
                                       if block_clip else 0.0)
                print(f"  [{label}] ep {ep_i:5d} | "
                      f"F {np.mean(block_cov):7.2f}/{env.max_cells:.0f} "
                      f"| eval {eval_cov_last:7.2f} | eps {epsilon:.2f}"
                      + (f" | clip {np.mean(block_clip):.3f}"
                         if block_clip else ""))
                block_cov, block_loss, block_err, block_clip = [], [], [], []
                out = Path(log_dir); out.mkdir(parents=True, exist_ok=True)
                with open(out / f"{label}.json", "w") as f:
                    json.dump(log, f, indent=2)

        pb = log["per_block"]
        log["summary"] = {
            "last10_eval": round(float(np.mean(pb["eval_coverage"][-10:])), 3),
            "final_eval": round(pb["eval_coverage"][-1], 3)}
        out = Path(log_dir); out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{label}.json", "w") as f:
            json.dump(log, f, indent=2)
        print(f"  -> {log_dir}/{label}.json | "
              f"last-10 eval {log['summary']['last10_eval']}")
        return log


# =============================================================================
# Wiring checks that do not need torch
# =============================================================================

def run_checks(verbose=True):
    from macs_fl import FacilityCoverageEnv
    from macs_v3 import SaturatingCoverageEnv
    import macs_v3  # noqa: F401  (rebinds M.compute_credits to the v3 form)

    install_facility_credits()
    install_facility_credits()          # idempotence
    assert getattr(M.compute_credits, "_facility_aware", False)

    rng = np.random.default_rng(0)

    # 1. facility credits go to the closed form at every k, including k > 4,
    #    and still satisfy efficiency there
    for k in (2, 4, 6, 8):
        env = FacilityCoverageEnv(size=16, k=k, horizon=10, rho=4.0, patch=0,
                                  seed=1)
        env.reset()
        _, r, _, info = env.step(rng.integers(0, 5, size=k))
        phi = M.compute_credits("MACS", r, info, env.F, k, rng=rng)
        exact = shapley_facility(info["S_prev"], info["patches"], env.Ffl)
        assert np.allclose(phi, exact, atol=1e-9), f"k={k} not routed"
        assert abs(phi.sum() - r) < 1e-8, f"k={k} efficiency"

    # 2. a non-facility F is untouched by the patch
    env = SaturatingCoverageEnv(size=12, k=3, horizon=10, seed=1)
    env.reset()
    _, r, _, info = env.step(rng.integers(0, 5, size=3))
    phi = M.compute_credits("MACS", r, info, env.F, 3, rng=rng)
    ref = M.shapley_exact(info["S_prev"], info["patches"], env.F)
    assert np.allclose(phi, ref, atol=1e-9), "saturating path perturbed"

    # 3. ObsTracker: shared broadcasts one map; private keeps k distinct ones,
    #    each a subset of the shared set, and together covering it
    for mode in ("shared", "private"):
        env = FacilityCoverageEnv(size=12, k=3, horizon=8, rho=3.0, patch=0,
                                  seed=2)
        (_, z) = env.reset()
        tr = ObsTracker(env, mode)
        for _ in range(env.horizon):
            zs, frs = tr.channels(z)
            assert zs.shape == (env.k, env.size, env.size)
            assert frs.shape == zs.shape and zs.dtype == np.bool_
            if mode == "shared":
                assert (zs[0] == zs[1]).all() and (zs[1] == zs[2]).all()
            else:
                for i in range(env.k):
                    assert (zs[i] & ~env.covered).sum() == 0, \
                        "private set must be a subset of the shared set"
                assert (np.logical_or.reduce(list(zs)) == env.covered).all(), \
                    "private sets must union to the shared set"
            (_, z), _, _, _ = env.step(rng.integers(0, 5, size=env.k))
            tr.update()

    # 4. private observation is strictly less informative than shared
    env = FacilityCoverageEnv(size=12, k=3, horizon=20, rho=3.0, patch=0,
                              seed=3)
    (_, z) = env.reset()
    tr = ObsTracker(env, "private")
    for _ in range(env.horizon):
        (_, z), _, _, _ = env.step(rng.integers(0, 5, size=env.k))
        tr.update()
    zs, _ = tr.channels(z)
    assert zs[0].sum() < env.covered.sum(), \
        "after 20 steps agent 0 should not have seen the whole set"

    # 5. the illegal combination is rejected
    if TORCH_OK:
        try:
            train_ablation(lambda: FacilityCoverageEnv(size=8, k=2),
                           "x", clip="budget", obs_mode="private")
        except ValueError:
            pass
        else:
            raise AssertionError("budget clip + private obs should be refused")

    if verbose:
        print("macs_ablation wiring checks passed:")
        print("  facility credits use the closed form at k = 2,4,6,8")
        print("  efficiency holds on the closed-form path at every k")
        print("  saturating credits unchanged by the patch")
        print("  shared obs broadcasts; private obs is k distinct subsets")
        print("  private sets union to the shared set (partition property)")
        print("  private observation is strictly weaker after 20 steps")
        if TORCH_OK:
            print("  budget clip + private observation refused")


if __name__ == "__main__":
    run_checks()
    if "--check" in sys.argv:
        sys.exit(0)
    if not TORCH_OK:
        print("\ntorch unavailable: checks passed, training skipped.")