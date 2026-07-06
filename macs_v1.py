"""
MACS v1 — Multi-Agent Coverage via Submodularity
Empirical validation of main_macs.tex (draft v0.5) on grid coverage.

Extends the sbrl_v16 codebase to k cooperative agents. Four credit-assignment
methods are compared under IDENTICAL independent DQN updates (Algorithm 2 of
the draft), so credit assignment is the only experimental variable:

  MACS     phi_i = exact Shapley credit (permutation form, Alg. 1)  proposed
  SHARED   phi_i = r_t   (every agent receives full team reward)    naive
  LOCAL    phi_i = F(S u P_i) - F(S)   (own marginal, ignores others)
           violates efficiency: sum_i phi_i >= r_t (double counting)
  VDN      joint TD on Q_tot = sum_i Q_i with team reward r_t
           (Sunehag et al. 2018, the paper's main baseline)

All methods use the decentralized parameterization Q_i(s^i, S): each agent
observes its own position and the shared coverage map ONLY (no other agents'
positions), matching the CTDE policy class of Prop. `Decentralized Execution
Sufficiency` in the draft.

Index convention (the draft is ambiguous between S_{t-1} and S_t in
Algorithm 2; the code fixes one consistent choice):
  reset(): initial patches are covered WITHOUT reward,  S_0 = P(s_0).
  step t : agents move to s_{t+1},
           M = ( u_i P(s_{t+1}^i) ) minus S_t
           r = F(S_t u M) - F(S_t)
           S_{t+1} = S_t u M
  Shapley is computed over the coverage game
           v(C) = F(S_t u  u_{i in C} P(s_{t+1}^i)) - F(S_t).

Usage:
  python macs_v1.py --test          run numpy-only unit tests (no torch)
  python macs_v1.py --smoke         tiny end-to-end run (2 min on CPU)
  python macs_v1.py                 full comparison, k in {2,3}
  python macs_v1.py --plot          plots from logs/macs_v1
"""

import json
import sys
import numpy as np
from itertools import permutations
from collections import defaultdict, deque
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_OK = False
    DEVICE = None


# =============================================================================
# Shapley credit (Algorithm 1 of the draft)
# =============================================================================

def F_cardinality(mask):
    """F(S) = |S|. mask: boolean (H,W)."""
    return float(mask.sum())


def make_F_weighted(weights):
    """F(S) = sum of cell weights, still modular hence submodular."""
    def F(mask): return float((mask * weights).sum())
    return F


def shapley_exact(S_prev, patches, F):
    """
    Exact Shapley via all k! permutations (Alg. 1, mode=EXACT).
    S_prev : boolean (H,W) coverage before the step
    patches: list of k boolean (H,W) masks, P(s^i) for each agent
    F      : set function on boolean masks
    Returns phi (k,) with sum_i phi_i = F(S_prev u all patches) - F(S_prev).
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


def shapley_mc(S_prev, patches, F, m, rng):
    """Monte Carlo Shapley (Alg. 1, mode=MC), m sampled permutations."""
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


def shapley_coverage_closed_form(S_prev, patches, weights=None):
    """
    Closed form valid when F is MODULAR (cardinality or weighted coverage):
    each newly covered cell's weight is split equally among the agents whose
    patch covers it. Used as an independent check against shapley_exact.
    """
    stack = np.stack(patches)                    # (k,H,W)
    new = stack & ~S_prev[None]                  # new cells per agent
    counts = new.sum(0).astype(np.float64)       # multiplicity per cell
    w = np.ones_like(counts) if weights is None else weights.astype(np.float64)
    share = np.divide(w, counts, out=np.zeros_like(w), where=counts > 0)
    return (new * share[None]).sum(axis=(1, 2))


# =============================================================================
# Environment — multi-agent coverage grid
# =============================================================================

class MultiAgentCoverageEnv:
    """
    Open-grid coverage with k agents, consistent with SubmodularCoverageEnv
    from sbrl_v16 (same patch semantics, same z_map convention).

    step(joint_action) returns:
      (positions (k,2) normalized, z_map copy), r_team, done, info
    info carries what credit assignment needs:
      info["S_prev"]  boolean (H,W)  coverage before the step
      info["patches"] list of k boolean (H,W)  each agent's patch AFTER moving
    """
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

    def __init__(self, size=12, horizon=40, k=2, patch=1, weights=None, seed=None):
        self.size = size; self.horizon = horizon; self.k = k; self.patch = patch
        self.n_actions = 5
        self.weights = weights            # None -> cardinality
        self.F = F_cardinality if weights is None else make_F_weighted(weights)
        self.rng = np.random.default_rng(seed)
        self.reset()

    # -- helpers ---------------------------------------------------------
    def _patch_mask(self, pos):
        m = np.zeros((self.size, self.size), dtype=bool)
        p = self.patch
        x0, x1 = max(0, pos[0]-p), min(self.size, pos[0]+p+1)
        y0, y1 = max(0, pos[1]-p), min(self.size, pos[1]+p+1)
        m[x0:x1, y0:y1] = True
        return m

    def _obs(self):
        return (self.pos.astype(np.float32) / self.size,
                self.covered.astype(np.float32))

    # -- API --------------------------------------------------------------
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

    @property
    def total_coverage(self): return int(self.covered.sum())
    @property
    def max_cells(self): return self.size * self.size


def compute_frontier(z_map):
    """Same as sbrl_v16: uncovered cells 4-adjacent to covered cells."""
    covered = (z_map == 1.0); uncovered = (z_map == 0.0)
    frontier = np.zeros_like(covered, dtype=bool)
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        s = np.zeros_like(covered)
        if   dx == -1: s[1:, :]  = covered[:-1, :]
        elif dx ==  1: s[:-1, :] = covered[1:, :]
        elif dy == -1: s[:, 1:]  = covered[:, :-1]
        elif dy ==  1: s[:, :-1] = covered[:, 1:]
        frontier |= s
    return (frontier & uncovered).astype(np.float32)


# =============================================================================
# Credit assignment dispatcher
# =============================================================================

def compute_credits(mode, r_team, info, F, k, mc_m=0, rng=None):
    """
    Returns phi (k,).
    MACS   exact Shapley for k<=4, MC with mc_m samples otherwise
    SHARED phi_i = r_team
    LOCAL  phi_i = F(S u P_i) - F(S)   own marginal, ignores overlap
    (VDN is not a credit rule; it is handled in the loss.)
    """
    if mode == "SHARED":
        return np.full(k, r_team)
    if mode == "LOCAL":
        S_prev = info["S_prev"]; f0 = F(S_prev)
        return np.array([F(S_prev | pm) - f0 for pm in info["patches"]])
    if mode == "MACS":
        if k <= 4:
            return shapley_exact(info["S_prev"], info["patches"], F)
        return shapley_mc(info["S_prev"], info["patches"], F, mc_m, rng)
    if mode == "VDN":
        return np.full(k, r_team)   # stored for logging; loss uses r_team
    raise ValueError(mode)


# =============================================================================
# Networks and training (torch)
# =============================================================================

if TORCH_OK:

    def _make_cnn(map_size, in_ch=3):
        cnn = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 1, 1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),    nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),    nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            cdim = cnn(torch.zeros(1, in_ch, map_size, map_size)).shape[1]
        return cnn, cdim

    def _orth(m):
        for l in m.modules():
            if isinstance(l, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(l.weight, np.sqrt(2))
                nn.init.constant_(l.bias, 0)

    class AgentQNet(nn.Module):
        """
        Q_i(s^i, S, .) — decentralized input:
          CNN channels: [z_map, frontier, own-position one-hot]
          MLP branch:   own position (2,) normalized
        No other agents' positions anywhere. This is the policy class of
        Prop. `Decentralized Execution Sufficiency`.
        """
        def __init__(self, map_size, n_actions=5):
            super().__init__()
            self.map_size = map_size
            cnn, cdim = _make_cnn(map_size, in_ch=3)
            self.cnn = cnn
            self.pos = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(cdim + 32, 128), nn.ReLU(),
                                      nn.Linear(128, 64), nn.ReLU(),
                                      nn.Linear(64, n_actions))
            _orth(self)
            nn.init.orthogonal_(self.head[-1].weight, 1.0)

        def forward(self, pos, z_map, frontier, pos_map):
            x = torch.stack([z_map, frontier, pos_map], dim=1)
            return self.head(torch.cat([self.cnn(x), self.pos(pos)], dim=1))

    def _pos_maps(pos_norm, size, device):
        """(B,2) normalized positions -> (B,H,W) one-hot maps."""
        B = pos_norm.shape[0]
        idx = (pos_norm * size).long().clamp(0, size - 1)
        m = torch.zeros(B, size, size, device=device)
        m[torch.arange(B), idx[:, 0], idx[:, 1]] = 1.0
        return m

    class Replay:
        def __init__(self, cap): self.buf = deque(maxlen=cap)
        def push(self, tr):      self.buf.append(tr)
        def __len__(self):       return len(self.buf)
        def sample(self, bs, rng):
            idx = rng.integers(0, len(self.buf), size=bs)
            return [self.buf[i] for i in idx]

    def train_macs(
        env_factory, env_name, mode,               # MACS | SHARED | LOCAL | VDN
        n_episodes=4000, gamma=0.99, lr=5e-4,
        batch_size=64, buffer_cap=60_000, warmup=1_500,
        eps_start=1.0, eps_end=0.05, eps_frac=0.6,
        target_every=50,                            # hard update, episodes (C)
        eval_every=100, n_eval=8,
        label=None, log_dir="logs/macs_v1", log_every=100, seed=0,
    ):
        env = env_factory()
        k, size, H = env.k, env.size, env.horizon
        if label is None: label = f"{env_name.replace(' ','_')}_k{k}_{mode}"
        rng = np.random.default_rng(seed)
        if TORCH_OK: torch.manual_seed(seed)

        nets    = [AgentQNet(size).to(DEVICE) for _ in range(k)]
        targets = [AgentQNet(size).to(DEVICE) for _ in range(k)]
        for q, qt in zip(nets, targets):
            qt.load_state_dict(q.state_dict())
            for p in qt.parameters(): p.requires_grad_(False)
        opts = [optim.Adam(q.parameters(), lr=lr, eps=1e-5) for q in nets]
        buf = Replay(buffer_cap)

        print(f"\n{'='*62}\n  {label} | k={k} size={size} H={H} mode={mode}"
              f"\n{'='*62}")

        log = {"label": label,
               "config": dict(method=mode, env_name=env_name, k=k,
                              map_size=size, horizon=H,
                              max_cells=env.max_cells, n_episodes=n_episodes),
               "per_block": {key: [] for key in
                 ("episode", "coverage_mean", "eval_coverage",
                  "loss", "epsilon", "credit_sum_err")},
               "summary": {}}
        hist = defaultdict(list)
        block_cov, block_loss, block_err = [], [], []
        eval_cov_last = 0.0
        gstep = 0

        def greedy_actions(pos_n, z, fr, epsilon):
            acts = np.zeros(k, dtype=np.int64)
            pt = torch.tensor(pos_n, dtype=torch.float32, device=DEVICE)
            zt = torch.tensor(z, device=DEVICE).unsqueeze(0).expand(k, -1, -1)
            ft = torch.tensor(fr, device=DEVICE).unsqueeze(0).expand(k, -1, -1)
            pm = _pos_maps(pt, size, DEVICE)
            with torch.no_grad():
                for i in range(k):
                    if rng.random() < epsilon:
                        acts[i] = rng.integers(env.n_actions)
                    else:
                        q = nets[i](pt[i:i+1], zt[i:i+1], ft[i:i+1], pm[i:i+1])
                        acts[i] = int(q.argmax())
            return acts

        def run_eval():
            covs = []
            for _ in range(n_eval):
                e = env_factory()
                (pn, z) = e.reset(); fr = compute_frontier(z)
                done = False
                while not done:
                    a = greedy_actions(pn, z, fr, 0.0)
                    (pn, z), _, done, _ = e.step(a)
                    fr = compute_frontier(z)
                covs.append(e.total_coverage)
            return float(np.mean(covs))

        for ep_i in range(1, n_episodes + 1):
            t = min(1.0, (ep_i - 1) / max(1, int(n_episodes * eps_frac)))
            epsilon = eps_start + t * (eps_end - eps_start)

            (pos_n, z) = env.reset(); fr = compute_frontier(z)
            done = False
            while not done:
                acts = greedy_actions(pos_n, z, fr, epsilon)
                (pos_n2, z2), r_team, done, info = env.step(acts)
                fr2 = compute_frontier(z2)
                phi = compute_credits(mode, r_team, info, env.F, k, rng=rng)
                if mode == "MACS":
                    block_err.append(abs(phi.sum() - r_team))
                buf.push((pos_n.copy(), z.copy(), fr.copy(), acts.copy(),
                          phi.astype(np.float32), np.float32(r_team),
                          pos_n2.copy(), z2.copy(), fr2.copy(),
                          np.float32(done)))
                pos_n, z, fr = pos_n2, z2, fr2
                gstep += 1

                # -- one gradient step per env step after warmup ----------
                if len(buf) >= warmup:
                    batch = buf.sample(batch_size, rng)
                    B = len(batch)
                    bp  = torch.tensor(np.stack([b[0] for b in batch]),
                                       dtype=torch.float32, device=DEVICE)  # (B,k,2)
                    bz  = torch.tensor(np.stack([b[1] for b in batch]),
                                       device=DEVICE)                        # (B,H,W)
                    bf  = torch.tensor(np.stack([b[2] for b in batch]),
                                       device=DEVICE)
                    ba  = torch.tensor(np.stack([b[3] for b in batch]),
                                       dtype=torch.long, device=DEVICE)      # (B,k)
                    bphi= torch.tensor(np.stack([b[4] for b in batch]),
                                       device=DEVICE)                        # (B,k)
                    br  = torch.tensor(np.stack([b[5] for b in batch]),
                                       device=DEVICE)                        # (B,)
                    bp2 = torch.tensor(np.stack([b[6] for b in batch]),
                                       dtype=torch.float32, device=DEVICE)
                    bz2 = torch.tensor(np.stack([b[7] for b in batch]),
                                       device=DEVICE)
                    bf2 = torch.tensor(np.stack([b[8] for b in batch]),
                                       device=DEVICE)
                    bd  = torch.tensor(np.stack([b[9] for b in batch]),
                                       device=DEVICE)                        # (B,)

                    if mode == "VDN":
                        # Q_tot = sum_i Q_i, single TD loss on team reward
                        q_sum, y_sum = 0.0, 0.0
                        for i in range(k):
                            pmi  = _pos_maps(bp[:, i], size, DEVICE)
                            pmi2 = _pos_maps(bp2[:, i], size, DEVICE)
                            qi = nets[i](bp[:, i], bz, bf, pmi)
                            q_sum = q_sum + qi.gather(
                                1, ba[:, i:i+1]).squeeze(1)
                            with torch.no_grad():
                                y_sum = y_sum + targets[i](
                                    bp2[:, i], bz2, bf2, pmi2).max(1).values
                        y = br + gamma * (1 - bd) * y_sum
                        loss = nn.functional.mse_loss(q_sum, y.detach())
                        for o in opts: o.zero_grad()
                        loss.backward()
                        for i in range(k):
                            nn.utils.clip_grad_norm_(nets[i].parameters(), 5.0)
                        for o in opts: o.step()
                        block_loss.append(loss.item())
                    else:
                        # independent per-agent TD with credit phi_i (Alg. 2)
                        losses = []
                        for i in range(k):
                            pmi  = _pos_maps(bp[:, i], size, DEVICE)
                            pmi2 = _pos_maps(bp2[:, i], size, DEVICE)
                            qi = nets[i](bp[:, i], bz, bf, pmi).gather(
                                1, ba[:, i:i+1]).squeeze(1)
                            with torch.no_grad():
                                yi = bphi[:, i] + gamma * (1 - bd) * targets[i](
                                    bp2[:, i], bz2, bf2, pmi2).max(1).values
                            li = nn.functional.mse_loss(qi, yi)
                            opts[i].zero_grad(); li.backward()
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
                print(f"  [{label}] ep {ep_i:5d} | "
                      f"cov {np.mean(block_cov):6.1f}/{env.max_cells} "
                      f"| eval {eval_cov_last:6.1f} | eps {epsilon:.2f}")
                hist["coverage"].append(float(np.mean(block_cov)))
                block_cov, block_loss, block_err = [], [], []
                out = Path(log_dir); out.mkdir(parents=True, exist_ok=True)
                with open(out / f"{label}.json", "w") as f:
                    json.dump(log, f, indent=2)

        pb = log["per_block"]
        log["summary"] = {
            "last5_train": round(float(np.mean(pb["coverage_mean"][-5:])), 2),
            "final_eval":  round(pb["eval_coverage"][-1], 2)}
        out = Path(log_dir); out.mkdir(parents=True, exist_ok=True)
        with open(out / f"{label}.json", "w") as f:
            json.dump(log, f, indent=2)
        print(f"  -> Log: {log_dir}/{label}.json | "
              f"final eval coverage {log['summary']['final_eval']}")
        return hist


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(log_dir="logs/macs_v1", out_dir="plots/macs_v1"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib"); return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    COLOR = {"MACS": "#2ecc71", "VDN": "#3498db",
             "SHARED": "#e67e22", "LOCAL": "#e74c3c"}
    ORDER = ["SHARED", "LOCAL", "VDN", "MACS"]
    groups = {}
    for p in sorted(Path(log_dir).glob("*.json")):
        with open(p) as f: d = json.load(f)
        key = (d["config"]["env_name"], d["config"]["k"])
        groups.setdefault(key, {})[d["config"]["method"]] = d
    for (env_name, k), methods in sorted(groups.items()):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for m in ORDER:
            if m not in methods: continue
            d = methods[m]; pb = d["per_block"]
            ax.plot(pb["episode"], pb["eval_coverage"], label=m,
                    color=COLOR[m], lw=2.0)
        ax.set_xlabel("Episode"); ax.set_ylabel("Eval coverage (cells)")
        ax.set_title(f"{env_name}, k={k}")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fp = Path(out_dir) / f"{env_name.replace(' ','_')}_k{k}.png"
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"Saved: {fp}"); plt.close()


# =============================================================================
# Unit tests (numpy only, no torch required) — validate theory-side claims
# =============================================================================

def run_unit_tests(verbose=True):
    rng = np.random.default_rng(7)
    S = 10

    def rand_state(k):
        S_prev = rng.random((S, S)) < 0.4
        patches = []
        for _ in range(k):
            pos = rng.integers(1, S - 1, size=2)
            m = np.zeros((S, S), dtype=bool)
            m[pos[0]-1:pos[0]+2, pos[1]-1:pos[1]+2] = True
            patches.append(m)
        return S_prev, patches

    # 1. Efficiency: sum_i phi_i == r  (Shapley axiom i)
    for k in (2, 3, 4):
        for _ in range(50):
            S_prev, patches = rand_state(k)
            phi = shapley_exact(S_prev, patches, F_cardinality)
            union = S_prev.copy()
            for pm in patches: union |= pm
            r = F_cardinality(union) - F_cardinality(S_prev)
            assert abs(phi.sum() - r) < 1e-9, "efficiency violated"

    # 2. Null agent: patch fully inside S_prev  ->  phi_i = 0  (axiom ii)
    S_prev, patches = rand_state(2)
    S_prev |= patches[1]                     # agent 2 covers nothing new
    phi = shapley_exact(S_prev, patches, F_cardinality)
    assert abs(phi[1]) < 1e-9, "null agent violated"

    # 3. Symmetry: two agents on the SAME new cell split it 1/2 each
    S_prev = np.zeros((S, S), dtype=bool)
    m = np.zeros((S, S), dtype=bool); m[4, 4] = True
    phi = shapley_exact(S_prev, [m, m.copy()], F_cardinality)
    assert abs(phi[0] - 0.5) < 1e-9 and abs(phi[1] - 0.5) < 1e-9

    # 4. Closed form == permutation form for modular F
    for k in (2, 3, 4):
        for _ in range(30):
            S_prev, patches = rand_state(k)
            a = shapley_exact(S_prev, patches, F_cardinality)
            b = shapley_coverage_closed_form(S_prev, patches)
            assert np.allclose(a, b, atol=1e-9), "closed form mismatch"
            w = rng.random((S, S)) * 3.0
            aw = shapley_exact(S_prev, patches, make_F_weighted(w))
            bw = shapley_coverage_closed_form(S_prev, patches, weights=w)
            assert np.allclose(aw, bw, atol=1e-7), "weighted mismatch"

    # 5. MC Shapley concentrates near exact (Prop MC error bound, sanity)
    S_prev, patches = rand_state(4)
    exact = shapley_exact(S_prev, patches, F_cardinality)
    approx = shapley_mc(S_prev, patches, F_cardinality, m=2000, rng=rng)
    assert np.max(np.abs(exact - approx)) < 0.5, "MC too far from exact"

    # 6. LOCAL over-counts exactly when patches overlap on new cells
    S_prev = np.zeros((S, S), dtype=bool)
    phi_local = np.array([F_cardinality(S_prev | m) for m in [m, m]]) \
                - F_cardinality(S_prev)
    assert phi_local.sum() > 1.0, "LOCAL should double count here"

    # 7. Environment: reward accounting matches coverage delta, every step
    env = MultiAgentCoverageEnv(size=8, horizon=15, k=3, seed=3)
    env.reset(); total_r = 0.0; c0 = env.total_coverage
    done = False
    while not done:
        a = np.random.default_rng().integers(0, 5, size=3)
        _, r, done, info = env.step(a)
        # env reward equals Shapley credit total at every step
        phi = shapley_exact(info["S_prev"], info["patches"], env.F)
        assert abs(phi.sum() - r) < 1e-9
        total_r += r
    assert abs((env.total_coverage - c0) - total_r) < 1e-9, \
        "telescoping F(S_T) - F(S_0) == sum r_t failed"

    if verbose:
        print("All unit tests passed:")
        print("  efficiency, null agent, symmetry (Shapley axioms i-iii)")
        print("  closed form == permutation form (cardinality and weighted)")
        print("  MC Shapley near exact; LOCAL double counts on overlap")
        print("  env reward telescopes: sum_t r_t = F(S_T) - F(S_0)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot_comparison(); sys.exit(0)

    run_unit_tests()

    if "--test" in sys.argv:
        sys.exit(0)

    if not TORCH_OK:
        print("\ntorch not available: unit tests passed, training skipped.")
        sys.exit(0)

    np.random.seed(42); torch.manual_seed(42)
    print(f"Device: {DEVICE}")

    SMOKE = "--smoke" in sys.argv
    if SMOKE:
        CFG = dict(n_episodes=30, warmup=200, eval_every=10,
                   n_eval=2, log_every=10, target_every=10)
        SIZE, HOR, KS = 8, 15, [2]
    else:
        CFG = dict(n_episodes=4000, eval_every=100, log_every=100)
        SIZE, HOR, KS = 12, 40, [2, 3]

    for k in KS:
        env_name = f"MA coverage {SIZE}x{SIZE}"
        factory = (lambda k=k: MultiAgentCoverageEnv(
            size=SIZE, horizon=HOR, k=k, patch=1))
        for mode in ["SHARED", "LOCAL", "VDN", "MACS"]:
            train_macs(factory, env_name=env_name, mode=mode, **CFG)

    print("\nDone. Run 'python macs_v1.py --plot' for figures.")