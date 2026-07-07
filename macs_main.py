"""
MACS v2 — Multi-Agent Coverage via Submodularity
Empirical validation of main_macs.tex (draft v0.6+) on grid coverage.

Changes from v1:
  + DR baseline        phi_i = F(S u all patches) - F(S u patches of others)
                       "difference rewards": marginal when arriving LAST.
                       Null-agent holds; efficiency violated DOWNWARD
                       (sum_i phi_i <= r_t, strict on overlap) — mirror of LOCAL.
  + MACS-CLIP (ours)   MACS credits + budget-clipped TD bootstrap:
                       boot <- min(boot, B(S')) with B(S') = F(V) - F(S').
                       Sound by Prop [Clipping is sound]: same fixed point,
                       same contraction, iterates confined to feasible region
                       after one application.
  + QMIX baseline      joint TD on Q_tot = g_s(Q_1..Q_k) with a monotone
                       mixing network (Rashid et al. 2018); hypernets condition
                       on the centralized state, mixer is TRAINING-ONLY and
                       discarded at execution.
  + Unit tests for the budget bound and DR <= Shapley <= LOCAL per agent
    (Remark [Ordering of credit rules under submodularity]).

Seven methods compared. All share the same per-agent networks, replay, and
update schedule; the experimental variables are the credit rule (SHARED /
LOCAL / DR / MACS), the mixing function (VDN / QMIX), and the backup
(MACS-CLIP):

  MACS       phi_i = exact Shapley credit (permutation form, Alg. 1)
  MACS-CLIP  MACS credits + clipped bootstrap                       proposed
  SHARED     phi_i = r_t   (every agent receives full team reward)  naive
  LOCAL      phi_i = F(S u P_i) - F(S)  first-arrival marginal, over-counts
  DR         phi_i = last-arrival marginal, under-counts on overlap
  VDN        joint TD on Q_tot = sum_i Q_i with team reward r_t
  QMIX       joint TD on Q_tot = g_s(Q_1..Q_k), monotone hypernet mixer;
             decomposition imposed by architecture, not derived from axioms

All methods use the decentralized parameterization Q_i(s^i, S): each agent
observes its own position and the shared coverage map ONLY (no other agents'
positions), matching the CTDE policy class of Prop. `Decentralized Execution
Sufficiency` in the draft.

Index convention (unchanged from v1):
  reset(): initial patches are covered WITHOUT reward,  S_0 = P(s_0).
  step t : agents move to s_{t+1},
           M = ( u_i P(s_{t+1}^i) ) minus S_t
           r = F(S_t u M) - F(S_t)
           S_{t+1} = S_t u M
  Shapley is computed over the coverage game
           v(C) = F(S_t u  u_{i in C} P(s_{t+1}^i)) - F(S_t).
  Budget stored with each transition: B(S_{t+1}) = F(V) - F(S_{t+1}),
  used only by MACS-CLIP to clip the bootstrap term of the TD target.

Usage:
  python macs_v2.py --test          run numpy-only unit tests (no torch)
  python macs_v2.py --smoke         tiny end-to-end run (2 min on CPU)
  python macs_v2.py                 full comparison, k in {2,3}
  python macs_v2.py --plot          plots from logs/macs_v2
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
# Environment — multi-agent coverage grid (unchanged from v1)
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

    def F_total(self):
        """F(V): value of the fully covered ground set. Used for B(S)."""
        return self.F(np.ones((self.size, self.size), dtype=bool))


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
    MACS / MACS-CLIP  exact Shapley for k<=4, MC with mc_m samples otherwise
                      (CLIP differs only in the TD backup, not the credits)
    SHARED            phi_i = r_team
    LOCAL             phi_i = F(S u P_i) - F(S)      first-arrival marginal
    DR                phi_i = F(S u all) - F(S u others)  last-arrival marginal
    (VDN and QMIX are not credit rules; both are handled in the loss.)

    For submodular F these obey, per agent and per step:
        DR <= Shapley <= LOCAL
    since the last marginal is the smallest, the first is the largest, and
    Shapley averages marginals over all positions (Remark credit-order).
    """
    if mode == "SHARED":
        return np.full(k, r_team)
    if mode == "LOCAL":
        S_prev = info["S_prev"]; f0 = F(S_prev)
        return np.array([F(S_prev | pm) - f0 for pm in info["patches"]])
    if mode == "DR":
        S_prev = info["S_prev"]; patches = info["patches"]
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
    if mode in ("MACS", "MACS-CLIP"):
        if k <= 4:
            return shapley_exact(info["S_prev"], info["patches"], F)
        return shapley_mc(info["S_prev"], info["patches"], F, mc_m, rng)
    if mode in ("VDN", "QMIX"):
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

    def _pos_maps_all(pos_norm_k, size, device):
        """(B,k,2) normalized positions -> (B,H,W) joint position map with
        ALL agents marked (value clamped at 1 where agents coincide).
        Centralized-state input for the QMIX hypernetworks only."""
        B, k, _ = pos_norm_k.shape
        idx = (pos_norm_k * size).long().clamp(0, size - 1)
        m = torch.zeros(B, size, size, device=device)
        for i in range(k):
            m[torch.arange(B), idx[:, i, 0], idx[:, i, 1]] = 1.0
        return m

    class QMixer(nn.Module):
        """
        QMIX mixing network (Rashid et al. 2018):
            Q_tot = w2(s)^T . ELU( w1(s)^T q + b1(s) ) + b2(s)
        with w1, w2 >= 0 (abs of hypernetwork outputs), so that
        dQ_tot / dQ_i >= 0: monotone mixing enforces IGM by construction.

        The hypernetworks condition on the CENTRALIZED state — coverage
        map, frontier, and the joint position map of all agents. This is
        training-only information, on the same footing as the Shapley
        computation in MACS (which also reads all arrival positions).
        The mixer is discarded at execution, so the executed policy class
        Q_i(s^i, S) is identical across all seven methods.
        """
        def __init__(self, k, map_size, embed=32, hyper_hidden=64):
            super().__init__()
            self.k = k; self.embed = embed
            cnn, cdim = _make_cnn(map_size, in_ch=3)
            self.state_enc = cnn
            self.hyper_w1 = nn.Sequential(
                nn.Linear(cdim, hyper_hidden), nn.ReLU(),
                nn.Linear(hyper_hidden, k * embed))
            self.hyper_b1 = nn.Linear(cdim, embed)
            self.hyper_w2 = nn.Sequential(
                nn.Linear(cdim, hyper_hidden), nn.ReLU(),
                nn.Linear(hyper_hidden, embed))
            self.hyper_b2 = nn.Sequential(
                nn.Linear(cdim, hyper_hidden), nn.ReLU(),
                nn.Linear(hyper_hidden, 1))

        def forward(self, q_agents, z_map, frontier, all_pos_map):
            """q_agents: (B, k) — each agent's Q at its chosen action."""
            s = self.state_enc(
                torch.stack([z_map, frontier, all_pos_map], dim=1))
            B = q_agents.shape[0]
            w1 = torch.abs(self.hyper_w1(s)).view(B, self.k, self.embed)
            b1 = self.hyper_b1(s).view(B, 1, self.embed)
            h  = nn.functional.elu(
                torch.bmm(q_agents.unsqueeze(1), w1) + b1)     # (B,1,embed)
            w2 = torch.abs(self.hyper_w2(s)).view(B, self.embed, 1)
            b2 = self.hyper_b2(s).view(B, 1, 1)
            return (torch.bmm(h, w2) + b2).view(B)

    class Replay:
        def __init__(self, cap): self.buf = deque(maxlen=cap)
        def push(self, tr):      self.buf.append(tr)
        def __len__(self):       return len(self.buf)
        def sample(self, bs, rng):
            idx = rng.integers(0, len(self.buf), size=bs)
            return [self.buf[i] for i in idx]

    def train_macs(
        env_factory, env_name, mode,   # MACS | MACS-CLIP | SHARED | LOCAL | DR | VDN
        n_episodes=4000, gamma=0.99, lr=5e-4,
        batch_size=64, buffer_cap=60_000, warmup=1_500,
        eps_start=1.0, eps_end=0.05, eps_frac=0.6,
        target_every=50,                            # hard update, episodes (C)
        eval_every=100, n_eval=8,
        label=None, log_dir="logs/macs_v2", log_every=100, seed=0,
    ):
        env = env_factory()
        k, size, H = env.k, env.size, env.horizon
        if label is None: label = f"{env_name.replace(' ','_')}_k{k}_{mode}"
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        clip_budget = (mode == "MACS-CLIP")
        F_total = env.F_total()          # F(V), fixed for the environment

        nets    = [AgentQNet(size).to(DEVICE) for _ in range(k)]
        targets = [AgentQNet(size).to(DEVICE) for _ in range(k)]
        for q, qt in zip(nets, targets):
            qt.load_state_dict(q.state_dict())
            for p in qt.parameters(): p.requires_grad_(False)
        opts = [optim.Adam(q.parameters(), lr=lr, eps=1e-5) for q in nets]

        mixer = target_mixer = mix_opt = None
        if mode == "QMIX":
            mixer = QMixer(k, size).to(DEVICE)
            target_mixer = QMixer(k, size).to(DEVICE)
            target_mixer.load_state_dict(mixer.state_dict())
            for p in target_mixer.parameters(): p.requires_grad_(False)
            mix_opt = optim.Adam(mixer.parameters(), lr=lr, eps=1e-5)

        buf = Replay(buffer_cap)

        print(f"\n{'='*62}\n  {label} | k={k} size={size} H={H} mode={mode}"
              f"{' (clipped backup)' if clip_budget else ''}"
              f"\n{'='*62}")

        log = {"label": label,
               "config": dict(method=mode, env_name=env_name, k=k,
                              map_size=size, horizon=H,
                              max_cells=env.max_cells, n_episodes=n_episodes,
                              clip_budget=clip_budget, F_total=F_total),
               "per_block": {key: [] for key in
                 ("episode", "coverage_mean", "eval_coverage",
                  "loss", "epsilon", "credit_sum_err", "clip_frac")},
               "summary": {}}
        hist = defaultdict(list)
        block_cov, block_loss, block_err, block_clip = [], [], [], []
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
                if mode in ("MACS", "MACS-CLIP"):
                    block_err.append(abs(phi.sum() - r_team))
                # remaining budget at the NEXT state: B(S') = F(V) - F(S')
                # (Lemma budget; used to clip the bootstrap, Remark clip-impl)
                bud2 = np.float32(F_total - env.F(env.covered))
                buf.push((pos_n.copy(), z.copy(), fr.copy(), acts.copy(),
                          phi.astype(np.float32), np.float32(r_team),
                          pos_n2.copy(), z2.copy(), fr2.copy(),
                          np.float32(done), bud2))
                pos_n, z, fr = pos_n2, z2, fr2
                gstep += 1

                # -- one gradient step per env step after warmup ----------
                if len(buf) >= warmup:
                    batch = buf.sample(batch_size, rng)
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
                    bbud= torch.tensor(np.stack([b[10] for b in batch]),
                                       device=DEVICE)                        # (B,)

                    if mode == "VDN":
                        # Q_tot = sum_i Q_i, single TD loss on team reward.
                        # NOTE: no budget clip here by design — VDN serves as
                        # the unclipped decomposition-by-architecture baseline;
                        # the bound B(S') is a MACS-side quantity (Lemma budget)
                        # and clipping VDN would confound the comparison.
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
                    elif mode == "QMIX":
                        # Q_tot = g_s(Q_1..Q_k), monotone mixer, joint TD on
                        # the team reward. Monotonicity makes the joint argmax
                        # decompose into per-agent argmaxes, so the target
                        # mixes each target net's per-agent max — the standard
                        # QMIX target. Like VDN, no budget clip by design:
                        # QMIX is the unclipped decomposition-by-architecture
                        # baseline.
                        pm_all  = _pos_maps_all(bp,  size, DEVICE)
                        pm_all2 = _pos_maps_all(bp2, size, DEVICE)
                        q_list = []
                        for i in range(k):
                            pmi = _pos_maps(bp[:, i], size, DEVICE)
                            q_list.append(nets[i](bp[:, i], bz, bf, pmi)
                                          .gather(1, ba[:, i:i+1]).squeeze(1))
                        q_tot = mixer(torch.stack(q_list, dim=1),
                                      bz, bf, pm_all)
                        with torch.no_grad():
                            y_list = []
                            for i in range(k):
                                pmi2 = _pos_maps(bp2[:, i], size, DEVICE)
                                y_list.append(targets[i](
                                    bp2[:, i], bz2, bf2, pmi2).max(1).values)
                            y_tot = target_mixer(torch.stack(y_list, dim=1),
                                                 bz2, bf2, pm_all2)
                            y = br + gamma * (1 - bd) * y_tot
                        loss = nn.functional.mse_loss(q_tot, y)
                        for o in opts: o.zero_grad()
                        mix_opt.zero_grad()
                        loss.backward()
                        for i in range(k):
                            nn.utils.clip_grad_norm_(nets[i].parameters(), 5.0)
                        nn.utils.clip_grad_norm_(mixer.parameters(), 5.0)
                        for o in opts: o.step()
                        mix_opt.step()
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
                                boot = targets[i](
                                    bp2[:, i], bz2, bf2, pmi2).max(1).values
                                if clip_budget:
                                    # y_i = phi_i + gamma*min(boot, B(S'))
                                    # Eq. (clip-target); sound by Prop clip.
                                    clipped = torch.minimum(boot, bbud)
                                    block_clip.append(
                                        (boot > bbud).float().mean().item())
                                    boot = clipped
                                yi = bphi[:, i] + gamma * (1 - bd) * boot
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
                if mixer is not None:
                    target_mixer.load_state_dict(mixer.state_dict())

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
                # fraction of bootstrap values the clip actually cut —
                # diagnostic for Remark clip-why (expect high early, decaying)
                pb["clip_frac"].append(
                    round(float(np.mean(block_clip)), 4) if block_clip else 0.0)
                print(f"  [{label}] ep {ep_i:5d} | "
                      f"cov {np.mean(block_cov):6.1f}/{env.max_cells} "
                      f"| eval {eval_cov_last:6.1f} | eps {epsilon:.2f}"
                      + (f" | clip {pb['clip_frac'][-1]:.2f}"
                         if clip_budget else ""))
                hist["coverage"].append(float(np.mean(block_cov)))
                block_cov, block_loss, block_err, block_clip = [], [], [], []
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

def plot_comparison(log_dir="logs/macs_v2", out_dir="plots/macs_v2"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib"); return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    COLOR = {"MACS": "#2ecc71", "MACS-CLIP": "#16a085",
             "VDN": "#3498db", "QMIX": "#34495e", "DR": "#9b59b6",
             "SHARED": "#e67e22", "LOCAL": "#e74c3c"}
    ORDER = ["SHARED", "LOCAL", "DR", "VDN", "QMIX", "MACS", "MACS-CLIP"]
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

    def team_r(S_prev, patches, F):
        union = S_prev.copy()
        for pm in patches: union |= pm
        return F(union) - F(S_prev)

    # 1. Efficiency: sum_i phi_i == r  (Shapley axiom i)
    for k in (2, 3, 4):
        for _ in range(50):
            S_prev, patches = rand_state(k)
            phi = shapley_exact(S_prev, patches, F_cardinality)
            r = team_r(S_prev, patches, F_cardinality)
            assert abs(phi.sum() - r) < 1e-9, "efficiency violated"

    # 2. Null agent: patch fully inside S_prev  ->  phi_i = 0  (axiom ii)
    S_prev, patches = rand_state(2)
    S_prev |= patches[1]                     # agent 2 covers nothing new
    phi = shapley_exact(S_prev, patches, F_cardinality)
    assert abs(phi[1]) < 1e-9, "null agent violated"

    # 3. Symmetry: two agents on the SAME new cell split it 1/2 each
    S0 = np.zeros((S, S), dtype=bool)
    m = np.zeros((S, S), dtype=bool); m[4, 4] = True
    phi = shapley_exact(S0, [m, m.copy()], F_cardinality)
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
    phi_local = np.array([F_cardinality(S0 | m) for m in [m, m]]) \
                - F_cardinality(S0)
    assert phi_local.sum() > 1.0, "LOCAL should double count here"

    # 7. DR: null agent holds; UNDER-counts on shared new cells
    #    (mirror image of LOCAL, Remark credit-order in the draft)
    info = {"S_prev": S0, "patches": [m, m.copy()]}
    phi_dr = compute_credits("DR", 1.0, info, F_cardinality, 2)
    # both agents arrive "last" onto a cell the other already covers -> 0
    assert abs(phi_dr[0]) < 1e-9 and abs(phi_dr[1]) < 1e-9
    assert phi_dr.sum() < 1.0 - 1e-9, "DR should under count here"
    S_prev, patches = rand_state(2)
    S_prev |= patches[1]                     # agent 2 covers nothing new
    info = {"S_prev": S_prev, "patches": patches}
    phi_dr = compute_credits("DR", 0.0, info, F_cardinality, 2)
    assert abs(phi_dr[1]) < 1e-9, "DR null agent violated"

    # 8. Credit ordering: DR <= Shapley <= LOCAL per agent, every state
    #    (Remark credit-order: last marginal <= average <= first marginal
    #     by submodularity / diminishing returns)
    for k in (2, 3, 4):
        for _ in range(50):
            S_prev, patches = rand_state(k)
            info = {"S_prev": S_prev, "patches": patches}
            r = team_r(S_prev, patches, F_cardinality)
            sh  = shapley_exact(S_prev, patches, F_cardinality)
            loc = compute_credits("LOCAL", r, info, F_cardinality, k)
            dr  = compute_credits("DR",    r, info, F_cardinality, k)
            assert np.all(dr <= sh + 1e-9),  "DR > Shapley somewhere"
            assert np.all(sh <= loc + 1e-9), "Shapley > LOCAL somewhere"
            # efficiency directions: DR under, LOCAL over, Shapley exact
            assert dr.sum() <= r + 1e-9 and loc.sum() >= r - 1e-9

    # 9. Budget bound (Lemma budget): remaining discounted credit from any
    #    reachable state never exceeds B(S_t) = F(V) - F(S_t), per agent,
    #    along random rollouts. This is the quantity MACS-CLIP clips against.
    env = MultiAgentCoverageEnv(size=8, horizon=15, k=2, seed=5)
    F_total = env.F_total()
    gamma = 0.99
    rng2 = np.random.default_rng(1)
    for _ in range(10):
        env.reset()
        S_list = [env.covered.copy()]
        phis = []
        done = False
        while not done:
            a = rng2.integers(0, 5, size=env.k)
            _, r, done, info = env.step(a)
            phis.append(shapley_exact(info["S_prev"], info["patches"], env.F))
            S_list.append(env.covered.copy())
        phis = np.array(phis)                       # (T, k)
        T = len(phis)
        for t in range(T):
            disc = gamma ** np.arange(T - t)
            remaining = (phis[t:] * disc[:, None]).sum(axis=0)   # (k,)
            B = F_total - env.F(S_list[t])
            assert np.all(remaining <= B + 1e-9), "budget bound violated"

    # 10. Environment: reward accounting matches coverage delta, every step
    env = MultiAgentCoverageEnv(size=8, horizon=15, k=3, seed=3)
    env.reset(); total_r = 0.0; c0 = env.total_coverage
    done = False
    while not done:
        a = np.random.default_rng().integers(0, 5, size=3)
        _, r, done, info = env.step(a)
        phi = shapley_exact(info["S_prev"], info["patches"], env.F)
        assert abs(phi.sum() - r) < 1e-9
        total_r += r
    assert abs((env.total_coverage - c0) - total_r) < 1e-9, \
        "telescoping F(S_T) - F(S_0) == sum r_t failed"

    if verbose:
        print("All unit tests passed:")
        print("  efficiency, null agent, symmetry (Shapley axioms i-iii)")
        print("  closed form == permutation form (cardinality and weighted)")
        print("  MC Shapley near exact")
        print("  LOCAL over-counts, DR under-counts, DR <= Shapley <= LOCAL")
        print("  budget bound: remaining discounted credit <= F(V) - F(S_t)")
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
        for mode in ["SHARED", "LOCAL", "DR", "VDN", "QMIX",
                     "MACS", "MACS-CLIP"]:
            train_macs(factory, env_name=env_name, mode=mode, **CFG)

    print("\nDone. Run 'python macs_v2.py --plot' for figures.")