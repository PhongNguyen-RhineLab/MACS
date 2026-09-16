"""
Networks, replay, and the tensor helpers. Torch is optional: importing this
module without torch installed succeeds and sets TORCH_OK = False, so the
numpy-only probes and tests keep working.

All methods share the decentralized parameterization Q_i(s^i, S): each
agent observes its own position and a coverage map only, never another
agent's position. That is the policy class of Prop. `Decentralized
Execution Sufficiency`. The QMIX mixer is the single exception, and it
conditions on the centralized state for TRAINING ONLY and is discarded at
execution, so the executed policy class is identical across every method.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:                                  # pragma: no cover
    torch = nn = optim = None
    TORCH_OK = False
    DEVICE = None


def require_torch(what="this operation"):
    if not TORCH_OK:
        raise RuntimeError(
            f"{what} needs PyTorch, which is not installed. "
            "pip install -r requirements.txt")


class Replay:
    """Uniform replay over transition tuples. RNG comes from the caller so
    that the sampling order is part of the run's single seeded stream."""

    def __init__(self, cap):
        from collections import deque
        self.buf = deque(maxlen=cap)

    def push(self, tr):
        self.buf.append(tr)

    def __len__(self):
        return len(self.buf)

    def sample(self, bs, rng):
        idx = rng.integers(0, len(self.buf), size=bs)
        return [self.buf[i] for i in idx]


if TORCH_OK:

    def make_cnn(map_size, in_ch=3):
        cnn = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, 1, 1), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            cdim = cnn(torch.zeros(1, in_ch, map_size, map_size)).shape[1]
        return cnn, cdim

    def orthogonal_init(m):
        for layer in m.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    class AgentQNet(nn.Module):
        """
        Q_i(s^i, S, .) -- decentralized input:
          CNN channels: [coverage map, frontier, own-position one-hot]
          MLP branch:   own position (2,) normalized
        No other agents' positions anywhere.
        """

        def __init__(self, map_size, n_actions=5):
            super().__init__()
            self.map_size = map_size
            cnn, cdim = make_cnn(map_size, in_ch=3)
            self.cnn = cnn
            self.pos = nn.Sequential(nn.Linear(2, 32), nn.ReLU(),
                                     nn.Linear(32, 32), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(cdim + 32, 128), nn.ReLU(),
                                      nn.Linear(128, 64), nn.ReLU(),
                                      nn.Linear(64, n_actions))
            orthogonal_init(self)
            nn.init.orthogonal_(self.head[-1].weight, 1.0)

        def forward(self, pos, z_map, frontier, pos_map):
            x = torch.stack([z_map, frontier, pos_map], dim=1)
            return self.head(torch.cat([self.cnn(x), self.pos(pos)], dim=1))

    class QMixer(nn.Module):
        """
        QMIX mixing network (Rashid et al. 2018):
            Q_tot = w2(s)^T . ELU( w1(s)^T q + b1(s) ) + b2(s)
        with w1, w2 >= 0 (abs of hypernetwork outputs), so dQ_tot/dQ_i >= 0:
        monotone mixing enforces IGM by construction.

        The hypernetworks condition on the CENTRALIZED state -- coverage
        map, frontier, and the joint position map of all agents. This is
        training-only information, on the same footing as the Shapley
        computation in MACS (which also reads all arrival positions).
        """

        def __init__(self, k, map_size, embed=32, hyper_hidden=64):
            super().__init__()
            self.k = k
            self.embed = embed
            cnn, cdim = make_cnn(map_size, in_ch=3)
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
            """q_agents: (B, k) -- each agent's Q at its chosen action."""
            s = self.state_enc(
                torch.stack([z_map, frontier, all_pos_map], dim=1))
            B = q_agents.shape[0]
            w1 = torch.abs(self.hyper_w1(s)).view(B, self.k, self.embed)
            b1 = self.hyper_b1(s).view(B, 1, self.embed)
            h = nn.functional.elu(
                torch.bmm(q_agents.unsqueeze(1), w1) + b1)     # (B,1,embed)
            w2 = torch.abs(self.hyper_w2(s)).view(B, self.embed, 1)
            b2 = self.hyper_b2(s).view(B, 1, 1)
            return (torch.bmm(h, w2) + b2).view(B)

    def pos_maps(pos_norm, size, device):
        """(B,2) normalized positions -> (B,H,W) one-hot maps."""
        B = pos_norm.shape[0]
        idx = (pos_norm * size).long().clamp(0, size - 1)
        m = torch.zeros(B, size, size, device=device)
        m[torch.arange(B), idx[:, 0], idx[:, 1]] = 1.0
        return m

    def pos_maps_all(pos_norm_k, size, device):
        """
        (B,k,2) normalized positions -> (B,H,W) joint position map with ALL
        agents marked (clamped at 1 where agents coincide). Centralized-state
        input for the QMIX hypernetworks only.
        """
        B, k, _ = pos_norm_k.shape
        idx = (pos_norm_k * size).long().clamp(0, size - 1)
        m = torch.zeros(B, size, size, device=device)
        for i in range(k):
            m[torch.arange(B), idx[:, i, 0], idx[:, i, 1]] = 1.0
        return m
