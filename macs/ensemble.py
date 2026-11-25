"""
All k agent networks evaluated and updated in one batched call.

The trainer's reference path loops over agents: k forward passes to act, and
for each learning step k separate forward/backward/Adam updates. Every one of
those is a handful of tiny kernels, so on a GPU the run is launch-bound and on
a CPU it pays Python and dispatch overhead k times. This module stacks the k
AgentQNet parameter sets along a leading agent axis (torch.func) and runs them
under vmap, so each of those operations happens once per step for the whole
team.

It is the same model and the same update, written in batched form:
  * initial weights are copied from the k individually constructed nets, so
    a run starts from exactly the weights the reference path would use;
  * the agents' parameters are disjoint, so the gradient of the summed
    per-agent losses with respect to agent i's parameters is the gradient of
    agent i's own loss;
  * Adam is elementwise and every agent steps once per learning call, so one
    Adam over the stacked tensors is k independent Adams;
  * gradient-norm clipping is computed per agent, matching
    nn.utils.clip_grad_norm_(net_i.parameters(), max_norm).
Results agree with the reference path up to floating-point summation order
(tests/test_ensemble.py checks one update step to tolerance), not bit for
bit, so a fused run is a statistically equivalent rerun, not a replay.
"""

import copy

import torch
from torch.func import functional_call, stack_module_state, vmap


class AgentEnsemble:
    """k AgentQNets with stacked parameters. q() returns (k, B, n_actions)."""

    def __init__(self, nets, requires_grad=True):
        self.k = len(nets)
        self.time_input = nets[0].time_input
        params, buffers = stack_module_state(nets)
        self.params = {n: p.detach().clone().requires_grad_(requires_grad)
                       for n, p in params.items()}
        self.buffers = {n: b.detach().clone() for n, b in buffers.items()}
        self._base = copy.deepcopy(nets[0]).to("meta")

        def one(p, b, pos, z, fr, pm, tau):
            return functional_call(self._base, (p, b),
                                   (pos, z, fr, pm, tau))

        self._vq = vmap(one, in_dims=(0, 0, 0, 0, 0, 0, None))

    # -- evaluation ----------------------------------------------------------
    def q(self, pos, z, fr, pm, tau=None):
        """pos (k,B,2), z/fr/pm (k,B,H,W), tau (B,) or None."""
        return self._vq(self.params, self.buffers, pos, z, fr, pm, tau)

    # -- optimisation ----------------------------------------------------------
    def parameters(self):
        return list(self.params.values())

    def clip_grad_norm_(self, max_norm, eps=1e-6):
        """Per-agent global-norm clipping, as clip_grad_norm_ on each net."""
        grads = [p.grad for p in self.params.values() if p.grad is not None]
        if not grads:
            return
        sq = torch.zeros(self.k, device=grads[0].device, dtype=grads[0].dtype)
        for g in grads:
            sq = sq + g.reshape(self.k, -1).pow(2).sum(1)
        coef = torch.clamp(max_norm / (sq.sqrt() + eps), max=1.0)
        for g in grads:
            g.mul_(coef.view(-1, *([1] * (g.dim() - 1))))

    # -- copying ---------------------------------------------------------------
    @torch.no_grad()
    def copy_from(self, other):
        for n, p in self.params.items():
            p.copy_(other.params[n])

    @torch.no_grad()
    def write_back(self, nets):
        """Copy the stacked weights into the per-agent modules."""
        for i, net in enumerate(nets):
            for n, p in net.named_parameters():
                p.copy_(self.params[n][i])
