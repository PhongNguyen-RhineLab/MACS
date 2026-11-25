"""
The fused ensemble path (train(..., fused=True)) against the reference
per-agent loop. With epsilon pinned at 1 both paths act at random through the
same RNG draws, so they see identical replay data and identical minibatches;
the per-episode TD loss and clip statistics must then agree to floating-point
tolerance for every learning step of the run.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from macs.envs import IslandEnv  # noqa: E402
from macs.ensemble import AgentEnsemble  # noqa: E402
from macs import nets as N  # noqa: E402


def tiny(k=3):
    def make(seed=None):
        return IslandEnv(size=12, horizon=8, k=k, rho=3.0, n_islands=3,
                         radius=1, min_sep=4, eps=0.0, layout_seed=0,
                         depot=True, seed=seed)
    return make


CFG = dict(n_episodes=12, warmup=40, batch_size=16, eval_every=6, n_eval=1,
           log_every=1, target_every=4, eps_start=1.0, eps_end=1.0, seed=3)


@pytest.mark.parametrize("mode,clip,tau", [
    ("MACS", None, True), ("MACS-CLIP", "reach", True),
    ("MACS-CLIP", "budget", False), ("LOCAL", None, True),
    ("VDN", None, True), ("QMIX", None, True), ("SHARED", None, False)])
def test_fused_matches_reference(tmp_path, mode, clip, tau):
    from macs.train import train
    logs = [train(tiny(), env_name="E", mode=mode, clip=clip, time_obs=tau,
                  fused=f, log_dir=tmp_path / str(f), **CFG)
            for f in (False, True)]
    a, b = (lg["per_block"] for lg in logs)
    la = [x for x in a["loss"] if x is not None]
    lb = [x for x in b["loss"] if x is not None]
    assert len(la) == len(lb) > 5
    # the first learning episodes agree to float32 round-off; later ones
    # drift as round-off is amplified by Adam's normalisation of tiny early
    # gradients (the exact-arithmetic check is test_update_is_exact_in_f64)
    np.testing.assert_allclose(la[:3], lb[:3], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(la, lb, rtol=5e-3, atol=1e-4)
    assert a["coverage_mean"] == b["coverage_mean"]          # same data
    for key in ("clip_frac", "floor_frac", "reach_ratio"):
        np.testing.assert_allclose(
            [np.nan if x is None else x for x in a[key]],
            [np.nan if x is None else x for x in b[key]], rtol=1e-4, atol=1e-5)
    assert logs[1]["config"]["fused"] is True


def test_ensemble_forward_and_clip_match_per_agent():
    torch.manual_seed(0)
    nets = [N.AgentQNet(12, time_input=True) for _ in range(4)]
    E = AgentEnsemble(nets)
    B = 5
    pos = torch.rand(4, B, 2)
    z, fr = torch.rand(4, B, 12, 12), torch.rand(4, B, 12, 12)
    pm = torch.rand(4, B, 12, 12)
    tau = torch.rand(B)
    q = E.q(pos, z, fr, pm, tau)
    for i, net in enumerate(nets):
        torch.testing.assert_close(q[i], net(pos[i], z[i], fr[i], pm[i], tau),
                                   rtol=1e-5, atol=1e-6)
    # per-agent gradient clipping equals clip_grad_norm_ on each net
    w = torch.randn(4, B, 5)
    (q * w).sum().backward()
    for i, net in enumerate(nets):
        (net(pos[i], z[i], fr[i], pm[i], tau) * w[i]).sum().backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
    E.clip_grad_norm_(0.1)
    for i, net in enumerate(nets):
        for n, p in net.named_parameters():
            torch.testing.assert_close(E.params[n].grad[i], p.grad,
                                       rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("loss_kind", ["per_agent", "team_sum"])
def test_update_is_exact_in_f64(loss_kind):
    """One full update (forward, loss, backward, per-agent clip, Adam) in
    float64: the fused and the per-agent paths give the same parameters to
    1e-10. per_agent is the credit-rule loss, team_sum the VDN loss."""
    torch.manual_seed(1)
    k, B = 3, 7
    nets = [N.AgentQNet(12, time_input=True).double() for _ in range(k)]
    E = AgentEnsemble(nets)
    opts = [torch.optim.Adam(n.parameters(), lr=1e-3, eps=1e-5) for n in nets]
    eopt = torch.optim.Adam(E.parameters(), lr=1e-3, eps=1e-5)
    g = torch.Generator().manual_seed(2)
    pos = torch.rand(k, B, 2, generator=g, dtype=torch.float64)
    z = torch.rand(k, B, 12, 12, generator=g, dtype=torch.float64)
    fr = torch.rand(k, B, 12, 12, generator=g, dtype=torch.float64)
    pm = torch.rand(k, B, 12, 12, generator=g, dtype=torch.float64)
    tau = torch.rand(B, generator=g, dtype=torch.float64)
    act = torch.randint(0, 5, (k, B), generator=g)
    y = 30 * torch.rand(k, B, generator=g, dtype=torch.float64)
    for _ in range(3):                      # three steps, so Adam state matters
        qi = torch.stack([nets[i](pos[i], z[i], fr[i], pm[i], tau)
                          .gather(1, act[i][:, None]).squeeze(1)
                          for i in range(k)])
        qe = E.q(pos, z, fr, pm, tau).gather(2, act[..., None]).squeeze(-1)
        if loss_kind == "per_agent":
            ref = ((qi - y) ** 2).mean(1).sum()
            fus = ((qe - y) ** 2).mean(1).sum()
        else:
            ref = ((qi.sum(0) - y.sum(0)) ** 2).mean()
            fus = ((qe.sum(0) - y.sum(0)) ** 2).mean()
        for o in opts:
            o.zero_grad()
        eopt.zero_grad()
        ref.backward()
        fus.backward()
        for n in nets:
            torch.nn.utils.clip_grad_norm_(n.parameters(), 5.0)
        E.clip_grad_norm_(5.0)
        for o in opts:
            o.step()
        eopt.step()
    for i, net in enumerate(nets):
        for n, p in net.named_parameters():
            torch.testing.assert_close(E.params[n][i], p, rtol=1e-10,
                                       atol=1e-12)
