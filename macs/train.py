"""
The trainer. One function for every arm the paper reports.

Before this refactor there were two trainers: macs_main.train_macs (VDN and
QMIX, shared observation only, budget clip only) and
macs_ablation.train_ablation (observation modes, constant clip, no mixing
baselines). They shared roughly 80% of their code and drifted: the
credit_sum_err column was logged for every credit rule in one and only for
MACS in the other, which made every non-MACS arm read as efficiency-exact.

The merge is behaviour-preserving. Both trainers consumed the single seeded
RNG stream in the same order -- epsilon-greedy draws per agent, then any
Monte Carlo Shapley draws, then the replay sample -- and constructed
networks in the same order under the same torch seed, so a credit-mode run
here reproduces either predecessor seed for seed. The observation tensors
agree too: the old shared path fed z directly and this one round-trips it
through int8, which is lossless for every map value the environments emit.

Axes:
  mode      MACS | MACS-CLIP | MACS-MC | SHARED | LOCAL | DR | VDN | QMIX
  obs_mode  shared | private                       (O9)
  clip      none | budget | const                  (O10)
  backend   auto | facility | permutation | subset | mc   (Shapley backend)

The budget clip replaces the bootstrap with min(boot, ceiling):
  budget  ceiling = B(S') = F(V) - F(S')   state-dependent (Lemma `budget`)
  const   ceiling = c, the mean of B(S') over the warmup window, then
          frozen -- the state-INDEPENDENT control for O10
Sound by Prop. `Clipping is sound`: same fixed point, same contraction,
iterates confined to the feasible region after one application.

VDN and QMIX never clip, by design: the bound is a MACS-side quantity and
clipping the decomposition-by-architecture baselines would confound the
comparison.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from macs import credits as C
from macs import nets as N
from macs import shapley
from macs.obs import ObsTracker
from macs.runlog import RunLog

CLIP_MODES = ("none", "budget", "const")


def _default_clip(mode):
    """MACS-CLIP means the budget clip unless the caller says otherwise."""
    return "budget" if mode == "MACS-CLIP" else "none"


def train(
    env_factory, env_name, mode="MACS",
    obs_mode="shared", clip=None, backend="auto", mc_m=shapley.MC_M,
    n_episodes=4000, gamma=0.99, lr=5e-4,
    batch_size=64, buffer_cap=40_000, warmup=1_500,
    eps_start=1.0, eps_end=0.05, eps_frac=0.6,
    target_every=50, eval_every=100, n_eval=8,
    label=None, log_dir="logs/macs", log_every=100, seed=0,
    grad_clip=5.0, diag=False, diag_episodes=32, credit_fn=None,
):
    """Train one arm and write its log. Returns the log dict."""
    # Dependency injection rather than module mutation: the scalability
    # sweep wraps this to time credit computation without patching anything.
    credit_fn = credit_fn or C.compute_credits

    if clip is None:
        clip = _default_clip(mode)
    if mode not in C.ALL_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected {C.ALL_MODES}")
    if clip not in CLIP_MODES:
        raise ValueError(f"clip must be one of {CLIP_MODES}, got {clip!r}")
    if mode in C.MIXING_MODES:
        if obs_mode != "shared":
            raise ValueError(
                f"{mode} mixes over the centralized state, so a private "
                "observation is not meaningful for it.")
        if clip != "none":
            raise ValueError(
                f"{mode} is the unclipped decomposition-by-architecture "
                "baseline; clipping it would confound the O10 comparison.")
    if clip == "budget" and obs_mode == "private":
        raise ValueError(
            "clip='budget' needs B(S') from the shared set, which a "
            "private-observation agent cannot see. Use clip='none' for the "
            "O9 ablation so the observation is the only difference.")

    # Every argument is checked above, so an impossible sweep fails in the
    # first second rather than after a network is built.
    N.require_torch("training")
    import torch
    import torch.nn as nn
    import torch.optim as optim

    env = env_factory()
    k, size, H = env.k, env.size, env.horizon
    if label is None:
        label = (f"{env_name.replace(' ', '_')}_k{k}_{mode}"
                 f"_{obs_mode}_clip-{clip}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    F_total = env.F_total()                      # F(V), fixed per environment
    resolved_backend = (shapley.resolve(env.F, k, backend)
                        if mode in C.SHAPLEY_MODES else None)

    nets = [N.AgentQNet(size).to(N.DEVICE) for _ in range(k)]
    targets = [N.AgentQNet(size).to(N.DEVICE) for _ in range(k)]
    for q, qt in zip(nets, targets):
        qt.load_state_dict(q.state_dict())
        for p in qt.parameters():
            p.requires_grad_(False)
    opts = [optim.Adam(q.parameters(), lr=lr, eps=1e-5) for q in nets]

    mixer = target_mixer = mix_opt = None
    if mode == "QMIX":
        mixer = N.QMixer(k, size).to(N.DEVICE)
        target_mixer = N.QMixer(k, size).to(N.DEVICE)
        target_mixer.load_state_dict(mixer.state_dict())
        for p in target_mixer.parameters():
            p.requires_grad_(False)
        mix_opt = optim.Adam(mixer.parameters(), lr=lr, eps=1e-5)

    buf = N.Replay(buffer_cap)
    const_c, const_acc = None, []       # constant clip level for O10

    run = RunLog(label, dict(
        method=mode, env_name=env_name, k=k, map_size=size, horizon=H,
        max_cells=float(env.max_cells), n_episodes=n_episodes,
        obs_mode=obs_mode, clip=clip, seed=seed, lr=lr, gamma=gamma,
        batch_size=batch_size, F_total=F_total,
        shapley_backend=resolved_backend,
        mc_m=(mc_m if resolved_backend == "mc" else None),
    ), log_dir=log_dir)

    print(f"\n{'=' * 66}\n  {label}\n  k={k} size={size} H={H} mode={mode} "
          f"obs={obs_mode} clip={clip}"
          + (f" shapley={resolved_backend}" if resolved_backend else "")
          + f"\n{'=' * 66}")

    block = defaultdict(list)
    eval_cov_last = 0.0

    # -- acting ------------------------------------------------------------
    def greedy_actions(pos_n, zs, frs, epsilon, nets_=None):
        nets_ = nets_ or nets
        acts = np.zeros(k, dtype=np.int64)
        pt = torch.tensor(pos_n, dtype=torch.float32, device=N.DEVICE)
        zt = torch.tensor(zs, dtype=torch.float32, device=N.DEVICE)
        ft = torch.tensor(frs, dtype=torch.float32, device=N.DEVICE)
        pm = N.pos_maps(pt, size, N.DEVICE)
        with torch.no_grad():
            for i in range(k):
                if rng.random() < epsilon:
                    acts[i] = rng.integers(env.n_actions)
                else:
                    q = nets_[i](pt[i:i + 1], zt[i:i + 1], ft[i:i + 1],
                                 pm[i:i + 1])
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

    # -- batch helpers -----------------------------------------------------
    def stack(batch, j, dtype=torch.float32):
        return torch.tensor(np.stack([b[j] for b in batch]), dtype=dtype,
                            device=N.DEVICE)

    def agent_q(net, bp, bz, bf, i, actions=None):
        """Q_i over the batch; gathered at `actions` if given, else max."""
        pm = N.pos_maps(bp[:, i], size, N.DEVICE)
        q = net(bp[:, i], bz[:, i], bf[:, i], pm)
        if actions is None:
            return q.max(1).values
        return q.gather(1, actions[:, i:i + 1]).squeeze(1)

    def step_optimizers(loss, extra=()):
        for o in opts:
            o.zero_grad()
        for o in extra:
            o.zero_grad()
        loss.backward()
        for q in nets:
            nn.utils.clip_grad_norm_(q.parameters(), grad_clip)
        if mixer is not None:
            nn.utils.clip_grad_norm_(mixer.parameters(), grad_clip)
        for o in opts:
            o.step()
        for o in extra:
            o.step()

    # -- learning ----------------------------------------------------------
    def learn(batch):
        nonlocal const_c
        bp, bz, bf = stack(batch, 0), stack(batch, 1), stack(batch, 2)
        ba = stack(batch, 3, torch.long)
        bphi, br = stack(batch, 4), stack(batch, 5)
        bp2, bz2, bf2 = stack(batch, 6), stack(batch, 7), stack(batch, 8)
        bd, bbud = stack(batch, 9), stack(batch, 10)

        if mode == "VDN":
            # Q_tot = sum_i Q_i, single TD loss on the team reward.
            q_sum = sum(agent_q(nets[i], bp, bz, bf, i, ba) for i in range(k))
            with torch.no_grad():
                y_sum = sum(agent_q(targets[i], bp2, bz2, bf2, i)
                            for i in range(k))
                y = br + gamma * (1 - bd) * y_sum
            loss = nn.functional.mse_loss(q_sum, y)
            step_optimizers(loss)
            return loss.item()

        if mode == "QMIX":
            # Monotonicity makes the joint argmax decompose into per-agent
            # argmaxes, so the target mixes each target net's per-agent max.
            pm_all = N.pos_maps_all(bp, size, N.DEVICE)
            pm_all2 = N.pos_maps_all(bp2, size, N.DEVICE)
            q_list = [agent_q(nets[i], bp, bz, bf, i, ba) for i in range(k)]
            q_tot = mixer(torch.stack(q_list, dim=1),
                          bz[:, 0], bf[:, 0], pm_all)
            with torch.no_grad():
                y_list = [agent_q(targets[i], bp2, bz2, bf2, i)
                          for i in range(k)]
                y_tot = target_mixer(torch.stack(y_list, dim=1),
                                     bz2[:, 0], bf2[:, 0], pm_all2)
                y = br + gamma * (1 - bd) * y_tot
            loss = nn.functional.mse_loss(q_tot, y)
            step_optimizers(loss, extra=(mix_opt,))
            return loss.item()

        # independent per-agent TD with credit phi_i (Alg. 2)
        losses = []
        for i in range(k):
            qi = agent_q(nets[i], bp, bz, bf, i, ba)
            with torch.no_grad():
                boot = agent_q(targets[i], bp2, bz2, bf2, i)
                if clip != "none":
                    ceil = (bbud if clip == "budget"
                            else torch.full_like(boot, const_c))
                    block["clip"].append((boot > ceil).float().mean().item())
                    boot = torch.minimum(boot, ceil)
                yi = bphi[:, i] + gamma * (1 - bd) * boot
            li = nn.functional.mse_loss(qi, yi)
            opts[i].zero_grad()
            li.backward()
            nn.utils.clip_grad_norm_(nets[i].parameters(), grad_clip)
            opts[i].step()
            losses.append(li.item())
        return float(np.mean(losses))

    # -- episode loop ------------------------------------------------------
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

            phi = credit_fn(mode, r_team, info, env.F, k,
                            backend=backend, mc_m=mc_m, rng=rng)
            if mode in C.CREDIT_MODES:
                # Measured for EVERY credit rule, not just Shapley: zero for
                # Shapley by construction, (k-1)*r_t for SHARED, and
                # overlap-dependent for LOCAL and DR.
                block["err"].append(C.efficiency_error(phi, r_team))

            # remaining budget at the NEXT state: B(S') = F(V) - F(S')
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
                    print(f"  [{label}] constant clip level c = {const_c:.3f} "
                          f"(mean B(S') over warmup, n={len(const_acc)})")
                    run.config["const_clip_level"] = const_c
                block["loss"].append(learn(buf.sample(batch_size, rng)))

        block["cov"].append(env.total_coverage)

        if ep_i % target_every == 0:
            for q, qt in zip(nets, targets):
                qt.load_state_dict(q.state_dict())
            if mixer is not None:
                target_mixer.load_state_dict(mixer.state_dict())

        if ep_i % eval_every == 0:
            eval_cov_last = run_eval()

        if ep_i % log_every == 0:
            run.append_block(episode=ep_i, block=block,
                             eval_coverage=eval_cov_last, epsilon=epsilon)
            print(f"  [{label}] ep {ep_i:5d} | "
                  f"F {np.mean(block['cov']):7.2f}/{env.max_cells:.0f} "
                  f"| eval {eval_cov_last:7.2f} | eps {epsilon:.2f}"
                  + (f" | clip {np.mean(block['clip']):.3f}"
                     if block["clip"] else ""))
            block = defaultdict(list)
            run.save()

    run.finish()
    if diag:
        run.log["diag"] = _run_diagnostic(env_factory, nets, size, obs_mode,
                                          k, H, diag_episodes, seed, label)
    run.save()
    print(f"  -> {run.path} | last-10 eval {run.log['summary']['last10_eval']}")
    return run.log


def _run_diagnostic(env_factory, nets, size, obs_mode, k, H,
                    episodes, seed, label):
    """
    O9 collapse diagnostic. Runs after training on greedy rollouts, so it
    cannot perturb the training RNG and the run still reproduces seed for
    seed against existing logs.

    The diagnostic module is optional and was never committed to this repo;
    the previous code did a bare `from diag_o9 import ...` inside the
    training function, so any --diag run died with an ImportError several
    hours in. Failing here with a clear message costs nothing and the run
    is already saved.
    """
    try:
        from macs.diagnostics import diagnose, torch_qfn, report
    except ImportError as exc:                          # pragma: no cover
        print(f"  [{label}] diag requested but macs.diagnostics is "
              f"unavailable ({exc}); training results are saved.")
        return None
    d = diagnose(env_factory, torch_qfn(nets, size), obs_mode, k,
                 episodes=episodes, seed=seed)
    d["cells_per_agent_ceiling"] = H + 1
    report(d, label)
    return d
