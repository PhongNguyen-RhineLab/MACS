"""Probe (run as python -m probes.reach): per-agent reachability budget vs team budget B(S)=F(V)-F(S) on islands.
Checks (a) the bound is never violated by realized discounted Shapley streams,
(b) how much tighter it is, (c) how often the team clip B(S') could ever bind."""
import numpy as np, sys
sys.path.insert(0, '.')
from macs.envs.facility import IslandEnv
from macs.config import ISLANDS
from macs import shapley, policies as P

def ball(size, p, r):
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    return (np.abs(xs-p[0]) + np.abs(ys-p[1])) <= r

def reach_budget(env, S, p, tau, gamma):
    # B_i^gamma = sum_j gamma^{j-1} (H(j)-H(j-1)), H(j)=F(S u Ball(p,j))-F(S)
    f0 = env.F(S); H_prev = 0.0; out = 0.0
    for j in range(1, tau+1):
        H = env.F(S | ball(env.size, p, j)) - f0
        out += gamma**(j-1) * (H - H_prev); H_prev = H
    return out, H_prev

def run(k, pol_name, eps=5, gamma=0.99, seed=0):
    rng = np.random.default_rng(seed)
    make = dict(P.ISLAND_LADDER)[pol_name]
    viol, rows = 0, []
    for e in range(eps):
        env = IslandEnv(k=k, seed=seed+e, **{kk:v for kk,v in ISLANDS.items()})
        env.reset(); pol = make(env); T = env.horizon
        traj = []  # (pos copy, S copy, t, phi)
        while True:
            pos, S, t = env.pos.copy(), env.covered.copy(), env.steps
            a = pol(env, rng)
            _, r, done, info = env.step(a)
            phi = shapley.compute(info['S_prev'], info['patches'], env.F, backend='auto')
            traj.append((pos, S, t, phi))
            if done: break
        FV = env.F_total()
        for idx,(pos,S,t,phi) in enumerate(traj):
            tau = T - t
            Bteam = FV - env.F(S)
            for i in range(k):
                G = sum(gamma**(u-idx)*traj[u][3][i] for u in range(idx, len(traj)))
                Bg, Bu = reach_budget(env, S, pos[i], tau, gamma)
                if G > Bg + 1e-9: viol += 1
                rows.append((t, G, Bg, Bu, Bteam))
    a = np.array(rows)
    print(f"k={k} {pol_name:9s} n={len(a)} violations={viol} | "
          f"mean realized={a[:,1].mean():6.1f}  reach_gamma={a[:,2].mean():6.1f}  "
          f"reach_undisc={a[:,3].mean():6.1f}  team B(S)={a[:,4].mean():6.1f} | "
          f"reach/team={np.mean(a[:,2]/np.maximum(a[:,4],1e-9)):.2f}  "
          f"late(t>=20) reach/team={np.mean((a[a[:,0]>=20,2])/np.maximum(a[a[:,0]>=20,4],1e-9)):.2f}  "
          f"frac reach==0: {np.mean(a[:,2]<1e-9):.2f}")

for k in (3, 6):
    for pn in ("random", "dec", "plan"):
        run(k, pn)
