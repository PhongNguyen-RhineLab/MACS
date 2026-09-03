"""
probe_ceiling.py -- two questions that can change a GPU decision.

  A. What is the actual ceiling of F-free policies? probe_free.py used a
     serpentine with rows spaced by rho, which was a guess. If tuning the
     spacing raises `sweep` substantially, the bar that extended MACS
     training must clear moves and the experiment may not be worth running.

  B. Sharpen the private-arm prediction. probe_cells.py inverted F = 152.58
     to |S_H| ~ 39 using random-walk layout. The collapse hypothesis says
     agents take a few steps then stall, which is a different geometry:
     k short self-avoiding walks, then nothing. Calibrate against that.
"""
import numpy as np
from macs_fl import FacilityCoverageEnv
from probe_fl import _next_pos

FAC = dict(size=24, horizon=60, rho=4.0, patch=0, demand="uniform")
MACS_SHARED = {2: 232.75, 3: 300.75, 4: 352.71, 6: 419.12, 8: 457.91}
FRONT = {2: 235.9, 3: 311.7, 4: 369.3, 6: 441.3, 8: 464.3}


def make_sweep(step, vertical=True):
    def mk(env):
        n, k = env.size, env.k
        wpts = []
        for i in range(k):
            a0 = int(round(i * n / k)); a1 = int(round((i + 1) * n / k)) - 1
            lanes = list(range(a0 + step // 2, a1 + 1, step)) or [a0]
            path, up = [], True
            for c in lanes:
                ys = list(range(step // 2, n, step))
                seg = [(c, y) for y in (ys if up else ys[::-1])]
                path += seg if vertical else [(y, x) for x, y in seg]
                up = not up
            wpts.append(path)
        idx = [0] * k

        def pol(env, rng):
            acts = np.zeros(env.k, dtype=np.int64)
            for i in range(env.k):
                path = wpts[i]
                if idx[i] >= len(path): idx[i] = 0
                tx, ty = path[idx[i]]
                if tuple(env.pos[i]) == (tx, ty):
                    idx[i] += 1
                    tx, ty = path[min(idx[i], len(path) - 1)]
                best, ba = None, 0
                for a in range(env.n_actions):
                    p = _next_pos(env, i, a)
                    d = abs(p[0] - tx) + abs(p[1] - ty)
                    if best is None or d < best: best, ba = d, a
                acts[i] = ba
            return acts
        return pol
    return mk


def rollout(cfg, mk, eps=8, seed=0):
    out = []
    for e in range(eps):
        rng = np.random.default_rng(seed * 7919 + e)
        env = FacilityCoverageEnv(**cfg, seed=seed * 1000 + e); env.reset()
        pol = mk(env)
        for _ in range(env.horizon): env.step(pol(env, rng))
        out.append(env.total_coverage)
    return float(np.mean(out))


print("A. sweep spacing sweep, uniform 24x24 H=60 (rho = 4)")
hdr = "  k " + " ".join(f"{'s=%d'%s:>8}" for s in (2,3,4,5,6)) + \
      f" {'best':>8} {'front':>8} {'MACS-sh':>8} {'best-MACS':>10}"
print(hdr); print("-"*len(hdr))
ceiling = {}
for k in (2,3,4,6,8):
    cfg = dict(k=k, **FAC); row = {}
    for s in (2,3,4,5,6):
        row[s] = max(rollout(cfg, make_sweep(s, True)),
                     rollout(cfg, make_sweep(s, False)))
    b = max(row.values()); ceiling[k] = max(b, FRONT[k])
    print(f"{k:>3} " + " ".join(f"{row[s]:>8.1f}" for s in (2,3,4,5,6)) +
          f" {b:>8.1f} {FRONT[k]:>8.1f} {MACS_SHARED[k]:>8.1f} "
          f"{ceiling[k]-MACS_SHARED[k]:>10.1f}")

print("\nB. stall calibration at k=6: agents self-avoid for L steps then stop")
print(f"{'L':>4} {'|S_H|':>7} {'F':>8}")
print("-"*22)
best = None
for L in (2,3,4,5,6,8,10,14,20):
    fs, cs = [], []
    for e in range(8):
        rng = np.random.default_rng(555+e)
        env = FacilityCoverageEnv(**dict(k=6, **FAC), seed=e); env.reset()
        own = [env._patch_mask(env.pos[i]).copy() for i in range(6)]
        for t in range(env.horizon):
            acts = np.zeros(6, dtype=np.int64)
            if t < L:
                for i in range(6):
                    cand = [a for a in range(env.n_actions)
                            if not own[i][_next_pos(env, i, a)]]
                    acts[i] = rng.choice(cand) if cand else 0
            env.step(acts)
            for i in range(6): own[i] |= env._patch_mask(env.pos[i])
        fs.append(env.total_coverage); cs.append(int(env.covered.sum()))
    f, c = float(np.mean(fs)), float(np.mean(cs))
    print(f"{L:>4} {c:>7.1f} {f:>8.1f}")
    if best is None or abs(f-152.58) < abs(best[1]-152.58): best = (L, f, c)
print(f"\nprivate F=152.58 matches L~{best[0]} stall steps, "
      f"|S_H|~{best[2]:.0f} ({best[2]/6:.1f} cells/agent)")