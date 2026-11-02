# MACS — Multi-Agent Coverage via Submodularity

Cooperative multi-agent RL for submodular coverage, where the cumulative
coverage set `S_t` is carried in the state, per-agent credit is the Shapley
value of the per-step coverage game, and the TD bootstrap is clipped at the
remaining budget `B(S') = F(V) - F(S')`.

```
reset(): initial footprints covered WITHOUT reward,  S_0 = P(s_0)
step t : agents move to s_{t+1}
         M = ( u_i P(s_{t+1}^i) ) \ S_t
         r = F(S_t u M) - F(S_t)
         S_{t+1} = S_t u M
credit : v(C) = F(S_t u  u_{i in C} P(s_{t+1}^i)) - F(S_t),  phi = Shapley(v)
```

## Install

```bash
pip install -r requirements.txt     # torch is needed for training only
python -m pytest                    # 156 tests, numpy only, ~5 s
```

Everything except training runs on numpy: the credit rules, the
environments, the probes and the whole test suite. That is deliberate — a
probe should be cheap enough to run before committing GPU time.

## Layout

```
macs/
  shapley.py    four backends + resolve(): permutation, subset, facility, mc
  credits.py    MACS, MACS-CLIP, MACS-MC, SHARED, LOCAL, DR (+ VDN/QMIX)
  envs/         base, saturating, facility (incl. the islands benchmark)
  nets.py       AgentQNet, QMixer, Replay          (torch optional)
  obs.py        shared vs private observation channels
  train.py      the trainer — one function, every arm
  policies.py   reference policies, one factory convention
  rollout.py    one rollout harness
  runlog.py     log schema, resume checks, arm keys
  stats.py      exact permutation and paired sign-flip tests
  sweep.py      plan / execute / inspect / aggregate
  plotting.py   figures
  config.py     every geometry and budget the paper reports
experiments/    the training sweeps
probes/         cheap numpy-only diagnostics
scripts/        figures
tests/
```

## Credit rules

| mode | `phi_i` | property |
|---|---|---|
| `MACS` | Shapley value of the step game | efficient, `sum_i phi_i = r_t` |
| `MACS-CLIP` | MACS credits, bootstrap clipped at `B(S')` | same fixed point |
| `MACS-MC` | Shapley forced through Monte Carlo | approximation-error ablation |
| `SHARED` | `r_t` to everyone | sums to `k * r_t` |
| `LOCAL` | `F(S u P_i) - F(S)` | first-arrival, over-counts on overlap |
| `DR` | `F(S u all) - F(S u others)` | last-arrival, under-counts |
| `VDN` / `QMIX` | no credit rule; joint TD on `r_t` | decomposition by architecture |

For submodular `F`, per agent and per step: `DR <= Shapley <= LOCAL`.

## Time-to-go and the reach clip

`train(..., time_obs=True)` feeds the normalized time-to-go `tau/T` to every
network (agent nets and the QMIX mixer), so the value class is
`q_i(s^i, S, tau, a^i)`. The horizon is finite and the bootstrap is
done-masked, so without `tau` the network is asked to fit one value to
states that differ only in time remaining. `time_obs=False` is the default
and reproduces every published run bit for bit.

`clip="reach"` (requires `time_obs=True`, shared observation) replaces the
team ceiling `B(S') = F(V) - F(S')` with a per-agent ceiling

```
B_i(s^i, S, tau) = sum_{j=1..tau} gamma^(j-1) [H(j) - H(j-1)],
H(j) = F(S u R_i(j)) - F(S),  R_i(j) = footprints reachable in j steps
```

and floors the bootstrap at 0. Both bounds hold for every joint policy, so
the fixed point is unchanged; `B_i <= B(S)` always. See `macs/budget.py`
for the proof sketch and `tests/test_budget.py` for the pathwise check.
Logged per block: `clip_frac`, `floor_frac`, and `reach_ratio`
(mean `B_i / B(S')`).

```bash
python -m experiments.facility --exp islands --time-obs --ks 3 4 6
python -m experiments.facility --aggregate --exp islands --time-obs
```

With `--time-obs` every label gets a `_tau` suffix, so these runs never
collide with existing logs, and the islands plan gains a
`MACS-CLIP/shared/reach/tau` arm beside the budget arm.

## Seeding

Every experiment factory takes `seed`, and the trainer derives it from the
run seed (`macs.train.env_seeds`): one seed for the training environment,
and a fixed seed per evaluation episode that is the same at every checkpoint
and for every arm. Two arms at run seed `s` therefore train on the same
spawn sequence and are scored on the same spawns, which is what the paired
sign-flip test assumes. Factories without a `seed` parameter (bare lambdas,
`o9_parallel`'s per-call generator) are called as before.

Earlier island and facility runs built the environment with `seed=None`, so
their spawns came from OS entropy and are not reproducible.
`experiments/saturating.py` and `experiments/scalability.py` still do this,
on purpose, so their published compact-label runs keep their meaning.

## Decoy islands

`IslandEnv(decoy_mult=m)` gives the island nearest the grid centre `m` times
the demand of each other island (`m = 1` is the original benchmark, bit for
bit); `depot_at=(x, y)` fixes the spawn block. `macs.config.DECOY` is the
islands layout with `m = 3` and horizon 20, chosen by

```bash
python -m probes.decoy --mults 1 1.5 2 3 --horizon 20
```

which reports headroom, coupling mass, and, per credit rule, whether one
agent gains by leaving a herd on the decoy. At the chosen configuration the
herd is stable under LOCAL and not under Shapley or DR, and it costs the
team about 6 points of F(V) in scripted play.

```bash
python -m experiments.facility --exp decoy --time-obs --seeds 0 1 2 3 4 5
python -m experiments.facility --aggregate --exp decoy --time-obs
```

Every run now also logs `eval_kappa`, the coupling mass
`sum(phi_LOC - phi) / sum(r)` on its own greedy evaluation rollouts. On the
decoy, a LOCAL arm that herds should show it there.

## Shapley backends

`--shapley` on every CLI. `auto` is the default and reproduces the
pre-refactor behaviour exactly.

| backend | cost | when |
|---|---|---|
| `facility` | `O(|D| k log k)` | `F` is facility location (islands included) |
| `permutation` | `k!` evaluations | the literal Alg. 1 form |
| `subset` | `2^k` evaluations, **shared across agents** | exact, cheapest for `5 <= k <= ~20` |
| `mc` | `m * k` evaluations | large `k`, non-facility `F` |

The subset form evaluates `F` once per coalition and reuses the table for
every agent, so it costs `2^k`, not `k * 2^(k-1)`. At `k=6` that is 64
evaluations against 384 for Monte Carlo at `m=64` — the cost argument for
MC inverts, and `--shapley subset` is the exact backend to ask for.

## Benchmarks

| suite | geometry | headroom |
|---|---|---|
| saturating | 12×12 (k≤4), 16×16 (k=6) | `coord - dec ≈ 0` |
| facility (uniform) | 24×24, ρ=4, H=60 | planning ≈ 0 |
| **islands** | 24×24, 8 discs, ρ=4, H=30, depot spawn | planning and assignment, both positive |

A credit rule can only be tested where the assignment headroom
`plan - plan_self` is clearly positive. On the saturating suite it is not,
so a separation measured there is about approximation difficulty in
learning rather than about task headroom. `H=30` on islands is
load-bearing: at `H=60` the assignment headroom collapses to ~2 points and
a stateless sweep recovers.

Measure it before trusting a result:

```bash
python -m probes.headroom --suite islands --ks 3 4 6
```

## Running things

```bash
# sweeps (all resumable — interrupt and relaunch)
python -m experiments.facility --exp islands --seeds 0 1 2 3 4 --ks 3 4 6
python -m experiments.facility --exp islandslr --ks 6
python -m experiments.facility --exp o9        # augmented-state ablation
python -m experiments.facility --exp o10       # budget clip vs constant clip
python -m experiments.saturating               # the Table 1 saturating columns
python -m experiments.o9_parallel --workers 8  # 25 seeds, one process each
python -m experiments.scalability --ks 8 10    # credit-cost accounting

# before trusting a sweep
python -m experiments.facility --exp islands --inspect
python -m experiments.facility --exp islands --floors-only
python -m experiments.facility --exp islands --aggregate

# probes
python -m probes.headroom --suite facility
python -m probes.iota --suite islands --check
python -m probes.desaturate
python -m probes.territory

# figures
python -m scripts.plot --log-dir logs/macs_islands
python -m scripts.viz_credits
```

`--dry-run` prints the plan, `--smoke` runs 30 episodes, `--pilot` runs one
seed at `k=6`.

## Reading a sweep

`--inspect` reports four things, in this order of importance:

- **`credit_err`** — `max |sum_i phi_i - r_t|`. Zero to machine precision
  for every exact backend. Nonzero on a MACS arm means the exact backend is
  not in use. Expected nonzero for `SHARED`, `LOCAL`, `DR`; `n/a` for VDN
  and QMIX, which have no credit rule.
- **`rising`** — still improving at the cutoff. If every arm is rising, the
  sweep is undertrained and a null difference is not evidence of no
  difference.
- **`DIVERGING`** — TD loss climbing against the run's midpoint *and*
  evaluation below its own peak. Compared against the midpoint, not the
  previous block, because a diverging run climbs steadily: the `k=6`
  private arm went 0.11 → 3.3 → 18.3 and a consecutive-thirds test missed
  it.
- **`clip_frac`** — what the ceiling actually cut. On islands `B(S')` sits
  well above the bootstrap values, so the ceiling never binds and the
  clipped arm runs the same backup as the unclipped one.

## Statistical claims

A paired sign-flip test has smallest attainable two-sided `p = 2/2^n`, so
**five seeds can never reach 0.05** and six is the minimum for a paired
claim at that level. `aggregate` prints the floor whenever it binds rather
than letting `p=0.0625` read as a null result.

`MACS vs SHARED` is not the headline comparison. `LOCAL` and `DR` are
per-agent credit rules too, and the `k=6` islands sweep put them within 2
and 19 points of MACS while MACS beat SHARED by ~165. The evidence supports
"per-agent credit decomposition beats a shared team reward"; the
Shapley-specific claim rests on `MACS vs LOCAL` and `MACS vs DR`.

## Reproducibility

Two lines decide whether a probe reproduces, and they are the same
everywhere:

```python
rng      = np.random.default_rng(seed * 7919 + episode)   # policy
env_seed = seed * 1000 + episode                          # layout, spawns
```

Training consumes one seeded stream in a fixed order: epsilon-greedy draws
per agent, then any Monte Carlo Shapley draws, then the replay sample.

## Notes for anyone who knew the old layout

- Nothing mutates another module's attributes any more. `import macs_v3`
  used to re-point `macs_main.compute_credits` as a side effect of import,
  and three further `install()` hooks patched the same targets, so what a
  run computed depended on import order. The backend is now an argument.
- One trainer replaces `train_macs` and `train_ablation`. The fork is why
  `credit_sum_err` was logged for every credit rule in one and only for
  MACS in the other, which made every non-MACS arm read as
  efficiency-exact.
- One `rollout` and one policy convention replace ten rollout copies and
  four different ways of asking "does this policy need binding to the
  environment".
- Deleted: `macs_v1.py` (superseded), `run_mc_sweep.py` (a stale fork of
  `run_facility.py`, down to its docstring), `testing plot.py` (Big-O
  notation plots, unrelated to this project), and the `install()` functions
  that nothing called.
- `logs/` and `plots/` are no longer tracked.
