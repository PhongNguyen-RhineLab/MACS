# Migration map

Old path -> new path, and what changed on the way.

## Deleted

| file | why |
|---|---|
| `macs_v1.py` | superseded by `macs_main.py`; same function names, older bodies |
| `run_mc_sweep.py` | a stale fork of `run_facility.py` — identical docstring, identical `python run_facility.py` usage strings, missing the paired test, the islands modes and the extended LR grid |
| `testing plot.py` | Big-O / Theta / Omega notation plots for a discrete-maths course; unrelated to this project, and a filename with a space in it |
| `macs_fl.install()` | never called by anything |
| `shapley_subset.install()` | never called by anything; the cheaper exact backend it installed was therefore never used in a training run |
| `macs_ablation.install_facility_credits()` | replaced by type-based dispatch in `macs.shapley.resolve` |

## Moved

| old | new |
|---|---|
| `macs_main.shapley_exact` | `macs.shapley.shapley_permutation` |
| `macs_main.shapley_mc` | `macs.shapley.shapley_mc` |
| `macs_main.shapley_coverage_closed_form` | `macs.shapley.shapley_modular_closed_form` |
| `shapley_subset.shapley_subset` | `macs.shapley.shapley_subset` |
| `macs_fl.shapley_facility` | `macs.shapley.shapley_facility` |
| `macs_main.compute_credits`, `macs_v3.compute_credits_v3` | `macs.credits.compute_credits` |
| `macs_main.MultiAgentCoverageEnv`, `F_cardinality`, `compute_frontier` | `macs.envs.base` |
| `macs_v3.SaturatingF`, `SaturatingCoverageEnv` | `macs.envs.saturating` |
| `macs_v3.TwoRoomCoverageEnv` | `macs.envs.base` |
| `macs_fl.FacilityLocationF`, `FacilityCoverageEnv`, `IslandF`, `IslandEnv`, `island_demand`, `linear_kernel` | `macs.envs.facility` |
| `macs_main.AgentQNet`, `QMixer`, `Replay`, `_pos_maps` | `macs.nets` |
| `macs_ablation.ObsTracker` | `macs.obs` |
| `macs_main.train_macs`, `macs_ablation.train_ablation` | `macs.train.train` |
| `probe_fl.pol_random/pol_dec/pol_coord`, `probe_blind.make_blind`, `probe_est.make_dec_est`, `probe_free.make_sweep/pol_front`, `probe_islands.make_plan` | `macs.policies` |
| the ten `rollout()` copies | `macs.rollout.rollout` |
| `run_multiseed_sat.exact_perm_test/r1/f1`, `run_facility.paired_sign_flip_test` | `macs.stats` |
| `run_facility.collect/inspect/aggregate/resolve_arm` | `macs.sweep`, `macs.runlog` |
| `run_facility.py` | `experiments/facility.py` |
| `run_multiseed_sat.py` | `experiments/saturating.py` |
| `run_o9.py` | `experiments/o9_parallel.py` |
| `run_scalability.py` | `experiments/scalability.py` |
| `probe_blind/est/cells/horizon/ceiling/free/svalue.py` | `probes/headroom.py --suite ...` |
| `probe_iota.py` | `probes/iota.py` |
| `probe_desaturate.py` | `probes/desaturate.py` |
| `probe_fl2.py` | `probes/territory.py` |
| `plot_macs.py`, `plot_multiseed.py` | `macs.plotting`, `scripts/plot.py` |
| `viz_credits.py` | `scripts/viz_credits.py` |
| `run_diag.py` | folded into `experiments/facility.py --exp o9 --pilot` |

## Bugs fixed on the way

1. **NaN credits at k>4.** `macs_main.compute_credits` defaulted `mc_m=0`
   and `train_macs` never passed it, so `shapley_mc(..., m=0, ...)` returned
   `0/0`. It only failed to fire because every runner happened to
   `import macs_v3` first, which rebound the dispatcher to one with
   `mc_m=64`. `macs.shapley.shapley_mc` now raises on `m <= 0`.

2. **`install_facility_credits` re-broke it.** Its wrapper declared its own
   `mc_m=0` default and forwarded it explicitly, overriding the 64. Live and
   reachable: `run_facility.py --exp o10` calls the installer at startup and
   defaults to `ks = [2,3,4,6]`, and at `k=6` the environment is saturating,
   not facility — so every O10 `k=6` MACS and MACS-CLIP run trained on NaN
   credits.

3. **`credit_sum_err` logged only on the MACS branch** in one of the two
   trainers, so every other arm's column read as an exact zero rather than
   as the over- or under-attribution it was.

4. **`diag_o9` was never committed** but `macs_ablation.py:358` imported it
   inside the `diag=True` path, so a `--diag` run died with an `ImportError`
   after training. The diagnostic is now optional, runs after the log is
   saved, and says so.

5. **Booleanizing the coverage map erased the two-room wall channel**
   (`-1.0`). `ObsTracker` stores `int8`, same byte cost, lossless.

6. **`CreditTimer` left its wrapper installed** if a run raised mid-sweep.
   Timing is now `train(credit_fn=...)`.

## Behaviour that is deliberately unchanged

Verified bit-identical against the old code:

- `compute_credits` across 4 objectives x k in {2,3,4,6} x 5 modes:
  max abs diff `0.000e+00`, MC path included (same RNG consumption order)
- every environment: identical positions, coverage maps, rewards, `info`
  and `F(V)` over 3 seeds x 20 steps
- every policy: `0.00e+00`, island planners within `1.4e-14`
- the exact Shapley backends agree within `3.5e-14`

## Still to verify locally

The trainer merge needs a differential run where torch is installed:

```bash
python -m experiments.saturating --smoke --ks 2 --modes MACS
# diff the emitted JSON against a log from the old train_macs
```

The merge was written for this to pass: both predecessors consumed the
single seeded RNG stream in the same order (epsilon-greedy draws per agent,
then Monte Carlo Shapley draws, then the replay sample) and built networks
in the same order under the same torch seed.
