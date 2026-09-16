"""
One rollout harness.

Every probe had its own copy of this loop -- probe_blind, probe_est,
probe_cells, probe_horizon, probe_free, probe_ceiling, probe_svalue,
probe_islands, probe_fl and probe_fl2 -- differing in which statistics they
returned and in how they decided whether a policy needed binding to the
environment. They agreed on the part that matters, which is the seeding:

    rng          = default_rng(seed * 7919 + e)     policy randomness
    env seed     = seed * 1000 + e                  layout and spawns

Those two lines are the reason a probe reproduces. They are kept exactly,
so every number already reported from a probe reproduces through this
function.

`stats` collects whatever a caller wants per episode; the defaults cover
what the probes actually reported.
"""

import numpy as np


def rollout(env_factory, policy, episodes=8, seed=0, stats=None,
            normalize=False):
    """
    Run `episodes` episodes and return a dict of per-episode statistics.

    env_factory : callable(episode_index, env_seed) -> env, already reset or
                  not; rollout resets it
    policy      : a factory, make(env) -> pol(env, rng) -> actions
                  (see macs.policies; wrap bare functions with `stateless`)
    normalize   : report coverage as a percentage of F(V) rather than raw
    stats       : optional {name: callable(env) -> float} evaluated at the
                  end of each episode, merged into the result
    """
    out = {"coverage": [], "sensed": []}
    extra = {name: [] for name in (stats or {})}
    for e in range(episodes):
        rng = np.random.default_rng(seed * 7919 + e)
        env = env_factory(e, seed * 1000 + e)
        env.reset()
        pol = policy(env)
        for _ in range(env.horizon):
            env.step(pol(env, rng))
        cov = env.total_coverage
        out["coverage"].append(100.0 * cov / env.max_cells if normalize
                               else cov)
        out["sensed"].append(float(env.covered.mean()))
        for name, fn in (stats or {}).items():
            extra[name].append(float(fn(env)))
    out.update(extra)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def mean_coverage(env_factory, policy, episodes=8, seed=0, normalize=False):
    """Just the mean, for the many callers that want one number."""
    return float(rollout(env_factory, policy, episodes, seed,
                         normalize=normalize)["coverage"].mean())


def factory(EnvCls, cfg, per_episode=None):
    """
    Build an env_factory from a class and a config dict.

    per_episode : optional callable(cfg_copy, episode_index) applied before
                  construction, for probes that vary the demand field or the
                  island layout from episode to episode
    """
    def make(e, env_seed):
        c = dict(cfg)
        if per_episode is not None:
            per_episode(c, e)
        env = EnvCls(**c, seed=env_seed)
        if "rho" in cfg and not hasattr(env, "rho_hint"):
            env.rho_hint = cfg["rho"]      # the sweep policy reads this
        return env
    return make


def ladder(env_factory, policies, episodes=8, seed=0, normalize=True):
    """{name: (mean, std)} over a sequence of (name, policy) pairs."""
    out = {}
    for name, pol in policies:
        r = rollout(env_factory, pol, episodes, seed,
                    normalize=normalize)["coverage"]
        out[name] = (float(r.mean()), float(r.std()))
    return out
