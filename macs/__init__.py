"""
MACS -- Multi-Agent Coverage via Submodularity.

Cooperative multi-agent RL where the cumulative coverage set S_t is carried
in the state, per-agent credit comes from the Shapley value of the per-step
coverage game, and the TD bootstrap is clipped at the remaining budget
B(S') = F(V) - F(S').

Layout:

    macs.envs       coverage environments and their objectives F
    macs.shapley    the four Shapley backends and the resolver
    macs.credits    the credit rules (MACS, SHARED, LOCAL, DR, ...)
    macs.nets       Q networks, QMIX mixer, replay (torch optional)
    macs.obs        shared vs private observation channels
    macs.train      the trainer -- one function, every arm
    macs.policies   reference policies, one factory convention
    macs.rollout    one rollout harness for every probe
    macs.runlog     log schema, resume checks, readers
    macs.stats      exact permutation and paired sign-flip tests
    macs.sweep      plan / execute / inspect / aggregate
    macs.config     every geometry and budget the paper reports

Entry points live in experiments/ and probes/; nothing in this package
mutates another module's attributes at import time.
"""

__version__ = "0.4.0"
