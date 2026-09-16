"""
Per-agent observation channels.

shared  : every agent gets the environment's cumulative set S_t
          (the setting of Prop. `decentralized`)
private : agent i gets S_t^i, the union of its own footprints so far

Both arms keep memory -- the private agent still sees its own trail -- so
the O9 comparison isolates SHARING, not memory. Reward and credits always
come from the shared environment; only the observation changes.

Returned arrays are (k, H, W) int8 in both modes, so the replay buffer and
the batch path have one shape and one dtype to deal with.

int8 rather than float32 keeps the buffer affordable: at k=6 on a 24x24
grid a transition costs ~7 KB instead of ~28 KB. int8 rather than bool
because the two-room map encodes walls as -1.0, and booleanizing would
silently erase that channel; int8 costs the same byte and stays faithful
to whatever the environment reports.
"""

import numpy as np

from macs.envs.base import compute_frontier

OBS_MODES = ("shared", "private")


class ObsTracker:
    def __init__(self, env, mode):
        if mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}, "
                             f"got {mode!r}")
        self.mode = mode
        self.env = env
        self.reset()

    def reset(self):
        env = self.env
        self.own = [env._patch_mask(env.pos[i]).copy() for i in range(env.k)]

    def update(self):
        """Call after env.step(): fold each agent's new footprint into its
        own set. The shared set is maintained by the environment itself."""
        env = self.env
        for i in range(env.k):
            self.own[i] |= env._patch_mask(env.pos[i])

    def channels(self, z_shared):
        """Return (coverage (k,H,W) int8, frontier (k,H,W) int8)."""
        env = self.env
        if self.mode == "shared":
            z = np.asarray(z_shared, dtype=np.float32)
            zb = z.astype(np.int8)
            fr = compute_frontier(z).astype(np.int8)
            return (np.repeat(zb[None], env.k, 0),
                    np.repeat(fr[None], env.k, 0))
        zs = np.stack([o.astype(np.int8) for o in self.own])
        frs = np.stack([compute_frontier(o.astype(np.float32)).astype(np.int8)
                        for o in self.own])
        return zs, frs
