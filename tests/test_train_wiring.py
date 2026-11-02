"""
Trainer wiring, without torch.

The tensor path needs a GPU-class dependency this suite deliberately does
not require, so what is tested here is everything around it: argument
validation, the log schema, the resume contract, and the replay sampling
order that makes a run reproduce.

The thing these tests cannot cover is that the merged trainer produces
numerically identical results to the two it replaced. That check has to be
run where torch is installed:

    python -m experiments.saturating --smoke --ks 2 --modes MACS
    # then diff the emitted JSON against a log from the old train_macs

The merge was written to make that check pass: both predecessors consumed
the single seeded RNG stream in the same order -- epsilon-greedy draws per
agent, then any Monte Carlo Shapley draws, then the replay sample -- and
built networks in the same order under the same torch seed.
"""

import numpy as np
import pytest

from macs import credits as C
from macs import nets
from macs.envs import SaturatingCoverageEnv
from macs.runlog import RunLog
from macs.sweep import Run
from macs.train import CLIP_MODES, _default_clip, train


def factory(k=2):
    return lambda: SaturatingCoverageEnv(size=12, horizon=10, k=k, patch=1,
                                         region=4, cap_frac=0.5)


# =============================================================================
# Argument validation happens before any expensive work
# =============================================================================

@pytest.mark.parametrize("kwargs,match", [
    (dict(mode="NOPE"), "unknown mode"),
    (dict(mode="MACS", clip="sideways"), "clip must be one of"),
    (dict(mode="VDN", obs_mode="private"), "private observation"),
    (dict(mode="QMIX", clip="budget"), "unclipped"),
    (dict(mode="MACS", obs_mode="private", clip="budget"), "shared set"),
    (dict(mode="MACS-CLIP", clip="reach"), "time_obs=True"),
    (dict(mode="MACS-CLIP", clip="reach", time_obs=True,
          obs_mode="private"), "shared set"),
])
def test_invalid_combinations_are_rejected(kwargs, match):
    """
    Each of these is a combination that cannot mean anything:

      VDN/QMIX with a private observation -- both mix over the centralized
      state, so there is nothing to be private about
      VDN/QMIX with a clip -- they are the unclipped
      decomposition-by-architecture baselines; clipping them would confound
      O10
      private + budget clip -- B(S') comes from the shared set, which a
      private-observation agent cannot see, so the O9 arms would differ in
      two ways at once

    They are rejected up front rather than after a network is built, so a
    typo in a sweep fails in the first second rather than the first hour.
    """
    with pytest.raises(ValueError, match=match):
        train(factory(), env_name="E", n_episodes=1, **kwargs)


def test_clip_defaults_follow_the_mode():
    assert _default_clip("MACS-CLIP") == "budget"
    assert _default_clip("MACS") == "none"
    assert _default_clip("VDN") == "none"
    assert set(CLIP_MODES) == {"none", "budget", "const", "reach"}


def test_training_without_torch_fails_loudly():
    if nets.TORCH_OK:
        pytest.skip("torch is installed")
    with pytest.raises(RuntimeError, match="PyTorch"):
        train(factory(), env_name="E", mode="MACS", n_episodes=1)


# =============================================================================
# Replay
# =============================================================================

def test_replay_sampling_is_part_of_the_seeded_stream():
    """
    The buffer takes the caller's rng rather than owning one, so the replay
    draw sits in the same seeded stream as the action draws. Two runs with
    the same seed therefore see the same batches in the same order.
    """
    buf = nets.Replay(100)
    for i in range(50):
        buf.push(i)
    a = buf.sample(8, np.random.default_rng(0))
    b = buf.sample(8, np.random.default_rng(0))
    c = buf.sample(8, np.random.default_rng(1))
    assert a == b
    assert a != c


def test_replay_evicts_oldest_first():
    buf = nets.Replay(4)
    for i in range(10):
        buf.push(i)
    assert len(buf) == 4
    assert sorted(buf.buf) == [6, 7, 8, 9]


# =============================================================================
# Log schema
# =============================================================================

def block(cov=None, loss=None, err=None, clip=None):
    from collections import defaultdict
    b = defaultdict(list)
    b["cov"] = cov or []
    b["loss"] = loss or []
    b["err"] = err or []
    b["clip"] = clip or []
    return b


def test_credit_error_is_none_not_zero_when_there_is_no_credit_rule(tmp_path):
    """
    An empty error list means the arm learns from the team reward and has no
    credit rule to measure. Writing 0.0 there makes VDN and QMIX read as
    efficiency-exact in the same column where it is a real claim for MACS.
    """
    run = RunLog("L", dict(method="VDN"), tmp_path)
    run.append_block(episode=100, block=block(cov=[1.0], loss=[0.5]),
                     eval_coverage=1.0, epsilon=0.5)
    assert run.log["per_block"]["credit_sum_err"] == [None]

    run2 = RunLog("L2", dict(method="MACS"), tmp_path)
    run2.append_block(episode=100,
                      block=block(cov=[1.0], loss=[0.5], err=[0.0, 0.25]),
                      eval_coverage=1.0, epsilon=0.5)
    assert run2.log["per_block"]["credit_sum_err"] == [0.25]


def test_credit_error_reports_the_worst_step_not_the_mean(tmp_path):
    """A rule that is exact 99 steps in 100 and wrong once is not exact."""
    run = RunLog("L", dict(method="LOCAL"), tmp_path)
    run.append_block(episode=100,
                     block=block(cov=[1.0], err=[0.0, 0.0, 4.0]),
                     eval_coverage=1.0, epsilon=0.5)
    assert run.log["per_block"]["credit_sum_err"] == [4.0]


def test_summary_carries_both_historic_keys(tmp_path):
    """
    The two predecessor trainers wrote different summary keys, so readers
    had to know which one produced a file. Both are written now.
    """
    run = RunLog("L", dict(method="MACS"), tmp_path)
    for i in range(12):
        run.append_block(episode=100 * (i + 1),
                         block=block(cov=[float(i)], loss=[1.0]),
                         eval_coverage=float(i), epsilon=0.1)
    run.finish()
    s = run.log["summary"]
    assert set(s) == {"last5_train", "last10_eval", "final_eval"}
    assert s["final_eval"] == pytest.approx(11.0)
    assert s["last10_eval"] == pytest.approx(np.mean(range(2, 12)))


def test_save_is_atomic(tmp_path):
    """
    A run is killed mid-sweep often enough that a half-written log is a real
    failure mode: the resume check reads it, fails to parse it, and reruns
    hours of training.
    """
    run = RunLog("L", dict(method="MACS"), tmp_path)
    run.append_block(episode=100, block=block(cov=[1.0]),
                     eval_coverage=1.0, epsilon=0.1)
    run.save()
    assert run.path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    import json
    json.loads(run.path.read_text())


# =============================================================================
# Run identity
# =============================================================================

def test_label_encodes_every_axis_that_varies():
    """
    Two runs differ in their label exactly when they differ in their
    configuration, which is what makes the resume check safe.
    """
    base = Run("Islands 24x24", 6, "MACS", 0)
    seen = {base.label}
    for other in (Run("Islands 24x24", 6, "MACS", 1),
                  Run("Islands 24x24", 6, "MACS", 0, obs_mode="private"),
                  Run("Islands 24x24", 6, "MACS", 0, clip="budget"),
                  Run("Islands 24x24", 6, "MACS", 0, lr=1e-4),
                  Run("Islands 24x24", 3, "MACS", 0),
                  Run("Islands 24x24", 6, "LOCAL", 0)):
        assert other.label not in seen
        seen.add(other.label)


def test_compact_label_matches_the_historic_format():
    r = Run("Saturating 12x12", 2, "MACS", 3, compact=True)
    assert r.label == "Saturating_12x12_k2_MACS_seed3"


def test_arm_ignores_the_seed():
    a = Run("E", 2, "MACS", 0, lr=5e-4)
    b = Run("E", 2, "MACS", 4, lr=5e-4)
    assert a.arm == b.arm == "MACS/shared/none/lr0.0005"


def test_credit_fn_injection_point_matches_the_dispatcher_signature():
    """
    The scalability sweep passes a timing wrapper as credit_fn instead of
    patching the module. The wrapper has to accept exactly what the trainer
    sends, so the call is exercised here rather than discovered at hour two
    of a sweep.
    """
    calls = []

    def wrapper(*a, **kw):
        calls.append((a, kw))
        return C.compute_credits(*a, **kw)

    env = factory(3)()
    env.reset()
    _, r, _, info = env.step([4, 4, 4])
    phi = wrapper("MACS", r, info, env.F, env.k,
                  backend="auto", mc_m=64, rng=np.random.default_rng(0))
    assert len(calls) == 1
    assert set(calls[0][1]) == {"backend", "mc_m", "rng"}
    assert phi.shape == (env.k,)
