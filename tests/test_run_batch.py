"""scripts.run_batch: job lists match the run plan, and the scheduler never
starts a run it expects to overrun the budget."""

from scripts import run_batch as RB


def test_batch_sizes_match_run_plan(tmp_path):
    n = {name: len(RB.build_jobs([name], tmp_path)) for name in RB.BATCHES}
    assert n["crowded_core"] == 72 and n["crowded_base"] == 36
    assert n["int_islands"] == 18 and n["int_decoy"] == 12
    assert n["h1_seeds"] == 54 and n["baselinelr"] == 8
    assert n["vdn16k"] == 6 and n["pipeline_check"] == 3
    j = RB.build_jobs(["vdn16k"], tmp_path)[0]
    assert j["cfg"]["n_episodes"] == 16000 and j["lr"] == 1e-5
    assert all(not j["done"] for j in RB.build_jobs(["int_islands"], tmp_path))


def test_budget_respected(tmp_path, monkeypatch, capsys):
    jobs = [dict(j, fake_s=0.3)
            for j in RB.build_jobs(["int_islands"], tmp_path)[:6]]
    monkeypatch.setattr(RB, "build_jobs", lambda *a, **k: jobs)
    # 6 jobs of ~0.3 s on 2 workers, budget ~0.65 s with a 0.3 s estimate:
    # the first wave and at most one more wave may start, never all six
    RB.run(["int_islands"], tmp_path, hours=0.65 / 3600, workers=2, gpus=[],
           first_run_hours=0.3 / 3600, worker_fn=RB._fake_worker)
    out = capsys.readouterr().out
    started = out.count("  start [")
    assert 2 <= started < 6, out
    assert "Budget reached" in out


def test_failure_does_not_stop_others(tmp_path, monkeypatch, capsys):
    jobs = RB.build_jobs(["int_islands"], tmp_path)[:3]
    jobs[0] = dict(jobs[0], fake_fail=True)
    monkeypatch.setattr(RB, "build_jobs", lambda *a, **k: jobs)
    RB.run(["int_islands"], tmp_path, hours=1.0, workers=2, gpus=[0, 1],
           worker_fn=RB._fake_worker)
    out = capsys.readouterr().out
    assert out.count("FAILED") == 1 and out.count("] done [") == 2
    assert "(gpu 0)" in out and "(gpu 1)" in out
