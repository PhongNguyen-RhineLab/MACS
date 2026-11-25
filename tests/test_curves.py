"""scripts.curves on synthetic logs: a faster arm must show a positive AUC
difference with every seed won, and identical arms a zero difference."""

import json

import numpy as np

from macs import runlog
from scripts import curves


def _write(tmp, mode, seed, rate, final=80.0, n=40):
    ep = list(range(100, 100 * (n + 1), 100))
    y = [final * (1 - np.exp(-rate * e / 1000)) for e in ep]
    log = {"label": f"x_{mode}_seed{seed}",
           "config": {"method": mode, "env_name": "Islands 24x24", "k": 6,
                      "seed": seed, "clip": "none", "F_total": 100.0},
           "per_block": {key: [None] * n for key in runlog.BLOCK_KEYS},
           "summary": {}}
    log["per_block"]["episode"] = ep
    log["per_block"]["eval_coverage"] = y
    (tmp / f"x_{mode}_seed{seed}.json").write_text(json.dumps(log))


def test_auc_detects_faster_arm(tmp_path):
    for s in range(6):
        _write(tmp_path, "MACS", s, rate=2.0 + 0.01 * s)
        _write(tmp_path, "DR", s, rate=1.0 + 0.01 * s)
        _write(tmp_path, "LOCAL", s, rate=2.0 + 0.01 * s)
    cur = curves.load_curves(tmp_path)[("Islands 24x24", 6)]
    row = curves.compare(cur, "MACS", "DR")
    assert row["auc"]["diff"] > 0 and row["auc"]["wins"] == 6
    assert abs(row["auc"]["p"] - 2 / 64) < 1e-12
    assert row["t90"]["diff"] < 0
    same = curves.compare(cur, "MACS", "LOCAL")
    assert abs(same["auc"]["diff"]) < 1e-9
