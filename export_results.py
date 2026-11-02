"""
Export MACS run logs for the paper.

  python export_results.py LOG_DIR [LOG_DIR ...] --out results

Reads every *seed*.json under the given directories (the trainer's log schema)
and writes, into --out:

  runs.csv      one row per run: env, k, method, clip, lr, tau, seed,
                final score (raw and % F(V)), coupling mass kappa, clip and
                floor fractions, reach ratio, still-rising flag, protocol checks
  arms.csv      one row per (env, k, arm): n, mean +- sd as % F(V)
  tests.txt     exact paired sign-flip tests, MACS arms vs every baseline,
                paired by seed (and only on seeds both arms have)
  report.txt    everything above in readable form -- send me this file

Numbers: score = mean greedy-eval F(S_T) over the last 10 logged checkpoints
(the paper's metric), reported as % of F(V). sd is the sample sd (ddof=1),
as in Table 1. kappa = mean of the last 10 logged eval_kappa values.
"""
import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def sd(x):
    if len(x) < 2:
        return float("nan")
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def last(vals, n=10):
    v = [x for x in (vals or []) if x is not None]
    return v[-n:]


def load_run(path):
    d = json.loads(path.read_text())
    c, pb = d["config"], d["per_block"]
    ev = last(pb.get("eval_coverage"))
    fv = float(c.get("F_total") or c.get("max_cells"))
    score = mean(ev)
    allev = [x for x in pb.get("eval_coverage", []) if x is not None]
    rising = len(allev) >= 4 and mean(allev[-2:]) > mean(allev[-4:-2])
    arm = f"{c['method']}/{c.get('obs_mode', 'shared')}/{c.get('clip', 'none')}"
    if c.get("lr") is not None:
        arm += f"/lr{c['lr']:g}"
    if c.get("time_obs"):
        arm += "/tau"
    return dict(
        file=path.name, env=c.get("env_name"), k=c.get("k"),
        method=c["method"], clip=c.get("clip"), lr=c.get("lr"),
        tau=bool(c.get("time_obs")), seed=c.get("seed"), arm=arm,
        episodes=c.get("n_episodes"), FV=fv,
        score_raw=round(score, 3), score_pct=round(100 * score / fv, 3),
        kappa=round(mean(last(pb.get("eval_kappa"))), 4) if pb.get("eval_kappa") else "",
        clip_frac=round(mean([x for x in pb.get("clip_frac", []) if x]), 4)
        if any(pb.get("clip_frac", []) or []) else 0.0,
        floor_frac=round(mean(last(pb.get("floor_frac"), 10**9)), 4) if last(pb.get("floor_frac")) else "",
        reach_ratio=round(mean(last(pb.get("reach_ratio"), 10**9)), 4) if last(pb.get("reach_ratio")) else "",
        still_rising=rising,
        seeded_env=bool(c.get("env_seeds")),
        credit_err=max([x for x in pb.get("credit_sum_err", []) if x is not None], default=""),
    )


def sign_flip(diffs):
    """Exact two-sided paired sign-flip p-value on the mean difference."""
    n = len(diffs)
    obs = abs(mean(diffs))
    hits = sum(abs(mean([s * d for s, d in zip(signs, diffs)])) >= obs - 1e-12
               for signs in itertools.product((1, -1), repeat=n))
    return hits / 2 ** n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in a.dirs:
        for p in sorted(Path(d).rglob("*seed*.json")):
            try:
                runs.append(load_run(p))
            except Exception as e:  # noqa: BLE001
                print(f"skip {p}: {e}")
    if not runs:
        raise SystemExit("no run logs found")

    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0]))
        w.writeheader()
        w.writerows(runs)

    groups = defaultdict(list)
    for r in runs:
        groups[(r["env"], r["k"], r["arm"])].append(r)

    lines = []
    with open(out / "arms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["env", "k", "arm", "n", "seeds", "mean_pct", "sd_pct",
                    "mean_raw", "kappa", "clip_frac", "reach_ratio",
                    "n_still_rising", "all_tau", "all_seeded"])
        for (env, k, arm), rs in sorted(groups.items(), key=lambda x: (str(x[0][0]), x[0][1], x[0][2])):
            pct = [r["score_pct"] for r in rs]
            kap = [r["kappa"] for r in rs if r["kappa"] != ""]
            rr = [r["reach_ratio"] for r in rs if r["reach_ratio"] != ""]
            row = [env, k, arm, len(rs), " ".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"])),
                   round(mean(pct), 2), round(sd(pct), 2),
                   round(mean([r["score_raw"] for r in rs]), 2),
                   round(mean(kap), 4) if kap else "",
                   round(mean([r["clip_frac"] for r in rs]), 4),
                   round(mean(rr), 4) if rr else "",
                   sum(r["still_rising"] for r in rs),
                   all(r["tau"] for r in rs), all(r["seeded_env"] for r in rs)]
            w.writerow(row)
            lines.append(f"{env:>22} k={k}  {arm:<42} n={len(rs)}  "
                         f"{mean(pct):6.2f} +- {sd(pct):5.2f} %F(V)"
                         + (f"  kappa={mean(kap):.3f}" if kap else "")
                         + (f"  rising={row[11]}/{len(rs)}" if row[11] else "")
                         + ("" if row[12] else "  [NO TAU]")
                         + ("" if row[13] else "  [UNSEEDED]"))

    tests = []
    by_cell = defaultdict(dict)
    for (env, k, arm), rs in groups.items():
        by_cell[(env, k)][arm] = {r["seed"]: r["score_pct"] for r in rs}
    for (env, k), arms in sorted(by_cell.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        for a_ in sorted(x for x in arms if x.startswith("MACS")):
            for b in sorted(x for x in arms if not x.startswith("MACS")):
                common = sorted(set(arms[a_]) & set(arms[b]))
                if len(common) < 2:
                    continue
                diffs = [arms[a_][s] - arms[b][s] for s in common]
                p = sign_flip(diffs)
                tests.append(f"{env:>22} k={k}  {a_}  vs  {b}:  "
                             f"diff {mean(diffs):+6.2f} pts  n={len(common)}  "
                             f"p={p:.4f}{'  *' if p < 0.05 else ''}")
    (out / "tests.txt").write_text("\n".join(tests) + "\n")
    report = ("ARMS (score = last-10 greedy eval, % of F(V), sample sd)\n"
              + "\n".join(lines)
              + "\n\nPAIRED SIGN-FLIP TESTS (MACS arms vs baselines, paired by seed;"
              " min attainable p with n seeds is 2/2^n)\n" + "\n".join(tests) + "\n")
    (out / "report.txt").write_text(report)
    print(report)
    print(f"wrote {out}/runs.csv, arms.csv, tests.txt, report.txt")


if __name__ == "__main__":
    main()