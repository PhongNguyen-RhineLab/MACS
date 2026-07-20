"""
viz_credits.py — credit-assignment illustration figure, no training needed.

Builds the pedagogical figure for the paper: a constructed cap-binding
scenario on the saturating F where LOCAL, Shapley, and DR visibly
disagree, rendered as (a) the grid state with agent footprints and
(b) a per-agent credit bar chart with the team reward line.

Scenario (12x12, regions 4x4, cap 8):
  Region (0,0) already has 6 of its 8 creditable cells covered.
  Agent 1 and Agent 2 both place their 3x3 footprints inside this
  nearly-saturated region, overlapping each other on 3 cells.
  Agent 3 covers fresh cells in an empty region.

What the figure shows:
  LOCAL pays agents 1 and 2 as if each were alone (each sees the same
  2 remaining credit units), so the team is over-paid; DR pays agents 1
  and 2 almost nothing (each is redundant given the other), under-paying
  the team; Shapley splits the 2 remaining units and sums exactly to r.
  Agent 3 is paid identically by all three rules, as it should be.

Also writes the numbers as JSON so the exact values can be quoted in the
text. Everything is verified against the axioms at runtime (efficiency
for Shapley, LOCAL >= Shapley >= DR per agent).

Usage:
  python viz_credits.py                 # writes plots/viz/credit_scenario.png
"""

import json
from pathlib import Path

import numpy as np

from macs_v3 import SaturatingF
from macs_main import shapley_exact

OUT_DIR = Path("plots/viz")
SIZE = 12


def build_scenario():
    Fsat = SaturatingF(SIZE, region=4, cap_frac=0.5)

    # Pre-coverage: 6 cells inside region (rows 0-3, cols 0-3), cap is 8.
    S_prev = np.zeros((SIZE, SIZE), dtype=bool)
    S_prev[0, 0:4] = True
    S_prev[1, 0:2] = True

    def patch(cx, cy):
        m = np.zeros((SIZE, SIZE), dtype=bool)
        m[max(0, cx-1):cx+2, max(0, cy-1):cy+2] = True
        return m

    # Agents 1 and 2 inside the near-saturated region, overlapping;
    # Agent 3 alone in an empty region.
    patches = [patch(2, 1), patch(2, 2), patch(9, 9)]
    return Fsat, S_prev, patches


def credits(Fsat, S_prev, patches):
    k = len(patches)
    union = S_prev.copy()
    for pm in patches:
        union |= pm
    r = Fsat(union) - Fsat(S_prev)

    f0 = Fsat(S_prev)
    local = np.array([Fsat(S_prev | pm) - f0 for pm in patches])

    dr = np.zeros(k)
    for i in range(k):
        rest = S_prev.copy()
        for j in range(k):
            if j != i:
                rest |= patches[j]
        dr[i] = Fsat(union) - Fsat(rest)

    sh = shapley_exact(S_prev, patches, Fsat)

    assert abs(sh.sum() - r) < 1e-9, "Shapley efficiency violated"
    assert np.all(dr <= sh + 1e-9) and np.all(sh <= local + 1e-9), \
        "credit ordering violated"
    return r, dict(LOCAL=local, Shapley=sh, DR=dr)


def draw(Fsat, S_prev, patches, r, rules):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agent_colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.6),
                                   gridspec_kw={"width_ratios": [1, 1.1]})

    # --- panel (a): grid state --------------------------------------------
    img = np.zeros((SIZE, SIZE, 3)) + 1.0
    img[S_prev] = (0.75, 0.75, 0.75)
    ax0.imshow(img, origin="upper")
    for g in range(0, SIZE + 1, 4):                    # region boundaries
        ax0.axhline(g - 0.5, color="k", lw=1.2)
        ax0.axvline(g - 0.5, color="k", lw=1.2)
    for i, (pm, c) in enumerate(zip(patches, agent_colors)):
        ys, xs = np.where(pm)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        ax0.add_patch(mpatches.Rectangle(
            (x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1,
            fill=False, edgecolor=c, lw=2.4, label=f"Agent {i+1}"))
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_title("(a) Coverage state\ngray = covered, cap 8 per region")
    ax0.legend(fontsize=8, loc="lower left")

    # --- panel (b): credits per rule --------------------------------------
    k = len(patches)
    rule_names = ["LOCAL", "Shapley", "DR"]
    x = np.arange(k)
    w = 0.26
    for j, rn in enumerate(rule_names):
        vals = rules[rn]
        bars = ax1.bar(x + (j - 1) * w, vals, w, label=rn,
                       color=["#c0392b", "#27ae60", "#8e44ad"][j],
                       alpha=0.85)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + 0.06,
                     f"{v:.2f}", ha="center", fontsize=7)
    totals = " | ".join(f"sum {rn} = {rules[rn].sum():.2f}"
                        for rn in rule_names)
    ax1.axhline(0, color="k", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Agent {i+1}" for i in range(k)])
    ax1.set_ylabel("Credit")
    ax1.set_title(f"(b) Per-agent credit, team reward r = {r:.0f}\n"
                  f"{totals}", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, axis="y", alpha=0.2)

    fp = OUT_DIR / "credit_scenario.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fp}")

    with open(OUT_DIR / "credit_scenario.json", "w") as f:
        json.dump({"team_reward": float(r),
                   **{rn: [float(v) for v in rules[rn]]
                      for rn in rule_names},
                   "sums": {rn: float(rules[rn].sum())
                            for rn in rule_names}}, f, indent=2)
    print(f"Saved: {OUT_DIR / 'credit_scenario.json'}")


if __name__ == "__main__":
    Fsat, S_prev, patches = build_scenario()
    r, rules = credits(Fsat, S_prev, patches)
    print(f"Team reward r = {r}")
    for rn in ("LOCAL", "Shapley", "DR"):
        v = rules[rn]
        print(f"  {rn:<8} {np.round(v, 3)}  sum = {v.sum():.3f}")
    draw(Fsat, S_prev, patches, r, rules)