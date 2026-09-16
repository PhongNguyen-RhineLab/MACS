"""
Draw the figures for a log directory.

Works on one seed or on twenty: a single-seed directory plots lines without
bands, a multi-seed one plots mean +- sd. This replaces plot_macs.py and
plot_multiseed.py, which differed only in that assumption.

Usage:
  python -m scripts.plot --log-dir logs/macs_islands
  python -m scripts.plot --log-dir logs/macs_v3_multiseed --window 10
  python -m scripts.plot --log-dir logs/macs_clipcontrol \\
      --headline MACS-CLIP/shared/budget MACS/shared/none
"""

import argparse
import sys
from pathlib import Path

from macs import plotting


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default="logs/macs_v3_multiseed")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--window", type=int, default=plotting.FINAL_WINDOW)
    ap.add_argument("--headline", nargs=2, metavar=("HI", "LO"),
                    default=["MACS/shared/none", "LOCAL/shared/none"],
                    help="arm pair to annotate on the bar chart with a "
                         "paired test")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    out_dir = args.out_dir or str(Path("plots") / Path(args.log_dir).name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data = plotting.load_runs(args.log_dir)
    if not data:
        print(f"No seed-labelled logs in {args.log_dir}.")
        return 1

    plotting.fig_learning_curves(data, out_dir)
    plotting.fig_final_bars(data, out_dir, args.window, tuple(args.headline))
    plotting.fig_clip_fraction(data, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
