#!/usr/bin/env python3
"""
plot.py - plot the CSV produced by scripts/evaluate.py.

Grouped bar chart:
  * one group of bars per problem size, one bar per OMP_TASKGRAPH_OPT setting
    (the legend), so the optimizations are compared side by side;
  * bar height  = average execution time per repetition (avg_ms column);
  * error bars  = standard deviation of the per-repetition time (stddev_ms);
  * x axis      = matrix size N (bottom) AND the theoretical FLOP count (top).

Usage:
    python3 scripts/plot.py results.csv                # -> results.png
    python3 scripts/plot.py results.csv -o plot.png
    python3 scripts/plot.py results.csv --show
    python3 scripts/plot.py results.csv --logy         # log y axis
"""

import argparse
import csv
import os
import sys


def _f(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fmt_flops(f):
    if f != f or f <= 0:   # nan / non-positive
        return "-"
    return "%.1e" % f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV produced by evaluate.py")
    ap.add_argument("-o", "--output", default=None,
                    help="image path (default: <csv>.png)")
    ap.add_argument("--logy", action="store_true", help="logarithmic y axis")
    ap.add_argument("--show", action="store_true", help="show the window instead of only saving")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")   # headless (e.g. on a compute node)
    import matplotlib.pyplot as plt

    # --- read rows (skip failed / incomplete runs) ---
    rows = []
    with open(args.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("returncode")) not in ("0", "0.0"):
                continue
            if r.get("avg_ms", "") == "" or r.get("nt_grid", "") == "":
                continue
            rows.append(r)
    if not rows:
        sys.exit("no usable rows in %s" % args.csv)

    # --- opts in first-appearance order (matches the sweep order) ---
    opts = []
    for r in rows:
        if r["opt"] not in opts:
            opts.append(r["opt"])

    # --- x categories: distinct tile grids, ascending ---
    nt_cats = sorted({int(r["nt_grid"]) for r in rows})
    flops_by, nsys_by = {}, {}
    for r in rows:
        k = int(r["nt_grid"])
        flops_by.setdefault(k, _f(r, "flops"))
        nsys_by.setdefault(k, r.get("n_sys", "") or "?")

    # --- data[opt][nt] = (avg_ms, stddev_ms) ---
    data = {opt: {} for opt in opts}
    for r in rows:
        data[r["opt"]][int(r["nt_grid"])] = (_f(r, "avg_ms"), _f(r, "stddev_ms"))

    # --- grouped bars ---
    x = list(range(len(nt_cats)))
    nopt = len(opts)
    width = 0.8 / nopt

    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(nt_cats) + 2.0), 5.0))
    for i, opt in enumerate(opts):
        heights, errs = [], []
        for nt in nt_cats:
            avg, std = data[opt].get(nt, (float("nan"), float("nan")))
            heights.append(avg)
            errs.append(0.0 if std != std else std)
        offs = [xi - 0.4 + width * (i + 0.5) for xi in x]
        ax.bar(offs, heights, width, yerr=errs, capsize=3, label=opt,
               error_kw={"elinewidth": 1, "alpha": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels(["N = %s\n(nt = %d)" % (nsys_by[nt], nt) for nt in nt_cats],
                       rotation=20, ha="right", fontsize=8)
    ax.set_xlabel("matrix size N  /  tile grid nt")
    ax.set_ylabel("avg execution time / repetition (ms)")
    if args.logy:
        ax.set_yscale("log")
    ax.grid(axis="y", ls=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="OMP_TASKGRAPH_OPT", fontsize=9)

    # --- top axis: theoretical FLOP count (same group positions) ---
    axtop = ax.twiny()
    axtop.set_xlim(ax.get_xlim())
    axtop.set_xticks(x)
    axtop.set_xticklabels([fmt_flops(flops_by[nt]) for nt in nt_cats],
                          rotation=40, ha="left", fontsize=8)
    axtop.set_xlabel("theoretical FLOPs")

    r0 = rows[0]
    title = "Cholesky  |  %s backend, taskgraph=%s  |  ts=%s, %s reps, %s thr (%s)" % (
        r0.get("backend", "?"), r0.get("taskgraph", "?"),
        r0.get("ts", r0.get("ts_arg", "?")), r0.get("reps", "?"),
        r0.get("omp_num_threads", "?"), r0.get("omp_places", "?"))
    ax.set_title(title, fontsize=10, pad=28)

    fig.tight_layout()

    out = args.output or (os.path.splitext(args.csv)[0] + ".png")
    fig.savefig(out, dpi=args.dpi)
    print("wrote %s" % out, file=sys.stderr)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
