#!/usr/bin/env python3
"""
plot.py - plot the CSV produced by scripts/evaluate.py.

Grouped bar chart:
  * one group of bars per problem size, one bar per OMP_TASKGRAPH_OPT setting
    (the legend), so the optimizations are compared side by side;
  * bar height  = average execution time per iteration (avg_ms column);
  * error bars  = standard deviation of the per-iteration time (stddev_ms);
  * x axis      = grid size n (bottom) AND the theoretical FLOP count (top).

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
    return "%.1e" % f      # e.g. 2.1e+08


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV produced by evaluate.py")
    ap.add_argument("-o", "--output", default=None,
                    help="image path (default: <csv>.png)")
    ap.add_argument("--solver", default=None,
                    help="only plot rows for this solver (e.g. cg.x)")
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
            if args.solver and r.get("solver") != args.solver:
                continue
            if str(r.get("returncode")) not in ("0", "0.0"):
                continue
            if r.get("avg_ms", "") == "" or r.get("n_grid", "") == "":
                continue
            rows.append(r)
    if not rows:
        sys.exit("no usable rows in %s" % args.csv)

    # --- opts in first-appearance order (matches the sweep order) ---
    opts = []
    for r in rows:
        if r["opt"] not in opts:
            opts.append(r["opt"])

    # --- x categories: distinct grid sizes, ascending ---
    n_cats = sorted({int(r["n_grid"]) for r in rows})
    flops_by_n = {}
    for r in rows:                       # FLOPs depend only on n -> one value per n
        flops_by_n.setdefault(int(r["n_grid"]), _f(r, "flops"))

    # --- data[opt][n] = (avg_ms, stddev_ms) ---
    data = {opt: {} for opt in opts}
    for r in rows:
        data[r["opt"]][int(r["n_grid"])] = (_f(r, "avg_ms"), _f(r, "stddev_ms"))

    # --- grouped bars ---
    x = list(range(len(n_cats)))
    nopt = len(opts)
    width = 0.8 / nopt

    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(n_cats) + 2.0), 5.0))
    for i, opt in enumerate(opts):
        heights, errs = [], []
        for n in n_cats:
            avg, std = data[opt].get(n, (float("nan"), float("nan")))
            heights.append(avg)
            errs.append(0.0 if std != std else std)   # nan err -> 0
        offs = [xi - 0.4 + width * (i + 0.5) for xi in x]
        ax.bar(offs, heights, width, yerr=errs, capsize=3, label=opt,
               error_kw={"elinewidth": 1, "alpha": 0.7})

    # --- bottom axis: grid size n ---
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_cats])
    ax.set_xlabel("grid size n  (system size = n$^3$)")
    ax.set_ylabel("avg execution time / iteration (ms)")
    if args.logy:
        ax.set_yscale("log")
    ax.grid(axis="y", ls=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="OMP_TASKGRAPH_OPT", fontsize=9)

    # --- top axis: theoretical FLOP count (same group positions) ---
    axtop = ax.twiny()
    axtop.set_xlim(ax.get_xlim())
    axtop.set_xticks(x)
    axtop.set_xticklabels([fmt_flops(flops_by_n[n]) for n in n_cats],
                          rotation=40, ha="left", fontsize=8)
    axtop.set_xlabel("theoretical FLOPs")

    # --- title from the (consistent) run configuration ---
    r0 = rows[0]
    solvers = sorted({r["solver"] for r in rows})
    title = "%s  |  %s backend, taskgraph=%s  |  -S %s, %s iters, %s thr (%s)" % (
        ",".join(solvers), r0.get("backend", "?"), r0.get("taskgraph", "?"),
        r0.get("stencil", "?"), r0.get("iters", "?"),
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
