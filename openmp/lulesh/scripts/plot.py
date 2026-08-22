#!/usr/bin/env python3
"""
plot.py - plot the CSV produced by scripts/evaluate.py.

Grouped bar chart (same style as the krylov / cholesky apps):
  * one group of bars per problem size, one bar per OMP_TASKGRAPH_OPT setting
    (the legend), so the optimizations are compared side by side;
  * bar height  = average execution time per iteration (avg_ms), or the figure
                  of merit with --metric fom (single-run FOM in z/s, no error bar);
  * error bars  = standard deviation of the per-iteration time (stddev_ms);
  * x axis      = mesh side s (bottom) AND the number of zones s^3 (top).

Usage:
    python3 scripts/plot.py results.csv                 # -> results.png (time)
    python3 scripts/plot.py results.csv --metric fom    # figure of merit
    python3 scripts/plot.py results.csv -o plot.png
    python3 scripts/plot.py results.csv --show --logy
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


def fmt_zones(z):
    if z != z or z <= 0:
        return "-"
    return "%.1e" % z


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV produced by evaluate.py")
    ap.add_argument("-o", "--output", default=None,
                    help="image path (default: <csv>.png)")
    ap.add_argument("--metric", choices=["time", "fom"], default="time",
                    help="bar height: avg time / iteration (default) or figure of merit")
    ap.add_argument("--logy", action="store_true", help="logarithmic y axis")
    ap.add_argument("--show", action="store_true", help="show the window instead of only saving")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.metric == "time":
        avg_col, std_col = "avg_ms", "stddev_ms"
        ylabel = "avg execution time / iteration (ms)"
    else:
        avg_col, std_col = "fom", None
        ylabel = "figure of merit (z/s)"

    # --- read rows (skip failed / incomplete runs) ---
    rows = []
    with open(args.csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("returncode")) not in ("0", "0.0"):
                continue
            if r.get(avg_col, "") == "" or r.get("s_grid", "") == "":
                continue
            rows.append(r)
    if not rows:
        sys.exit("no usable rows in %s" % args.csv)

    opts = []
    for r in rows:
        if r["opt"] not in opts:
            opts.append(r["opt"])

    s_cats = sorted({int(r["s_grid"]) for r in rows})
    zones_by = {}
    for r in rows:
        k = int(r["s_grid"])
        zones_by.setdefault(k, _f(r, "zones"))

    data = {opt: {} for opt in opts}
    for r in rows:
        std = _f(r, std_col) if std_col else float("nan")
        data[r["opt"]][int(r["s_grid"])] = (_f(r, avg_col), std)

    x = list(range(len(s_cats)))
    nopt = len(opts)
    width = 0.8 / nopt

    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(s_cats) + 2.0), 5.0))
    for i, opt in enumerate(opts):
        heights, errs = [], []
        for s in s_cats:
            avg, std = data[opt].get(s, (float("nan"), float("nan")))
            heights.append(avg)
            errs.append(0.0 if std != std else std)
        offs = [xi - 0.4 + width * (i + 0.5) for xi in x]
        ax.bar(offs, heights, width, yerr=errs, capsize=3, label=opt,
               error_kw={"elinewidth": 1, "alpha": 0.7})

    ax.set_xticks(x)
    ax.set_xticklabels(["s = %d" % s for s in s_cats], fontsize=9)
    ax.set_xlabel("mesh side s")
    ax.set_ylabel(ylabel)
    if args.logy:
        ax.set_yscale("log")
    ax.grid(axis="y", ls=":", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="OMP_TASKGRAPH_OPT", fontsize=9)

    # --- top axis: number of zones (s^3) ---
    axtop = ax.twiny()
    axtop.set_xlim(ax.get_xlim())
    axtop.set_xticks(x)
    axtop.set_xticklabels([fmt_zones(zones_by[s]) for s in s_cats],
                          rotation=40, ha="left", fontsize=8)
    axtop.set_xlabel("zones (s^3)")

    r0 = rows[0]
    title = "LULESH  |  %s backend, taskgraph=%s  |  %s iters, -nb %s, %s thr (%s)" % (
        r0.get("backend", "?"), r0.get("taskgraph", "?"),
        r0.get("iters", "?"), r0.get("nb", "?"),
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
