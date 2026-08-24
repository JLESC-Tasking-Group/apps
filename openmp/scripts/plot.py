#!/usr/bin/env python3
"""
plot.py - plot the sweep produced by scripts/evaluate.py.

Reads results/runs.csv (and optionally results/cgstats.csv) and writes PNGs under
results/figures:

  * time-<app>[-<variant>].png : grouped bars of avg execution time / iteration
    (ms) with stddev error bars, one group per problem size and one bar per
    configuration (synchronous / no-taskgraph / taskgraph:none / taskgraph:<opt>).
    The problem size is on the bottom axis and the work (FLOPs / zones / tokens)
    on the top axis.
  * graph-<app>.png : per-CGIR-pass command-graph reduction (nodes & edges
    before -> after) and per-pass wall time, from cgstats.csv joined on the run
    tag. Only the taskgraph configurations contribute here.

Uses only matplotlib + the standard library.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

APPS_OPENMP = Path(__file__).resolve().parent.parent

# Canonical CGIR pipeline order for the graph-stats x axis.
PASS_ORDER = ["copy-normalize", "copy-fuse", "reduce-node", "reduce-edge",
              "prog-fuse", "jit", "sequence", "batch"]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def median(xs):
    xs = sorted(v for v in xs if v is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def load_runs(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") not in ("ok", "", None):
                continue
            rows.append(r)
    return rows


def ordered_unique(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def plot_time(rows, figdir, dpi, logy):
    import matplotlib.pyplot as plt

    groups = defaultdict(list)  # (app, variant) -> rows
    for r in rows:
        if r.get("avg_ms", "") == "" or r.get("size", "") == "":
            continue
        groups[(r["app"], r.get("variant", ""))].append(r)

    for (app, variant), grp in sorted(groups.items()):
        sizes = sorted({int(r["size"]) for r in grp})
        configs = ordered_unique([r["config"] for r in grp])
        # data[config][size] = (avg, std); work_by[size]
        data = {c: {} for c in configs}
        work_by, wlabel = {}, ""
        for r in grp:
            s = int(r["size"])
            data[r["config"]][s] = (fnum(r["avg_ms"]), fnum(r.get("stddev_ms")) or 0.0)
            work_by.setdefault(s, fnum(r.get("work")))
            wlabel = r.get("work_label", "") or wlabel

        x = list(range(len(sizes)))
        width = 0.8 / max(len(configs), 1)
        fig, ax = plt.subplots(figsize=(max(7.0, 1.3 * len(sizes) + 2.0), 5.0))
        for i, c in enumerate(configs):
            heights = [(data[c].get(s) or (float("nan"), 0.0))[0] for s in sizes]
            errs = [(data[c].get(s) or (float("nan"), 0.0))[1] for s in sizes]
            offs = [xi - 0.4 + width * (i + 0.5) for xi in x]
            ax.bar(offs, heights, width, yerr=errs, capsize=2, label=c,
                   error_kw={"elinewidth": 1, "alpha": 0.7})

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("problem size")
        ax.set_ylabel("avg execution time / iteration (ms)")
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", ls=":", alpha=0.6)
        ax.set_axisbelow(True)
        ax.legend(title="configuration", fontsize=8, ncol=2)

        if any(work_by.get(s) for s in sizes):
            axtop = ax.twiny()
            axtop.set_xlim(ax.get_xlim())
            axtop.set_xticks(x)
            axtop.set_xticklabels(["%.1e" % work_by[s] if work_by.get(s) else "-" for s in sizes],
                                  rotation=40, ha="left", fontsize=8)
            axtop.set_xlabel(wlabel or "work")

        title = app + (f" / {variant}" if variant else "")
        r0 = grp[0]
        title += f"  |  {r0.get('backend','?')} backend, {r0.get('iters','?')} iters"
        ax.set_title(title, fontsize=10, pad=28)
        fig.tight_layout()
        name = "time-" + app + (f"-{variant}" if variant else "")
        _save(fig, figdir, name, dpi)


def plot_graph_stats(rows, cgstats_path, figdir, dpi):
    import matplotlib.pyplot as plt

    tag_app = {r["run_id"]: r["app"] for r in rows}
    # (app, pass) -> {metric: [values]}
    acc = defaultdict(lambda: defaultdict(list))
    metrics = ("nodes_before", "nodes_after", "edges_before", "edges_after", "pass_ms")
    with open(cgstats_path, newline="") as fh:
        for r in csv.DictReader(fh):
            app = tag_app.get(r.get("tag", ""))
            if not app:
                continue
            for m in metrics:
                v = fnum(r.get(m))
                if v is not None:
                    acc[(app, r["pass"])][m].append(v)

    apps = sorted({a for (a, _) in acc})
    for app in apps:
        passes = sorted({p for (a, p) in acc if a == app},
                        key=lambda p: PASS_ORDER.index(p) if p in PASS_ORDER else 99)
        if not passes:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
        for ax, (lo, hi, ttl) in zip(
                axes[:2], [("nodes_before", "nodes_after", "nodes"),
                           ("edges_before", "edges_after", "edges")]):
            xs = range(len(passes))
            before = [median(acc[(app, p)].get(lo, [])) or 0 for p in passes]
            after = [median(acc[(app, p)].get(hi, [])) or 0 for p in passes]
            ax.bar([x - 0.2 for x in xs], before, width=0.4, label="before")
            ax.bar([x + 0.2 for x in xs], after, width=0.4, label="after")
            ax.set_xticks(list(xs))
            ax.set_xticklabels(passes, rotation=30, ha="right", fontsize=7)
            ax.set_ylabel(ttl)
            ax.legend(fontsize=7)
            ax.grid(axis="y", ls=":", alpha=0.6)
            ax.set_axisbelow(True)
        # third panel: per-pass wall time
        axt = axes[2]
        ms = [median(acc[(app, p)].get("pass_ms", [])) or 0 for p in passes]
        axt.bar(list(range(len(passes))), ms, width=0.6, color="tab:gray")
        axt.set_xticks(list(range(len(passes))))
        axt.set_xticklabels(passes, rotation=30, ha="right", fontsize=7)
        axt.set_ylabel("pass_ms")
        axt.grid(axis="y", ls=":", alpha=0.6)
        axt.set_axisbelow(True)
        fig.suptitle(f"{app}: CGIR command-graph reduction per pass", fontsize=10)
        fig.tight_layout()
        _save(fig, figdir, f"graph-{app}", dpi)


def _save(fig, figdir, name, dpi):
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / f"{name}.png"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"wrote {path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(APPS_OPENMP / "results"))
    ap.add_argument("--runs", default="", help="runs.csv (default: <outdir>/runs.csv)")
    ap.add_argument("--cgstats", default="", help="cgstats.csv (default: <outdir>/cgstats.csv)")
    ap.add_argument("--figdir", default="", help="figure dir (default: <outdir>/figures)")
    ap.add_argument("--logy", action="store_true", help="logarithmic y axis (time plot)")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    outdir = Path(args.outdir)
    runs_csv = Path(args.runs) if args.runs else outdir / "runs.csv"
    cgstats_csv = Path(args.cgstats) if args.cgstats else outdir / "cgstats.csv"
    figdir = Path(args.figdir) if args.figdir else outdir / "figures"

    if not runs_csv.exists():
        ap.error(f"no runs.csv at {runs_csv} (run evaluate.py first)")
    rows = load_runs(runs_csv)
    if not rows:
        ap.error("runs.csv has no successful rows")

    plot_time(rows, figdir, args.dpi, args.logy)
    if cgstats_csv.exists():
        plot_graph_stats(rows, cgstats_csv, figdir, args.dpi)
    else:
        print(f"(no {cgstats_csv}; skipping CGIR graph-stats plot)", file=sys.stderr)


if __name__ == "__main__":
    main()
