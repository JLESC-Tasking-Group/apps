#!/usr/bin/env python3
"""
plot.py - plot the sweep produced by scripts/evaluate.py.

Reads results/runs.csv (and optionally results/cgstats.csv) and writes figures
under results/figures (PDF by default; see --format):

  * time-<app>[-<variant>] : grouped bars of avg execution time / iteration
    (ms) with stddev error bars, one group per problem size and one bar per
    configuration (synchronous / no-taskgraph / taskgraph:none / taskgraph:<opt>).
    The problem size is on the bottom axis and the work (FLOPs / zones / tokens)
    on the top axis.
  * graph-<app> : per-CGIR-pass command-graph reduction (nodes & edges
    before -> after) and per-pass wall time, from cgstats.csv joined on the run
    tag. Only the taskgraph configurations contribute here.

A per-(app, variant) coverage summary is printed first, so runs that failed (and
are therefore not plottable) are reported rather than silently omitted.

Figures follow the conference camera-ready guidance: all text is >=10pt (default
leading is ~1.2x, i.e. >=12pt), and series are distinguished by hatch patterns +
black edges (not colour alone) so they stay readable when printed in grayscale
without magnification. Uses only matplotlib + the standard library.

A series' colour and hatch are derived from the *name* of the configuration it
shows (its set of CGIR optimization passes), never from its position in the
figure, so a configuration looks the same in every figure -- including figures
that omit some of the other passes. Configurations are styled once for the whole
input file, so all figures of a single run are mutually consistent; see
assign_styles() and --style-salt.
"""

import argparse
import csv
import math
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path

APPS_OPENMP = Path(__file__).resolve().parent.parent

# Canonical CGIR pipeline order for the graph-stats x axis.
PASS_ORDER = ["copy-fuse", "reduce-node", "transitive-reduction",
              "prog-fuse", "jit", "sequence", "batch"]

# Every <name>_before/<name>_after metric pair cgstats.csv reports per pass, in
# the order they are printed by --dump-cgstats. The graph-<app> figure only ever
# shows nodes/edges/pass_ms, so the rest are visible in the dump alone.
CG_METRICS = ["nodes", "edges", "empty", "command", "graph", "prog",
              "copy1d", "copy2d", "batch"]

OK_STATUS = ("ok", "", None)

# Publication style: >=10pt fonts everywhere (default leading is ~1.2x -> >=12pt),
# and hatch patterns + black edges so grouped bars stay distinguishable when the
# figure is printed in grayscale without magnification. See the conference guide.
STYLE = {
    "font.size":            11,
    "axes.titlesize":       12,
    "axes.labelsize":       11,
    "xtick.labelsize":      10,
    "ytick.labelsize":      10,
    "legend.fontsize":      10,
    "legend.title_fontsize": 10,
    "figure.titlesize":     13,
    "hatch.linewidth":      0.6,
    "savefig.bbox":         "tight",
}
# Distinct hatches so each series is identifiable both in color and in grayscale;
# the hatch is the primary discriminator when the figure is printed in grayscale.
HATCHES = ["", "//", "\\\\", "xx", "..", "oo", "++", "--", "||", "OO"]
# Explicit palette. The bar color must be a function of the configuration *name*,
# so it cannot come from the implicit rcParams color cycle (which advances once
# per ax.bar() call, i.e. depends on how many series precede it in the figure).
# Okabe-Ito colorblind-safe core (black swapped for a gray so the hatch stays
# visible on the fill) plus four extras.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
          "#F0E442", "#999999", "#6A3D9A", "#B15928", "#7FBC41", "#DE77AE"]
BAR_EDGE = dict(edgecolor="black", linewidth=0.6)
ERR_KW = dict(elinewidth=1.0, ecolor="black")


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


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load_runs(path):
    """All rows (including failed ones, so coverage can be reported)."""
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def ordered_unique(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


# --------------------------------------------------------------------------- #
# Series styling. A configuration must look the same in every figure, so its
# (color, hatch) is derived from its *name* -- never from its position in the
# figure -- otherwise a plot that omits a pass restyles every series after it.
# --------------------------------------------------------------------------- #
def canon_config(label):
    """Style key for a configuration label: equal pass sets -> equal key.

    `taskgraph:transitive-reduction,reduce-node` and
    `taskgraph:reduce-node,transitive-reduction` are the same series, and an
    empty opt list is `taskgraph:none`. Only the *key* is canonicalized; the
    legend keeps the CSV label verbatim, so the incremental "+ one pass" reading
    order of a cumulative sweep survives.
    """
    head, _, opt = label.partition(":")
    head = head.strip()
    if head != "taskgraph":
        return head or label.strip()
    passes = [p for p in re.split(r"[,\s]+", opt.strip()) if p and p != "none"]
    if not passes:
        return "taskgraph:none"
    # Canonical pipeline order first, unknown pass names last (alphabetically).
    passes = sorted(set(passes),
                    key=lambda p: (PASS_ORDER.index(p) if p in PASS_ORDER
                                   else len(PASS_ORDER), p))
    return "taskgraph:" + ",".join(passes)


def assign_styles(labels, salt=0):
    """Map every configuration label to a stable {color, hatch}, keyed by name.

    Each key hashes to a base (color, hatch) slot; a key whose base color is
    already taken probes forward for a free one, so no two series in the
    assignment share a color (with more keys than colors, only the
    (color, hatch) *pair* is guaranteed unique). Keys are visited in sorted
    order, so the result is a pure function of the *set* of labels and does not
    depend on the order they appear in runs.csv.

    Callers pass the union of every configuration in the input, so all figures
    from one run share one assignment and are coherent by construction. zlib.crc32
    is used rather than hash(), which is per-process randomized by PYTHONHASHSEED
    and would restyle the figures on every invocation.
    """
    nc, nh = len(COLORS), len(HATCHES)
    styles, used_color, used_pair = {}, set(), set()
    for key in sorted({canon_config(x) for x in labels}):
        h = zlib.crc32(("%d\0%s" % (salt, key)).encode("utf-8"))
        base_c, base_h = h % nc, (h // nc) % nh
        slot = None
        for j in range(nc):                 # prefer a color nobody else uses
            c = (base_c + j) % nc
            if c not in used_color:
                slot = (c, base_h)
                break
        if slot is None:                    # >nc configs: keep the pair unique
            for j in range(nc * nh):
                pair = ((base_c + j) % nc, (base_h + j // nc) % nh)
                if pair not in used_pair:
                    slot = pair
                    break
        if slot is None:                    # >nc*nh configs: unavoidable repeat
            slot = (base_c, base_h)
        used_color.add(slot[0])
        used_pair.add(slot)
        styles[key] = {"color": COLORS[slot[0]], "hatch": HATCHES[slot[1]]}
    return styles


def report_styles(styles):
    """Print the name -> style assignment so figures are auditable/reproducible."""
    print("series styles (keyed by configuration name, stable across figures):",
          file=sys.stderr)
    width = max((len(k) for k in styles), default=0)
    for key, st in sorted(styles.items()):
        print(f"  {key:<{width}}  {st['color']}  {st['hatch'] or '(solid)'}",
              file=sys.stderr)


def warn_style_collisions(app, variant, configs, styles):
    """Warn if a figure ends up with two same-color series. Only reachable when
    the input holds more distinct configurations than COLORS has entries."""
    seen = {}
    for c in configs:
        st = styles[canon_config(c)]
        prev = seen.setdefault(st["color"], c)
        if prev is not c:
            name = f"{app}/{variant}" if variant else app
            same = " (and the same hatch: the bars are indistinguishable)" \
                if styles[canon_config(prev)]["hatch"] == st["hatch"] else ""
            print(f"  WARNING: {name}: '{prev}' and '{c}' share a color{same}; "
                  f"add entries to COLORS or re-roll with --style-salt.",
                  file=sys.stderr)


def report_coverage(rows):
    """Print per-(app,variant) plottable/total, warning about groups that have
    runs but nothing plottable (e.g. all failed) -- so they aren't silently
    dropped from the figures. Returns the list of plottable-status rows."""
    groups = {}
    for r in rows:
        key = (r["app"], r.get("variant", ""))
        g = groups.setdefault(key, {"total": 0, "ok": 0, "plottable": 0, "rc": defaultdict(int)})
        g["total"] += 1
        if r.get("status") in OK_STATUS:
            g["ok"] += 1
            if r.get("avg_ms", "") != "":
                g["plottable"] += 1
        else:
            g["rc"][str(r.get("returncode", "?"))] += 1

    print("run coverage (plottable / total):", file=sys.stderr)
    for (app, variant), g in sorted(groups.items()):
        name = f"{app}/{variant}" if variant else app
        line = f"  {name:18s} {g['plottable']}/{g['total']}"
        nfail = g["total"] - g["ok"]
        if nfail:
            line += f"  ({nfail} failed, returncodes {dict(g['rc'])})"
        print(line, file=sys.stderr)
        if g["total"] and not g["plottable"]:
            print(f"    WARNING: {name} has no plottable runs -> omitted from the figures "
                  f"(all failed/empty; check --sizes and the build).", file=sys.stderr)

    return [r for r in rows if r.get("status") in OK_STATUS]

def report_speedups(rows, reference):
    """Print (to stdout) the average speedup of every configuration against the
    reference configuration, per (app, variant). The speedup at a given size is
    ref_time / config_time (>1 = faster than the reference); the reported value is
    the geometric mean over the sizes the two configurations have in common."""
    # groups[(app,variant)][config][size] = avg_ms ; keep sweep order of configs.
    groups = defaultdict(lambda: defaultdict(dict))
    order = defaultdict(list)
    for r in rows:
        if r.get("avg_ms", "") == "" or r.get("size", "") == "":
            continue
        v = fnum(r["avg_ms"])
        if v is None:
            continue
        key = (r["app"], r.get("variant", ""))
        c = r["config"]
        groups[key][c][int(r["size"])] = v
        if c not in order[key]:
            order[key].append(c)

    print(f"average speedup vs reference '{reference}' "
          f"(geomean over sizes; >1 = faster):")
    if not groups:
        print("  (no plottable rows)")
        return
    for key in sorted(groups):
        app, variant = key
        name = f"{app}/{variant}" if variant else app
        data = groups[key]
        if reference not in data:
            print(f"  {name}: reference '{reference}' not found "
                  f"(available: {', '.join(order[key])})")
            continue
        ref = data[reference]
        width = max(len(c) for c in order[key])
        print(f"  {name}:")
        for c in order[key]:
            sp = [ref[s] / data[c][s] for s in data[c]
                  if s in ref and data[c][s] > 0 and ref[s] > 0]
            g = geomean(sp)
            tag = "  (reference)" if c == reference else ""
            cell = "   n/a" if g is None else f"{g:6.2f}x"
            print(f"    {c:<{width}}  {cell}{tag}")


# --------------------------------------------------------------------------- #
# cgstats.csv as text. The graph-<app> figure only draws nodes/edges/pass_ms and
# medians over sizes and configurations; this dump keeps every metric and every
# pipeline separate, which is what you want when checking what a pass did.
# --------------------------------------------------------------------------- #
def _cell(before, after):
    """`before->after`, or just the value when the pass left the metric alone,
    so the entries that actually changed stand out when scanning a column."""
    if before is None and after is None:
        return "-"
    fmt = lambda v: "-" if v is None else ("%d" % v if float(v).is_integer()
                                           else "%g" % v)
    return fmt(before) if before == after else f"{fmt(before)}->{fmt(after)}"


def _widths(headers, *rowsets):
    """Column widths and per-column numeric-ness, shared by several tables so
    they stay aligned with each other and not just internally."""
    rows = [r for rs in rowsets for r in rs]
    w = [max([len(str(headers[i]))] + [len(str(r[i])) for r in rows])
         for i in range(len(headers))]
    num = [all(re.fullmatch(r"[-+0-9.eE>=%]*", str(r[i])) for r in rows)
           for i in range(len(headers))]
    return w, num


def _table(headers, rows, indent="  ", widths=None):
    """Render an aligned ASCII table (left-aligned text, right-aligned numbers)."""
    if not rows:
        return []
    w, num = widths or _widths(headers, rows)
    def line(cells):
        return indent + " ".join(
            (str(c).rjust(w[i]) if num[i] else str(c).ljust(w[i]))
            for i, c in enumerate(cells)).rstrip()
    return [line(headers), indent + "-" * (sum(w) + len(w) - 1)] + \
           [line(r) for r in rows]


def dump_cgstats(rows, cgstats_path, show_all=False):
    """Print every per-pass metric in cgstats.csv as ASCII tables on stdout.

    One table per pipeline -- a (app, variant, config, size) quadruple, i.e. one
    recorded taskgraph -- with the passes in the order they actually ran (`seq`,
    not PASS_ORDER), followed by a one-line-per-pipeline summary. Metrics that
    are zero everywhere are hidden unless show_all, since cgir reports a fixed
    set of counters whether or not a given app exercises them.
    """
    meta = {r["run_id"]: r for r in rows}
    recs, orphans = [], set()
    with open(cgstats_path, newline="") as fh:
        for r in csv.DictReader(fh):
            m = meta.get(r.get("tag", ""))
            if m is None:
                orphans.add(r.get("tag", ""))
            else:
                recs.append((m, r))

    print(f"\ncgstats: per-pass metrics from {cgstats_path}")
    if not recs:
        print("  (no rows join runs.csv on run_id == tag)")
        return

    # Drop counters this run never exercised, so the table stays narrow.
    shown = CG_METRICS if show_all else [
        m for m in CG_METRICS
        if any(fnum(r.get(m + "_before")) or fnum(r.get(m + "_after"))
               for _, r in recs)]
    hidden = [m for m in CG_METRICS if m not in shown]

    pipes = defaultdict(list)
    for m, r in recs:
        pipes[(m["app"], m.get("variant", ""), m["config"], int(m["size"]))].append(r)

    print("  a metric shows as `before->after` when the pass changed it, "
          "and as a bare value when it did not.")

    # Build every table first, so all of them can share one set of column widths
    # and stay comparable by eye from one pipeline to the next.
    blocks, summary = [], []
    for key in sorted(pipes, key=lambda k: (k[0], k[1], k[3], k[2])):
        app, variant, config, size = key
        passes = sorted(pipes[key], key=lambda r: int(r.get("seq") or 0))
        name = f"{app}/{variant}" if variant else app
        body = [[r.get("seq", ""), r.get("pass", ""),
                 "%.3f" % (fnum(r.get("pass_ms")) or 0.0)] +
                [_cell(fnum(r.get(m + "_before")), fnum(r.get(m + "_after")))
                 for m in shown]
                for r in passes]
        total = sum(fnum(r.get("pass_ms")) or 0.0 for r in passes)
        body.append(["", "TOTAL", "%.3f" % total] + [""] * len(shown))
        blocks.append((f"\n  {name}  size={size}  {config}", body))

        row = {"app": name, "size": size, "config": config, "n": len(passes),
               "ms": total}
        for m in ("nodes", "edges"):        # net effect of the whole pipeline
            row[m] = (fnum(passes[0].get(m + "_before")),
                      fnum(passes[-1].get(m + "_after")))
        summary.append(row)

    head = ["seq", "pass", "pass_ms"] + shown
    widths = _widths(head, *[b for _, b in blocks])
    for title, body in blocks:
        print(title)
        for ln in _table(head, body, indent="    ", widths=widths):
            print(ln)

    print("\n  summary (net effect of the whole pipeline, per recorded taskgraph):")
    body = []
    for s in summary:
        cells = [s["app"], s["size"], s["n"], "%.3f" % s["ms"]]
        for m in ("nodes", "edges"):
            b, a = s[m]
            # No percentage when nothing moved: the bare value already says so.
            pct = "" if not b or a == b else "  (%+.1f%%)" % (100.0 * (a - b) / b)
            cells.append(f"{_cell(b, a)}{pct}")
        cells.append(s["config"])
        body.append(cells)
    for ln in _table(["app", "size", "passes", "total_ms", "nodes", "edges",
                      "config"], body, indent="    "):
        print(ln)

    if hidden:
        print(f"\n  hidden (zero in every row): {', '.join(hidden)} "
              f"-- use --dump-cgstats-all to show them.")
    if orphans:
        print(f"  WARNING: {len(orphans)} cgstats tag(s) match no run_id in "
              f"runs.csv and were skipped.")


def plot_time(rows, figdir, dpi, logy, fmt, show, styles):
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
        fig, ax = plt.subplots(figsize=(max(9.0, 1.6 * len(sizes) + 3.0), 5.5))
        # `i` only positions the bar within its group: the style comes from the
        # configuration name, so it is identical in every figure that shows it.
        for i, c in enumerate(configs):
            heights = [(data[c].get(s) or (float("nan"), 0.0))[0] for s in sizes]
            errs = [(data[c].get(s) or (float("nan"), 0.0))[1] for s in sizes]
            offs = [xi - 0.4 + width * (i + 0.5) for xi in x]
            ax.bar(offs, heights, width, yerr=errs, capsize=3, label=c,
                   error_kw=ERR_KW, **styles[canon_config(c)], **BAR_EDGE)
        warn_style_collisions(app, variant, configs, styles)

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("problem size")
        ax.set_ylabel("avg execution time / iteration (ms)")
        if logy:
            ax.set_yscale("log")
        ax.grid(axis="y", ls=":", alpha=0.6)
        ax.set_axisbelow(True)
        # Legend below the axes so >=10pt entries do not overflow the plot area.
        ncol = 2 if len(configs) > 4 else 1
        ax.legend(title="configuration", ncol=ncol, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        # ax.legend(title="configuration", fontsize=8, ncol=2)

        if any(work_by.get(s) for s in sizes):
            axtop = ax.twiny()
            axtop.set_xlim(ax.get_xlim())
            axtop.set_xticks(x)
            axtop.set_xticklabels(["%.1e" % work_by[s] if work_by.get(s) else "-" for s in sizes],
                                  rotation=40, ha="left")
            axtop.set_xlabel(wlabel or "work")

        title = app + (f" / {variant}" if variant else "")
        r0 = grp[0]
        title += f"  |  {r0.get('backend','?')} backend, {r0.get('iters','?')} iters"
        ax.set_title(title, pad=28)
        fig.tight_layout()
        name = "time-" + app + (f"-{variant}" if variant else "")
        _save(fig, figdir, name, dpi, fmt, show)


def plot_graph_stats(rows, cgstats_path, figdir, dpi, fmt, show):
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
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
        for ax, (lo, hi, ttl) in zip(
                axes[:2], [("nodes_before", "nodes_after", "nodes"),
                           ("edges_before", "edges_after", "edges")]):
            xs = range(len(passes))
            before = [median(acc[(app, p)].get(lo, [])) or 0 for p in passes]
            after = [median(acc[(app, p)].get(hi, [])) or 0 for p in passes]
            # grayscale-safe: distinct grays + a hatch on "after".
            ax.bar([x - 0.2 for x in xs], before, width=0.4, label="before",
                   color="0.80", **BAR_EDGE)
            ax.bar([x + 0.2 for x in xs], after, width=0.4, label="after",
                   color="0.45", hatch="//", **BAR_EDGE)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(passes, rotation=30, ha="right")
            ax.set_ylabel(ttl)
            ax.legend()
            ax.grid(axis="y", ls=":", alpha=0.6)
            ax.set_axisbelow(True)
        # third panel: per-pass wall time
        axt = axes[2]
        ms = [median(acc[(app, p)].get("pass_ms", [])) or 0 for p in passes]
        axt.bar(list(range(len(passes))), ms, width=0.6, color="0.6", **BAR_EDGE)
        axt.set_xticks(list(range(len(passes))))
        axt.set_xticklabels(passes, rotation=30, ha="right")
        axt.set_ylabel("pass_ms")
        axt.grid(axis="y", ls=":", alpha=0.6)
        axt.set_axisbelow(True)
        fig.suptitle(f"{app}: CGIR command-graph reduction per pass")
        fig.tight_layout()
        _save(fig, figdir, f"graph-{app}", dpi, fmt, show)


def _save(fig, figdir, name, dpi, fmt, show):
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    path = figdir / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi)
    print(f"wrote {path}", file=sys.stderr)
    if not show:                 # keep figures open for an interactive plt.show()
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=str(APPS_OPENMP / "results"))
    ap.add_argument("--runs", default="", help="runs.csv (default: <outdir>/runs.csv)")
    ap.add_argument("--cgstats", default="", help="cgstats.csv (default: <outdir>/cgstats.csv)")
    ap.add_argument("--figdir", default="", help="figure dir (default: <outdir>/figures)")
    ap.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"],
                    help="output image format (default: pdf)")
    ap.add_argument("--show", action="store_true",
                    help="display the figures interactively (in addition to writing them)")
    ap.add_argument("--logy", action="store_true", help="logarithmic y axis (time plot)")
    ap.add_argument("--reference", default="", help="print (to stdout) the geomean "
                    "speedup of every configuration vs this reference config, per app "
                    "(e.g. --reference no-taskgraph)")
    ap.add_argument("--dump-cgstats", action="store_true",
                    help="print (to stdout) every per-pass metric in cgstats.csv as "
                    "ASCII tables: one per recorded taskgraph, plus a summary. Shows "
                    "the metrics the graph-<app> figure leaves out")
    ap.add_argument("--dump-cgstats-all", action="store_true",
                    help="as --dump-cgstats, but keep the counters that are zero in "
                    "every row")
    ap.add_argument("--no-figures", action="store_true",
                    help="skip the figures and only print the text reports")
    ap.add_argument("--style-salt", type=int, default=0, help="re-roll the "
                    "configuration -> (color, hatch) assignment; applies to every "
                    "figure at once, so they stay consistent with each other")
    ap.add_argument("--style-universe", action="append", default=[], metavar="CSV",
                    help="extra runs.csv whose configurations also take part in the "
                    "style assignment (repeatable). Figures of one runs.csv are always "
                    "mutually consistent; point every invocation at the same superset "
                    "file to keep figures plotted from *different* files consistent too")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")   # headless: only write files
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)  # >=10pt fonts + grayscale-friendly hatch width

    outdir = Path(args.outdir)
    runs_csv = Path(args.runs) if args.runs else outdir / "runs.csv"
    cgstats_csv = Path(args.cgstats) if args.cgstats else outdir / "cgstats.csv"
    figdir = Path(args.figdir) if args.figdir else outdir / "figures"

    if not runs_csv.exists():
        ap.error(f"no runs.csv at {runs_csv} (run evaluate.py first)")

    allrows = load_runs(runs_csv)
    if not allrows:
        ap.error(f"{runs_csv} is empty")
    rows = report_coverage(allrows)
    if not any(r.get("avg_ms", "") != "" for r in rows):
        ap.error("no plottable rows (see the coverage report above)")

    if args.reference:
        report_speedups(rows, args.reference)

    if args.dump_cgstats or args.dump_cgstats_all:
        if cgstats_csv.exists():
            dump_cgstats(rows, cgstats_csv, args.dump_cgstats_all)
        else:
            ap.error(f"no cgstats.csv at {cgstats_csv} (it is written by "
                     f"evaluate.py unless --no-stats)")

    if args.no_figures:
        return

    # One assignment over every configuration in the input, not one per figure:
    # a configuration then keeps its style even in figures that omit some of the
    # others, which is what makes the figures comparable side by side.
    # Same filter on every source: the universe is the configurations that can be
    # drawn, so styling a file directly and styling it via --style-universe agree.
    universe = [r["config"] for r in rows if r.get("avg_ms", "")]
    for extra in args.style_universe:
        if not Path(extra).exists():
            ap.error(f"--style-universe: no such file {extra}")
        universe += [r["config"] for r in load_runs(extra) if r.get("avg_ms", "")]
    styles = assign_styles(universe, args.style_salt)
    report_styles(styles)

    plot_time(rows, figdir, args.dpi, args.logy, args.format, args.show, styles)
    if cgstats_csv.exists():
        plot_graph_stats(rows, cgstats_csv, figdir, args.dpi, args.format, args.show)
    else:
        print(f"(no {cgstats_csv}; skipping CGIR graph-stats plot)", file=sys.stderr)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
