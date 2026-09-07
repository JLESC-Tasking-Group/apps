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

The two above are the exploratory views: one figure per app, every metric, no
editorial choices. The paper's two artifacts are opt-in, and make choices:

  * --paper -> paper-speedup : one panel per app, speedup over --baseline, the
    incremental pipeline as bars and the references (synchronous, and the
    hand-written CUDA implementations passed with --external) as markers.
  * --latex-table PATH : the statistics table, joining runs.csv with cgstats.csv
    and jitstats.csv -- the recorded graph, what each pass did to it, what the
    passes cost, and how many replays repay that cost against --baseline.

Both normalize against the same --baseline, so they cannot disagree.

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

A run's `unroll` (iterations folded into ONE taskgraph instance) becomes a ` u<N>`
suffix on the configuration label when it is >1, so an unroll sweep plots as extra
series next to the configuration it unrolls; see load_runs().
"""

import argparse
import csv
import math
import re
import sys
import zlib
from collections import OrderedDict, defaultdict
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

# Statuses whose numbers are usable. `ok_crashed_at_exit` is a run that printed
# its complete result and then died on the way out (a teardown bug in the app):
# the measurement is there and throwing it away would lose data, so it is kept
# and reported. `wrong_answer` is deliberately absent -- see report_answers().
OK_STATUS = ("ok", "ok_crashed_at_exit", "", None)

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
    """All rows (including failed ones, so coverage can be reported).

    The unroll -- iterations folded into ONE taskgraph instance -- is folded into
    the configuration label here, at the single point where runs.csv is read, so
    every consumer downstream (grouping, styling, the speedup tables, the figures)
    treats `taskgraph:none u4` as its own series without further plumbing. u=1 is
    left unsuffixed: it is the un-unrolled behaviour and the label older CSVs
    (which have no `unroll` column at all) already carry."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        try:
            u = int(r.get("unroll") or 1)
        except ValueError:
            u = 1
        if u > 1:
            r["config"] = f"{r['config']} u{u}"
    return rows


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
_UNROLL_SUFFIX = re.compile(r"\s+u(\d+)$")


def canon_config(label):
    """Style key for a configuration label: equal pass sets -> equal key.

    `taskgraph:transitive-reduction,reduce-node` and
    `taskgraph:reduce-node,transitive-reduction` are the same series, and an
    empty opt list is `taskgraph:none`. Only the *key* is canonicalized; the
    legend keeps the CSV label verbatim, so the incremental "+ one pass" reading
    order of a cumulative sweep survives.

    A trailing ` u<N>` (the unroll, appended by load_runs) is held aside and
    re-attached after canonicalization, so it stays a distinct series instead of
    being mistaken for a CGIR pass name.
    """
    label = label.strip()
    m = _UNROLL_SUFFIX.search(label)
    unroll = ""
    if m:
        unroll, label = " u" + m.group(1), label[:m.start()]
    return _canon_passes(label) + unroll


def canon_pipeline(label):
    """canon_config() without the unroll: the pass set alone.

    The unroll is a property of the *run*, not of the pipeline, so anything that
    selects "the rows of pipeline X" (the table) must ignore it, while anything
    that draws one series per run (the figures) must not."""
    return _canon_passes(_UNROLL_SUFFIX.sub("", label.strip()))


def _canon_passes(label):
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

def select_evaluated(rows):
    """Keep only the app variants the app registry still lists.

    appspecs.py is what defines the evaluation, so a variant dropped from an
    AppSpec is dropped from the figures and tables too -- without re-running the
    sweep, and without the two ever disagreeing about what was evaluated. Data
    for an unknown app is left alone: the registry describes the apps it drives,
    not every CSV it may be pointed at."""
    try:
        from appspecs import APPS
    except ImportError:
        return rows

    kept, dropped = [], defaultdict(int)
    for r in rows:
        spec = APPS.get(r["app"])
        v = r.get("variant", "")
        if spec is None or not v or v in spec.variants:
            kept.append(r)
        else:
            dropped[f"{r['app']}/{v}"] += 1
    for name, n in sorted(dropped.items()):
        print(f"  {name} is no longer in the app registry: {n} row(s) omitted",
              file=sys.stderr)
    return kept


def aggregate_repeats(rows):
    """Reduce the repeats of a configuration (evaluate.py --repeat) to one row.

    A repeat is a whole separate process, so it re-measures everything a single
    run holds fixed by accident: JIT output, page cache, clock and power state.
    That is the variance that matters on short problems -- LULESH at n=16 moved
    26% between repeats of a pass that provably changes nothing -- and a run's
    own `stddev_ms`, taken across the instances inside one process, cannot see
    any of it.

    The survivor carries the MEDIAN `avg_ms`, not the mean: repeats are a small
    sample and their outliers are one-sided (a process can be slowed by the
    machine, never sped up by it), so one bad repeat would drag a mean but not a
    median. `lo_ms`/`hi_ms` keep the extremes for the error bars and `nrep`
    records how many runs are behind the number.

    Rows are keyed on everything that defines a measurement, so anything that
    is not the same experiment stays separate. Runs with no repeat column (any
    CSV written before --repeat existed) key on their own run_id and so pass
    through untouched, as nrep=1."""
    groups = OrderedDict()
    for r in rows:
        if r.get("rep", "") == "":
            key = ("__single__", r.get("run_id", id(r)))
        else:
            key = (r.get("app", ""), r.get("variant", ""), r.get("config", ""),
                   r.get("opt", ""), r.get("backend", ""), r.get("size", ""),
                   r.get("unroll", ""), r.get("grain", ""), r.get("iters", ""),
                   r.get("tag", ""), r.get("env", ""))
        groups.setdefault(key, []).append(r)

    out, nrep_seen = [], defaultdict(int)
    for key, grp in groups.items():
        times = sorted(t for t in (fnum(g.get("avg_ms")) for g in grp)
                       if t is not None and t > 0)
        if not times:
            out.append(dict(grp[0], nrep="0")
                       if len(grp) == 1 else dict(grp[0], nrep=str(len(grp))))
            continue
        # Keep the repeat whose time is the median, so every other column
        # (answer, residual, fom, run_id) stays consistent with the time
        # reported next to it rather than being blended across processes.
        med = times[len(times) // 2]
        rep = min(grp, key=lambda g: abs((fnum(g.get("avg_ms")) or 1e30) - med))
        out.append(dict(rep, avg_ms=f"{med:.6g}",
                        lo_ms=f"{times[0]:.6g}", hi_ms=f"{times[-1]:.6g}",
                        nrep=str(len(times)),
                        # cgstats/jitstats join on run_id, so the survivor has to
                        # carry its siblings' ids for the cost table to reduce
                        # the per-pass times over the same runs as the replay.
                        rep_run_ids=" ".join(g.get("run_id", "") for g in grp)))
        nrep_seen[len(times)] += 1

    if nrep_seen and set(nrep_seen) != {1}:
        counts = ", ".join(f"{n} repeat(s) x{c}" for n, c in sorted(nrep_seen.items()))
        print(f"  repeats collapsed to medians: {counts}", file=sys.stderr)
        # A configuration measured fewer times than the rest is a single sample
        # wearing the same error bar as the others; say which, rather than let
        # it look equally well measured.
        most = max(nrep_seen, key=lambda n: nrep_seen[n])
        thin = sorted({n for n in nrep_seen if n < most})
        if thin:
            print(f"  WARNING: some configurations have only {thin} repeat(s) "
                  f"where most have {most}: their spread is not comparable",
                  file=sys.stderr)
    return out


def report_answers(rows, reference):
    """Drop every run that did not compute the same thing as the reference.

    An optimization pass may only make a program faster, never make it compute
    something else -- so a configuration whose answer differs from the reference
    configuration's is not a data point, it is a bug report. This exists because
    it did happen: prog-fuse used to collapse two `omp target` launches into one
    without checking the dependence, which removed the device-wide barrier
    between them and moved Krylov CG's residual from 4e-15 to 2.6e-03 while
    reporting a healthy speedup. Nothing in the harness noticed.

    Comparison is against the reference configuration of the SAME (app, variant,
    size, backend) rather than against a fixed threshold, because "did the solver
    converge" is a property of the problem while "did the passes change the
    result" is the property under test. An app that fails to converge in every
    configuration is reported separately and kept.

    Returns the rows that agree. Also honours a run's own verdict, if it has one.
    """
    try:
        from appspecs import APPS
    except ImportError:
        APPS = {}

    def key(r):
        return (r["app"], r.get("variant", ""), r.get("size", ""), r.get("backend", ""))

    refs = {}
    for r in rows:
        if canon_pipeline(r["config"]) == canon_pipeline(reference):
            v = fnum(r.get("answer"))
            if v is not None:
                refs[key(r)] = v

    kept, rejected, unchecked = [], [], 0
    for r in rows:
        verdict = str(r.get("verdict", "")).strip().lower()
        if verdict in ("fail", "failed"):
            rejected.append((r, f"the app's own verdict is '{verdict}'"))
            continue

        got = fnum(r.get("answer"))
        want = refs.get(key(r))
        if got is None or want is None:
            unchecked += 1
            kept.append(r)
            continue

        spec = APPS.get(r["app"])
        rtol = getattr(spec, "answer_rtol", 1e-6)
        atol = getattr(spec, "answer_atol", 0.0)
        # Relatively close, OR both below the level at which the quantity stops
        # meaning anything. The second clause is what makes this usable on a
        # converged solver's residual, whose correct value is zero and whose
        # digits are therefore round-off -- see AppSpec.answer_atol.
        close = abs(got - want) <= rtol * max(abs(want), abs(got))
        both_negligible = abs(got) <= atol and abs(want) <= atol
        if close or both_negligible:
            kept.append(r)
        else:
            rejected.append((r, f"answer {got:.6g} != {want:.6g} "
                                f"(reference '{reference}', rtol {rtol:g}, "
                                f"atol {atol:g})"))

    print("answer check (every configuration must reproduce the reference's result):",
          file=sys.stderr)
    if not refs:
        print(f"  WARNING: no '{reference}' run carries an answer -- nothing was "
              f"checked. The speedups below are unverified.", file=sys.stderr)
    print(f"  {len(kept)} verified/unchecked, {len(rejected)} rejected"
          f"{f', {unchecked} had no answer to compare' if unchecked else ''}",
          file=sys.stderr)
    for r, whys in rejected:
        name = f"{r['app']}/{r['variant']}" if r.get("variant") else r["app"]
        print(f"  REJECTED {name} n={r.get('size','?')} u={r.get('unroll','?')} "
              f"[{r['config']}]: {whys}", file=sys.stderr)
    return kept


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


# --------------------------------------------------------------------------- #
# The paper figure and the paper table.
#
# plot_time / plot_graph_stats above are the exploratory views: one figure per
# app, every metric, no editorial choices. The two functions below are the
# opposite -- they produce the two artifacts the evaluation section has room for,
# and therefore make choices (which baseline, which panels, which columns).
# --------------------------------------------------------------------------- #

# Short legend labels for the incremental pipeline, keyed by canonical pass set.
# The figure has four bars per group and no room for a pass list in each; the
# `opt` column of runs.csv stays the ground truth.
PAPER_LABELS = {
    "taskgraph:none":                                                "record/replay",
    "taskgraph:reduce-node,transitive-reduction":                    "+reduce",
    "taskgraph:reduce-node,transitive-reduction,jit":                "+jit",
    "taskgraph:reduce-node,transitive-reduction,prog-fuse,jit":      "+prog-fuse",
    "taskgraph:reduce-node,transitive-reduction,prog-fuse,jit,sequence,batch":
                                                                     "+packing",
}

# Configurations drawn as point markers rather than bars: they are references,
# not steps of the pipeline, and a bar each would double the width of the figure.
MARKERS = {
    "synchronous": dict(marker="x", color="black", s=42, zorder=5,
                        label="synchronous"),
    "no-taskgraph": dict(marker="_", color="black", s=64, zorder=5,
                         label="no-taskgraph (=1)"),
}


def paper_label(config):
    """Legend text for a configuration.

    The paper's pipelines get their short name. Anything else is abbreviated to
    "+<last pass>" -- the incremental sweep reads as "this bar adds that pass to
    the one on its left", so the last pass is the only informative part, and a
    full pass list would make the legend wider than the figure."""
    key = canon_pipeline(config)
    if key in PAPER_LABELS:
        return PAPER_LABELS[key]
    head, _, opt = key.partition(":")
    if head != "taskgraph" or not opt:
        return config
    passes = [p for p in opt.split(",") if p]
    return "+" + passes[-1] if passes else config


def load_external(path):
    """Read the hand-written reference implementations, which no sweep produces.

    These are the CUDA baselines that ship with two of the apps (MNMGDatalog's
    v2_cudagraph in tc.cu). They are not configurations of
    our stack -- different source, different toolchain -- so they are not in
    runs.csv and are supplied as their own small file:

        app,variant,size,label,avg_ms
        mnmg,,7035,CUDA graph (hand-written),0.061

    Returns {(app, variant, size): (label, avg_ms)}."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            v = fnum(r.get("avg_ms"))
            if v is None or v <= 0:
                continue
            out[(r["app"], r.get("variant", "") or "", int(r["size"]))] = \
                (r.get("label", "reference"), v)
    return out


def _panel_series(grp, xkey):
    """Reduce a panel's rows to {config: {x: avg_ms}} plus the ordered x values.

    `xkey` is "size" or "variant": which of the two the panel varies. Whichever
    it is NOT must be constant within the panel; the caller guarantees that by
    filtering, so a leftover duplicate (a re-run) simply takes the last value.

    Also returns the repeat extremes as {config: {x: (lo, hi)}}, from the
    lo_ms/hi_ms that aggregate_repeats() left on the row. A row measured once
    has no spread, so it reports its own value for both -- an interval of zero
    width, which is the truthful statement that nothing is known about how much
    it would move on a second run."""
    data, spread = defaultdict(dict), defaultdict(dict)
    order = []
    for r in grp:
        v = fnum(r.get("avg_ms"))
        if v is None or v <= 0:
            continue
        x = int(r["size"]) if xkey == "size" else (r.get("variant") or "")
        data[r["config"]][x] = v
        spread[r["config"]][x] = (fnum(r.get("lo_ms")) or v, fnum(r.get("hi_ms")) or v)
        if x not in order:
            order.append(x)
    return data, sorted(order, key=lambda x: (isinstance(x, str), x)), spread


def _speedup_err(heights, xs, cfg, data, spread, ref, ref_spread):
    """Asymmetric yerr for a row of speedup bars, or None if nothing varied.

    A speedup is a ratio of two measured times, so its uncertainty comes from
    both: the widest ratio consistent with the repeats is the fastest baseline
    over the slowest pipeline run, and the narrowest is the reverse. Taking the
    extremes of both is deliberately conservative -- it is the range of
    speedups you could have reported by pairing any baseline repeat with any
    pipeline repeat, so a bar whose interval clears 1.0 clears it under every
    pairing, not just the flattering one.

    Returns None when every interval has zero width (no repeats anywhere), so a
    single-sample figure draws no bars rather than a row of degenerate ticks
    implying a precision that was never measured."""
    lo_err, hi_err, any_spread = [], [], False
    for h, x in zip(heights, xs):
        if h != h or x not in data.get(cfg, {}) or x not in ref:
            lo_err.append(0.0); hi_err.append(0.0)
            continue
        t_lo, t_hi = spread.get(cfg, {}).get(x, (data[cfg][x], data[cfg][x]))
        r_lo, r_hi = ref_spread.get(x, (ref[x], ref[x]))
        s_lo, s_hi = r_lo / t_hi, r_hi / t_lo
        if t_hi > t_lo or r_hi > r_lo:
            any_spread = True
        # Clamp: the median ratio need not sit inside the corner-to-corner
        # range, and a negative yerr would draw the bar inside out.
        lo_err.append(max(h - s_lo, 0.0))
        hi_err.append(max(s_hi - h, 0.0))
    return [lo_err, hi_err] if any_spread else None


EXTERNAL_LEGEND = "hand-written CUDA"


def plot_paper_speedup(rows, figdir, dpi, fmt, show, styles, baseline,
                       external, panel_size, apps_order, figsize=None,
                       unroll=None):
    """The end-to-end figure: one panel per app, speedup over `baseline`.

    Speedup rather than time because the four apps differ by two orders of
    magnitude in absolute time; a shared y axis of milliseconds would compress
    three panels into a line. The baseline (`no-taskgraph` by default: the same
    program, same task granularity, without record/replay) is the honest
    reference for "what does the command graph buy", and is drawn at y=1.

    Exactly one unroll is drawn (`unroll`, default the largest in the input).
    Drawing several would double the bars per group while the legend -- which
    names pipelines, not runs -- could not tell them apart; the unroll sweep is
    reported in the table instead.
    """
    import matplotlib.pyplot as plt

    # load_runs() folded the unroll into the config label; undo that selection
    # here rather than plumb a second key through _panel_series().
    def run_unroll(r):
        try:
            return int(r.get("unroll") or 1)
        except ValueError:
            return 1

    us = {run_unroll(r) for r in rows}
    keep = unroll if unroll in us else (max(us) if us else 1)

    # Show each series at the requested unroll, or -- when it has no run there --
    # at the largest one it does have. An app that cannot be unrolled is not an
    # app to leave out of the figure: GMRES is pinned to u=1 because a restarted
    # solver ends each restart with a host solve the next one consumes, so two
    # restarts cannot share a graph instance. Filtering on the unroll alone drew
    # its panel empty.
    best = {}
    for r in rows:
        u = run_unroll(r)
        if u > keep:
            continue
        k = (r["app"], r.get("variant", ""), r.get("size", ""),
             canon_pipeline(r["config"]))
        if k not in best or u > run_unroll(best[k]):
            best[k] = r

    # Drop the ` u<N>` that load_runs() folded into the label. The figure now
    # shows one series per pipeline, each at its own best unroll, so the unroll is
    # no longer what tells two series apart -- and leaving it in would make a
    # pipeline that ran at u=1 a *different* series from the same pipeline at
    # u=8, giving the group extra bars and pushing them out of position. Copy
    # rather than edit in place: these rows are shared with the other figures.
    rows = []
    for r in best.values():
        r = dict(r)
        r["config"] = _UNROLL_SUFFIX.sub("", r["config"])
        rows.append(r)

    if len(us) > 1:
        print(f"  paper figure: unroll {keep} where available (of {sorted(us)})",
              file=sys.stderr)

    try:
        from appspecs import APPS
    except ImportError:
        APPS = {}

    # Group rows into panels. An app whose panel varies the variant is pinned to
    # one size (the largest it has, or --panel-size) so the five solvers are
    # compared on equal work; an app whose panel varies the size keeps them all.
    panels = []
    for app in apps_order:
        grp = [r for r in rows if r["app"] == app]
        if not grp:
            continue
        spec = APPS.get(app)
        xkey = getattr(spec, "panel_x", "size") if spec else "size"
        if xkey == "variant":
            sizes = sorted({int(r["size"]) for r in grp if r.get("size", "")})
            if not sizes:
                continue
            pinned = panel_size if panel_size in sizes else sizes[-1]
            grp = [r for r in grp if int(r["size"]) == pinned]
            sub = f"n={pinned}"
        else:
            variants = sorted({r.get("variant", "") for r in grp})
            if len(variants) > 1:
                # A size-panel of an app with variants would overlay them; keep
                # the first so the figure cannot silently average two solvers.
                grp = [r for r in grp if r.get("variant", "") == variants[0]]
                sub = variants[0]
            else:
                sub = variants[0] if variants[0] else ""
        title = (getattr(spec, "pretty", None) or app) + (f" ({sub})" if sub else "")
        panels.append((app, title, xkey, grp))

    if not panels:
        print("  (no rows for the paper figure)", file=sys.stderr)
        return

    # Sized for a two-column figure* whose final width is the ~7in text block:
    # the aspect ratio is what survives the scaling, and a wider-than-3.5:1 figure
    # would end up too short to read once shrunk.
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(figsize or (2.8 * len(panels), 3.1)),
                             squeeze=False)
    axes = axes[0]
    handles = {}

    for ax, (app, title, xkey, grp) in zip(axes, panels):
        data, xs, spread = _panel_series(grp, xkey)
        ref = next((v for c, v in data.items()
                    if canon_pipeline(c) == canon_pipeline(baseline)), None)
        ref_spread = next((v for c, v in spread.items()
                           if canon_pipeline(c) == canon_pipeline(baseline)), {})
        if not ref:
            ax.set_title(f"{title}\n(no '{baseline}')")
            print(f"  WARNING: {app}: no '{baseline}' run -> panel left empty",
                  file=sys.stderr)
            continue
        # An x value the baseline does not cover cannot be turned into a speedup;
        # dropping it is honest, keeping an empty slot would read as "no gain".
        dropped = [x for x in xs if x not in ref]
        xs = [x for x in xs if x in ref]
        if dropped:
            print(f"  WARNING: {app}: no '{baseline}' at {dropped} -> omitted",
                  file=sys.stderr)
        if not xs:
            ax.set_title(f"{title}\n(no '{baseline}')")
            continue

        # Bars: the pipeline, in the order the sweep produced it (which is the
        # order passes are added), skipping the reference series.
        bars = [c for c in ordered_unique(r["config"] for r in grp)
                if canon_pipeline(c) != canon_pipeline(baseline) and c not in MARKERS]
        width = 0.8 / max(len(bars), 1)
        idx = list(range(len(xs)))
        whisker_top = 0.0
        for i, c in enumerate(bars):
            h = [(ref[x] / data[c][x]) if (x in data[c] and x in ref) else float("nan")
                 for x in xs]
            offs = [xi - 0.4 + width * (i + 0.5) for xi in idx]
            b = ax.bar(offs, h, width, label=paper_label(c),
                       **styles[canon_config(c)], **BAR_EDGE)
            handles.setdefault(paper_label(c), b)
            err = _speedup_err(h, xs, c, data, spread, ref, ref_spread)
            if err is not None:
                ax.errorbar(offs, h, yerr=err, fmt="none", ecolor="black",
                            elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=5)
                whisker_top = max([whisker_top]
                                  + [y + e for y, e in zip(h, err[1]) if y == y])

        # References as markers, at the group centre.
        for c, kw in MARKERS.items():
            if c not in data or canon_pipeline(c) == canon_pipeline(baseline):
                continue
            ys = [(ref[x] / data[c][x]) if (x in data[c] and x in ref) else float("nan")
                  for x in xs]
            kw = dict(kw)
            lbl = kw.pop("label")
            sc = ax.scatter(idx, ys, **kw)
            handles.setdefault(lbl, sc)

        # The speedups this panel is actually about: our own bars.
        finite = [ref[x] / data[c][x] for c in bars for x in xs
                  if x in data[c] and data[c][x] > 0]
        # Freeze the y range on them (and on the markers already drawn). An
        # external implementation far outside it would flatten every bar in the
        # panel, so it is clipped -- and named, so the clipping is never silent.
        # Whiskers included: a clipped one reads as a shorter one, i.e. as more
        # certainty than was measured.
        top = max(ax.get_ylim()[1], (max(finite) if finite else 0.0) * 1.15,
                  whisker_top * 1.05, 1.2)
        ax.set_ylim(0, top)

        # The hand-written CUDA implementation of this app, where there is one.
        ext = [external.get((app, grp[0].get("variant", ""), x)) if xkey == "size"
               else external.get((app, x, int(grp[0]["size"]))) for x in xs]
        if any(e for e in ext):
            ys = [(ref[x] / e[1]) if (e and x in ref) else float("nan")
                  for x, e in zip(xs, ext)]
            over = [f"{x}:{y:.2g}" for x, y in zip(xs, ys) if y == y and y > top]
            if over:
                print(f"  WARNING: {app}: hand-written reference off scale at "
                      f"{', '.join(over)} (axis capped at {top:.2g})",
                      file=sys.stderr)
            sc = ax.scatter(idx, ys, marker="*", s=90, color="black",
                            zorder=6, label=EXTERNAL_LEGEND, clip_on=True)
            handles.setdefault(EXTERNAL_LEGEND, sc)

        # A whole panel one or two orders of magnitude away from 1 is almost never
        # a real result: it means the baseline measured something else -- typically
        # an asynchronous configuration whose tasks were still in flight when it
        # stopped the clock. Say so, rather than let the panel into the paper.
        m = median(finite)
        if m is not None and not (0.05 <= m <= 20.0):
            print(f"  WARNING: {app}: median speedup vs '{baseline}' is {m:.3g} "
                  f"-- implausible; check that '{baseline}' drains its tasks before "
                  f"it stops the clock", file=sys.stderr)

        ax.axhline(1.0, color="black", lw=0.8, ls="--", zorder=1)
        ax.set_xticks(idx)
        labels = [str(x) for x in xs]
        # Solver names are long enough to collide at this panel width.
        rot = 25 if max((len(l) for l in labels), default=0) > 4 else 0
        ax.set_xticklabels(labels, rotation=rot,
                           ha="right" if rot else "center")
        ax.set_xlabel("problem size" if xkey == "size" else "solver")
        ax.set_title(title)
        ax.grid(axis="y", ls=":", alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(f"speedup over\n{baseline}")
    # One legend for the whole figure, below it: four panels each carrying the
    # same five entries would cost more area than the panels.
    fig.legend(handles.values(), handles.keys(), loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=min(len(handles), 4), frameon=False)
    fig.tight_layout()
    _save(fig, figdir, "paper-speedup", dpi, fmt, show)


def plot_paper_unroll(rows, figdir, dpi, fmt, show, styles, baseline,
                      apps_order, tag=None, figsize=None):
    """Speedup against `baseline` as a function of the unroll, one panel per app.

    A taskgraph instance ends in a taskwait: every task it recorded must finish
    before the next instance starts. The un-recorded program has no such point
    and lets consecutive iterations overlap, so record/replay begins one barrier
    per iteration behind it, and on an app whose iteration does not by itself
    fill the device that deficit is larger than anything the passes recover.
    Unrolling folds `u` iterations into one instance, which pays the barrier
    once per `u` instead of once per iteration.

    This is the figure that says whether a command graph is worth recording at
    all for a given app: where the curve crosses 1.0 is the unroll at which the
    graph stops costing more synchronization than it saves.

    The baseline runs no taskgraph, so it has no unroll to vary (evaluate.py
    pins it to u=1) -- it is the same horizontal reference at every x."""
    import matplotlib.pyplot as plt

    def run_unroll(r):
        try:
            return int(r.get("unroll") or 1)
        except ValueError:
            return 1

    # One sweep only. The unroll sweep is usually tagged so it does not merge
    # with the main one, and mixing them here would put two measurements of the
    # same unroll on the same curve -- the second silently replacing the first.
    if tag is not None:
        rows = [r for r in rows if (r.get("tag") or "") == tag]
        if not rows:
            print(f"  no runs tagged '{tag or '(untagged)'}': skipping the "
                  f"unroll figure", file=sys.stderr)
            return
    else:
        tags = {(r.get("tag") or "") for r in rows}
        if len(tags) > 1:
            print(f"  WARNING: unroll figure spans {len(tags)} sweeps "
                  f"({', '.join(sorted(t or '(untagged)' for t in tags))}): two "
                  f"runs at the same unroll will collide. Pass --unroll-tag",
                  file=sys.stderr)

    panels = []
    for app in apps_order:
        grp = [r for r in rows if r["app"] == app and fnum(r.get("avg_ms"))]
        # One variant per panel: two solvers' curves overlaid in one axes would
        # read as one app measured twice.
        variants = sorted({r.get("variant", "") for r in grp})
        if len(variants) > 1:
            grp = [r for r in grp if r.get("variant", "") == variants[0]]
        # And one size: the crossing point moves with problem size, so mixing
        # sizes into a single curve would average two different answers.
        sizes = sorted({int(r["size"]) for r in grp if r.get("size", "")})
        if not sizes:
            continue
        grp = [r for r in grp if int(r["size"]) == sizes[-1]]
        if len({run_unroll(r) for r in grp}) < 2:
            continue                      # nothing to say about unroll here
        panels.append((app, variants[0], sizes[-1], grp))

    if not panels:
        print("  (no app has more than one unroll: skipping the unroll figure. "
              "Run evaluate.py --unroll 1,2,4,8,16)", file=sys.stderr)
        return

    try:
        from appspecs import APPS
    except ImportError:
        APPS = {}

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(figsize or (2.9 * len(panels), 2.9)),
                             squeeze=False)
    axes = axes[0]
    handles = {}

    for ax, (app, variant, size, grp) in zip(axes, panels):
        # {pipeline: {unroll: (median, lo, hi)}}, the unroll taken from the
        # column rather than the label suffix load_runs() added.
        series = defaultdict(dict)
        ref = None
        for r in grp:
            pipe = canon_pipeline(_UNROLL_SUFFIX.sub("", r["config"]))
            v = fnum(r.get("avg_ms"))
            lo, hi = fnum(r.get("lo_ms")) or v, fnum(r.get("hi_ms")) or v
            if pipe == canon_pipeline(baseline):
                ref = (v, lo, hi)
                continue
            # The other references (synchronous, ...) run no taskgraph either, so
            # they have no unroll to vary: evaluate.py pins them to u=1 and a
            # "curve" through their single point would suggest they were measured
            # across the axis and found flat.
            if pipe in MARKERS:
                continue
            series[pipe][run_unroll(r)] = (v, lo, hi)
        if ref is None:
            ax.set_title(f"{app}\n(no '{baseline}')")
            print(f"  WARNING: {app}: no '{baseline}' run -> unroll panel empty",
                  file=sys.stderr)
            continue

        for pipe, by_u in series.items():
            us = sorted(by_u)
            ys = [ref[0] / by_u[u][0] for u in us]
            # Same corner-to-corner interval as the bar figure.
            err = [[max(y - ref[1] / by_u[u][2], 0.0) for y, u in zip(ys, us)],
                   [max(ref[2] / by_u[u][1] - y, 0.0) for y, u in zip(ys, us)]]
            st = styles[canon_config(pipe)]
            ln = ax.errorbar(us, ys, yerr=(err if any(any(e) for e in err) else None),
                             marker="o", ms=4, lw=1.4, capsize=2,
                             color=st["color"], label=paper_label(pipe))
            handles.setdefault(paper_label(pipe), ln)

        ax.axhline(1.0, color="black", lw=0.8, ls="--", zorder=1)
        ax.set_xscale("log", base=2)
        all_u = sorted({u for by_u in series.values() for u in by_u})
        ax.set_xticks(all_u)
        ax.set_xticklabels([str(u) for u in all_u])
        ax.minorticks_off()
        ax.set_xlabel("iterations per taskgraph instance")
        spec = APPS.get(app)
        pretty = (getattr(spec, "pretty", None) or app)
        sub = f"{variant} " if variant else ""
        ax.set_title(f"{pretty} ({sub}n={size})")
        ax.grid(ls=":", alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(f"speedup over\n{baseline}")
    fig.legend(handles.values(), handles.keys(), loc="upper center",
               bbox_to_anchor=(0.5, 0.0), ncol=min(len(handles), 4), frameon=False)
    fig.tight_layout()
    _save(fig, figdir, "paper-unroll", dpi, fmt, show)


def _cg_by_tag(cgstats_path):
    """cgstats.csv indexed as {tag: {pass: row}} -- the per-pass graph metrics of
    one run, keyed by the run_id it joins on."""
    out = defaultdict(dict)
    with open(cgstats_path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r.get("tag", "")][r.get("pass", "")] = r
    return out


def _jit_by_tag(jitstats_path):
    out = {}
    if not Path(jitstats_path).exists():
        return out
    with open(jitstats_path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r.get("tag", "")] = r
    return out


def _fmt(v, prec=0):
    if v is None:
        return "--"
    return f"{v:.{prec}f}" if prec else f"{int(round(v))}"


# The canonical order the library applies passes in, and the label each gets in
# the cost table. reduce-node and transitive-reduction are two passes in the
# implementation and one -- `reduce` -- in the paper, so their times are summed.
COST_PASSES = [
    ("reduce",     ["reduce-node", "transitive-reduction"]),
    ("prog-fuse",  ["prog-fuse"]),
    ("JIT",        ["jit"]),
    ("packing",    ["sequence", "batch"]),
]


def _cost_rows(rows, cgstats_path, jitstats_path, baseline, baseline_tag,
               full_pipeline, tag):
    """Per-problem cost and amortization for the runs of `full_pipeline` tagged
    `tag` (an empty tag matches the untagged main sweep).

    Returns {(app, variant, size, backend): dict}, each holding the per-pass
    milliseconds, the total, and the replays needed to repay it against
    `baseline`. Split out so the same computation serves the main sweep and the
    warm-cache sweep, which differ only in which rows they read.

    The baseline is taken from `baseline_tag` alone, not from whichever matching
    row happens to come last. It does not run a CGIR pass, so every sweep that
    includes it measures the same thing -- but not to the same digits, and using
    a different one per regime would put run-to-run noise into the difference
    between the columns, which is the comparison the table exists to make."""
    cg = _cg_by_tag(cgstats_path) if Path(cgstats_path).exists() else {}

    def key(r):
        return (r["app"], r.get("variant", ""), r.get("size", ""), r.get("backend", ""))

    base = {}
    for r in rows:
        if (canon_pipeline(r["config"]) == canon_pipeline(baseline)
                and (r.get("tag") or "") == baseline_tag):
            v = fnum(r.get("avg_ms"))
            if v:
                base[key(r)] = v
    if not base:
        print(f"  no '{baseline}' run tagged '{baseline_tag or '(untagged)'}': "
              f"break-even cannot be computed (see --baseline-tag)", file=sys.stderr)

    out = {}
    for r in rows:
        if canon_pipeline(r["config"]) != canon_pipeline(full_pipeline):
            continue
        if (r.get("tag") or "") != tag:
            continue
        # Every repeat of this configuration that cgstats knows about. Optimizing
        # is measured per run like replaying is, so a single run's pass times are
        # a single sample of it too.
        ids = [i for i in (r.get("rep_run_ids") or r["run_id"]).split() if i in cg]
        if not ids:
            continue
        passes = cg.get(r["run_id"]) or cg[ids[0]]

        def pass_ms(name):
            """Median of `name` over the repeats, or None if none recorded it."""
            vs = sorted(v for v in (fnum(cg[i][name].get("pass_ms"))
                                    for i in ids if name in cg[i]) if v is not None)
            return vs[len(vs) // 2] if vs else None

        per = {}
        for label, names in COST_PASSES:
            ms = [pass_ms(n) for n in names]
            per[label] = sum(v for v in ms if v is not None)
        # Summed from the same medians as the columns, so the total is the total
        # of what the table shows rather than a separately-medianed number the
        # columns do not add up to.
        all_names = {n for i in ids for n in cg[i]}
        total = sum(v for v in (pass_ms(n) for n in sorted(all_names)) if v is not None)

        t_new = fnum(r.get("avg_ms"))
        t_ref = base.get(key(r))
        gain = (t_ref - t_new) if (t_ref and t_new) else None
        per["total"] = total
        per["replay"] = t_new
        per["breakeven"] = (total / gain) if (gain and gain > 0) else None
        # Keep the best (largest-unroll) row per problem, as the graph table does.
        k = key(r)
        if k not in out or (fnum(r.get("unroll")) or 1) >= out[k].get("_unroll", 0):
            per["_unroll"] = fnum(r.get("unroll")) or 1
            out[k] = per
    return out


def _emit_table(caption, label, colspec, header, body, out):
    lines = [
        "% Generated by scripts/plot.py. Do not edit by hand.",
        "\\begin{table*}[t]",
        "  \\centering",
        "  \\caption{" + caption + "}",
        "  \\label{" + label + "}",
        "  {\\footnotesize",
        "  \\begin{tabular}{" + colspec + "}",
        "    \\toprule",
        "    " + " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\",
        "    \\midrule",
    ]
    lines += ["    " + " & ".join(c) + " \\\\" for c in body]
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table*}"]
    text = "\n".join(lines) + "\n"
    if out == "-":
        print(text)
    else:
        Path(out).write_text(text)
        print(f"wrote {out}", file=sys.stderr)


def latex_table_graph(rows, cgstats_path, full_pipeline, out):
    """The applications and what the passes do to their command graph.

    One row per problem: the graph as recorded, after the reduction passes, and
    after packing. This is the evidence for section 5.2 -- what the passes
    change -- and it doubles as the table describing the applications."""
    try:
        from appspecs import APPS
    except ImportError:
        APPS = {}

    cg = _cg_by_tag(cgstats_path) if Path(cgstats_path).exists() else {}

    # One row per problem, taken from its largest unroll -- the same recording the
    # cost table and the figure report. Collected into a dict first: appending as
    # rows arrive would emit one row per unroll.
    best = {}
    for r in rows:
        if canon_pipeline(r["config"]) != canon_pipeline(full_pipeline):
            continue
        if not cg.get(r["run_id"]):
            continue
        k = (r["app"], r.get("variant", ""), r.get("size", ""))
        u = fnum(r.get("unroll")) or 1
        if k not in best or u > (fnum(best[k].get("unroll")) or 1):
            best[k] = r

    body = []
    for k in sorted(best, key=lambda k: (k[0], k[1], int(k[2] or 0))):
        r = best[k]
        passes = cg[r["run_id"]]

        def before(p, m): return fnum(passes.get(p, {}).get(m + "_before"))
        def after(p, m):  return fnum(passes.get(p, {}).get(m + "_after"))

        # Every command graph is bracketed by a virtual entry and exit node --
        # an implementation convenience that gives each dependence a single
        # attachment point, not work the device performs. Reporting them would
        # put a floor of two on every count and make a fully packed graph read as
        # three nodes instead of one, so they are subtracted here.
        def user_nodes(v): return None if v is None else max(v - 2, 0)

        first = next((p for p in PASS_ORDER if p in passes), None)
        v0, e0 = user_nodes(before(first, "nodes")), before(first, "edges")
        v1 = user_nodes(after("transitive-reduction", "nodes")
                        or after("reduce-node", "nodes"))
        e1 = after("transitive-reduction", "edges") or after("reduce-node", "edges")
        vp = user_nodes(after("batch", "nodes") or after("sequence", "nodes"))

        spec = APPS.get(r["app"])
        body.append([
            (getattr(spec, "pretty", None) or r["app"])
            + (f" {r['variant']}" if r.get("variant") else ""),
            getattr(spec, "klass", "") or "",
            str(r.get("size", "")),
            f"{_fmt(v0)}/{_fmt(e0)}",
            f"{_fmt(v1)}/{_fmt(e1)}",
            _fmt(vp),
        ])

    _emit_table(
        "Applications, and what the passes do to the command graph recorded for "
        "one iteration. Nodes are given before any pass, after the two reduction "
        "steps, and after packing; the virtual entry and exit nodes that bracket "
        "every graph are excluded, so a fully packed graph is one node.",
        "tbl:apps", "@{}l l r r r r@{}",
        ["Application", "Class", "Size", "$|V|/|E|$", "after \\code{reduce}",
         "after \\code{packing}"],
        body, out)


def latex_table_cost(rows, cgstats_path, jitstats_path, baseline, baseline_tag,
                     full_pipeline, cached_tag, out):
    """What the passes cost, and how many replays repay it.

    Two break-even columns. `cold` is what a run pays when nothing has been
    compiled before; `cached` is what every later run of the same application
    pays, once the JIT's on-disk cache holds the compiled kernels. They differ by
    orders of magnitude because JIT dominates the cost, so reporting only one of
    them would misrepresent the system in one direction or the other."""
    try:
        from appspecs import APPS
    except ImportError:
        APPS = {}

    cold   = _cost_rows(rows, cgstats_path, jitstats_path, baseline, baseline_tag,
                        full_pipeline, "")
    cached = _cost_rows(rows, cgstats_path, jitstats_path, baseline, baseline_tag,
                        full_pipeline, cached_tag)
    if not cached:
        print(f"  no runs tagged '{cached_tag}': the cached break-even column will "
              f"be empty (see --cached-tag)", file=sys.stderr)

    body = []
    for k in sorted(cold, key=lambda k: (k[0], k[1], int(k[2] or 0))):
        c = cold[k]
        w = cached.get(k)
        spec = APPS.get(k[0])
        body.append([
            (getattr(spec, "pretty", None) or k[0]) + (f" {k[1]}" if k[1] else ""),
            str(k[2]),
            _fmt(c["reduce"], 1),
            _fmt(c["prog-fuse"], 1),
            _fmt(c["JIT"], 1),
            _fmt(c["packing"], 1),
            _fmt(c["total"] / 1000.0, 2),
            _fmt(c["replay"], 3),
            _fmt(c["breakeven"]),
            _fmt(w["breakeven"]) if w else "--",
        ])

    _emit_table(
        "Cost of one run of the pipeline, per pass and in total, next to the time "
        "of the single replay it is optimizing, and the number of replays that "
        "repays it against \\texttt{no-taskgraph}. Per-pass times and one replay "
        "are in milliseconds. The break-even column is a first run, which compiles "
        "every kernel; \\emph{cached} is any later run of the same application, "
        "served from the JIT's on-disk cache.",
        "tbl:cost", "@{}l r r r r r r r r r@{}",
        ["Application", "Size", "\\code{reduce}", "\\code{prog-fuse}",
         "\\code{JIT}", "\\code{packing}", "total (s)", "one replay",
         "break-even", "\\;cached"],
        body, out)

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
    ap.add_argument("--answer-reference", default="synchronous",
                    help="configuration whose computed answer every other one must "
                    "reproduce (default: synchronous, the simplest schedule)")
    ap.add_argument("--no-answer-check", action="store_true",
                    help="do not drop runs whose answer disagrees with the reference. "
                    "Only for inspecting a known-broken sweep -- results produced with "
                    "this flag are unverified")
    ap.add_argument("--paper", action="store_true",
                    help="also render the paper figure (paper-speedup.<fmt>): one panel "
                    "per app, speedup over --baseline, the pipeline as bars and the "
                    "references as markers")
    ap.add_argument("--paper-unroll-figure", action="store_true",
                    help="also render paper-unroll.<fmt>: speedup over --baseline "
                    "against the number of iterations folded into one taskgraph "
                    "instance. A taskgraph instance ends in a taskwait while the "
                    "un-recorded program overlaps iterations freely, so this is "
                    "where the curve crosses 1.0 that says whether recording a "
                    "graph pays for that barrier. Needs evaluate.py --unroll with "
                    "more than one value")
    ap.add_argument("--unroll-tag", default=None, metavar="TAG",
                    help="runs.csv `tag` the unroll figure reads, to keep it to the "
                    "one sweep that varied the unroll (\"\" selects the untagged "
                    "main sweep). Without it the figure spans every sweep and two "
                    "runs at the same unroll collide")
    ap.add_argument("--baseline", default="no-taskgraph",
                    help="configuration the paper figure and the break-even column "
                    "normalize against (default: no-taskgraph)")
    ap.add_argument("--panel-size", type=int, default=0,
                    help="problem size at which an app whose panel varies the variant "
                    "(krylov) is compared; default: its largest size")
    ap.add_argument("--apps", default="krylov,lulesh,mnmg",
                    help="comma list fixing the panel order of the paper figure")
    ap.add_argument("--external", default="",
                    help="CSV of hand-written reference implementations to overlay "
                    "(app,variant,size,label,avg_ms); see load_external()")
    ap.add_argument("--paper-unroll", type=int, default=0,
                    help="unroll the paper figure shows (default: the largest in the "
                    "input). One only: the legend names pipelines, not runs")
    ap.add_argument("--paper-figsize", default="", metavar="W,H",
                    help="override the paper figure size in inches (default: "
                    "2.8 per panel x 3.1)")
    ap.add_argument("--latex-tables", default="", metavar="DIR",
                    help="write the paper's two LaTeX tables into DIR: "
                    "generated-table-graph.tex (what the passes do to the graph) and "
                    "generated-table-cost.tex (what they cost, and the replays that "
                    "repay it). Joins runs.csv with cgstats.csv/jitstats.csv")
    ap.add_argument("--baseline-tag", default="",
                    help="runs.csv `tag` whose --baseline rows are used as THE baseline "
                    "for every regime (default: the untagged main sweep). Pinning it "
                    "keeps the cost table's columns comparable when several sweeps "
                    "measured the same reference")
    ap.add_argument("--cached-tag", default="jit-disk-warm",
                    help="runs.csv `tag` identifying the warm on-disk-JIT-cache sweep, "
                    "which supplies the cost table's cached break-even column. Absent "
                    "from the input, that column is left empty")
    ap.add_argument("--pipeline",
                    default="taskgraph:reduce-node,transitive-reduction,jit,prog-fuse,"
                            "sequence,batch",
                    help="configuration the LaTeX table reports (default: the full "
                    "pipeline)")
    ap.add_argument("--jitstats", default="", help="jitstats.csv (default: <outdir>/jitstats.csv)")
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
    jitstats_csv = Path(args.jitstats) if args.jitstats else outdir / "jitstats.csv"
    figdir = Path(args.figdir) if args.figdir else outdir / "figures"

    if not runs_csv.exists():
        ap.error(f"no runs.csv at {runs_csv} (run evaluate.py first)")

    allrows = load_runs(runs_csv)
    if not allrows:
        ap.error(f"{runs_csv} is empty")
    rows = select_evaluated(report_coverage(allrows))
    if not any(r.get("avg_ms", "") != "" for r in rows):
        ap.error("no plottable rows (see the coverage report above)")

    # Before anything is plotted or tabulated: a run that computed the wrong
    # answer is not a slower or faster data point, it is not a data point.
    if not args.no_answer_check:
        rows = report_answers(rows, args.answer_reference)
        if not rows:
            ap.error("every run was rejected by the answer check")

    # After the answer check (a rejected repeat must not enter a median) and
    # before anything reads a time, so the figures and both tables see exactly
    # one row per configuration and need no notion of repeats themselves.
    rows = aggregate_repeats(rows)

    if args.reference:
        report_speedups(rows, args.reference)

    if args.dump_cgstats or args.dump_cgstats_all:
        if cgstats_csv.exists():
            dump_cgstats(rows, cgstats_csv, args.dump_cgstats_all)
        else:
            ap.error(f"no cgstats.csv at {cgstats_csv} (it is written by "
                     f"evaluate.py unless --no-stats)")

    if args.latex_tables:
        if not cgstats_csv.exists():
            ap.error(f"--latex-tables needs {cgstats_csv} (written by evaluate.py "
                     f"unless --no-stats)")
        d = Path(args.latex_tables)
        d.mkdir(parents=True, exist_ok=True)
        latex_table_graph(rows, cgstats_csv, args.pipeline,
                          str(d / "generated-table-graph.tex"))
        latex_table_cost(rows, cgstats_csv, jitstats_csv, args.baseline,
                         args.baseline_tag, args.pipeline, args.cached_tag,
                         str(d / "generated-table-cost.tex"))

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
    if args.paper:
        if args.external and not Path(args.external).exists():
            ap.error(f"--external: no such file {args.external}")
        external = load_external(args.external) if args.external else {}
        figsize = None
        if args.paper_figsize:
            try:
                w, h = (float(x) for x in args.paper_figsize.split(","))
                figsize = (w, h)
            except ValueError:
                ap.error("--paper-figsize: expected W,H in inches")
        plot_paper_speedup(rows, figdir, args.dpi, args.format, args.show, styles,
                           args.baseline, external, args.panel_size,
                           [a.strip() for a in args.apps.split(",") if a.strip()],
                           figsize, args.paper_unroll or None)
    if args.paper_unroll_figure and not args.no_figures:
        plot_paper_unroll(rows, figdir, args.dpi, args.format, args.show, styles,
                          args.baseline,
                          [a.strip() for a in args.apps.split(",") if a.strip()],
                          args.unroll_tag)
    if cgstats_csv.exists():
        plot_graph_stats(rows, cgstats_csv, figdir, args.dpi, args.format, args.show)
    else:
        print(f"(no {cgstats_csv}; skipping CGIR graph-stats plot)", file=sys.stderr)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
