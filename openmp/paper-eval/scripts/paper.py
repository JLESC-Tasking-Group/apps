#!/usr/bin/env python3
"""paper.py <paper dir> [machine for the tables, default gh200]: the paper's generated
artifacts, from the sweeps of <paper-eval>/{a100,gh200,gb300} (sweep.sh).

Writes, in <paper dir>:
  sections/generated-table-speedup.tex table*: speedup over XKOMP synchronous, per problem,
                                      one column group per machine
  sections/generated-table-cost.tex   recorded graph size, replay time, pass costs and
                                      break-evens (table machine)
and prints, on stdout, every number the text quotes (geomeans, ranges, ratios).

Every time is the steady-state mean of one run (instances W.. of <m>/warmup.env), as in
table.py; a problem whose synchronous run is missing is skipped. Answers are not
re-checked here: table.py does it (section "Checks" of <m>/table.md).
"""
import csv, sys, math, statistics as st, collections
from pathlib import Path

PE = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1])
TM = sys.argv[2] if len(sys.argv) > 2 else "gh200"
MACHINES = [("a100", "A100"), ("gh200", "GH200"), ("gb300", "GB300")]

R = "reduce-node,transitive-reduction"
CFG = {
    "synchronous": "sync",
    "no-taskgraph": "notg",
    "taskgraph:none": "tg",
    f"taskgraph:{R}": "red",
    f"taskgraph:{R},jit": "jit",
    f"taskgraph:{R},jit,prog-fuse": "pf",
    f"taskgraph:{R},jit,prog-fuse,batch": "full",
    f"taskgraph:{R},batch": "rp",
}
# application families of the figure, in order; key -> (label, app, variant or None)
FAM = [("krylov", "Krylov", "krylov", None), ("lulesh", "LULESH", "lulesh", None),
       ("mnmg", "MNMG", "mnmg", None), ("cloverleaf", "CloverLeaf", "cloverleaf", None),
       ("sw4lite", "SW4lite", "sw4lite", None), ("miniweather", "miniWeather", "miniweather", None),
       ("j1d", "jacobi-1d", "polybench", "jacobi-1d"), ("j2d", "jacobi-2d", "polybench", "jacobi-2d"),
       ("h3d", "heat-3d", "polybench", "heat-3d"), ("fdtd", "fdtd-2d", "polybench", "fdtd-2d")]
PRETTY = {"krylov": "Krylov", "lulesh": "LULESH", "mnmg": "MNMG", "cloverleaf": "CloverLeaf",
          "sw4lite": "SW4lite", "miniweather": "miniWeather", "polybench": "PolyBench"}
f = lambda x: float(x) if x not in ("", None) else None
geo = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def load(m):
    """runs[(app, variant, size)][(cfg, tag, vanilla)] = row; cg[run_id][pass] = row"""
    runs = collections.defaultdict(dict)
    for sub, van in (("xkomp", False), ("vanilla", True)):
        p = PE / m / sub / "runs.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            c = CFG.get(r["config"])
            if c is None or r["status"] not in ("ok", "ok_crashed_at_exit") or f(r["avg_ms"]) is None:
                continue
            k = (r["app"], r["variant"], int(r["size"]))
            runs[k][(c, r.get("tag") or "", van)] = r
    cg = collections.defaultdict(dict)
    p = PE / m / "xkomp" / "cgstats.csv"
    for r in csv.DictReader(open(p)):
        cg[r["tag"]][r["pass"]] = r
    return runs, cg


def t(runs, k, c, tag="", van=False):
    r = runs[k].get((c, tag, van))
    return f(r["avg_ms"]) if r else None


def cost(runs, cg, k, c, tag=""):
    """{pass: ms} and total, of run (k, c, tag)"""
    r = runs[k].get((c, tag, False))
    ps = cg.get(r["run_id"]) if r else None
    if not ps:
        return None, None
    per = collections.defaultdict(float)
    for n, row in ps.items():
        per[n] += f(row["pass_ms"]) or 0.0
    return per, sum(per.values())


def nodes(ps, p, when):
    v = f(ps.get(p, {}).get(f"nodes_{when}")) if ps else None
    return None if v is None else max(int(v) - 2, 0)     # minus the virtual entry/exit


def edges(ps, p, when):
    v = f(ps.get(p, {}).get(f"edges_{when}")) if ps else None
    return None if v is None else int(v)


def num(x, d=0):
    if x is None:
        return "--"
    s = f"{x:,.{d}f}"
    return s.replace(",", "{,}")


data = {m: load(m) for m, _ in MACHINES}

# --------------------------------------------------------------------------- speedup table
# speedup table: one row per problem, one column group per machine
TCOLS = [("vnotg", "\\shortstack{KMP\\\\no-tg}"), ("notg", "\\shortstack{no-\\\\tg}"), ("tg", "tg"),
         ("red", "\\shortstack{+re-\\\\duce}"), ("jit", "\\shortstack{+\\\\JIT}"), ("pf", "\\shortstack{+prog-\\\\fuse}"),
         ("full", "\\shortstack{+pa-\\\\cking}"), ("rp", "\\shortstack{red.+\\\\pack.}")]
summary = {}


def sp(runs, k, s):
    b = t(runs, k, "sync")
    x = t(runs, k, "notg", van=True) if s == "vnotg" else t(runs, k, s)
    return (b / x) if (b and x) else None


def fmt_sp(x):
    if x is None:
        return "--"
    return f"{x:.1f}" if x >= 9.995 else f"{x:.2f}"


ORDER = lambda k: (PRETTY[k[0]].lower(), k[1], k[2])
allk = sorted({k for m, _ in MACHINES for k in data[m][0]}, key=ORDER)
rows = []
prev = None
prevv = None
for k in allk:
    app = PRETTY[k[0]]
    label = app if app != prev else ""
    var = k[1] if (k[0], k[1]) != prevv else ""
    if prev is not None and app != prev:
        rows.append("    \\midrule")
    prev, prevv = app, (k[0], k[1])
    cells = [label, var, num(k[2])]
    for m, _ in MACHINES:
        runs = data[m][0]
        best = max((sp(runs, k, s) or 0) for s, _ in TCOLS)
        for s, _ in TCOLS:
            v = sp(runs, k, s)
            c = fmt_sp(v)
            cells.append(f"\\textbf{{{c}}}" if v and v == best else c)
    rows.append("    " + " & ".join(cells) + " \\\\")
geos = ["\\textbf{geomean}", "", ""]
for m, _ in MACHINES:
    runs = data[m][0]
    gs = []
    for s, _ in TCOLS:
        v = [sp(runs, k, s) for k in runs if sp(runs, k, s)]
        gs.append(geo(v))
        for key, _, app, var in FAM:
            vf = [sp(runs, k, s) for k in runs if k[0] == app and (var is None or k[1] == var) and sp(runs, k, s)]
            summary[(m, key, s)] = geo(vf)
    best = max(g for g in gs if g)
    geos += [f"\\textbf{{{fmt_sp(g)}}}" if g == best else fmt_sp(g) for g in gs]
n = len(TCOLS)
lines = ["% Generated by paper-eval/scripts/paper.py. Do not edit by hand.",
         "\\begin{table*}[t]", "  \\centering",
         "  \\caption{Speedup of the replay over XKOMP \\code{synchronous}, per problem and platform. "
         "KMP no-tg: the LLVM OpenMP runtime (\\code{target nowait} tasks; it does not support "
         "\\code{taskgraph}). From \\code{tg} (\\code{taskgraph}, no pass) to \\code{+packing}, the "
         "passes \\code{reduce}, \\code{JIT}, \\code{prog-fuse} and \\code{packing} are added "
         "cumulatively; \\code{red.+pack.} runs \\code{reduce} and \\code{packing} only. "
         "In bold, the best configuration of each problem and platform.}",
         "  \\label{tbl:speedup}", "  {\\scriptsize", "  \\setlength{\\tabcolsep}{1.7pt}",
         "  \\begin{tabular}{@{}l l r " + " ".join(["|" + " r" * n] * len(MACHINES)) + "@{}}", "    \\toprule",
         "     & & & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{M}}}}}" for _, M in MACHINES) + " \\\\",
         "    " + " ".join(f"\\cmidrule(lr){{{4 + i * n}-{3 + (i + 1) * n}}}" for i in range(len(MACHINES))),
         "    \\textbf{Application} & \\textbf{Variant} & $s$ & " + " & ".join(lbl for _ in MACHINES for _, lbl in TCOLS) + " \\\\",
         "    \\midrule"] + rows + ["    \\midrule", "    " + " & ".join(geos) + " \\\\",
         "    \\bottomrule", "  \\end{tabular}}", "\\end{table*}"]
(OUT / "sections" / "generated-table-speedup.tex").write_text("\n".join(lines) + "\n")

# --------------------------------------------------------------------------- tables
runs, cg = data[TM]
order = sorted(runs, key=ORDER)
order = [k for k in order if t(runs, k, "sync")]


def ms(x):
    """a cost in ms, compact: integers from 100 ms, one decimal below"""
    return "--" if x is None else num(x, 0 if x >= 99.95 else 1)


def rt(x):
    """a replay time in ms, three significant digits"""
    return "--" if x is None else num(x, 3 if x < 1 else 2 if x < 10 else 1 if x < 100 else 0)


def recorded(k):
    """#nodes/#edges of the command graph as recorded (before any pass, full pipeline run)"""
    r = runs[k].get(("full", "", False))
    ps = cg.get(r["run_id"]) if r else None
    first = next((p for p in ("reduce-node", "transitive-reduction", "jit", "prog-fuse", "pack") if ps and p in ps), None)
    return nodes(ps, first, "before"), edges(ps, first, "before")



body = []
incremental = []     # (problem, cold, cached): printed, no longer a column
name_of = lambda k: f"{PRETTY[k[0]]} {k[1]} {k[2]}".replace("  ", " ")
prev = prevv = None
for k in order:
    app = PRETTY[k[0]]
    label = app if app != prev else ""
    var = k[1] if (k[0], k[1]) != prevv else ""
    if prev is not None and app != prev:
        body.append("    \\midrule")
    prev, prevv = app, (k[0], k[1])
    v0, e0 = recorded(k)
    nt = t(runs, k, "notg")
    tf, trp = t(runs, k, "full"), t(runs, k, "rp")
    pc, tc = cost(runs, cg, k, "full")
    pw, tw = cost(runs, cg, k, "full", "jit-disk-warm")
    pr, trc = cost(runs, cg, k, "rp")
    def be(total, tnew, tref):
        g = (tref - tnew) if (tref is not None and tnew is not None) else None
        return (total / g) if (total is not None and g and g > 0) else None
    red = (pc["reduce-node"] + pc["transitive-reduction"]) if pc else None
    # incremental break-even of JIT+prog-fuse over reduce+packing
    inc_c = be((tc - trc) if (tc is not None and trc is not None) else None, tf, trp)
    inc_w = be((tw - trc) if (tw is not None and trc is not None) else None, tf, trp)
    body.append("    " + " & ".join([
        label, var, num(k[2]), f"{num(e0)}/{num(v0)}", rt(tf), ms(red),
        ms(pc.get("prog-fuse") if pc else None), ms(pw.get("prog-fuse") if pw else None),
        ms(pc.get("jit") if pc else None), ms(pw.get("jit") if pw else None),
        ms(pc.get("pack") if pc else None),
        ms(tc), ms(tw), num(be(tc, tf, nt)), num(be(tw, tf, nt))]) + " \\\\")
    if trp and tf and trp > 1.02 * tf:
        incremental.append((name_of(k), inc_c, inc_w))
costt = ["% Generated by paper-eval/scripts/paper.py. Do not edit by hand.",
         "\\begin{table*}[t]", "  \\centering",
         f"  \\caption{{Problems ({dict(MACHINES)[TM]}): size of the recorded command graph (\\#edge/\\#node), replay time "
         f"of one iteration with the full pipeline, and optimization cost, in milliseconds. "
         f"\\emph{{cold}}: first run, \\emph{{cached}}: a later run reading the on-disk cache of "
         f"\\code{{prog-fuse}} and \\code{{JIT}}. Break-even: replayed iterations to repay the "
         f"optimization cost against \\code{{no-taskgraph}} (--: never).}}",
         "  \\label{tbl:cost}", "  {\\scriptsize", "  \\setlength{\\tabcolsep}{2.2pt}",
         "  \\begin{tabular}{@{}l l r r r r r r r r r r r r r@{}}", "    \\toprule",
         "     & & & & & & \\multicolumn{2}{c}{\\code{prog-fuse}} & \\multicolumn{2}{c}{\\code{JIT}} & & "
         "\\multicolumn{2}{c}{\\textbf{total}} & \\multicolumn{2}{c}{\\textbf{break-even}} \\\\",
         "    \\cmidrule(lr){7-8} \\cmidrule(lr){9-10} \\cmidrule(lr){12-13} \\cmidrule(lr){14-15}",
         "    \\textbf{Application} & \\textbf{Variant} & $s$ & \\textbf{\\#edge/\\#node} & \\textbf{replay} & "
         "\\code{reduce} & cold & cached & cold & cached & \\code{packing} & cold & cached & cold & cached \\\\",
         "    \\midrule"] + body + ["    \\bottomrule", "  \\end{tabular}}", "\\end{table*}"]
(OUT / "sections" / "generated-table-cost.tex").write_text("\n".join(costt) + "\n")

# --------------------------------------------------------------------------- numbers for the text
print("== geomean speedup over XKOMP sync, all problems")
for m, M in MACHINES:
    runs, _ = data[m]
    ks = [k for k in runs if t(runs, k, "sync")]
    row = []
    for s in ("vsync", "vnotg", "notg", "tg", "red", "jit", "pf", "full", "rp"):
        v = []
        for k in ks:
            b = t(runs, k, "sync")
            x = (t(runs, k, s[1:], van=True) if s.startswith("v") else t(runs, k, s))
            if b and x:
                v.append(b / x)
        row.append(f"{s}={geo(v):.2f}[{min(v):.2f},{max(v):.2f}]n{len(v)}")
    print(M, " ".join(row))
print("== geomean over no-taskgraph / over KMP sync / over KMP no-tg, full and rp")
for m, M in MACHINES:
    runs, _ = data[m]
    out = []
    for ref, van in (("notg", False), ("sync", True), ("notg", True)):
        for s in ("full", "rp"):
            v = [t(runs, k, ref, van=van) / t(runs, k, s) for k in runs if t(runs, k, ref, van=van) and t(runs, k, s)]
            out.append(f"{s}/{'K' if van else ''}{ref}={geo(v):.2f}[{min(v):.2f},{max(v):.2f}]")
    print(M, " ".join(out))
print("== per pass step ratio (geomean, [min,max]) over the previous config")
for m, M in MACHINES:
    runs, _ = data[m]
    out = []
    for a, b in (("notg", "sync"), ("tg", "notg"), ("red", "tg"), ("jit", "red"), ("pf", "jit"), ("full", "pf"), ("rp", "red"), ("full", "rp")):
        v = [t(runs, k, b) / t(runs, k, a) for k in runs if t(runs, k, a) and t(runs, k, b)]
        out.append(f"{a}/{b}={geo(v):.2f}[{min(v):.2f},{max(v):.2f}]med{st.median(v):.2f}")
    print(M, " ".join(out))
print("== per family (machine: family -> full, rp, full/rp)")
for m, M in MACHINES:
    print(M, " ".join(f"{key}:{summary[(m,key,'full')]:.2f}/{summary[(m,key,'rp')]:.2f}" for key, *_ in FAM))
print("== costs per machine: sum over problems of total cold / cached, full and rp (s)")
for m, M in MACHINES:
    runs, cg = data[m]
    tc = sum((cost(runs, cg, k, "full")[1] or 0) for k in runs)
    tw = sum((cost(runs, cg, k, "full", "jit-disk-warm")[1] or 0) for k in runs)
    tr = sum((cost(runs, cg, k, "rp")[1] or 0) for k in runs)
    print(f"{M} full cold {tc/1e3:.1f} s, full cached {tw/1e3:.1f} s, rp {tr/1e3:.1f} s")

print("== in-run stability: stddev/mean of the steady state (%), per config, [median, p90, max] over problems")
for m, M in MACHINES:
    runs, _ = data[m]
    out = []
    for c in ("notg", "tg", "full", "rp"):
        v = sorted(100 * f(r["stddev_ms"]) / f(r["avg_ms"]) for k in runs for (cc, tag, van), r in runs[k].items()
                   if cc == c and not tag and not van and f(r["stddev_ms"]) is not None and f(r["avg_ms"]))
        out.append(f"{c}: med {st.median(v):.1f} p90 {v[int(0.9*len(v))]:.1f} max {v[-1]:.1f}")
    print(M, " | ".join(out))

print("== incremental break-even of JIT+prog-fuse over reduce+packing (iterations; only where full is >2% faster)")
for p, c, w in incremental:
    print(f"{p}: cold {c:.0f} cached {w:.0f}" if c is not None and w is not None else f"{p}: --")
