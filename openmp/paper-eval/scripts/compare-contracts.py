#!/usr/bin/env python3
"""compare-contracts.py <A machine dir> <B machine dir>: one machine swept under two
CGIR contract settings (sweep.sh NOALIAS_MODE / BARRIER_MODE), e.g. gb300 (prove,
prove) against gb300-trusted (distinct-base, distinct-base). Markdown on stdout.

  1. execution: steady-state time per iteration, A / B for every CGIR configuration
     (> 1: B runs faster), with the full pipeline's speedup over synchronous in each
  2. compilation: pass cost of the full pipeline (ms, CGIR pass wall time, median of
     the repeats), cold (main sweep, no on-disk cache) and warm (cache read back),
     prog-fuse and JIT separately and in total, A against B
  3. answers: B's runs against B's synchronous (appspecs tolerances), as table.py

Steady times are the steady-state mean of each run (instances W.. of warmup.env);
the two sweeps ran at different times, so a ratio within a few % is noise
(GB300: README, gb300/variance.md)."""
import csv, sys, collections, statistics as st
from pathlib import Path

A, B = Path(sys.argv[1]), Path(sys.argv[2])
R = "reduce-node,transitive-reduction"
CFG = [(f"taskgraph:{R}", "+reduce"), (f"taskgraph:{R},jit", "+jit"),
       (f"taskgraph:{R},jit,prog-fuse", "+prog-fuse"), (f"taskgraph:{R},jit,prog-fuse,batch", "+packing"),
       (f"taskgraph:{R},batch", "reduce+packing")]
FULL = CFG[3][0]
med = lambda v: st.median(v) if v else None
fnum = lambda x: float(x) if x not in ("", None) else None


def load(d):
    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(d / "xkomp/runs.csv")):
        if r["status"] not in ("ok", "ok_crashed_at_exit") or fnum(r["avg_ms"]) is None:
            continue
        k = (r["app"] + (f"/{r['variant']}" if r["variant"] else ""), float(r["size"] or 0), r["size"])
        runs[k][(r["config"], r.get("tag") or "")].append(r)
    cg = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in csv.DictReader(open(d / "xkomp/cgstats.csv")):
        cg[r["tag"]][r["pass"]] += float(r["pass_ms"] or 0)
    return runs, cg


ra, ca = load(A)
rb, cb = load(B)


def t(runs, k, c, tag=""):
    return med([fnum(r["avg_ms"]) for r in runs[k].get((c, tag), [])])


def cost(runs, cg, k, tag, p=None):
    v = []
    for r in runs[k].get((FULL, tag), []):
        ps = cg.get(r["run_id"])
        if ps:
            v.append(sum(ps.values()) if p is None else ps.get(p, 0.0))
    return med(v)


def fmt(x, d=2):
    return "–" if x is None else (f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.{d}f}")


def ratio(a, b, d=2):
    return fmt(a / b, d) if (a and b) else "–"


keys = sorted(set(ra) & set(rb))
print(f"# CGIR contracts: {A.name} vs {B.name}\n")
for d in (A, B):
    s = d / "stack.txt"
    line = next((l for l in s.read_text().splitlines() if l.startswith("CGIR_NOALIAS_MODE")), None) if s.exists() else None
    print(f"- `{d.name}`: {line or 'CGIR_NOALIAS_MODE=prove CGIR_PROG_FUSE_BARRIER_MODE=prove (default; not recorded)'}")
print()

print(f"## 1. Execution: steady time {A.name} / {B.name} (> 1: {B.name} faster)\n")
print("Speedup columns: full pipeline (`+packing`) over XKOMP `synchronous` in each sweep.\n")
print("| app | size | " + " | ".join(l for _, l in CFG) + f" | full speedup {A.name} | full speedup {B.name} |")
print("|---|---:|" + "---:|" * (len(CFG) + 2))
gm = collections.defaultdict(list)
for k in keys:
    cells = []
    for c, l in CFG:
        a, b = t(ra, k, c), t(rb, k, c)
        cells.append(ratio(a, b))
        if a and b:
            gm[l].append(a / b)
    sa, sb = t(ra, k, "synchronous"), t(rb, k, "synchronous")
    fa, fb = t(ra, k, FULL), t(rb, k, FULL)
    print(f"| {k[0]} | {k[2]} | " + " | ".join(cells) +
          f" | {ratio(sa, fa)} | {ratio(sb, fb)} |")
geo = lambda v: (st.geometric_mean(v) if v else None)
print("| **geomean** | | " + " | ".join(f"**{fmt(geo(gm[l]))}**" for _, l in CFG) + " | | |")

print(f"\n## 2. Compilation: pass cost of the full pipeline (ms), {A.name} vs {B.name}\n")
print("cold = main sweep (no on-disk cache); warm = on-disk JIT cache read back.\n")
print(f"| app | size | prog-fuse cold {A.name} | {B.name} | JIT cold {A.name} | {B.name} | "
      f"total cold {A.name} | {B.name} | {A.name}/{B.name} | prog-fuse warm {A.name} | {B.name} | "
      f"total warm {A.name} | {B.name} | {A.name}/{B.name} |")
print("|---|---:|" + "---:|" * 12)
W = "jit-disk-warm"
tot = collections.defaultdict(float)
for k in keys:
    pa, pb = cost(ra, ca, k, "", "prog-fuse"), cost(rb, cb, k, "", "prog-fuse")
    ja, jb = cost(ra, ca, k, "", "jit"), cost(rb, cb, k, "", "jit")
    ta, tb = cost(ra, ca, k, ""), cost(rb, cb, k, "")
    wpa, wpb = cost(ra, ca, k, W, "prog-fuse"), cost(rb, cb, k, W, "prog-fuse")
    wa, wb = cost(ra, ca, k, W), cost(rb, cb, k, W)
    for n, v in (("ta", ta), ("tb", tb), ("wa", wa), ("wb", wb)):
        tot[n] += v or 0
    print(f"| {k[0]} | {k[2]} | {fmt(pa, 1)} | {fmt(pb, 1)} | {fmt(ja, 1)} | {fmt(jb, 1)} | "
          f"{fmt(ta, 1)} | {fmt(tb, 1)} | {ratio(ta, tb)} | {fmt(wpa, 1)} | {fmt(wpb, 1)} | "
          f"{fmt(wa, 1)} | {fmt(wb, 1)} | {ratio(wa, wb)} |")
print(f"| **sum** | | | | | | **{fmt(tot['ta'], 0)}** | **{fmt(tot['tb'], 0)}** | **{ratio(tot['ta'], tot['tb'])}** "
      f"| | | **{fmt(tot['wa'], 0)}** | **{fmt(tot['wb'], 0)}** | **{ratio(tot['wa'], tot['wb'])}** |")

print(f"\n## 3. Answers of {B.name} against its synchronous\n")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
try:
    from appspecs import APPS as SPECS
except Exception:
    SPECS = {}
bad, n = [], 0
for k in sorted(rb):
    spec = SPECS.get(k[0].split("/")[0])
    rtol, atol = (getattr(spec, "answer_rtol", 1e-6), getattr(spec, "answer_atol", 0.0)) if spec else (1e-6, 0.0)
    sync = [r for r in rb[k].get(("synchronous", ""), []) if r.get("answer")]
    if not sync:
        continue
    ref = st.median(float(r["answer"]) for r in sync)
    for (c, tag), rs in rb[k].items():
        for r in rs:
            if not r.get("answer") or r.get("iters") != sync[0].get("iters"):
                continue
            n += 1
            v = float(r["answer"])
            if abs(v - ref) > atol + rtol * abs(ref):
                bad.append(f"{k[0]} {k[2]} {c}{'@' + tag if tag else ''}: {v:.6g} != {ref:.6g}")
print(f"- runs checked: {n}; answers that differ from synchronous: {len(bad)}")
for b in bad:
    print(f"  - {b}")
