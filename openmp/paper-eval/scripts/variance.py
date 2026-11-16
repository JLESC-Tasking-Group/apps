#!/usr/bin/env python3
"""variance.py <machine-dir> [subdir]: run-to-run spread of <dir>/<subdir>/runs.csv (default
subdir `variance`; REPEAT=n runs).

Per problem and configuration: the n steady-state means (ms/iteration), their spread
(min..max, CV = stddev/mean across runs), the speedup over the median synchronous run,
and the speedup of the single run in the main sweep (<dir>/xkomp/runs.csv) for comparison.
The verdict compares each pass with the previous one: a real loss (or gain) is when the
ranges of their speedups over the repeats are disjoint; overlapping ranges are noise."""
import collections, csv, statistics as st, sys
from pathlib import Path

D = Path(sys.argv[1])
SHORT = {"synchronous": "sync", "taskgraph:none": "tg"}
A = "reduce-node,transitive-reduction"
for c, n in ((A, "+reduce"), (A + ",jit", "+jit"), (A + ",jit,prog-fuse", "+prog-fuse"),
             (A + ",jit,prog-fuse,batch", "+packing")):
    SHORT["taskgraph:" + c] = n
ORDER = ["sync", "tg", "+reduce", "+jit", "+prog-fuse", "+packing"]
OK = ("ok", "ok_crashed_at_exit")

def load(p):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    if p.exists():
        for r in csv.DictReader(open(p)):
            c = SHORT.get(r["config"])
            if c and r["status"] in OK and r["avg_ms"] and not r.get("tag"):
                d[(r["app"] + ("/" + r["variant"] if r["variant"] else ""), r["size"])][c].append(float(r["avg_ms"]))
    return d

SUB = sys.argv[2] if len(sys.argv) > 2 else "variance"
var, main = load(D / SUB / "runs.csv"), load(D / "xkomp" / "runs.csv")
print(f"# Run-to-run variance, {D.name} ({SUB})\n")
print("Speedups over the median `synchronous` run; `range` is min..max over the repeats.\n")
for k in sorted(var):
    v = var[k]
    s0 = st.median(v["sync"]) if v.get("sync") else None
    m0 = st.median(main[k]["sync"]) if main.get(k, {}).get("sync") else None
    print(f"## {k[0]} {k[1]}\n")
    print("| config | n | mean ms | CV | speedup (median) | range | main sweep | verdict vs previous |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    prev = None
    for c in ORDER:
        x = v.get(c)
        if not x:
            continue
        mean = st.mean(x); cv = st.stdev(x) / mean if len(x) > 1 else 0.0
        sp = [s0 / t for t in x] if s0 else []
        ms = (m0 / main[k][c][0]) if (m0 and main.get(k, {}).get(c)) else None
        verdict = ""
        if prev and c != "sync" and prev[0] != "sync":
            lo, hi = min(sp), max(sp); plo, phi = prev[1]
            verdict = ("slower (ranges disjoint)" if hi < plo else
                       "faster (ranges disjoint)" if lo > phi else "same within noise")
        print(f"| {c} | {len(x)} | {mean:.4f} | {cv:.1%} | {st.median(sp):.2f} | "
              f"{min(sp):.2f}..{max(sp):.2f} | {'' if ms is None else f'{ms:.2f}'} | {verdict} |")
        prev = (c, (min(sp), max(sp)))
    print()
