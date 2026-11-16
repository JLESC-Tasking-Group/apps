#!/usr/bin/env python3
"""table.py <machine-dir>: Markdown summary of one machine's sweep (sweep.sh).

Every time is the steady-state mean of a run (instances W.., W per problem from
<dir>/warmup.env; ms per iteration),
reduced over the repeats by the median. Reads <dir>/xkomp/{runs,cgstats}.csv
(untagged main sweep, plus the jit-disk-cold / jit-disk-warm full-pipeline runs)
and <dir>/vanilla/runs.csv.

  1. speedup over XKOMP synchronous, per configuration
  2. the full pipeline: steady mean, its in-run stddev, the spread over repeats,
     instance 0 (record) and instance 1 (graph build + optimize + first replay)
  3. pass cost (CGIR pass wall time, ms): cold = no on-disk JIT cache (main sweep),
     warm = read from a populated on-disk cache; break-even = iterations for the
     full pipeline to repay its passes against no-taskgraph
  4. reduce+packing (reduce, then batch: no JIT, no prog-fuse) against the full
     pipeline: what JIT and prog-fuse add on top of packing, and what they cost
"""
import csv, sys, math, statistics as st, collections
from pathlib import Path

D = Path(sys.argv[1])
A = "reduce-node,transitive-reduction"
FULL = f"taskgraph:{A},jit,prog-fuse,batch"
RP = f"taskgraph:{A},batch"
SHORT = {"synchronous": "sync", "no-taskgraph": "no-tg", "taskgraph:none": "tg",
         f"taskgraph:{A}": "+reduce", f"taskgraph:{A},jit": "+jit",
         f"taskgraph:{A},jit,prog-fuse": "+prog-fuse", FULL: "+packing", RP: "reduce+packing"}
COLS = ["no-tg", "tg", "+reduce", "+jit", "+prog-fuse", "+packing", "reduce+packing", "V sync", "V no-tg"]
med = lambda v: st.median(v) if v else None
f = lambda x: float(x) if x not in ("", None) else None

def key(r):
    app = r["app"] + (f"/{r['variant']}" if r["variant"] else "")
    return (app, float(r["size"] or 0), r["size"])

runs = collections.defaultdict(lambda: collections.defaultdict(list))   # (key)(cfg) -> rows
fails = collections.Counter()
crashed = collections.Counter()
for sub, pre in (("xkomp", ""), ("vanilla", "V ")):
    p = D / sub / "runs.csv"
    if not p.exists():
        continue
    for r in csv.DictReader(open(p)):
        c = SHORT.get(r["config"])
        if c is None:
            continue
        tag = r.get("tag") or ""
        c = pre + c + ("" if not tag else f"@{tag}")
        if r["status"] not in ("ok", "ok_crashed_at_exit") or f(r["avg_ms"]) is None:
            fails[(key(r)[0], key(r)[2], c, r["status"])] += 1
            continue
        if r["status"] == "ok_crashed_at_exit":   # complete report, then died in teardown: kept
            crashed[(key(r)[0], key(r)[2], c, r["returncode"])] += 1
        runs[key(r)][c].append(r)

cg = collections.defaultdict(dict)        # run_id -> pass -> ms
p = D / "xkomp" / "cgstats.csv"
if p.exists():
    for r in csv.DictReader(open(p)):
        cg[r["tag"]][r["pass"]] = cg[r["tag"]].get(r["pass"], 0.0) + float(r["pass_ms"] or 0)

def t(k, c, field="avg_ms"):
    return med([f(r[field]) for r in runs[k].get(c, []) if f(r[field]) is not None])

def fmt(x, d=2):
    return "–" if x is None else (f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.{d}f}")

out = []
P = out.append
P(f"# {D.name}\n")
if (D / "stack.txt").exists():
    P("```\n" + (D / "stack.txt").read_text().strip() + "\n```\n")
if (D / "grains.env").exists():
    P("Grains (tiles per loop, per size): " + " ".join(l.split()[-1] for l in open(D / "grains.env")) + "\n")

P("## 1. Speedup over XKOMP `synchronous`\n")
P("Steady-state mean (instances W.., W per problem: `warmup.env`), median of the repeats. `V` = vanilla clang/KMP.\n")
P("| app | size | sync ms | " + " | ".join(COLS) + " |")
P("|---|---:|---:|" + "---:|" * len(COLS))
for k in sorted(runs):
    b = t(k, "sync")
    cells = [fmt(b / t(k, c)) if (b and t(k, c)) else "–" for c in COLS]
    P(f"| {k[0]} | {k[2]} | {fmt(b, 3)} | " + " | ".join(cells) + " |")

P("\n## 2. Full pipeline (`+packing`): stability and first instances\n")
P("| app | size | steady ms | in-run stddev % | repeats min–max % | record ms (inst. 0) | inst. 1 ms | n |")
P("|---|---:|---:|---:|---:|---:|---:|---:|")
for k in sorted(runs):
    rs = runs[k].get("+packing", [])
    if not rs:
        continue
    m = t(k, "+packing"); sd = t(k, "+packing", "stddev_ms")
    v = [f(r["avg_ms"]) for r in rs]
    spread = 100 * (max(v) - min(v)) / m if m else None
    P(f"| {k[0]} | {k[2]} | {fmt(m, 3)} | {fmt(100 * sd / m if m and sd is not None else None, 1)} | "
      f"{fmt(spread, 1)} | {fmt(t(k, '+packing', 'iter0_ms'))} | {fmt(t(k, '+packing', 'iter1_ms'))} | {len(rs)} |")

P("\n## 3. Pass cost of the full pipeline (ms), cold vs warm on-disk JIT cache\n")
P("cold = main sweep (no on-disk cache); populate = first run with an empty cache "
  "(`CGIR_JIT_CACHE_MODE=w`); warm = cache read back (`r`). Median of the repeats. "
  "Break-even = total / (no-tg − full) per iteration, in iterations.\n")
P("| app | size | reduce | prog-fuse | JIT cold | JIT populate | JIT warm | packing | total cold | total warm | break-even cold | break-even warm | warm steady ms |")
P("|---|---:|" + "---:|" * 12)
def cost(k, c):
    per = collections.defaultdict(list); tot = []
    for r in runs[k].get(c, []):
        ps = cg.get(r["run_id"])
        if not ps:
            continue
        for n, ms in ps.items():
            per[n].append(ms)
        tot.append(sum(ps.values()))
    return {n: med(v) for n, v in per.items()}, med(tot)
for k in sorted(runs):
    if "+packing" not in runs[k]:
        continue
    pc, tc = cost(k, "+packing")
    pp, _ = cost(k, "+packing@jit-disk-cold")
    pw, tw = cost(k, "+packing@jit-disk-warm")
    gain = (t(k, "no-tg") - t(k, "+packing")) if t(k, "no-tg") else None
    be = lambda tot: (tot / gain) if (tot is not None and gain and gain > 0) else None
    red = (pc.get("reduce-node") or 0) + (pc.get("transitive-reduction") or 0) if pc else None
    P(f"| {k[0]} | {k[2]} | {fmt(red, 1)} | {fmt(pc.get('prog-fuse'), 1)} | "
      f"{fmt(pc.get('jit'), 1)} | {fmt(pp.get('jit'), 1)} | {fmt(pw.get('jit'), 1)} | {fmt(pc.get('pack'), 1)} | "
      f"{fmt(tc, 1)} | {fmt(tw, 1)} | {fmt(be(tc), 0)} | {fmt(be(tw), 0)} | {fmt(t(k, '+packing@jit-disk-warm'), 3)} |")

P("\n## 4. reduce+packing (no JIT, no prog-fuse) vs the full pipeline\n")
P("reduce+packing = reduce-node, transitive-reduction, batch: the graph packed without "
  "the JIT and prog-fuse. Speedups over XKOMP synchronous, as in section 1 (the full "
  "pipeline's is its `+packing` column there). JIT+prog-fuse gain = reduce+packing time / "
  "full pipeline time (> 1: they speed up the packed graph). Pass cost of the main (cold) sweep; "
  "break-even against no-taskgraph, as in section 3.\n")
P("| app | size | reduce+packing speedup | full speedup | JIT+prog-fuse gain | reduce+packing pass cost | full cost cold | full cost warm | "
  "break-even reduce+packing | break-even full cold | break-even full warm |")
P("|---|---:|" + "---:|" * 9)
for k in sorted(runs):
    if "reduce+packing" not in runs[k]:
        continue
    mr, mf = t(k, "reduce+packing"), t(k, "+packing")
    _, cr = cost(k, "reduce+packing")
    _, cf = cost(k, "+packing")
    _, cw = cost(k, "+packing@jit-disk-warm")
    nt = t(k, "no-tg")
    b = t(k, "sync")
    sp = lambda m: (b / m) if (b and m) else None
    def be2(tot, m):
        g = (nt - m) if (nt is not None and m is not None) else None
        return (tot / g) if (tot is not None and g and g > 0) else None
    P(f"| {k[0]} | {k[2]} | {fmt(sp(mr))} | {fmt(sp(mf))} | {fmt(mr / mf if mr and mf else None)} | "
      f"{fmt(cr, 1)} | {fmt(cf, 1)} | {fmt(cw, 1)} | {fmt(be2(cr, mr), 0)} | {fmt(be2(cf, mf), 0)} | {fmt(be2(cw, mf), 0)} |")

# answers: every configuration of a problem must agree with synchronous
P("\n## 5. Checks\n")
bad = []
for k in sorted(runs):
    ref = {r["answer"] for r in runs[k].get("sync", []) if r["answer"]}
    for c, rs in runs[k].items():
        for r in rs:
            if r.get("verdict") not in ("", "pass"):
                bad.append(f"{k[0]} {k[2]} {c}: verdict {r['verdict']}")
P(f"- runs with a failing verdict: {len(bad)}" + "".join(f"\n  - {b}" for b in bad[:20]))
# Each run's answer against synchronous, with the app's tolerance (appspecs.py), as
# plot.py checks it: a run can pass its own verdict and still compute something else.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
try:
    from appspecs import APPS as _SPECS
except Exception:
    _SPECS = {}
wrong = []
for k in sorted(runs):
    spec = _SPECS.get(k[0].split("/")[0])
    rtol, atol = (getattr(spec, "answer_rtol", 1e-6), getattr(spec, "answer_atol", 0.0)) if spec else (1e-6, 0.0)
    sync = [r for r in runs[k].get("sync", []) if r.get("answer")]
    if not sync:
        continue
    ref = st.median(float(r["answer"]) for r in sync)
    for c, rs in runs[k].items():
        for r in rs:
            # only runs of the same length: the cache populate pass is a short run
            if not r.get("answer") or r.get("iters") != sync[0].get("iters"):
                continue
            v = float(r["answer"])
            if abs(v - ref) > atol + rtol * abs(ref):
                wrong.append(f"{k[0]} {k[2]} {c}: answer {v:.6g} != sync {ref:.6g}")
P(f"- runs whose answer differs from synchronous (rtol/atol of appspecs.py): {len(wrong)}"
  + "".join(f"\n  - {w}" for w in wrong[:40]))
# A NaN answer ("-nan") does not parse, so the comparison above skips it: list every
# run of an app that reports answers whose own answer is missing or not finite.
has_ans = {k[0].split("/")[0] for k in runs for rs in runs[k].values() for r in rs if r.get("answer")}
noans = collections.Counter()
for k in sorted(runs):
    if k[0].split("/")[0] not in has_ans:
        continue
    for c, rs in runs[k].items():
        for r in rs:
            a = r.get("answer")
            if not a or not math.isfinite(float(a)):
                noans[(k[0], k[2], c)] += 1
P(f"- runs without a finite answer (NaN/Inf, or none printed): {sum(noans.values())}"
  + "".join(f"\n  - {a} {s} {c} x{n}" for (a, s, c), n in sorted(noans.items())[:60]))
P(f"- failed runs: {sum(fails.values())}" + "".join(f"\n  - {a} {s} {c}: {st_} x{n}" for (a, s, c, st_), n in sorted(fails.items())))
P(f"- runs kept although they crashed at exit, after their complete report: {sum(crashed.values())}"
  + "".join(f"\n  - {a} {s} {c}: rc {rc} x{n}" for (a, s, c, rc), n in sorted(crashed.items())))
print("\n".join(out))
