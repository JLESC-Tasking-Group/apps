#!/usr/bin/env python3
"""warmup.py <machine> <runs.csv>... [--seconds S] [--steady N]

Sizes each problem's warm-up -- the instances its runs leave out of the statistics
(TG_WARMUP, see tasking.h) -- and writes <machine>/warmup.env for sweep.sh.

Why: the first replay of a `jit` configuration compiles for hundreds of ms while the
GPU idles, and the GPU clocks down meanwhile (A100: 1410 -> 765 MHz). It takes ~0.4 s
of load to clock back up, so a problem whose instance lasts ~1 ms reached its steady
state long after instance 5, and the old window (instances 5..54) was measured at the
low clock: +jit looked 14-30% slower than without it on A100, while at full clock it is
faster. The warm-up is therefore sized in time: W instances keep the GPU busy for at
least S seconds after instance 1 (the compile), at the speed of the problem's fastest
taskgraph configuration measured in <runs.csv> (a previous sweep on the same machine):

    W = max(5, 2 + ceil(S / (unroll * t_fastest)))

Every configuration of a problem uses the same W and the same iteration count
8*(W+N), so they stay comparable. The table is written once and kept with the results,
so a sweep is reproducible from it."""
import argparse, collections, csv, math, os, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("machine")
ap.add_argument("runs", nargs="+", help="runs.csv of a previous sweep on that machine")
ap.add_argument("--seconds", type=float, default=1.0, help="GPU busy time after instance 1 (default 1.0)")
ap.add_argument("--steady", type=int, default=50, help="steady instances (for the run-time estimate)")
ap.add_argument("--unroll", type=int, default=8)
a = ap.parse_args()

rows = []
for p in a.runs:
    rows += [r for r in csv.DictReader(open(p)) if r["status"] == "ok" and r["avg_ms"] and r["iters"] not in ("", "0")]

key = lambda r: (r["app"], r["variant"], r["size"])
fast = collections.defaultdict(lambda: math.inf)
slow = collections.defaultdict(float)
for r in rows:
    t = float(r["avg_ms"])
    if r["config"].startswith("taskgraph"):
        fast[key(r)] = min(fast[key(r)], t)
    slow[key(r)] = max(slow[key(r)], t)

W = {k: max(5, 2 + math.ceil(a.seconds * 1e3 / (a.unroll * t))) for k, t in fast.items()}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", a.machine, "warmup.env")
with open(out, "w") as f:
    f.write(f"# warm-up instances per problem (TG_WARMUP), written by scripts/warmup.py:\n"
            f"# {a.seconds} s of GPU load after instance 1 at the fastest taskgraph config's speed in\n")
    for p in a.runs:
        f.write(f"#   {os.path.relpath(p, os.path.dirname(out))}\n")
    f.write("# keys app:variant:size; sweep.sh runs 8*(W+STEADY) iterations\n")
    for k in sorted(W):
        f.write(f"WU[{':'.join(k)}]={W[k]}\n")
print(f"{out}: {len(W)} problems")
for k in sorted(W, key=lambda k: -W[k]):
    it = a.unroll * (W[k] + a.steady)
    print(f"  {':'.join(k):28s} fastest {fast[k]:8.4f} ms  W={W[k]:6d}  iters={it:7d}"
          f"  slowest config ~{it * slow[k] / 1e3:6.1f} s")
