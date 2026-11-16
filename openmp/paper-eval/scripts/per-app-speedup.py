"""One speedup bar figure per (machine, app), reusing plot.py's paper figure.

Usage: per-app-speedup.py <paper-eval dir> <machine>... [--format pdf,png]
Writes <paper-eval>/<machine>/figures/speedup-<app>.<fmt>.
PolyBench kernels are split into one figure each (x = problem size);
Krylov keeps its solver panel (one size per solver).
Vanilla clang/KMP runs (<machine>/vanilla/runs.csv) are drawn as the two
leftmost bars of each group. All bars are speedups over XKOMP `synchronous`;
whiskers are +-1 steady-state stddev of each run (plot.py).
"""
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"   # openmp/scripts
sys.path.insert(0, str(SCRIPTS))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot as P
_ORIG_SAVE = P._save

args = [a for a in sys.argv[1:] if not a.startswith("--")]
fmts = ["pdf", "png"]
for a in sys.argv[1:]:
    if a.startswith("--format="):
        fmts = a.split("=", 1)[1].split(",")
root, machines = Path(args[0]), args[1:]
plt.rcParams.update(P.STYLE)

# Legend = the pass each bar adds, in sweep order (plot.py canonicalizes the
# pass order, so its labels would name prog-fuse's bar "+jit").
NAMES = {"reduce-node,transitive-reduction": "+reduce", "critical-path": "+crit-path",
         "jit": "+jit", "prog-fuse": "+prog-fuse", "batch": "+packing"}
def paper_label(config):
    c = P._UNROLL_SUFFIX.sub("", config)
    if c == "taskgraph:none":
        return "taskgraph"
    head, _, opt = c.partition(":")
    if c in VANILLA.values():
        return c
    if head != "taskgraph":
        return c
    if opt == "reduce-node,transitive-reduction":
        return NAMES[opt]
    if opt == "reduce-node,transitive-reduction,batch":
        return "reduce+packing"     # not cumulative: packing without JIT/prog-fuse
    return NAMES.get(opt.split(",")[-1], "+" + opt.split(",")[-1])
P.paper_label = paper_label

# Vanilla configurations, renamed so they are not mistaken for XKOMP's.
VANILLA = {"synchronous": "vanilla sync", "no-taskgraph": "vanilla no-tg"}
VANILLA_STYLE = {"vanilla sync":  {"color": "#FFFFFF", "hatch": "\\\\"},
                 "vanilla no-tg": {"color": "#BBBBBB", "hatch": "\\\\"}}

for m in machines:
    runs = root / m / "xkomp" / "runs.csv"
    rows = P.load_runs(runs)
    vruns = root / m / "vanilla" / "runs.csv"
    if vruns.exists():
        van = [dict(r, config=VANILLA[r["config"]]) for r in P.load_runs(vruns)
               if r["config"] in VANILLA and not r.get("tag")]
        rows = van + rows           # first rows -> leftmost bars
    rows = P.select_evaluated(rows)
    rows = P.report_answers(rows, "synchronous")
    rows = P.aggregate_repeats(rows)
    # PolyBench: each kernel becomes its own size-panel app.
    for r in rows:
        if r["app"] == "polybench" and r.get("variant"):
            r["app"] = f"PolyBench {r['variant']}"
            r["variant"] = ""
    universe = [r["config"] for r in rows if r.get("avg_ms", "")]
    series = [P._UNROLL_SUFFIX.sub("", r["config"]) for r in rows
              if not r.get("tag") and r.get("avg_ms", "")]
    styles = P.assign_styles(universe, 0, series)
    styles.update(VANILLA_STYLE)
    figdir = root / m / "figures"
    def _save(fig, d, name, dpi, fmt, show, _m=m, _orig=_ORIG_SAVE):
        ax = fig.axes[0]
        ax.set_title(f"{ax.get_title()} -- {_m}")
        _orig(fig, d, name, dpi, fmt, show)
    P._save = _save
    for app in P.ordered_unique(r["app"] for r in rows):
        for fmt in fmts:
            P.plot_paper_speedup(rows, figdir, 200, fmt, False, styles,
                                 "synchronous", {}, 0, [app],
                                 figsize=(5.2, 3.4), tag="")
            src = figdir / f"paper-speedup.{fmt}"
            src.rename(figdir / f"speedup-{app.replace('PolyBench ', 'polybench-')}.{fmt}")
