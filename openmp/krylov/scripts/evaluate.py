#!/usr/bin/env python3
"""
evaluate.py - sweep a Krylov solver over taskgraph-optimization settings and
problem sizes, and record every run to a CSV.

It reproduces this manual launch for each configuration:

    XKRT_STATS=0 OMP_PLACES="cores" OMP_NUM_THREADS=4 \
    OMP_TASKGRAPH_OPT="<opt>" XKRT_DRIVERS="host,2;cuda,1" \
    ./cg.x -i 100 -S 7 -n <n> -t 1 -s 1

Sweep (defaults):
  * OMP_TASKGRAPH_OPT in { "reduce-node,reduce-edge",
                           "batch",
                           "reduce-node,reduce-edge,batch" }
  * grid size n : 20 points from 8 to 512 chosen so the *theoretical FLOP count*
                  (which scales like n^3 for the stencil matrix) is linearly
                  spaced -- i.e. n = cbrt(linspace(8^3, 512^3, 20)). Use
                  --spacing grid for a plain linear-in-n sweep instead.
  * fixed: 100 iterations, 7-point stencil, 4 threads on cores, one task (-t 1 -s 1).

The solver must already be built (this script does not compile). For the
OMP_TASKGRAPH_OPT / XKRT_DRIVERS knobs to have any effect the binary must be
built with the taskgraph + XKOMP/XKRT backend (see the Makefile).

Each run's configuration and the values the solver prints (n, nnz, residual,
error, total time, theoretical FLOPs, GFLOP/s, and the per-iteration timing
breakdown) are written as one CSV row.
"""

import argparse
import csv
import datetime
import os
import re
import shutil
import subprocess
import sys

# ----------------------------------------------------------------------------
# Defaults matching the evaluation request.
# ----------------------------------------------------------------------------
DEFAULT_OPTS = [
    "reduce-node,reduce-edge",
    "batch",
    "reduce-node,reduce-edge,batch",
]
DEFAULT_ENV = {
    "XKRT_STATS":    "0",
    "OMP_PLACES":    "cores",
    "XKRT_DRIVERS":  "host,2;cuda,1",
}


def grid_sizes(n_min, n_max, points, spacing):
    """Return `points` grid dimensions in [n_min, n_max].

    spacing == "flop": n^3 (proportional to the FLOP count) is linearly spaced,
                       so the theoretical FLOPs grow linearly across the points.
    spacing == "grid": n itself is linearly spaced.
    Endpoints are pinned to n_min / n_max and the list is made strictly
    increasing (rounding collisions removed).
    """
    ns = []
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        if spacing == "grid":
            val = n_min + (n_max - n_min) * t
        else:  # "flop": linear in n^3
            f = (n_min ** 3) + ((n_max ** 3) - (n_min ** 3)) * t
            val = f ** (1.0 / 3.0)
        ns.append(max(n_min, min(n_max, int(round(val)))))
    ns[0], ns[-1] = n_min, n_max
    out = []
    for n in ns:
        if not out or n > out[-1]:
            out.append(n)
    return out


# ----------------------------------------------------------------------------
# Output parsing: the solver prints "  <label> : <value>" lines.
# ----------------------------------------------------------------------------
def _grab(text, pattern, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (ValueError, IndexError):
        return None


def parse_output(text):
    F = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"  # a float
    return {
        "backend":          _grab(text, r"backend\s*:\s*(\S+)", str),
        "taskgraph":        _grab(text, r"taskgraph\s*:\s*(\S+)", str),
        "n_sys":            _grab(text, r"grid\s*:.*\(n\s*=\s*(\d+)", int),
        "nnz":              _grab(text, r"nnz\s*=\s*(\d+)", int),
        "omp_num_threads":  _grab(text, r"omp threads:\s*(\d+)", int),
        "rel_residual":     _grab(text, r"relative residual\s*:\s*" + F),
        "rel_error":        _grab(text, r"relative error\s*:\s*" + F),
        "total_time_s":     _grab(text, r"total solve time\s*:\s*" + F),
        "flops":            _grab(text, r"theoretical flops\s*:\s*" + F),
        "gflops":           _grab(text, r"performance\s*:\s*" + F),
        "iter0_ms":         _grab(text, r"(?:iteration|restart) 0.*:\s*" + F + r"\s*ms"),
        "iter1_ms":         _grab(text, r"(?:iteration|restart) 1.*:\s*" + F + r"\s*ms"),
        "avg_ms":           _grab(text, r"(?:iterations|restarts) 2\.\..*\(avg\)\s*:\s*" + F + r"\s*ms"),
        "stddev_ms":        _grab(text, r"(?:iterations|restarts) 2\.\..*\(stddev\)\s*:\s*" + F + r"\s*ms"),
    }


CSV_FIELDS = [
    "timestamp", "solver", "opt", "spacing",
    "omp_num_threads", "omp_places", "xkrt_drivers", "xkrt_stats",
    "stencil", "iters", "t1", "t2", "n_grid",
    "backend", "taskgraph", "n_sys", "nnz",
    "rel_residual", "rel_error",
    "total_time_s", "flops", "gflops",
    "iter0_ms", "iter1_ms", "avg_ms", "stddev_ms",
    "returncode", "command",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", default="./cg.x",
                    help="path to the solver binary (default: ./cg.x)")
    ap.add_argument("--output", default=None,
                    help="CSV output path (default: evaluate_<solver>_<timestamp>.csv)")
    ap.add_argument("--iters", type=int, default=100, help="-i (default 100)")
    ap.add_argument("--stencil", type=int, default=7, help="-S (default 7)")
    ap.add_argument("--t1", type=int, default=1, help="-t (default 1)")
    ap.add_argument("--t2", type=int, default=1, help="-s (default 1)")
    ap.add_argument("--threads", type=int, default=4, help="OMP_NUM_THREADS (default 4)")
    ap.add_argument("--places", default="cores", help="OMP_PLACES (default cores)")
    ap.add_argument("--drivers", default=DEFAULT_ENV["XKRT_DRIVERS"],
                    help='XKRT_DRIVERS (default "%(default)s")')
    ap.add_argument("--nmin", type=int, default=8, help="min grid dim (default 8)")
    ap.add_argument("--nmax", type=int, default=512, help="max grid dim (default 512)")
    ap.add_argument("--points", type=int, default=20, help="number of grid sizes (default 20)")
    ap.add_argument("--spacing", choices=["flop", "grid"], default="flop",
                    help="'flop': linear in FLOPs (n^3); 'grid': linear in n (default flop)")
    ap.add_argument("--opts", nargs="+", default=DEFAULT_OPTS,
                    help="OMP_TASKGRAPH_OPT values to sweep")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="per-run timeout in seconds (0 = none)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args()

    solver = args.solver
    if not os.path.sep in solver and not solver.endswith(".x"):
        solver = "./" + solver + ".x"
    if not args.dry_run and not (os.path.isfile(solver) and os.access(solver, os.X_OK)):
        sys.exit("error: solver binary '%s' not found or not executable "
                 "(build it first, e.g. `make cg`)" % solver)

    ns = grid_sizes(args.nmin, args.nmax, args.points, args.spacing)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.output or "evaluate_%s_%s.csv" % (
        os.path.basename(solver).replace(".x", ""), stamp)

    total = len(args.opts) * len(ns)
    print("solver     : %s" % solver, file=sys.stderr)
    print("opt cases  : %s" % args.opts, file=sys.stderr)
    print("grid sizes : %s (%s-spaced, %d pts)" % (ns, args.spacing, len(ns)), file=sys.stderr)
    print("runs       : %d  ->  %s" % (total, out_path), file=sys.stderr)

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()

        run = 0
        for opt in args.opts:
            for n in ns:
                run += 1
                env = os.environ.copy()
                env.update(DEFAULT_ENV)
                env["OMP_NUM_THREADS"]  = str(args.threads)
                env["OMP_PLACES"]       = args.places
                env["XKRT_DRIVERS"]     = args.drivers
                env["OMP_TASKGRAPH_OPT"] = opt

                cmd = [solver, "-i", str(args.iters), "-S", str(args.stencil),
                       "-n", str(n), "-t", str(args.t1), "-s", str(args.t2)]
                pretty = ('XKRT_STATS=%s OMP_PLACES="%s" OMP_NUM_THREADS=%s '
                          'OMP_TASKGRAPH_OPT="%s" XKRT_DRIVERS="%s" %s'
                          % (env["XKRT_STATS"], args.places, args.threads, opt,
                             args.drivers, " ".join(cmd)))
                print("[%d/%d] %s" % (run, total, pretty), file=sys.stderr)
                if args.dry_run:
                    continue

                row = {k: "" for k in CSV_FIELDS}
                row.update({
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    "solver": os.path.basename(solver),
                    "opt": opt, "spacing": args.spacing,
                    "omp_num_threads": args.threads, "omp_places": args.places,
                    "xkrt_drivers": args.drivers, "xkrt_stats": env["XKRT_STATS"],
                    "stencil": args.stencil, "iters": args.iters,
                    "t1": args.t1, "t2": args.t2, "n_grid": n,
                    "command": pretty,
                })
                try:
                    p = subprocess.run(cmd, env=env,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=(args.timeout or None))
                    row["returncode"] = p.returncode
                    row.update({k: v for k, v in parse_output(p.stdout).items() if v is not None})
                    if p.returncode != 0:
                        sys.stderr.write(p.stderr[-2000:] + "\n")
                except subprocess.TimeoutExpired:
                    row["returncode"] = "timeout"
                    print("      -> timeout after %.0fs" % args.timeout, file=sys.stderr)

                w.writerow(row)
                fh.flush()

    print("wrote %s" % out_path, file=sys.stderr)


if __name__ == "__main__":
    main()
