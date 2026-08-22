#!/usr/bin/env python3
"""
evaluate.py - sweep LULESH over taskgraph-optimization settings and problem
sizes, and record every run to a CSV.

It reproduces this manual launch for each configuration:

    XKRT_STATS=0 OMP_PLACES="cores" OMP_NUM_THREADS=4 \
    OMP_TASKGRAPH_OPT="<opt>" XKRT_DRIVERS="host,2;cuda,1" \
    ./lulesh -i <iters> -s <size> -r <regions> -b <balance> -c <cost> -nb <tasks>

Sweep (defaults):
  * OMP_TASKGRAPH_OPT in { "none",
                           "reduce-node,reduce-edge",
                           "reduce-node,reduce-edge,batch" }
  * mesh side s : points chosen so the *work* (number of zones ~ s^3) is linearly
                  spaced -- i.e. s = cbrt(linspace(s_min^3, s_max^3)). Use
                  --spacing grid for a plain linear-in-s sweep.
  * fixed: iteration count, regions, balance, cost, tasks-per-loop (-nb).

LULESH now prints a per-iteration wall time and the record / first-replay /
steady-replay breakdown (mean and stddev of iterations 2..N), so a single run per
configuration yields the average/stddev used for the plot's error bars -- exactly
like the krylov / cholesky apps.

The binary must already be built (this script does not compile). For the
OMP_TASKGRAPH_OPT / XKRT_DRIVERS knobs to matter it must be built with the
taskgraph + XKOMP backend (see the Makefile).
"""

import argparse
import csv
import datetime
import os
import re
import subprocess
import sys

DEFAULT_OPTS = [
    "none",
    "reduce-node,reduce-edge",
    "reduce-node,reduce-edge,batch",
]
DEFAULT_ENV = {
    "XKRT_STATS":   "0",
    "OMP_PLACES":   "cores",
    "XKRT_DRIVERS": "host,2;cuda,1",
}


def sweep_sizes(n_min, n_max, points, spacing):
    ns = []
    for i in range(points):
        t = i / (points - 1) if points > 1 else 0.0
        if spacing == "grid":
            val = n_min + (n_max - n_min) * t
        else:  # "flop": linear in s^3 (number of zones)
            f = (n_min ** 3) + ((n_max ** 3) - (n_min ** 3)) * t
            val = f ** (1.0 / 3.0)
        ns.append(max(n_min, min(n_max, int(round(val)))))
    ns[0], ns[-1] = n_min, n_max
    out = []
    for n in ns:
        if not out or n > out[-1]:
            out.append(n)
    return out


def _grab(text, pattern, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (ValueError, IndexError):
        return None


def parse_output(text):
    F = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    return {
        "backend":         _grab(text, r"backend=(.+?)\s+taskgraph=", str),
        "taskgraph":       _grab(text, r"taskgraph=(\d+)", int),
        "nb":              _grab(text, r"Num tasks per loop:\s*(\d+)", int),
        "problem_size":    _grab(text, r"Problem size\s*=\s*(\d+)", int),
        "iteration_count": _grab(text, r"Iteration count\s*=\s*(\d+)", int),
        "final_energy":    _grab(text, r"Final Origin Energy\s*=\s*" + F),
        "elapsed_s":       _grab(text, r"Elapsed time\s*=\s*" + F),
        "grind_overall":   _grab(text, r"Grind time.*\(\s*" + F + r"\s*overall\)"),
        "fom":             _grab(text, r"FOM\s*=\s*" + F + r"\s*\(z/s\)"),
        "fom_per_watt":    _grab(text, r"FOM/watt\s*=\s*" + F),
        "tdg_s":           _grab(text, r"TDG creation time\s*=\s*" + F),
        "iter0_ms":        _grab(text, r"iteration 0\s*:\s*" + F + r"\s*ms"),
        "iter1_ms":        _grab(text, r"iteration 1\s*:\s*" + F + r"\s*ms"),
        "avg_ms":          _grab(text, r"iterations 2\.\..*\(avg\)\s*:\s*" + F + r"\s*ms"),
        "stddev_ms":       _grab(text, r"iterations 2\.\..*\(stddev\)\s*:\s*" + F + r"\s*ms"),
    }


CSV_FIELDS = [
    "timestamp", "opt", "spacing",
    "omp_num_threads", "omp_places", "xkrt_drivers", "xkrt_stats",
    "iters", "regions", "balance", "cost", "nb", "s_grid",
    "backend", "taskgraph", "problem_size", "iteration_count", "zones",
    "iter0_ms", "iter1_ms", "avg_ms", "stddev_ms",
    "elapsed_s", "fom", "grind_overall", "tdg_s", "final_energy",
    "returncode", "command",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="./lulesh",
                    help="path to the lulesh binary (default: ./lulesh)")
    ap.add_argument("--output", default=None,
                    help="CSV output path (default: evaluate_lulesh_<timestamp>.csv)")
    ap.add_argument("--iters", type=int, default=30, help="-i (default 30)")
    ap.add_argument("--regions", type=int, default=11, help="-r (default 11)")
    ap.add_argument("--balance", type=int, default=1, help="-b (default 1)")
    ap.add_argument("--cost", type=int, default=1, help="-c (default 1)")
    ap.add_argument("--nb", type=int, default=32, help="-nb tasks per loop (default 32)")
    ap.add_argument("--threads", type=int, default=4, help="OMP_NUM_THREADS (default 4)")
    ap.add_argument("--places", default="cores", help="OMP_PLACES (default cores)")
    ap.add_argument("--drivers", default=DEFAULT_ENV["XKRT_DRIVERS"],
                    help='XKRT_DRIVERS (default "%(default)s")')
    ap.add_argument("--smin", type=int, default=16, help="min mesh side (default 16)")
    ap.add_argument("--smax", type=int, default=90, help="max mesh side (default 90)")
    ap.add_argument("--points", type=int, default=8, help="number of mesh sizes (default 8)")
    ap.add_argument("--spacing", choices=["flop", "grid"], default="flop",
                    help="'flop': linear in zones (s^3); 'grid': linear in s (default flop)")
    ap.add_argument("--opts", nargs="+", default=DEFAULT_OPTS,
                    help="OMP_TASKGRAPH_OPT values to sweep")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="per-run timeout in seconds (0 = none)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args()

    binary = args.bin
    if not args.dry_run and not (os.path.isfile(binary) and os.access(binary, os.X_OK)):
        sys.exit("error: binary '%s' not found or not executable (build it first: `make`)" % binary)

    ss = sweep_sizes(args.smin, args.smax, args.points, args.spacing)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.output or "evaluate_lulesh_%s.csv" % stamp

    total = len(args.opts) * len(ss)
    print("binary     : %s" % binary, file=sys.stderr)
    print("opt cases  : %s" % args.opts, file=sys.stderr)
    print("mesh sizes : %s (%s-spaced, %d pts)" % (ss, args.spacing, len(ss)), file=sys.stderr)
    print("runs       : %d  ->  %s" % (total, out_path), file=sys.stderr)

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()

        run = 0
        for opt in args.opts:
            for s in ss:
                run += 1
                env = os.environ.copy()
                env.update(DEFAULT_ENV)
                env["OMP_NUM_THREADS"]   = str(args.threads)
                env["OMP_PLACES"]        = args.places
                env["XKRT_DRIVERS"]      = args.drivers
                env["OMP_TASKGRAPH_OPT"] = opt

                cmd = [binary, "-i", str(args.iters), "-s", str(s),
                       "-r", str(args.regions), "-b", str(args.balance),
                       "-c", str(args.cost), "-nb", str(args.nb)]
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
                    "opt": opt, "spacing": args.spacing,
                    "omp_num_threads": args.threads, "omp_places": args.places,
                    "xkrt_drivers": args.drivers, "xkrt_stats": env["XKRT_STATS"],
                    "iters": args.iters, "regions": args.regions,
                    "balance": args.balance, "cost": args.cost, "nb": args.nb,
                    "s_grid": s, "zones": s * s * s, "command": pretty,
                })
                try:
                    p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, universal_newlines=True,
                                       timeout=(args.timeout or None))
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
