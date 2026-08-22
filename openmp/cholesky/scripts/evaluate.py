#!/usr/bin/env python3
"""
evaluate.py - sweep the tiled Cholesky app over taskgraph-optimization settings
and problem sizes, and record every run to a CSV.

It reproduces this manual launch for each configuration:

    XKRT_STATS=0 OMP_PLACES="cores" OMP_NUM_THREADS=4 \
    OMP_TASKGRAPH_OPT="<opt>" XKRT_DRIVERS="host,2;cuda,1" \
    ./cholesky <nt> <ts> <reps> <check>

Sweep (defaults):
  * OMP_TASKGRAPH_OPT in { "none",
                           "reduce-node,reduce-edge",
                           "reduce-node,reduce-edge,batch" }
  * tile grid nt : points chosen so the *theoretical FLOP count* (~ (nt*ts)^3 / 3)
                   is linearly spaced -- i.e. nt = cbrt(linspace(nt_min^3, nt_max^3)).
                   Use --spacing grid for a plain linear-in-nt sweep.
  * fixed: tile size ts, number of repetitions reps (record + replays).

The binary must already be built (this script does not compile). For the
OMP_TASKGRAPH_OPT / XKRT_DRIVERS knobs to matter the binary must be built with
the taskgraph + XKOMP backend (see the Makefile).

Each run's configuration and the values the app prints (nt, ts, N, backend,
taskgraph, total time, theoretical FLOPs, GFLOP/s, and the record / first-replay
/ steady-replay timing breakdown) are written as one CSV row.
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
    """`points` tile-grid sizes in [n_min, n_max], endpoints pinned, strictly
    increasing. spacing == "flop": n^3 (proportional to the FLOP count) linearly
    spaced; spacing == "grid": n itself linearly spaced."""
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
        "backend":         _grab(text, r"backend=(.+?)\s+taskgraph=", str),
        "taskgraph":       _grab(text, r"taskgraph=(\d+)", int),
        "nt":              _grab(text, r"\bnt=(\d+)", int),
        "ts":              _grab(text, r"\bts=(\d+)", int),
        "n_sys":           _grab(text, r"\bN=(\d+)", int),
        "total_time_s":    _grab(text, r"total solve time\s*:\s*" + F),
        "flops":           _grab(text, r"theoretical flops\s*:\s*" + F),
        "gflops":          _grab(text, r"performance\s*:\s*" + F),
        "rep0_ms":         _grab(text, r"repetition 0\s*:\s*" + F + r"\s*ms"),
        "rep1_ms":         _grab(text, r"repetition 1\s*:\s*" + F + r"\s*ms"),
        "avg_ms":          _grab(text, r"repetitions 2\.\..*\(avg\)\s*:\s*" + F + r"\s*ms"),
        "stddev_ms":       _grab(text, r"repetitions 2\.\..*\(stddev\)\s*:\s*" + F + r"\s*ms"),
        "maxdiff":         _grab(text, r"max \|A - ref\|.*=\s*" + F),
    }


CSV_FIELDS = [
    "timestamp", "opt", "spacing",
    "omp_num_threads", "omp_places", "xkrt_drivers", "xkrt_stats",
    "ts_arg", "reps", "check", "nt_grid",
    "backend", "taskgraph", "nt", "ts", "n_sys",
    "total_time_s", "flops", "gflops",
    "rep0_ms", "rep1_ms", "avg_ms", "stddev_ms", "maxdiff",
    "returncode", "command",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="./cholesky",
                    help="path to the cholesky binary (default: ./cholesky)")
    ap.add_argument("--output", default=None,
                    help="CSV output path (default: evaluate_cholesky_<timestamp>.csv)")
    ap.add_argument("--ts", type=int, default=256, help="tile size (default 256)")
    ap.add_argument("--reps", type=int, default=10,
                    help="repetitions per run: record + replays (default 10)")
    ap.add_argument("--check", type=int, default=0, help="verify each run (default 0)")
    ap.add_argument("--threads", type=int, default=4, help="OMP_NUM_THREADS (default 4)")
    ap.add_argument("--places", default="cores", help="OMP_PLACES (default cores)")
    ap.add_argument("--drivers", default=DEFAULT_ENV["XKRT_DRIVERS"],
                    help='XKRT_DRIVERS (default "%(default)s")')
    ap.add_argument("--ntmin", type=int, default=2, help="min tile grid nt (default 2)")
    ap.add_argument("--ntmax", type=int, default=24, help="max tile grid nt (default 24)")
    ap.add_argument("--points", type=int, default=12, help="number of nt sizes (default 12)")
    ap.add_argument("--spacing", choices=["flop", "grid"], default="flop",
                    help="'flop': linear in FLOPs (nt^3); 'grid': linear in nt (default flop)")
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

    nts = sweep_sizes(args.ntmin, args.ntmax, args.points, args.spacing)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.output or "evaluate_cholesky_%s.csv" % stamp

    total = len(args.opts) * len(nts)
    print("binary     : %s" % binary, file=sys.stderr)
    print("opt cases  : %s" % args.opts, file=sys.stderr)
    print("tile grids : %s (%s-spaced, %d pts), ts=%d" % (nts, args.spacing, len(nts), args.ts), file=sys.stderr)
    print("runs       : %d  ->  %s" % (total, out_path), file=sys.stderr)

    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()

        run = 0
        for opt in args.opts:
            for nt in nts:
                run += 1
                env = os.environ.copy()
                env.update(DEFAULT_ENV)
                env["OMP_NUM_THREADS"]   = str(args.threads)
                env["OMP_PLACES"]        = args.places
                env["XKRT_DRIVERS"]      = args.drivers
                env["OMP_TASKGRAPH_OPT"] = opt

                cmd = [binary, str(nt), str(args.ts), str(args.reps), str(args.check)]
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
                    "ts_arg": args.ts, "reps": args.reps, "check": args.check,
                    "nt_grid": nt, "command": pretty,
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
