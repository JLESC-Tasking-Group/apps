#!/usr/bin/env python3
"""
evaluate.py - shared evaluation sweep for the apps/openmp taskgraph apps
(krylov, lulesh, llm.c). For each app x variant x configuration x problem size it
builds the right binary, runs it, parses the per-iteration timing, and appends a
row to results/runs.csv. Per-pass CGIR command-graph stats are collected
alongside via CGIR_STATS_CSV (auto, unless --no-stats).

Configurations always include the three references (synchronous, no-taskgraph,
taskgraph:none) plus one taskgraph:<opt> per CGIR optimization combo (see
appspecs.py / --opts). The synchronous / no-taskgraph / taskgraph split is a
compile-time choice, so binaries are rebuilt per configuration; the CGIR pass
within a taskgraph build is the run-time OMP_TASKGRAPH_OPT. CPU vs GPU (--target)
is orthogonal and applied to every build.

Examples
--------
  ./scripts/evaluate.py --list
  ./scripts/evaluate.py                      # all apps, CPU, default sizes
  ./scripts/evaluate.py --apps lulesh --sizes 30,45,60
  ./scripts/evaluate.py --apps krylov --variants cg,cr --target gpu
  ./scripts/evaluate.py --dry-run

Nothing runs on the GPU by itself: it shells out to `make` and each app binary,
which the caller must be able to build/run (taskgraph clang + XKOMP in PATH /
LD_LIBRARY_PATH). Use --dry-run to inspect the plan first.
"""

import argparse
import csv
import datetime
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from appspecs import APPS, DEFAULT_OPTS, default_configs  # noqa: E402

APPS_OPENMP = Path(__file__).resolve().parent.parent

DEFAULT_ENV = {
    "XKRT_STATS":   "0",
    "OMP_PLACES":   "cores",
    "XKRT_DRIVERS": "host,2;cuda,1",
}

CSV_FIELDS = [
    "run_id", "timestamp", "machine",
    "app", "variant", "config", "opt", "build_vars", "backend",
    "size", "work", "work_label", "iters",
    "avg_ms", "stddev_ms", "iter0_ms", "elapsed_s", "fom", "flops", "gflops",
    "residual", "error",
    "returncode", "status", "cmd",
]


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")


def build_cmd(app, variant, cfg, size, iters, backend_vars):
    variables = dict(cfg.build)
    variables.update(backend_vars)
    argv = ["make", "-C", app.directory, app.make_target(variant)]
    argv += [f"{k}={v}" for k, v in variables.items()]
    if app.rebuild_per_size and app.llmc_defs:
        argv.append("LLMC_DEFS=" + app.llmc_defs(size, iters, app.batch))
    return argv


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps", default="", help="comma list (default: all)")
    ap.add_argument("--variants", default="", help="comma list to filter variants")
    ap.add_argument("--sizes", default="", help="problem sizes: a global comma list "
                    "(e.g. '8,16,24') and/or per-app 'app=list' items, separated by ';' "
                    "(e.g. 'krylov=32,48,64;lulesh=30,45,60'). A per-app list overrides the "
                    "global list, which overrides each app's built-in default.")
    ap.add_argument("--iters", type=int, default=0, help="override iterations")
    ap.add_argument("--opts", default="", help="semicolon-separated CGIR opt combos, each "
                    "a comma/space list of passes (e.g. 'reduce-node,reduce-edge;batch'); "
                    "each combo -> one taskgraph:<opt> config. Default from appspecs.")
    ap.add_argument("--target", choices=["cpu", "gpu"], default="cpu",
                    help="backend for every build (USE_TARGET); default cpu")
    ap.add_argument("--threads", type=int, default=0, help="OMP_NUM_THREADS (0=leave unset)")
    ap.add_argument("--places", default=DEFAULT_ENV["OMP_PLACES"], help="OMP_PLACES")
    ap.add_argument("--drivers", default=DEFAULT_ENV["XKRT_DRIVERS"], help="XKRT_DRIVERS")
    ap.add_argument("--outdir", default=str(APPS_OPENMP / "results"))
    ap.add_argument("--out", default="", help="runs.csv path (default: <outdir>/runs.csv)")
    ap.add_argument("--no-stats", action="store_true",
                    help="do not collect CGIR_STATS_CSV per-pass graph stats")
    ap.add_argument("--timeout", type=float, default=0.0, help="per-run timeout s (0=none)")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("apps:")
        for name, spec in APPS.items():
            print(f"  {name:8s} variants={spec.variants} sizes={spec.sizes} iters={spec.iters}"
                  f"{' (rebuild per size)' if spec.rebuild_per_size else ''}")
        opts = _parse_opts(args.opts)
        print("\nconfigurations:")
        for c in default_configs(opts):
            print(f"  {c.label:34s} build={c.build} opt={c.opt!r}"
                  f"{' grain1' if c.grain1 else ''}")
        return 0

    selected = [a.strip() for a in args.apps.split(",") if a.strip()] or list(APPS)
    for a in selected:
        if a not in APPS:
            ap.error(f"unknown app '{a}' (known: {', '.join(APPS)})")
    variant_filter = {v.strip() for v in args.variants.split(",") if v.strip()}
    size_default, size_by_app = _parse_sizes(args.sizes)
    for a in size_by_app:
        if a not in APPS:
            ap.error(f"unknown app '{a}' in --sizes (known: {', '.join(APPS)})")
    configs = default_configs(_parse_opts(args.opts))
    backend_vars = {"USE_TARGET": "1" if args.target == "gpu" else "0"}

    outdir = Path(args.outdir)
    runs_csv = Path(args.out) if args.out else outdir / "runs.csv"
    stats_csv = outdir / "cgstats.csv"
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    machine = socket.gethostname()
    ts_run = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    new_file = not runs_csv.exists() or runs_csv.stat().st_size == 0
    fh = None if args.dry_run else open(runs_csv, "a", newline="")
    writer = None
    if fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()

    built = {}
    n_ok = n_fail = 0

    def do_build(app, variant, cfg, size, iters):
        key = (app.name, variant, cfg.label, args.target)
        if app.rebuild_per_size:
            key = key + (size,)
        if key in built:
            return built[key]
        cmd = build_cmd(app, variant, cfg, size, iters, backend_vars)
        print("[build] " + " ".join(cmd), file=sys.stderr)
        if args.dry_run or args.skip_build:
            built[key] = True
            return True
        p = subprocess.run(cmd, cwd=str(APPS_OPENMP),
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        ok = p.returncode == 0
        if not ok:
            sys.stderr.write(p.stdout[-2000:] + "\n")
        built[key] = ok
        return ok

    for app_name in selected:
        app = APPS[app_name]
        # A variant-less app (variants == [""]) always runs; the --variants filter
        # only applies to apps that expose real variants (e.g. krylov's solvers).
        if app.variants == [""]:
            variants = [""]
        else:
            variants = [v for v in app.variants if not variant_filter or v in variant_filter]
        sizes = size_by_app.get(app_name) or size_default or app.sizes
        iters = args.iters or app.iters

        for variant in variants:
            for cfg in configs:
                for size in sizes:
                    ok = do_build(app, variant, cfg, size, iters)
                    work, work_label = app.work(size)
                    vtag = f"-{variant}" if variant else ""
                    disp = f"{app_name}/{variant}" if variant else app_name
                    run_id = sanitize(f"{app_name}{vtag}-{args.target}-{cfg.label}-n{size}-{ts_run}")
                    argv = [app.binary(variant)] + list(app.run_args(variant, size, iters, cfg))
                    workdir = APPS_OPENMP / app.directory

                    env = dict(os.environ)
                    env.update(DEFAULT_ENV)
                    env["OMP_PLACES"] = args.places
                    env["XKRT_DRIVERS"] = args.drivers
                    if args.threads:
                        env["OMP_NUM_THREADS"] = str(args.threads)
                    if cfg.opt is not None:
                        env["OMP_TASKGRAPH_OPT"] = cfg.opt
                    # CGIR per-pass stats: only taskgraph configs produce passes.
                    if not args.no_stats and cfg.opt is not None:
                        env["CGIR_STATS_CSV"] = str(stats_csv)
                        env["CGIR_STATS_TAG"] = run_id

                    pretty = " ".join(argv)
                    print(f"[run ] {cfg.label:34s} {disp} n={size} : {pretty}",
                          file=sys.stderr)

                    row = {k: "" for k in CSV_FIELDS}
                    row.update({
                        "run_id": run_id,
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                        "machine": machine, "app": app_name, "variant": variant,
                        "config": cfg.label, "opt": ("" if cfg.opt is None else cfg.opt),
                        "build_vars": " ".join(f"{k}={v}" for k, v in
                                               {**cfg.build, **backend_vars}.items()),
                        "backend": args.target, "size": size,
                        "work": work, "work_label": work_label, "iters": iters,
                        "cmd": pretty,
                    })

                    if args.dry_run:
                        continue
                    if not ok:
                        row["status"] = "build_fail"
                        row["returncode"] = 1
                        n_fail += 1
                        writer.writerow(row); fh.flush()
                        continue

                    try:
                        p = subprocess.run(argv, cwd=str(workdir), env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, timeout=(args.timeout or None))
                        row["returncode"] = p.returncode
                        row["status"] = "ok" if p.returncode == 0 else "run_fail"
                        if p.returncode == 0:
                            for k, v in app.parse(p.stdout).items():
                                if v is not None:
                                    row[k] = v
                            n_ok += 1
                        else:
                            n_fail += 1
                            sys.stderr.write(p.stdout[-2000:] + "\n")
                    except subprocess.TimeoutExpired:
                        row["status"] = "timeout"
                        row["returncode"] = -1
                        n_fail += 1
                        print(f"      -> timeout after {args.timeout:.0f}s", file=sys.stderr)

                    writer.writerow(row); fh.flush()

    if fh:
        fh.close()
    print("", file=sys.stderr)
    print(f"ok={n_ok} fail={n_fail}", file=sys.stderr)
    if not args.dry_run:
        print(f"runs    -> {runs_csv}", file=sys.stderr)
        if not args.no_stats:
            print(f"cgstats -> {stats_csv}", file=sys.stderr)
    return 0 if n_fail == 0 else 1


def _parse_opts(arg):
    if not arg:
        return list(DEFAULT_OPTS)
    return [o.strip() for o in re.split(r"[;]", arg) if o.strip()]


def _parse_sizes(arg):
    """Parse --sizes into (global_default_or_None, {app: [sizes]}).

    Items are ';'-separated; an item 'app=8,16,24' sets that app's sizes, a bare
    item '8,16,24' sets the global default. E.g. '8,16;lulesh=30,45' -> default
    [8,16] with lulesh overridden to [30,45]. Empty -> (None, {})."""
    default, by_app = None, {}
    for chunk in arg.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            app, _, lst = chunk.partition("=")
            by_app[app.strip()] = [int(s) for s in lst.split(",") if s.strip()]
        else:
            default = [int(s) for s in chunk.split(",") if s.strip()]
    return default, by_app


if __name__ == "__main__":
    sys.exit(main())
