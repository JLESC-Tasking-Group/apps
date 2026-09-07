#!/usr/bin/env python3
"""
evaluate.py - shared evaluation sweep for the apps/openmp taskgraph apps
(krylov, lulesh, mnmg). For each app x variant x configuration x problem size it
builds the right binary, runs it, parses the per-iteration timing, and appends a
row to results/runs.csv. Alongside (auto, unless --no-stats), for taskgraph
configs CGIR writes two side files that join runs.csv on run_id == tag:
per-pass command-graph stats via CGIR_STATS_CSV -> results/cgstats.csv, and the
per-run JIT compile breakdown + cache reuse via CGIR_JIT_STATS_CSV ->
results/jitstats.csv (rows only for opts that include the `jit` pass).

Configurations always include the three references (synchronous, no-taskgraph,
taskgraph:none) plus one taskgraph:<opt> per CGIR optimization combo (see
appspecs.py / --opts). The synchronous / no-taskgraph / taskgraph split is a
compile-time choice, so binaries are rebuilt per configuration; the CGIR pass
within a taskgraph build is a run-time environment variable. The backend
(--target cpu | gpu | ompss) is orthogonal and applied to every build; `ompss`
runs the OmpSs-2 / NODES port of the same source and takes its pass list through
NODES_TASKITER_CGIR_OPT, so one --opts string is comparable across runtimes.

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
import itertools
import math
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from appspecs import (APPS, BACKENDS, CGIR_PASSES, DEFAULT_OPTS,  # noqa: E402
                       REFERENCE_CONFIGS, default_configs)

APPS_OPENMP = Path(__file__).resolve().parent.parent

DEFAULT_ENV = {
    "XKRT_STATS":   "0",
    "OMP_PLACES":   "cores",
    "XKRT_DRIVERS": "host,2;cuda,1",
}

CSV_FIELDS = [
    "run_id", "timestamp", "machine", "tag",
    "app", "variant", "config", "opt", "build_vars", "backend", "env",
    "size", "work", "work_label", "iters", "unroll", "grain",
    "avg_ms", "stddev_ms", "iter0_ms", "iter1_ms", "elapsed_s", "fom", "flops", "gflops",
    "residual", "error", "answer", "verdict",
    "returncode", "status", "cmd",
]


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")


def ordered_unique(seq):
    """De-duplicate while preserving order (a set would reorder the sweep)."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_cmd(app, variant, cfg, size, iters, backend_vars, grain, unroll):
    variables = dict(cfg.build)
    variables.update(backend_vars)
    argv = ["make", "-C", app.directory, "clean", app.make_target(variant)]
    argv += [f"{k}={v}" for k, v in variables.items()]
    if app.rebuild_per_size and app.build_defs:
        # An app whose problem size is a compile-time macro contributes extra
        # make variables here; on sync the grain is 1/loop.
        g = None if cfg.grain1 else grain
        argv += app.build_defs(size, iters, g, unroll)
    return argv


def parse_env(items, ap):
    """Parse repeated --env K=V into an ordered dict, rejecting malformed items.

    An unparsable item is fatal rather than skipped: the point of --env is to
    change what is measured (a JIT cache regime, say), so silently dropping one
    would produce a row that is labelled as one regime and ran as another."""
    env = {}
    for item in items:
        if "=" not in item:
            ap.error(f"--env: '{item}' is not K=V")
        k, _, v = item.partition("=")
        k = k.strip()
        if not k:
            ap.error(f"--env: '{item}' has an empty variable name")
        env[k] = v
    return env


def effective_unroll(app, backend, variant, unroll):
    """Clamp an unroll to what this (backend, variant) supports (see AppSpec)."""
    caps = [c for c in (app.max_unroll.get(backend),
                        app.variant_max_unroll.get(variant)) if c]
    return min([unroll] + caps)


def variant_iters(app, variant, iters):
    """Scale an iteration count to what one `-i` unit buys for this variant.

    See AppSpec.variant_iters_div: a restarted solver's `-i` counts cycles of
    several inner steps, so the same --iters must be divided to mean the same
    amount of work. Never returns 0 -- a sweep that asks for few iterations
    should run one unit, not none."""
    div = app.variant_iters_div.get(variant)
    return max(1, int(round(iters / float(div)))) if (div and iters) else iters


def effective_iters(iters, unroll):
    """Round an iteration count up to a whole number of taskgraph instances.

    A trailing partial instance is a *different* graph -- it would record instead
    of replay -- so the apps refuse or truncate it. Rounding up here keeps the
    instance count exact and identical across configurations, at the price of at
    most unroll-1 extra iterations; the reported time is per iteration, so the
    metric is unaffected. The rounded value is what lands in the `iters` column."""
    if unroll <= 1 or iters <= 0:
        return iters
    return int(math.ceil(iters / float(unroll))) * unroll


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apps", default="", help="comma list (default: all)")
    ap.add_argument("--variants", default="", help="comma list to filter variants")
    ap.add_argument("--sizes", default="", help="problem sizes: a global comma list "
                    "(e.g. '8,16,24') and/or per-app 'app=list' items, separated by ';' "
                    "(e.g. 'krylov=32,48,64;lulesh=30,45,60'). A per-app list overrides the "
                    "global list, which overrides each app's built-in default.")
    ap.add_argument("--iters", default="", help="iterations: a global value (e.g. '200') "
                    "and/or per-app 'app=N' items separated by ';' (e.g. 'krylov=200;lulesh=30'). "
                    "A per-app value overrides the global one, which overrides each app's default.")
    ap.add_argument("--grain", default="", help="tasks-per-loop granularity, ONE entry per "
                    "problem size: per-app 'app=<e1>,<e2>,...' items separated by ';' (e.g. "
                    "'lulesh=1,8,16;krylov=4:4,8:2,2:8'); a bare list is the global default. "
                    "Entries pair positionally with --sizes; a single entry is applied to every "
                    "size. Each entry is a ':'-separated list of that app's knobs: krylov 's:t' "
                    "-> -s/-t, lulesh 'nb' -> -nb; mnmg has none. "
                    "Applies to the async configs only (sync is "
                    "always 1 task/loop). Unset -> each app's default granularity.")
    ap.add_argument("--unroll", default="", help="iterations folded into ONE taskgraph "
                    "instance -- a sweep dimension: a global comma list (e.g. '1,2,4,8') "
                    "and/or per-app 'app=list' items separated by ';' (e.g. "
                    "'1,2,4;lulesh=1,8'). A taskgraph instance carries an implicit taskgroup, "
                    "so instances cannot overlap; unrolling recovers that overlap inside the "
                    "graph. Swept for the taskgraph configurations only -- elsewhere it is "
                    "inert by construction, so synchronous / no-taskgraph run once, at the "
                    "first value. Iteration counts are rounded up to a whole number of "
                    "instances. Default: 1 (one iteration per instance).")
    ap.add_argument("--no-taskgraphloop", action="store_true",
                    help="build with USE_TASKGRAPHLOOP=0: the apps then record one instance "
                    "PER ITERATION instead of one per --unroll group, keeping the iteration "
                    "count and the epilogue cadence identical. This is the A/B baseline the "
                    "taskgraphloop construct is measured against.")
    ap.add_argument("--opts", default="", help="semicolon-separated CGIR opt combos, each "
                    "a comma/space list of passes (e.g. 'reduce-node,transitive-reduction;batch'); "
                    "each combo -> one taskgraph:<opt> config. Default from appspecs.")
    ap.add_argument("--target", choices=sorted(BACKENDS), default="cpu",
                    help="backend for every build; 'cpu'/'gpu' are OpenMP/XKOMP host "
                    "tasks vs target offload, 'ompss' is the OmpSs-2/NODES port "
                    "(host only, and only for apps that have one). Default cpu")
    ap.add_argument("--env", action="append", default=[], metavar="K=V",
                    help="extra environment variable applied to every run (repeatable). "
                    "Applied after the harness' own variables, so it can override them "
                    "-- which is how the JIT cache regimes are swept, e.g. "
                    "--env CGIR_JIT_CACHE=0 (cold) or --env CGIR_JIT_CACHE_DIR=/tmp/jitc "
                    "(persistent). Recorded in the `env` column of runs.csv.")
    ap.add_argument("--omit", default="", metavar="LIST",
                    help="comma list of reference configurations NOT to run, from "
                    + ", ".join(REFERENCE_CONFIGS) + ". None of them runs a CGIR pass, "
                    "so a follow-up sweep that only varies the passes need not measure "
                    "them again -- and re-running them leaves a second copy of each in "
                    "the results. Omitting 'no-taskgraph' means this sweep carries no "
                    "baseline of its own, so the analysis has to take one from another "
                    "sweep of the same problems (plot.py --baseline-tag)")
    ap.add_argument("--tag", default="", help="free-form string written to the `tag` column "
                    "of runs.csv, to mark a sweep (e.g. 'jit-cold') so several sweeps can "
                    "share one results file and still be told apart")
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
                  f" backends={spec.backends}"
                  f"{' (rebuild per size)' if spec.rebuild_per_size else ''}")
        print("\nbackends:")
        for name, b in sorted(BACKENDS.items()):
            print(f"  {name:8s} build={b.build} passes via {b.opt_env}"
                  f"{' env=' + str(b.env) if b.env else ''}")
        opts = _parse_opts(args.opts, ap)
        print("\nconfigurations:")
        for c in default_configs(opts):
            print(f"  {c.label:34s} build={c.build} opt={c.opt!r}"
                  f"{' grain1' if c.grain1 else ''}"
                  f"{'' if c.taskgraph else ' (no unroll sweep)'}")
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
    iters_default, iters_by_app = _parse_ints(args.iters)
    for a in iters_by_app:
        if a not in APPS:
            ap.error(f"unknown app '{a}' in --iters (known: {', '.join(APPS)})")
    unroll_default, unroll_by_app = _parse_sizes(args.unroll)   # same list syntax
    for a in unroll_by_app:
        if a not in APPS:
            ap.error(f"unknown app '{a}' in --unroll (known: {', '.join(APPS)})")
    for name, lst in [("--unroll", unroll_default)] + list(unroll_by_app.items()):
        for u in lst or []:
            if u < 1:
                ap.error(f"--unroll: '{u}' is not a positive iteration count")
    grain_default, grain_by_app = _parse_grain(args.grain)   # per-app list of per-size entries
    for a in grain_by_app:
        if a not in APPS:
            ap.error(f"unknown app '{a}' in --grain (known: {', '.join(APPS)})")
    omit = [o.strip() for o in args.omit.split(",") if o.strip()]
    for o in omit:
        if o not in REFERENCE_CONFIGS:
            ap.error(f"--omit: '{o}' is not a reference configuration "
                     f"(known: {', '.join(REFERENCE_CONFIGS)})")
    if "no-taskgraph" in omit:
        print("[note ] --omit no-taskgraph: this sweep records no baseline, so "
              "speedups and break-even must be computed against another sweep "
              "(plot.py --baseline-tag)", file=sys.stderr)
    configs = default_configs(_parse_opts(args.opts, ap), omit)
    backend = BACKENDS[args.target]
    backend_vars = dict(backend.build)
    if args.no_taskgraphloop:
        backend_vars["USE_TASKGRAPHLOOP"] = "0"
    extra_env = parse_env(args.env, ap)
    env_col = " ".join(f"{k}={v}" for k, v in extra_env.items())

    # An app without a port for this backend is skipped, not built: its task
    # constructs would compile to nothing and the run would silently measure a
    # serial program.
    skipped = [a for a in selected if args.target not in APPS[a].backends]
    selected = [a for a in selected if args.target in APPS[a].backends]
    if skipped:
        print(f"[skip ] no {args.target} port: {', '.join(skipped)}", file=sys.stderr)
    if not selected:
        ap.error(f"none of the selected apps has a '{args.target}' port")

    outdir = Path(args.outdir)
    runs_csv = Path(args.out) if args.out else outdir / "runs.csv"
    stats_csv = outdir / "cgstats.csv"
    jit_csv = outdir / "jitstats.csv"
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    machine = socket.gethostname()
    ts_run = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    new_file = not runs_csv.exists() or runs_csv.stat().st_size == 0
    if not args.dry_run and not new_file:
        _check_csv_header(runs_csv, ap)
    fh = None if args.dry_run else open(runs_csv, "a", newline="")
    writer = None
    if fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()

    built = {}
    n_ok = n_fail = 0
    fail_by_app = {}   # app_name -> failed run count

    def do_build(app, variant, cfg, size, iters, grain, unroll):
        key = (app.name, variant, cfg.label, args.target)
        if app.rebuild_per_size:
            # size, grain (GRAN_TMP/...) and unroll (UNROLL) are all compile-time
            # for such an app, so each combination is its own binary; tuple()
            # to stay hashable. iters is compiled in too, but it is a pure
            # function of (size, unroll) here, so it needs no key of its own.
            key = key + (size, tuple(grain) if grain else None, unroll)
        if key in built:
            return built[key]
        cmd = build_cmd(app, variant, cfg, size, iters, backend_vars, grain, unroll)
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
        iters = iters_by_app.get(app_name) or iters_default or app.iters
        # One grain entry per size (None -> the app's own default granularity).
        grains = _grain_for_sizes(app, grain_by_app.get(app_name) or grain_default,
                                  sizes, app_name in grain_by_app, ap)
        unrolls = unroll_by_app.get(app_name) or unroll_default or [1]

        for variant in variants:
            for cfg in configs:
                # Only a taskgraph has the per-instance barrier that unrolling
                # amortizes. The other configurations are the reference the
                # taskgraph rows are compared against, so they run once, at
                # unroll 1 -- pinned rather than taking unrolls[0], so that
                # `--unroll 2,4` cannot label the baseline series "no-taskgraph u2".
                cfg_unrolls = unrolls if cfg.taskgraph else [1]
                # Clamp to what the app accepts here, then de-duplicate: on a
                # backend capped at 1, `--unroll 1,2,4` must run once, not thrice.
                cfg_unrolls = ordered_unique(
                    effective_unroll(app, args.target, variant, u) for u in cfg_unrolls)
                # product() materializes its arguments, so the one-shot zip is safe
                for unroll, (size, grain) in itertools.product(cfg_unrolls,
                                                               zip(sizes, grains)):
                    eff_iters = effective_iters(variant_iters(app, variant, iters),
                                                unroll)
                    ok = do_build(app, variant, cfg, size, eff_iters, grain, unroll)
                    work, work_label = app.work(size)
                    vtag = f"-{variant}" if variant else ""
                    disp = f"{app_name}/{variant}" if variant else app_name
                    # The tag is part of the id, not only of its own column: the
                    # two CGIR side files join on run_id, so two sweeps that
                    # differ only by --env (the JIT cache regimes) must not
                    # collide there.
                    tagpart = f"-{sanitize(args.tag)}" if args.tag else ""
                    run_id = sanitize(f"{app_name}{vtag}-{args.target}-{cfg.label}"
                                      f"-n{size}-u{unroll}{tagpart}-{ts_run}")
                    argv = [app.binary(variant)] + list(
                        app.run_args(variant, size, eff_iters, cfg, grain, unroll))
                    workdir = APPS_OPENMP / app.directory

                    env = dict(os.environ)
                    env.update(DEFAULT_ENV)
                    env.update(backend.env)
                    env["OMP_PLACES"] = args.places
                    env["XKRT_DRIVERS"] = args.drivers
                    if args.threads:
                        env["OMP_NUM_THREADS"] = str(args.threads)
                    if cfg.opt is not None:
                        # Same pass names for every backend; only the variable the
                        # runtime reads them from differs (see appspecs.Backend).
                        env[backend.opt_env] = cfg.opt
                    # CGIR stats: only taskgraph configs produce passes. cgstats
                    # is per-pass command-graph stats; jitstats is the per-run JIT
                    # compile breakdown + cache reuse (populated only for opts that
                    # include the `jit` pass). Both join runs.csv on run_id == tag.
                    if not args.no_stats and cfg.opt is not None:
                        env["CGIR_STATS_CSV"] = str(stats_csv)
                        env["CGIR_STATS_TAG"] = run_id
                        env["CGIR_JIT_STATS_CSV"] = str(jit_csv)
                    # Last, so a sweep can override anything above -- notably the
                    # CGIR_JIT_CACHE* knobs whose regimes are the point of --env.
                    env.update(extra_env)

                    # The apps drop instance 0 (record) and 1 (build + first
                    # replay) from the steady-state window, so a run with fewer
                    # than three instances has no steady state at all and a few
                    # more has a mean of two or three samples. Say so rather than
                    # let a stddev over 2 points into the paper.
                    ninst = (eff_iters // unroll) if (eff_iters and unroll) else 0
                    if cfg.taskgraph and 0 < ninst < 5:
                        print(f"      -> only {ninst} instances "
                              f"({eff_iters} iters / u{unroll}): "
                              f"{max(ninst - 2, 0)} steady-state samples",
                              file=sys.stderr)

                    pretty = " ".join(argv)
                    utag = f" u={unroll}" if unroll != 1 else ""
                    print(f"[run ] {cfg.label:34s} {disp} n={size}{utag} : {pretty}",
                          file=sys.stderr)

                    row = {k: "" for k in CSV_FIELDS}
                    row.update({
                        "run_id": run_id,
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                        "machine": machine, "tag": args.tag,
                        "app": app_name, "variant": variant,
                        "config": cfg.label, "opt": ("" if cfg.opt is None else cfg.opt),
                        "build_vars": " ".join(f"{k}={v}" for k, v in
                                               {**cfg.build, **backend_vars}.items()),
                        "backend": args.target, "env": env_col, "size": size,
                        "work": work, "work_label": work_label, "iters": eff_iters,
                        "unroll": unroll,
                        "grain": (":".join(map(str, grain)) if grain else ""),
                        "cmd": pretty,
                    })

                    if args.dry_run:
                        continue
                    if not ok:
                        row["status"] = "build_fail"
                        row["returncode"] = 1
                        n_fail += 1
                        fail_by_app[app_name] = fail_by_app.get(app_name, 0) + 1
                        writer.writerow(row); fh.flush()
                        continue

                    try:
                        p = subprocess.run(argv, cwd=str(workdir), env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, timeout=(args.timeout or None))
                        row["returncode"] = p.returncode
                        row["status"] = "ok" if p.returncode == 0 else "run_fail"
                        metrics = app.parse(p.stdout)
                        for k, v in metrics.items():
                            if v is not None:
                                row[k] = v

                        if p.returncode == 0:
                            n_ok += 1
                        elif metrics.get("avg_ms") is not None:
                            # The run produced its complete result and then died,
                            # typically in teardown. Keeping the numbers but
                            # flagging the row beats throwing away a measurement
                            # that is there -- and beats pretending it is clean.
                            row["status"] = "ok_crashed_at_exit"
                            n_ok += 1
                            print(f"      -> completed, then exited with "
                                  f"{p.returncode}: kept as ok_crashed_at_exit",
                                  file=sys.stderr)
                        else:
                            n_fail += 1
                            fail_by_app[app_name] = fail_by_app.get(app_name, 0) + 1
                            sys.stderr.write(p.stdout[-2000:] + "\n")

                        # The app's own verdict is authoritative and immediate: a
                        # run that says it computed the wrong thing must never be
                        # reported as a data point, however fast it was.
                        if str(row.get("verdict", "")).lower() in ("fail", "failed"):
                            row["status"] = "wrong_answer"
                            print(f"      -> WRONG ANSWER: {app_name} reported "
                                  f"verdict '{row['verdict']}'", file=sys.stderr)

                        # One line per run in the log, carrying what the run
                        # computed and not only how fast. A sweep whose log shows
                        # only timings cannot be audited afterwards -- which is
                        # exactly the position a silently-corrupted campaign
                        # leaves you in.
                        summary = [f"{row['status']}"]
                        if row.get("avg_ms") != "":
                            summary.append(f"avg={row['avg_ms']} ms")
                        if row.get("verdict") != "":
                            summary.append(f"verdict={row['verdict']}")
                        if row.get("answer") != "":
                            summary.append(f"answer={row['answer']}")
                        print("      -> " + "  ".join(summary), file=sys.stderr)
                    except subprocess.TimeoutExpired:
                        row["status"] = "timeout"
                        row["returncode"] = -1
                        n_fail += 1
                        fail_by_app[app_name] = fail_by_app.get(app_name, 0) + 1
                        print(f"      -> timeout after {args.timeout:.0f}s", file=sys.stderr)

                    writer.writerow(row); fh.flush()

    if fh:
        fh.close()
    print("", file=sys.stderr)
    print(f"ok={n_ok} fail={n_fail}", file=sys.stderr)
    if fail_by_app:
        print("fail by app: " + ", ".join(f"{a}={n}" for a, n in sorted(fail_by_app.items())),
              file=sys.stderr)
    if not args.dry_run:
        print(f"runs     -> {runs_csv}", file=sys.stderr)
        if not args.no_stats:
            print(f"cgstats  -> {stats_csv}", file=sys.stderr)
            print(f"jitstats -> {jit_csv}", file=sys.stderr)
    return 0 if n_fail == 0 else 1


def _check_csv_header(path, ap):
    """Refuse to append rows that do not match an existing runs.csv header.

    The header is only written for a NEW file, so appending after CSV_FIELDS has
    changed would silently write rows of a different width than the header they
    live under -- and csv.DictReader maps by position, so every column after the
    first difference would be misread, for the old rows and the new ones alike.
    Corrupting a results file that way is much worse than refusing to write."""
    try:
        with open(path, newline="") as fh:
            header = next(csv.reader(fh), None)
    except OSError as e:
        ap.error(f"cannot read {path}: {e}")
    if header is None or header == CSV_FIELDS:
        return
    added   = [c for c in CSV_FIELDS if c not in header]
    removed = [c for c in header if c not in CSV_FIELDS]
    detail  = ", ".join(filter(None, [
        f"added {added}" if added else "",
        f"removed {removed}" if removed else "",
        "same columns, different order" if not added and not removed else "",
    ]))
    ap.error(f"{path} has a stale header ({detail}). Appending would misalign "
             f"every column after the change. Move it aside (e.g. `mv {path} "
             f"{path}.bak`) and rerun, or pass --out to write elsewhere.")


def _parse_opts(arg, ap=None):
    """Split --opts into pipelines and reject any unknown pass name.

    Rejecting is the whole point: a runtime that meets a name it does not know
    warns and carries on, so a typo silently measures a different pipeline than
    the one the results are labelled with. Better to refuse to start.

    Whitespace inside a pipeline is a separator like a comma (both runtimes
    tokenize on ", \t"), so a value that picked up a stray newline -- e.g. from a
    shell line continuation inside single quotes -- is normalized here rather
    than turned into a bogus pass name."""
    if not arg:
        return list(DEFAULT_OPTS)
    out = []
    for chunk in arg.split(";"):
        passes = [p for p in re.split(r"[,\s]+", chunk.strip()) if p]
        if not passes:
            continue
        unknown = [p for p in passes if p not in CGIR_PASSES and p != "none"]
        if unknown and ap is not None:
            ap.error(f"--opts: unknown CGIR pass(es) {unknown} in '{chunk.strip()}'. "
                     f"Known passes: {', '.join(sorted(CGIR_PASSES))}, none. "
                     f"(A runtime would warn and ignore them, silently running a "
                     f"different pipeline than the one your results claim.)")
        out.append(",".join(passes))
    return out


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


def _parse_grain(arg):
    """Parse --grain into (global_default_or_None, {app: [entry, ...]}).

    Items are ';'-separated; an item 'app=<spec>' sets that app's grain, a bare
    item sets the global default. A <spec> is a ','-separated list holding ONE
    entry per problem size, and each entry is a ':'-separated list of that app's
    granularity knobs. E.g. 'lulesh=1,8,16;krylov=4:4,8:2,2:8' gives lulesh
    -nb 1/8/16 and krylov (-s 4 -t 4)/(-s 8 -t 2)/(-s 2 -t 8) for their
    respective 1st/2nd/3rd --sizes. Empty -> (None, {})."""
    def entries(spec):
        return [[int(v) for v in e.split(":") if v.strip()]
                for e in spec.split(",") if e.strip()]

    default, by_app = None, {}
    for chunk in arg.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            app, _, spec = chunk.partition("=")
            by_app[app.strip()] = entries(spec)
        else:
            default = entries(chunk)
    return default, by_app


def _grain_for_sizes(app, entries, sizes, explicit, ap):
    """Align a parsed grain spec with `sizes`, returning one entry per size.

    `entries` is None (unset -> the app's own default), a single entry (applied
    to every size) or exactly one entry per size. `explicit` says whether the
    spec came from an 'app=' item rather than the global default, so that a
    global default can be silently ignored for apps that have no knob."""
    if not entries:
        return [None] * len(sizes)

    if app.grain_arity == 0:
        if explicit:
            ap.error(f"--grain: app '{app.name}' has no granularity knob")
        return [None] * len(sizes)

    # Over-long entries are an error when the app was named explicitly (the user
    # meant that app, so the arity is wrong), but merely trimmed when they come
    # from the bare global default, which must stay usable across apps whose
    # knob counts differ (e.g. --grain "3:5" over krylov 's:t' and lulesh 'nb').
    for e in entries:
        if len(e) > app.grain_arity and explicit:
            ap.error(f"--grain for '{app.name}': entry '{':'.join(map(str, e))}' has "
                     f"{len(e)} components but the app takes at most {app.grain_arity}")
    entries = [e[:app.grain_arity] for e in entries]

    # Stale flat syntax: 'krylov=4,4' used to mean one list [4,4] for every size;
    # it now reads as two single-knob entries. Reject it rather than silently
    # running a different granularity than intended.
    if (app.grain_arity >= 2 and len(sizes) > 1 and len(entries) == len(sizes)
            and all(len(e) == 1 for e in entries)):
        ap.error(f"--grain for '{app.name}': ambiguous spec "
                 f"'{','.join(str(e[0]) for e in entries)}' -- entries are now per SIZE and "
                 f"their knobs are ':'-separated. Write e.g. '4:4,8:2' for one entry per size, "
                 f"or a single entry like '4:4' to apply it to all sizes.")

    if len(entries) == 1:
        return [entries[0]] * len(sizes)
    if len(entries) != len(sizes):
        ap.error(f"--grain for '{app.name}': {len(entries)} entries but {len(sizes)} sizes "
                 f"({','.join(map(str, sizes))}) -- give one entry per size, or a single "
                 f"entry to apply to all")
    return entries


def _parse_ints(arg):
    """Parse an --iters style spec into (global_default_or_None, {app: int}),
    like _parse_sizes but with a single int per entry. E.g. 'krylov=200;lulesh=30'
    or '200'."""
    default, by_app = None, {}
    for chunk in str(arg).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            app, _, v = chunk.partition("=")
            by_app[app.strip()] = int(v)
        else:
            default = int(chunk)
    return default, by_app


if __name__ == "__main__":
    sys.exit(main())
