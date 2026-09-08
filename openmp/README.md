# apps/openmp

OpenMP task / target benchmarks used to evaluate the `taskgraph` construct and
its CGIR command-graph optimizations. Each app is a single source expressing two
backends (CPU tasks vs GPU target offload) plus a synchronous baseline, selected
by compile-time toggles shared through `common.mk`.

| App                              | CPU tasks | GPU target | OmpSs-2 | synchronous | taskgraph | in harness |
| -------------------------------- | :-------: | :--------: | :-----: | :---------: | :-------: | :--------: |
| Krylov (cg/cr/bicgstab/minres/gmres) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| LULESH                           | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MNMG (Datalog TC)                | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cholesky                         | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| llm.c                            | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

llm.c is **not** in the harness. Its OpenMP-target port is a transliteration of
the reference implementation -- a scalar triple-loop matmul, fp32, no tensor
cores, 860 kernels per training step of which 92 run on a single thread block --
so it reaches ~0.4 % of a GH200's fp32 peak and every configuration takes the
same time. The passes target launch overhead and inter-kernel memory traffic;
neither is where its time goes, so it measures the port rather than the runtime.
(Its self-reported "MFU" is also misleading: `train_gpt2.c` divides by a
dual-socket Sapphire Rapids FP64 peak, not the GPU's.) It is kept as a
standalone benchmark: `make -C llm.c`.

Cholesky is kept as a standalone tasks/target benchmark but is **not** a
record/replay taskgraph example (its tiled DAG changes shape every step), so it
is excluded from the shared harness; build it with `make -C cholesky`.

## Building

One top-level `Makefile` dispatches to each app (which shares `common.mk`):

```sh
make                 # build all harness apps (CPU tasks + taskgraph, defaults)
make krylov          # just the krylov solvers   (make lulesh / make llmc)
make USE_TARGET=1    # GPU target offload for all apps
make USE_SYNC=1      # synchronous blocking baseline
make USE_TASKGRAPH=0 # plain tasks/target, no record/replay
make USE_OMPSS=1 OMPSS_CC=<ompss-2 clang++>   # OmpSs-2 / NODES host tasks
make clean
```

The backend/schedule toggles (`USE_TARGET`, `USE_TASKGRAPH`, `USE_SYNC`,
`USE_REPLAYABLE`, `USE_OMPSS`) live in `common.mk` and propagate to the per-app
Makefiles, which remain usable directly (e.g. `make -C lulesh run`,
`make -C krylov`). `USE_OMPSS=1` also switches the compiler to `$(OMPSS_CC)`
and the flags to `-fompss-2=libnodes`; it is mutually exclusive with
`USE_TARGET=1`.

## Evaluating

`scripts/evaluate.py` sweeps each app across problem sizes and configurations
(synchronous / no-taskgraph / taskgraph:none / taskgraph:\<opt\>), rebuilding the
right binary per configuration, and writes `results/runs.csv`. Per-pass CGIR
command-graph stats are collected via `CGIR_STATS_CSV` into `results/cgstats.csv`
(auto; disable with `--no-stats`). `scripts/plot.py` renders the figures.

```sh
./scripts/evaluate.py --list                     # apps + configurations
./scripts/evaluate.py                            # all apps, CPU, default sizes
./scripts/evaluate.py --apps lulesh --sizes 30,45,60
./scripts/evaluate.py --apps krylov --variants cg --target gpu
./scripts/plot.py                                # -> results/figures/*.png
```

`plot.py` produces `time-<app>.pdf` (avg time / iteration with stddev error bars,
one bar per configuration, work on the top axis) and `graph-<app>.pdf` (CGIR
per-pass command-graph reduction and pass wall time). With `--paper` it also
produces `paper-speedup.pdf`, and with `--latex-table` the paper's statistics
table; see below.

## Reproducing the paper's evaluation

`results/sweep1-single/` holds the earlier single-run sweep, kept because it is
what the pre-`--repeat` numbers came from and it still replots with
`--outdir results/sweep1-single`. It is not the paper's data: one run per
configuration cannot separate a pass from the machine (see `--repeat` below).

Three artifacts, three sweeps. Sweeps 4 and 5 below are not part of the paper:
the OmpSs-2/CPU integration is preliminary and appears only as future work, and
the hand-written baselines are deferred. `$OPTS` is the incremental pipeline (the default of
`appspecs.py`, spelled out here so a sweep is self-describing):

```sh
OPTS='reduce-node,transitive-reduction'
OPTS="$OPTS;reduce-node,transitive-reduction,jit"
OPTS="$OPTS;reduce-node,transitive-reduction,jit,prog-fuse"
OPTS="$OPTS;reduce-node,transitive-reduction,jit,prog-fuse,batch"
```

Note the absence of `\` line continuations: inside single quotes a backslash is
literal, so `'a;\<newline>b'` puts a backslash and a newline *into the value*.
That once cost a whole overnight sweep -- both runtimes warned about the unknown
pass and carried on, so `reduce-node` never ran in three of the four pipelines
while the CSV said it had. `evaluate.py` now validates every pass name against
CGIR's list and refuses to start, and both runtimes now treat an unknown name as
fatal, so the same mistake fails loudly instead.

**1. End-to-end, GPU (GH200)** -- the figure, and the graph statistics and pass
costs of the table. `cgstats.csv` / `jitstats.csv` are written automatically.

```sh
SZ='krylov=64;lulesh=16,60,100;mnmg=7035,23874'
IT='krylov=200;lulesh=208'
GR='krylov=4:4;lulesh=1,4,8'
./scripts/evaluate.py --target gpu --apps krylov,lulesh,mnmg \
    --opts "$OPTS" --unroll 1,8 --sizes "$SZ" --iters "$IT" --grain "$GR" \
    --repeat 5
```

`--repeat 5` is not optional. A run's own `stddev_ms` is taken over the instances
*inside* one process, so it cannot see what changes between processes -- the code
the JIT emitted, the page cache, the clock and power state. On the smaller
problems that is the larger variance by far: with one run per configuration,
`prog-fuse` appeared to cost 26% on LULESH at `n=16` while the pass provably did
nothing at all (node count 442 -> 442, zero chains fused). Repeats are what tell
a pass apart from the machine; `plot.py` reduces them to a median and draws the
extremes as whiskers. Repeats are interleaved, not consecutive, so machine drift
is spread across configurations instead of landing on one.

Each configuration is still built once. Because every app cleans its whole
directory before building -- krylov's `clean` is `rm -f *.x`, which takes all
four solvers -- the binary is copied to `results/.binstash/` as soon as it is
built, and later repeats run that copy; the in-tree one is gone by then. The
stash is removed when the sweep finishes, and kept if any run failed. This is
also why `--skip-build` is refused with `--repeat > 1`: nothing is built, so
nothing is stashed, and the tree holds only the last binary of each app.
`scripts/test_repeat.py` covers it with a fake app and needs no compiler.

`--grain` is what makes these graphs parallel -- four tasks per Krylov vector
operation, and a task count per LULESH size. It also removes every fusible chain,
which is why `prog-fuse` is flat in the results; a run at `krylov=0:0` (one task
per operation) is the configuration in which fusion has something to do.

Krylov gets a single size on purpose: its panel of the figure puts the five
solvers on the x axis (`AppSpec.panel_x`), which says more about generality than
one solver at three sizes, and the section has room for one panel per app.

Two per-variant adjustments happen automatically and are visible in the `cmd`
column: GMRES is a *restarted* solver, so its `-i` counts restart cycles of 30
inner steps (`-i 7` here $\approx$ the others' 200 iterations) and it is pinned
to `u=1`, because each restart ends with a host solve the next one consumes.
The harness also warns when a run would have fewer than five graph instances --
instances 0 and 1 are the record and the build, so a short run has a
steady-state mean over two or three samples. A clean dry-run prints no such
warning.

**2. The unroll sweep** -- `figures/paper-unroll.pdf`, the figure behind
§5.2's instance-barrier paragraph. A taskgraph instance ends in a taskwait, so
consecutive instances cannot overlap while `no-taskgraph` overlaps iterations
freely; unrolling pays that barrier once per group instead of once per iteration.
The curve's crossing of 1.0 is what says whether recording a graph pays for
itself on a given app -- LULESH is *slower* than `no-taskgraph` until several
iterations share an instance.

```sh
./scripts/evaluate.py --target gpu --apps lulesh,krylov,mnmg --variants cg \
    --opts "$OPTS" --unroll 1,2,4,8,16 --iters "$IT" --repeat 5 --tag unroll \
    --sizes 'krylov=64;lulesh=100;mnmg=23874' --grain 'krylov=4:4;lulesh=8'
```

One size per app, the largest: the crossing point moves with problem size, so a
curve may only carry one, and the figure pins each panel to the largest size it
finds. Sweeping `$SZ` here instead would measure 405 extra runs and discard them.

Then `./scripts/plot.py --paper-unroll-figure --unroll-tag unroll`. The tag is
what keeps the figure to this sweep: sweep 1 also ran `u=1` and `u=8`, and
without it both would land on the same curve with the second silently replacing
the first. No `--omit` either -- the sweep needs its own `no-taskgraph`
reference, which the harness pins to `u=1` by itself (it records no graph, so it
has no unroll to vary).

**3. The on-disk JIT cache** -- the cost table's `cached` break-even column.

`CGIR_JIT_CACHE_DIR` is opt-in, so sweep 1 pays a cold compile in every run, and
JIT is 75-99% of the optimization cost. This sweep measures what a *later* run of
the same application costs, once the compiled kernels are on disk.

Run the full pipeline twice against the same cache directory. The first pass uses
`CGIR_JIT_CACHE_MODE=w`, which writes to the cache and never reads it: without
that, a populating run reuses artifacts it produced moments earlier and is not a
first run at all. The second pass uses `r`, so it measures against a cache it
does not modify and can be repeated. `plot.py --cached-tag` reads the second.

```sh
FULL='reduce-node,transitive-reduction,jit,prog-fuse,batch'
D=$PWD/results/jitcache && rm -rf $D
run () {   # $1 = cache mode, $2 = tag
  ./scripts/evaluate.py --target gpu --apps krylov,lulesh,mnmg --opts "$FULL" \
      --unroll 8 --sizes "$SZ" --iters "$IT" --grain "$GR" \
      --env CGIR_JIT_CACHE_DIR=$D --env CGIR_JIT_CACHE_MODE=$1 --tag $2
}
run w jit-disk-cold      # fills the cache, reads nothing: a true first run
run r jit-disk-warm      # reads it, writes nothing: the measurement
```

Add `--omit synchronous,no-taskgraph` to both if sweep 1 already measured them on
the same problems: they run no CGIR pass, so re-running them only produces a
second copy of each, and `plot.py --main-tag` then has to be told which one
to use. The cost table's *cold* column comes from sweep 1 rather than from
`jit-disk-cold`, so that it excludes the cost of writing the cache out.

Check it worked in `results/jitstats.csv`: the `jit-disk-cold` rows must have
`device_disk_reuse = 0` (it read nothing) and the `jit-disk-warm` rows must have
`device_compiled = 0` (it compiled nothing).

**4. OmpSs-2 / NODES, CPU** (not in the paper) -- the second runtime. Needs a NODES built
`--with-cgir`; the harness turns the CGIR path on (`taskiter.opt.use_cgir`) and
passes the same pass names through `NODES_TASKITER_CGIR_OPT`.

Run both host backends, not just OmpSs-2: the same application and the same
passes under two runtimes is the paired comparison.

`sequence` is added back for this sweep only. It groups a same-device chain into
one serial BATCH the runtime replays as a single super-task, which is a host
construct: on the GPU graphs it batched nothing (measured, on all three
applications), which is why it is not in `$OPTS`, but the host backends are what
it was written for and the only place it can pay. Dropping it from the default
should not also drop the one sweep that can show it working.

```sh
export OMPSS_CC=<ompss-2 clang++>
HOST_OPTS="$OPTS;reduce-node,transitive-reduction,jit,prog-fuse,sequence,batch"
for T in cpu ompss; do
  ./scripts/evaluate.py --target $T --apps krylov --variants cg \
      --opts "$HOST_OPTS" --unroll 1,8 --sizes 'krylov=64' --iters 'krylov=200' \
      --repeat 5
done
```

On this backend `tasking.h` records through `#pragma oss taskiter`, and `--unroll`
keeps its meaning -- iterations per recorded instance -- so the same value is
comparable across the two runtimes.

**5. Hand-written references** (not in the paper) -- built and run by hand (different sources and
toolchains, so not part of the sweep), then written into a small CSV that the
figure overlays:

```sh
make -C MNMGDatalog/MNMGDatalog-reference tc_benchmark   # v1_baseline, v2_cudagraph
cat > results/external.csv <<'EOF'
app,variant,size,label,avg_ms
mnmg,,7035,CUDA graph (hand-written),<measured>
mnmg,,23874,CUDA graph (hand-written),<measured>
EOF
```

**Rendering:**

```sh
./scripts/plot.py --paper --latex-tables ../../paper/sections \
                  --pipeline "taskgraph:$FULL" \
                  --main-tag '' --cached-tag jit-disk-warm
./scripts/plot.py --paper-unroll-figure --unroll-tag unroll
```

That writes `figures/paper-speedup.pdf` (copy it to `paper/figures/eval-speedup.pdf`),
`figures/paper-unroll.pdf` (-> `paper/figures/eval-unroll.pdf`) and the section's
two tables: `generated-table-graph.tex` (what the passes do to the graph) and
`generated-table-cost.tex` (what they cost, and the replays that repay it).
`--pipeline` must name the pipeline actually swept -- including `sequence` if the
results predate its removal from the default (the CSVs shipped here do; it cost
0.1 ms and batched nothing, so the measurements are unaffected and `plot.py`
labels both spellings `+packing`).

Bars are medians over `--repeat` runs and whiskers span the extreme ratios --
fastest baseline over slowest replay, and the reverse -- so a bar clears 1.0
only if it does so under every pairing of the runs behind it. A configuration
measured fewer times than the rest is reported, because its whisker is not
comparable to the others'.

The baseline of both the figure and the break-even column is `no-taskgraph`
(`--baseline`), so the two artifacts always tell the same story.

**The answer check runs first, and is not optional.** Every configuration must
reproduce the answer of the `--answer-reference` configuration (default
`synchronous`) for the same problem: the relative residual for Krylov, the
transitive-closure size for MNMG, LULESH's own `Verification` line plus its
`TotalAbsDiff`. A run that disagrees is dropped and named, because a speedup from
a run that computed something else is not a speedup. This exists because it
happened: `prog-fuse` used to merge two `omp target` launches without checking
the dependence between them, which removed the device-wide barrier and moved CG's
residual from 4e-15 to 2.6e-03 while reporting a healthy speedup.
