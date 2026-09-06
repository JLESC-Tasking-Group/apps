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

Two artifacts, five sweeps. `$OPTS` is the incremental pipeline (the default of
`appspecs.py`, spelled out here so a sweep is self-describing):

```sh
OPTS='reduce-node,transitive-reduction'
OPTS="$OPTS;reduce-node,transitive-reduction,jit"
OPTS="$OPTS;reduce-node,transitive-reduction,jit,prog-fuse"
OPTS="$OPTS;reduce-node,transitive-reduction,jit,prog-fuse,sequence,batch"
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
./scripts/evaluate.py --target gpu --apps krylov,lulesh,mnmg \
    --opts "$OPTS" --unroll 1,8 \
    --sizes 'krylov=64;lulesh=16,60,100;mnmg=7035,23874' \
    --iters 'krylov=200;lulesh=104'
```

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

**2. `taskgraphloop` control** -- same iteration count and epilogue cadence, one
recorded instance per iteration. The difference against sweep 1 is the gain that
comes from cross-iteration overlap rather than from fewer instances.

```sh
./scripts/evaluate.py --target gpu --apps lulesh,krylov --variants cg \
    --opts "$OPTS" --unroll 1,8 --no-taskgraphloop --tag no-tgl
```

**3. JIT cache regimes** -- the same full pipeline three times. `--tag` keeps the
three apart in one results file (it is part of `run_id`, so the CGIR side files
stay joinable).

```sh
FULL='reduce-node,transitive-reduction,jit,prog-fuse,sequence,batch'
./scripts/evaluate.py --target gpu --opts "$FULL" --unroll 8 \
    --env CGIR_JIT_CACHE=0            --tag jit-cold
./scripts/evaluate.py --target gpu --opts "$FULL" --unroll 8 \
    --tag jit-mem                      # in-process cache: the default
./scripts/evaluate.py --target gpu --opts "$FULL" --unroll 8 \
    --env CGIR_JIT_CACHE_DIR=$PWD/results/jitcache --tag jit-disk   # run twice
```

**4. OmpSs-2 / NODES, CPU** -- the second runtime. Needs a NODES built
`--with-cgir`; the harness turns the CGIR path on (`taskiter.opt.use_cgir`) and
passes the same pass names through `NODES_TASKITER_CGIR_OPT`.

```sh
export OMPSS_CC=<ompss-2 clang++>
./scripts/evaluate.py --target ompss --apps krylov --variants cg \
    --opts "$OPTS" --unroll 1,8 --sizes 'krylov=64' --iters 'krylov=200'
```

On this backend `tasking.h` records through `#pragma oss taskiter`, and `--unroll`
keeps its meaning -- iterations per recorded instance -- so the same value is
comparable across the two runtimes.

**5. Hand-written references** -- built and run by hand (different sources and
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
./scripts/plot.py --paper --external results/external.csv \
                  --latex-table ../../paper/sections/generated-table.tex
```

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
