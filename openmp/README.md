# apps/openmp

OpenMP task / target benchmarks used to evaluate the `taskgraph` construct and
its CGIR command-graph optimizations. Each app is a single source expressing two
backends (CPU tasks vs GPU target offload) plus a synchronous baseline, selected
by compile-time toggles shared through `common.mk`.

| App                              | CPU tasks | GPU target | OmpSs-2 | synchronous | taskgraph | in harness |
| -------------------------------- | :-------: | :--------: | :-----: | :---------: | :-------: | :--------: |
| Krylov (cg/cr/bicgstab/minres/gmres) | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| LULESH                           | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| llm.c                            | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| MNMG (Datalog TC)                | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| Cholesky                         | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |

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
make USE_OMPSS=1 OMPSS_CC=<ompss-2 clang++>   # OmpSs-2 / NODES host tasks (llm.c)
make clean
```

The backend/schedule toggles (`USE_TARGET`, `USE_TASKGRAPH`, `USE_SYNC`,
`USE_REPLAYABLE`, `USE_OMPSS`) live in `common.mk` and propagate to the per-app
Makefiles, which remain usable directly (e.g. `make -C lulesh run`,
`make -C llm.c test`). `USE_OMPSS=1` also switches the compiler to `$(OMPSS_CC)`
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
OPTS='reduce-node,transitive-reduction;\
reduce-node,transitive-reduction,jit;\
reduce-node,transitive-reduction,jit,prog-fuse;\
reduce-node,transitive-reduction,jit,prog-fuse,sequence,batch'
```

**1. End-to-end, GPU (GH200)** -- the figure, and the graph statistics and pass
costs of the table. `cgstats.csv` / `jitstats.csv` are written automatically.

```sh
./scripts/evaluate.py --target gpu --apps krylov,lulesh,llm.c,mnmg \
    --opts "$OPTS" --unroll 1,8 \
    --sizes 'krylov=64;lulesh=16,60,100;llm.c=64,128,256;mnmg=7035,23874' \
    --iters 'krylov=200;lulesh=104'
```

Krylov gets a single size on purpose: its panel of the figure puts the five
solvers on the x axis (`AppSpec.panel_x`), which says more about generality than
one solver at three sizes, and the section has room for one panel per app.
This is 156 runs.

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
./scripts/evaluate.py --target ompss --apps llm.c --opts "$OPTS"
```

**5. Hand-written references** -- built and run by hand (different sources and
toolchains, so not part of the sweep), then written into a small CSV that the
figure overlays:

```sh
make -C MNMGDatalog/MNMGDatalog-reference tc_benchmark   # v1_baseline, v2_cudagraph
make -C llm.c train_gpt2cu
cat > results/external.csv <<'EOF'
app,variant,size,label,avg_ms
mnmg,,7035,CUDA graph (hand-written),<measured>
llm.c,,256,llm.c CUDA,<measured>
EOF
```

**Rendering:**

```sh
./scripts/plot.py --paper --external results/external.csv \
                  --latex-table ../../paper/sections/generated-table.tex
```

The baseline of both the figure and the break-even column is `no-taskgraph`
(`--baseline`), so the two artifacts always tell the same story.
