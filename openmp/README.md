# apps/openmp

OpenMP task / target benchmarks used to evaluate the `taskgraph` construct and
its CGIR command-graph optimizations. Each app is a single source expressing two
backends (CPU tasks vs GPU target offload) plus a synchronous baseline, selected
by compile-time toggles shared through `common.mk`.

| App                              | CPU tasks | GPU target | synchronous | taskgraph | in harness |
| -------------------------------- | :-------: | :--------: | :---------: | :-------: | :--------: |
| Krylov (cg/cr/bicgstab/minres/gmres) | ✅ | ✅ | ✅ | ✅ | ✅ |
| LULESH                           | ✅ | ✅ | ✅ | ✅ | ✅ |
| llm.c                            | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cholesky                         | ✅ | ✅ | ✅ | ❌ | ❌ |

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
make clean
```

The backend/schedule toggles (`USE_TARGET`, `USE_TASKGRAPH`, `USE_SYNC`,
`USE_REPLAYABLE`) live in `common.mk` and propagate to the per-app Makefiles,
which remain usable directly (e.g. `make -C lulesh run`, `make -C llm.c test`).

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

`plot.py` produces `time-<app>.png` (avg time / iteration with stddev error bars,
one bar per configuration, work on the top axis) and `graph-<app>.png` (CGIR
per-pass command-graph reduction and pass wall time).
