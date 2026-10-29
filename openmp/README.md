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
| PolyBench (jacobi-1d/2d, heat-3d, fdtd-2d) | ✅ | ✅ | ✅ | ✅ | ✅ |
| miniWeather                      | ✅ | ✅ | ✅ | ✅ | ✅ |
| sw4lite (proxy)                  | ✅ | ✅ | ✅ | ✅ | ✅ |
| CloverLeaf                       | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cholesky                         | ✅ | ✅ | ✅ | ❌ | ❌ |

Cholesky is kept as a standalone tasks/target benchmark but is **not** a
record/replay taskgraph example (its tiled DAG changes shape every step), so it
is excluded from the shared harness; build it with `make -C cholesky`.

Notes on the four most-recently-ported apps:

* **PolyBench** — only the four fully-parallel stencils are in the harness
  (`jacobi-1d`, `jacobi-2d`, `heat-3d`, `fdtd-2d`). `seidel-2d` and `adi` have
  loop-carried wavefront / tridiagonal dependences that are not clean
  record/replay examples (like Cholesky), so they are intentionally excluded; the
  pristine upstream sources remain under `polybench-stencils/`. A small vendored
  `polybench/pb_common.h` replaces the (absent) upstream PolyBench utilities.
* **miniWeather** — ported from the serial reference (`miniWeather/c/…`) with MPI
  and parallel-netcdf stripped (single node, periodic halo, no file output). The
  x/z Strang-splitting alternates each step, so two taskgraphs (one per parity)
  are recorded. Correctness is the native mass / total-energy conservation check.
* **sw4lite** — a self-contained *proxy* of the LLNL sw4lite seismic mini-app:
  the real 4th-order elastic-wave stencil (`rhs4sg_rev.C`) + predictor/corrector
  time integrator (`ew-cfromfort.C`), on a uniform Cartesian box without the
  MPI/BLAS/LAPACK/supergrid machinery of the full code. The upstream build is
  kept as `sw4lite/Makefile.upstream`.
* **CloverLeaf** — an *in-place* port: the model-independent `cloverleaf/driver/`
  is reused unchanged and a new `cloverleaf/src/tasking/` model (cloned from
  `src/omp-target`, with `context.h` routing the conditional-offload macro through
  `tasking.h`) provides the kernels; the per-timestep kernel chain is wrapped in a
  record/replay taskgraph in `driver/hydro.cpp`. A flat `cloverleaf/Makefile`
  bypasses the upstream CMake build (kept as `CMakeLists.txt`). Run it with a deck,
  e.g. `./cloverleaf --file InputDecks/clover_bm16_short.in --device 0`.

## Building

One top-level `Makefile` dispatches to each app (which shares `common.mk`):

```sh
make                 # build all harness apps (CPU tasks + taskgraph, defaults)
make krylov          # just the krylov solvers
make lulesh          # (also: make llmc / polybench / miniweather / sw4lite / cloverleaf)
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
./scripts/evaluate.py --apps polybench --variants heat-3d,jacobi-2d
./scripts/evaluate.py --apps miniweather,sw4lite,cloverleaf
./scripts/plot.py                                # -> results/figures/*.png
```

`plot.py` produces `time-<app>.png` (avg time / iteration with stddev error bars,
one bar per configuration, work on the top axis) and `graph-<app>.png` (CGIR
per-pass command-graph reduction and pass wall time).
