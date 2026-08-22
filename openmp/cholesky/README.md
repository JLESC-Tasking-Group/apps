Tiled dense Cholesky factorization (A = L Lᵀ) as an OpenMP task graph.

One source, two backends selected at compile time (see `tasking.h`):

* **CPU tasks** (`USE_TARGET=0`, default): each tile kernel is a `#pragma omp task`.
* **GPU target tasks** (`USE_TARGET=1`): each tile kernel is offloaded with
  `#pragma omp target ... nowait depend(...)`, and the matrix is staged to the
  device via `target enter data` + per-tile `target update` pipelines.

`USE_TASKGRAPH` records the per-repetition task region once and replays it on
later repetitions (`reps` argument). The four tile kernels (potrf/trsm/syrk/gemm)
are hand-written in `kernels.h` (no BLAS/LAPACK) so the same code offloads to the
device unchanged.

Files: `main.c` (driver + DAG), `kernels.h` (tile kernels), `tasking.h` (task /
target / taskgraph macros), `kalloc.h`/`kalloc.c` (pinned host allocator).

Build and run:

    make
    ./cholesky [nt] [ts] [reps] [check]

Please, see the Makefile for build configuration (CPU vs GPU, defaults, flags).

Evaluation (like the krylov app): `scripts/evaluate.py` sweeps the taskgraph
optimizations (`OMP_TASKGRAPH_OPT`) and problem sizes into a CSV, and
`scripts/plot.py` renders a grouped bar chart (avg time / repetition, with
record vs replay stddev error bars):

    make
    ./scripts/evaluate.py --ts 256 --reps 10          # -> evaluate_cholesky_<ts>.csv
    ./scripts/plot.py evaluate_cholesky_*.csv          # -> .png

Unified from the earlier `task/` (CPU + LAPACK) and `target-nowait-depend/` (GPU)
variants; the MPI/MPC multi-node path has been removed.
