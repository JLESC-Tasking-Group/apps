Tiled dense Cholesky factorization (A = L Lᵀ) as an OpenMP task DAG.

One source, two backends selected at compile time (see `tasking.h`):

* **CPU tasks** (`USE_TARGET=0`, default): each tile kernel is a `#pragma omp task`.
* **GPU target tasks** (`USE_TARGET=1`): each tile kernel is offloaded with
  `#pragma omp target ... nowait depend(...)`, and the matrix is staged to the
  device via `target enter data` + per-tile `target update` pipelines.

The four tile kernels (potrf/trsm/syrk/gemm) are hand-written in `kernels.h` (no
BLAS/LAPACK) so the same code offloads to the device unchanged.

NOTE: Cholesky is intentionally **not** a record/replay taskgraph example and is
**not** part of the shared `apps/openmp` evaluation harness. Its tiled DAG changes
shape at every step of the factorization, so there is no fixed per-step graph to
record and replay. It is kept as a standalone tasks/target benchmark; the `reps`
argument simply re-runs the factorization to average the timing.

Files: `main.c` (driver + DAG), `kernels.h` (tile kernels), `tasking.h` (task /
target macros), `kalloc.h`/`kalloc.c` (pinned host allocator).

Build and run:

    make
    ./cholesky [nt] [ts] [reps] [check]

Please, see the Makefile for build configuration (CPU vs GPU, defaults, flags).

Unified from the earlier `task/` (CPU + LAPACK) and `target-nowait-depend/` (GPU)
variants; the MPI/MPC multi-node path has been removed.
