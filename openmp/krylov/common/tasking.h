/*
 * tasking.h - OpenMP task / target / taskgraph abstraction for the Krylov
 * solvers, in the spirit of llm.c: one source expresses two backends, selected
 * at compile time by -DUSE_TARGET.
 *
 *   USE_TARGET == 0 (default): every computational loop is tiled into CPU tasks
 *                    -> "#pragma omp task depend(...)".
 *   USE_TARGET == 1: every computational loop becomes one offloaded GPU task
 *                    -> "#pragma omp target teams distribute parallel for
 *                        nowait depend(...) map(...)".
 *
 * USE_TASKGRAPH wraps the (loop-invariant) per-iteration task region with
 * TASKGRAPH_BEGIN/END so it is recorded once and replayed on later iterations.
 */
#ifndef KRYLOV_TASKING_H
#define KRYLOV_TASKING_H

/* ---- Compile-time control variables (override with -D on the compiler) ---- */

#ifndef USE_TARGET          /* 0: host CPU tasks     1: GPU target tasks */
# define USE_TARGET 0
#endif

#ifndef USE_TASKGRAPH       /* 1: record/replay the per-iteration task graph */
# define USE_TASKGRAPH 0
#endif

#ifndef USE_XKOMP           /* 1: use XKOMP's taskgraph API instead of LLVM's */
# define USE_XKOMP 0
#endif

/*
 * Tasks are created inside helper functions (spmv/dot/axpy/...), i.e. not
 * lexically inside the taskgraph region, so they must be marked replayable to
 * be captured. Enable it automatically whenever the taskgraph is enabled.
 */
#ifndef USE_REPLAYABLE
# define USE_REPLAYABLE USE_TASKGRAPH
#endif

#if USE_XKOMP
# include <xkomp/xkomp.h>
# include <xkomp/xkomp++.h>
#endif

/* ---- Pragma stringization helpers (macros expand only through _Pragma) ---- */
#define KR_PRAGMA(...)  _Pragma(#__VA_ARGS__)
#define KR_XPRAGMA(...) KR_PRAGMA(__VA_ARGS__)

/* replayable(1) is emitted on every task-generating construct when recording. */
#if USE_REPLAYABLE
# define REPLAYABLE_CLAUSE replayable(1)
#else
# define REPLAYABLE_CLAUSE
#endif

/* ----------------------------------------------------------------------------
 * Task / kernel emission macros. Exactly one of OMP_TASK / OMP_TARGET_LOOP_TASK
 * is non-empty per backend, so the SAME loop nest compiles to per-block host
 * tasks (CPU) or a single offloaded parallel-for (GPU):
 *
 *     OMP_TARGET_LOOP_TASK(coarse deps + MAP)   // GPU: the parallel-for ; CPU: empty
 *     for (blk = 0; blk < n; blk += bs) {        // bs == 1 on GPU (one row/thread)
 *         OMP_TASK(fine per-block deps)          // CPU: the task ; GPU: empty
 *         { ... work on rows [blk, blk+bs) ... }
 *     }
 *
 * OMP_TARGET_TASK(...) is loop-less device work (a GPU target task; a host task
 * on the CPU) -- used for the tiny scalar updates (alpha, beta, ...).
 * ------------------------------------------------------------------------- */
#if USE_TARGET

# define OMP_TASK(...)                                                          /* nothing: work is done by OMP_TARGET_LOOP_TASK */
# define OMP_TARGET_LOOP_TASK(...) KR_XPRAGMA(omp target teams distribute parallel for REPLAYABLE_CLAUSE nowait __VA_ARGS__)
# define OMP_TARGET_TASK(...)      KR_XPRAGMA(omp target REPLAYABLE_CLAUSE nowait __VA_ARGS__)

#else

# define OMP_TASK(...)             KR_XPRAGMA(omp task REPLAYABLE_CLAUSE __VA_ARGS__)
# define OMP_TARGET_LOOP_TASK(...)                                              /* nothing: tiling is done by OMP_TASK */
# define OMP_TARGET_TASK(...)      KR_XPRAGMA(omp task REPLAYABLE_CLAUSE __VA_ARGS__)

#endif /* USE_TARGET */

/* Always a host task (never offloaded), regardless of USE_TARGET. Used for
 * host-side work such as the optional per-iteration debug print. It is created
 * outside the taskgraph region, so it is not marked replayable. */
#define OMP_HOST_TASK(...) KR_XPRAGMA(omp task __VA_ARGS__)

/* ---- Dependency-clause abstraction (matches llm.c naming) ----
 *   DEPEND(in, a[x:y], b)           -> depend(in: a[x:y], b)
 *   DEPEND_MULTI(in, (i=0:N), a[i]) -> depend(iterator(i=0:N), in: a[i])
 * The iterator argument must be parenthesized to shield its internal commas. */
#define DEPEND(dir, ...)              depend(dir: __VA_ARGS__)
#define DEPEND_MULTI(dir, iters, ...) depend(iterator iters, dir: __VA_ARGS__)

/* Only the GPU backend needs map() clauses; expands to nothing on the host so
 * the same call site serves both (host tasks operate directly on host memory). */
#if USE_TARGET
# define MAP(...) map(__VA_ARGS__)
#else
# define MAP(...)
#endif

# define DEFAULT_NONE default(none)

/* ---- Device data-management directives ----
 * On the host backend (USE_TARGET == 0) they expand to nothing (buffers already
 * live in host memory); on the device backend they emit the matching
 * "#pragma omp target ..." directive. Used for the one-time H2D/D2H staging and
 * the per-iteration residual read-back. */
#if USE_TARGET
# define OMP_TARGET_ENTER_DATA(...) KR_XPRAGMA(omp target enter data __VA_ARGS__)
# define OMP_TARGET_EXIT_DATA(...)  KR_XPRAGMA(omp target exit data __VA_ARGS__)
# define OMP_TARGET_UPDATE(...)     KR_XPRAGMA(omp target update __VA_ARGS__)
#else
# define OMP_TARGET_ENTER_DATA(...)
# define OMP_TARGET_EXIT_DATA(...)
# define OMP_TARGET_UPDATE(...)
#endif

/* ----------------------------------------------------------------------------
 * Taskgraph record/replay wrapper for one iteration body:
 *
 *     TASKGRAPH_BEGIN
 *     {
 *         ... spawn the (loop-invariant) tasks of one iteration ...
 *     }
 *     TASKGRAPH_END
 *
 * With USE_TASKGRAPH the region is recorded on the first encounter and replayed
 * afterwards. Two backends are supported: LLVM's "#pragma omp taskgraph" and
 * XKOMP's function/lambda form. Without USE_TASKGRAPH the macros vanish and the
 * tasks are simply created every iteration.
 * ------------------------------------------------------------------------- */
#if USE_TASKGRAPH
# if USE_XKOMP
#  define TASKGRAPH_BEGIN pragma_omp_taskgraph(0, XKOMP_TASKGRAPH_FLAG_NONE, [&] (void)
#  define TASKGRAPH_END   );
# else
#  define TASKGRAPH_BEGIN KR_XPRAGMA(omp taskgraph graph_id(0))
#  define TASKGRAPH_END
# endif
#else
# define TASKGRAPH_BEGIN
# define TASKGRAPH_END
#endif

#endif /* KRYLOV_TASKING_H */
