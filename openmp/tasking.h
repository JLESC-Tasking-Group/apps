/*
 * tasking.h - shared OpenMP task / target / taskgraph abstraction for the
 * apps/openmp benchmarks (krylov, lulesh, llm.c). One source expresses several
 * backends, selected at compile time by the toggles below:
 *
 *   USE_TARGET == 0 (default): every computational loop is tiled into CPU tasks
 *                    -> "#pragma omp task depend(...)".
 *   USE_TARGET == 1: every computational loop becomes one offloaded GPU task
 *                    -> "#pragma omp target teams distribute parallel for
 *                        nowait depend(...) map(...)".
 *   USE_OMPSS  == 1: host tasks are emitted as OmpSs-2 "#pragma oss task ..."
 *                    (mutually exclusive with USE_TARGET).
 *
 * USE_TASKGRAPH wraps the (loop-invariant) per-iteration task region with
 * TASKGRAPH_BEGIN/END so it is recorded once and replayed on later iterations.
 *
 * USE_SYNC switches from the asynchronous task schedule to a *synchronous* one:
 * each kernel runs to completion before the next (classic blocking "omp target"
 * with no nowait / depend / tasks / taskgraph). This is what most OpenMP offload
 * codes do; it is provided to compare against the default async task/taskgraph
 * schedule. On the host backend it degenerates to a serial single-thread run.
 *
 * Macro model (used across the apps):
 *   OMP_TASK(...)                  one CPU host task (empty on the GPU backend,
 *                                  where the offloaded loop does the work)
 *   OMP_TARGET_LOOP_TASK(...)      one offloaded parallel-for (GPU); empty on the
 *                                  host, where tiling is done by OMP_TASK
 *   OMP_TILE(deps, mp, fp)         a tiled loop as ONE task: the offloaded
 *                                  parallel-for (GPU) or one host task (CPU)
 *   OMP_TARGET_TASK(...)           loop-less device work (GPU target task; a host
 *                                  task on the CPU / OmpSs backends)
 *   OMP_HOST_TASK(...)             always a host task (replayable, so in-taskgraph
 *                                  host tasks are captured)
 *   DEPEND / DEPEND_MULTI          dependency clauses (vanish under USE_SYNC)
 *   MAP                            map() clauses (GPU only)
 *   ATOMIC                         "#pragma omp/oss atomic"
 *   OMP_TARGET_ENTER_DATA / _EXIT_DATA / _UPDATE   device data management
 *   TASKGRAPH_BEGIN / _END         record/replay wrapper (OpenMP / XKOMP)
 */
#ifndef OPENMP_TASKING_H
#define OPENMP_TASKING_H

/* ---- Compile-time control variables (override with -D on the compiler) ---- */

#ifndef USE_TARGET          /* 0: host CPU tasks     1: GPU target tasks */
# define USE_TARGET 0
#endif

#ifndef USE_TASKGRAPH       /* 1: record/replay the per-iteration task graph */
# define USE_TASKGRAPH 0
#endif

#ifndef USE_SYNC            /* 0: asynchronous tasks   1: synchronous blocking */
# define USE_SYNC 0
#endif

#ifndef USE_XKOMP           /* 1: use XKOMP's taskgraph API instead of LLVM's */
# define USE_XKOMP 0
#endif

#ifndef USE_OMPSS           /* 1: emit OmpSs-2 (#pragma oss ...) host tasks */
# define USE_OMPSS 0
#endif

/*
 * Tasks are created inside helper functions (spmv/dot/axpy/...), i.e. not
 * lexically inside the taskgraph region, so they must be marked replayable to
 * be captured. Enable it automatically whenever the taskgraph is enabled.
 */
#ifndef USE_REPLAYABLE
# define USE_REPLAYABLE USE_TASKGRAPH
#endif

#if USE_OMPSS && USE_TARGET
# error "USE_OMPSS=1 is incompatible with USE_TARGET=1: OmpSs-2 does not support OpenMP target (GPU) tasks. Set USE_TARGET=0 for the OmpSs-2 host backend."
#endif

#if USE_XKOMP
# include <xkomp/xkomp.h>
# include <xkomp/xkomp++.h>
#endif

/* ---- Pragma stringization helpers (macros expand only through _Pragma) ---- */
#define TG_PRAGMA(...)  _Pragma(#__VA_ARGS__)
#define TG_XPRAGMA(...) TG_PRAGMA(__VA_ARGS__)

/* replayable(1) is emitted on every task-generating construct when recording.
 * Synchronous mode has no tasks/taskgraph, and OmpSs-2 has no replayable clause,
 * so it is never emitted for those backends. */
#if USE_REPLAYABLE && !USE_SYNC && !USE_OMPSS
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
 * on the CPU / OmpSs backends) -- used for the tiny scalar updates (alpha, ...).
 * ------------------------------------------------------------------------- */
#if USE_SYNC

/* Synchronous: each kernel blocks until complete; no tasks, no nowait. On GPU
 * the loops are still offloaded (blocking `omp target`); on the host they are
 * plain serial loops (the macros vanish). */
# if USE_TARGET
#  define OMP_TASK(...)                                                          /* nothing: the loop is offloaded whole */
#  define OMP_TARGET_LOOP_TASK(...) TG_XPRAGMA(omp target teams distribute parallel for __VA_ARGS__)
#  define OMP_TARGET_TASK(...)      TG_XPRAGMA(omp target __VA_ARGS__)
# else
#  define OMP_TASK(...)                                                          /* nothing: serial loop */
#  define OMP_TARGET_LOOP_TASK(...)                                              /* nothing: serial loop */
#  define OMP_TARGET_TASK(...)                                                   /* nothing: serial block */
# endif

#elif USE_TARGET

# define OMP_TASK(...)                                                           /* nothing: work is done by OMP_TARGET_LOOP_TASK */
# define OMP_TARGET_LOOP_TASK(...) TG_XPRAGMA(omp target teams distribute parallel for REPLAYABLE_CLAUSE nowait __VA_ARGS__)
# define OMP_TARGET_TASK(...)      TG_XPRAGMA(omp target REPLAYABLE_CLAUSE nowait __VA_ARGS__)

#elif USE_OMPSS

# define OMP_TASK(...)             TG_XPRAGMA(oss task REPLAYABLE_CLAUSE __VA_ARGS__)
# define OMP_TARGET_LOOP_TASK(...)                                               /* nothing: tiling is done by OMP_TASK */
# define OMP_TARGET_TASK(...)      TG_XPRAGMA(oss task REPLAYABLE_CLAUSE __VA_ARGS__)

#else

# define OMP_TASK(...)             TG_XPRAGMA(omp task REPLAYABLE_CLAUSE __VA_ARGS__)
# define OMP_TARGET_LOOP_TASK(...)                                               /* nothing: tiling is done by OMP_TASK */
# define OMP_TARGET_TASK(...)      TG_XPRAGMA(omp task REPLAYABLE_CLAUSE __VA_ARGS__)

#endif /* USE_SYNC / USE_TARGET / USE_OMPSS */

/* ----------------------------------------------------------------------------
 * One tile of a decomposed loop as ONE task. The SAME tiled loop serves every
 * backend: each tile becomes one host task (CPU / OmpSs) or one offloaded
 * parallel-for over the tile's sub-range (GPU).
 *
 *     for (blk = 0; blk < n; blk += BS) {
 *         const idx_t begin = blk, end = MIN(blk + BS, n);
 *         OMP_TILE(DEPEND(in, x[begin]) DEPEND(out, y[begin]),
 *                  MAP(present: x[0:n], y[0:n]),
 *                  firstprivate(x, y, begin, end))
 *         for (idx_t i = begin; i < end; i++) y[i] = x[i];
 *     }
 *
 * Three arguments (each keeps its commas parenthesis-shielded):
 *   deps : DEPEND(...) / DEPEND_MULTI(...) clauses (common to all backends)
 *   mp   : GPU-only clauses -- MAP(present: ...) and, where needed,
 *          num_teams()/thread_limit()/collapse() (empty on the host backends)
 *   fp   : firstprivate(...) clause (host backends only; may be left empty to
 *          rely on the implicit-firstprivate default). Target scalars/pointers
 *          are implicitly firstprivate/mapped, and firstprivate on a target
 *          construct trips a clang codegen assertion, so fp is omitted on GPU.
 * ------------------------------------------------------------------------- */
#if USE_SYNC
# if USE_TARGET
#  define OMP_TILE(deps, mp, fp) TG_XPRAGMA(omp target teams distribute parallel for deps mp)
# else
#  define OMP_TILE(deps, mp, fp)                                                 /* nothing: serial loop */
# endif
#elif USE_TARGET
# define OMP_TILE(deps, mp, fp) TG_XPRAGMA(omp target teams distribute parallel for REPLAYABLE_CLAUSE nowait deps mp)
#elif USE_OMPSS
# define OMP_TILE(deps, mp, fp) TG_XPRAGMA(oss task REPLAYABLE_CLAUSE fp deps)
#else
# define OMP_TILE(deps, mp, fp) TG_XPRAGMA(omp task REPLAYABLE_CLAUSE fp deps)
#endif

/* Host-side work such as the optional per-iteration debug print / timing, and
 * host-resident reductions. In the asynchronous modes it is a real host task
 * (replayable so in-taskgraph host tasks are captured; depend-synchronized so it
 * fires after the iteration's tasks); in synchronous mode it vanishes and the
 * block runs inline -- correct because the preceding kernels already completed. */
#if USE_SYNC
# define OMP_HOST_TASK(...)                                                      /* nothing: runs inline */
#elif USE_OMPSS
# define OMP_HOST_TASK(...) TG_XPRAGMA(oss task REPLAYABLE_CLAUSE __VA_ARGS__)
#else
# define OMP_HOST_TASK(...) TG_XPRAGMA(omp task REPLAYABLE_CLAUSE __VA_ARGS__)
#endif

/* `nowait` on a construct in the asynchronous modes; nothing in synchronous mode
 * (used on the residual target-update so it becomes a blocking D2H under -p). */
#if USE_SYNC
# define NOWAIT
#else
# define NOWAIT nowait
#endif

/* ---- Dependency-clause abstraction ----
 *   OpenMP (USE_OMPSS == 0):
 *     DEPEND(in, a[x:y], b)           -> depend(in: a[x:y], b)
 *     DEPEND_MULTI(in, (i=0:N), a[i]) -> depend(iterator(i=0:N), in: a[i])
 *   OmpSs-2 (USE_OMPSS == 1):
 *     DEPEND(in, a[x:y], b)           -> in(a[x:y], b)
 *     DEPEND_MULTI(in, (i=0:N), a[i]) -> in({ a[i], i=0:N })
 * The iterator argument must be parenthesized to shield its internal commas.
 * In synchronous mode there are no tasks, so dependences expand to nothing --
 * program order is the schedule. */
#define UNWRAP(...) __VA_ARGS__

#if USE_SYNC
# define DEPEND(dir, ...)
# define DEPEND_MULTI(dir, iters, ...)
# define ATOMIC TG_PRAGMA(omp atomic)
#elif USE_OMPSS
/* Dependency-direction keyword mapping, OpenMP -> OmpSs-2 (only `inoutset`
 * differs -- OmpSs calls it `concurrent`; the rest map to themselves). */
# define OSS_DIR(dir)      OSS_DIR__##dir
# define OSS_DIR__in       in
# define OSS_DIR__out      out
# define OSS_DIR__inout    inout
# define OSS_DIR__inoutset concurrent
# define DEPEND(dir, ...)              OSS_DIR(dir)(__VA_ARGS__)
# define DEPEND_MULTI(dir, iters, ...) OSS_DIR(dir)({ __VA_ARGS__, UNWRAP iters })
# define ATOMIC TG_PRAGMA(oss atomic)
#else
# define DEPEND(dir, ...)              depend(dir: __VA_ARGS__)
# define DEPEND_MULTI(dir, iters, ...) depend(iterator iters, dir: __VA_ARGS__)
# define ATOMIC TG_PRAGMA(omp atomic)
#endif

/* Only the GPU backend needs map() clauses; expands to nothing on the host so
 * the same call site serves both (host tasks operate directly on host memory). */
#if USE_TARGET
# define MAP(...) map(__VA_ARGS__)
#else
# define MAP(...)
#endif

/* Host-only shared() clause -- the counterpart of MAP for the host backends.
 * With DEFAULT_NONE, every variable used in a task needs an explicit data-sharing
 * attribute. On the GPU backend that comes from the map() clause, so SHARED
 * expands to nothing; on the host/OmpSs backends it emits shared(...) for
 * variables whose lifetime is protected by the task dependences (e.g. the
 * solver-scope scalar buffers, live for the whole solve and ordered by depend).
 * Per-task values that must be captured by value (loop indices, tile bounds) use
 * firstprivate instead. Assumes SHARED is only used on host-executed tasks. */
#if USE_TARGET
# define SHARED(...)
#else
# define SHARED(...) shared(__VA_ARGS__)
#endif

# define DEFAULT_NONE default(none)

/* ---- Device data-management directives ----
 * On the host backends they expand to nothing (buffers already live in host
 * memory); on the device backend they emit the matching "#pragma omp target ..."
 * directive. Used for the one-time H2D/D2H staging and per-iteration read-back. */
#if USE_TARGET
# define OMP_TARGET_ENTER_DATA(...) TG_XPRAGMA(omp target enter data __VA_ARGS__)
# define OMP_TARGET_EXIT_DATA(...)  TG_XPRAGMA(omp target exit data __VA_ARGS__)
# define OMP_TARGET_UPDATE(...)     TG_XPRAGMA(omp target update __VA_ARGS__)
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
 * XKOMP's function/lambda form. Without USE_TASKGRAPH (or under USE_SYNC /
 * USE_OMPSS, which use their own schedules) the macros vanish and the tasks are
 * simply created every iteration.
 * ------------------------------------------------------------------------- */
#if USE_TASKGRAPH && !USE_SYNC && !USE_OMPSS
# if USE_XKOMP
#  define TASKGRAPH_BEGIN pragma_omp_taskgraph(0, XKOMP_TASKGRAPH_FLAG_NONE, [&] (void)
#  define TASKGRAPH_END   );
# else
#  define TASKGRAPH_BEGIN TG_XPRAGMA(omp taskgraph graph_id(0))
#  define TASKGRAPH_END
# endif
#else
# define TASKGRAPH_BEGIN
# define TASKGRAPH_END
#endif

#endif /* OPENMP_TASKING_H */
