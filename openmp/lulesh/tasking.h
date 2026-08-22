/*
 * tasking.h - OpenMP task / target / taskgraph abstraction for LULESH, in the
 * spirit of the krylov / llm.c / cholesky apps: one source expresses two
 * backends, selected at compile time by -DUSE_TARGET.
 *
 *   USE_TARGET == 0 (default): every per-block computational loop becomes one
 *                    CPU task -> "#pragma omp task depend(...)".
 *   USE_TARGET == 1: every per-block loop becomes one offloaded GPU task
 *                    -> "#pragma omp target teams distribute parallel for
 *                        num_teams(..) thread_limit(..) nowait depend(...)".
 *
 * LULESH is already written as per-block loops (for b in 0..n step EBS/NBS) with
 * fine-grained per-block depend(...) clauses, so each loop maps directly onto
 * OMP_TARGET_LOOP_TASK: it is the offloaded parallel-for on the GPU, or one host
 * task running the block serially on the CPU.
 *
 * USE_TASKGRAPH wraps the per-iteration task region with TASKGRAPH_BEGIN/END so
 * it is recorded once and replayed on later iterations. USE_SYNC switches to a
 * synchronous schedule (blocking target on GPU; serial on the host).
 *
 * Macro model (shared with the cholesky app):
 *   OMP_TARGET_LOOP_TASK(clauses)  a per-block parallel loop as ONE task
 *                                  GPU -> target teams distribute parallel for
 *                                  CPU -> omp task (the whole block runs in it)
 *   OMP_TARGET_TASK(clauses)       loop-less device work / host task
 *                                  GPU -> omp target ; CPU -> omp task
 *   OMP_HOST_TASK(clauses)         always a host task (host-resident reductions,
 *                                  prints; the D2H'd data they read is on host)
 *   DEPEND / DEPEND_MULTI          dependency clauses (vanish under USE_SYNC)
 *   MAP                            map() clauses (GPU only)
 *   GPU_CLAUSES                    extra loop clauses, e.g. num_teams() /
 *                                  thread_limit() / collapse() (GPU only)
 *   OMP_TARGET_ENTER_DATA / _EXIT_DATA / _UPDATE   device data management
 *   TASKGRAPH_BEGIN / _END         record/replay wrapper
 *
 * Variables referenced in a CPU task are firstprivate by default (the block's
 * start/end and the domain pointers), so no explicit firstprivate is needed --
 * the same implicit capture the target regions already relied on.
 */
#ifndef LULESH_TASKING_H
#define LULESH_TASKING_H

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

/* Tasks are created inside helper regions, not lexically inside the taskgraph
 * region, so they must be marked replayable to be captured. Enable it
 * automatically whenever the taskgraph is enabled. */
#ifndef USE_REPLAYABLE
# define USE_REPLAYABLE USE_TASKGRAPH
#endif

#if USE_XKOMP
# include <xkomp/xkomp.h>
# include <xkomp/xkomp++.h>
#endif

/* ---- Pragma stringization helpers (macros expand only through _Pragma) ---- */
#define LU_PRAGMA(...)  _Pragma(#__VA_ARGS__)
#define LU_XPRAGMA(...) LU_PRAGMA(__VA_ARGS__)

/* replayable(1) is emitted on every task-generating construct when recording.
 * Synchronous mode has no tasks/taskgraph, so it is never replayable. */
#if USE_REPLAYABLE && !USE_SYNC
# define REPLAYABLE_CLAUSE replayable(1)
#else
# define REPLAYABLE_CLAUSE
#endif

/* ----------------------------------------------------------------------------
 * Task / kernel emission macros. The SAME per-block loop / block compiles to a
 * host task (CPU) or an offloaded target task (GPU):
 *
 *     OMP_TARGET_LOOP_TASK(GPU_CLAUSES(num_teams(elem_teams) thread_limit(THREADS))
 *                          DEPEND(in, q[start], p[start]) DEPEND(out, sigxx[start]))
 *     for (i = start; i < end; i++) { ... }        // one task / one kernel
 *
 *     OMP_TARGET_TASK(DEPEND(inout, deltatime[0]))
 *     { ... }                                       // loop-less device/host work
 * ------------------------------------------------------------------------- */
#if USE_SYNC

# if USE_TARGET
#  define OMP_TARGET_LOOP_TASK(...) LU_XPRAGMA(omp target teams distribute parallel for __VA_ARGS__)
#  define OMP_TARGET_TASK(...)      LU_XPRAGMA(omp target __VA_ARGS__)
# else
#  define OMP_TARGET_LOOP_TASK(...)                                              /* nothing: serial loop */
#  define OMP_TARGET_TASK(...)                                                   /* nothing: serial block */
# endif

#elif USE_TARGET

# define OMP_TARGET_LOOP_TASK(...) LU_XPRAGMA(omp target teams distribute parallel for REPLAYABLE_CLAUSE nowait __VA_ARGS__)
# define OMP_TARGET_TASK(...)      LU_XPRAGMA(omp target REPLAYABLE_CLAUSE nowait __VA_ARGS__)

#else

# define OMP_TARGET_LOOP_TASK(...) LU_XPRAGMA(omp task REPLAYABLE_CLAUSE __VA_ARGS__)
# define OMP_TARGET_TASK(...)      LU_XPRAGMA(omp task REPLAYABLE_CLAUSE __VA_ARGS__)

#endif /* USE_SYNC / USE_TARGET */

/* Host-side work (prints / host-resident time-constraint reductions). Real host
 * task in the asynchronous modes; runs inline in synchronous mode. */
#if USE_SYNC
# define OMP_HOST_TASK(...)                                                      /* nothing: runs inline */
#else
# define OMP_HOST_TASK(...) LU_XPRAGMA(omp task REPLAYABLE_CLAUSE __VA_ARGS__)
#endif

/* `nowait` on a construct in the asynchronous modes; nothing in synchronous. */
#if USE_SYNC
# define NOWAIT
#else
# define NOWAIT nowait
#endif

/* ---- Dependency-clause abstraction (matches krylov / llm.c naming) ----
 *   DEPEND(in, a, b)                -> depend(in: a, b)
 *   DEPEND_MULTI(in, (i=0:N), a[i]) -> depend(iterator(i=0:N), in: a[i])
 * The iterator argument must be parenthesized to shield its internal commas.
 * In synchronous mode there are no tasks, so dependences expand to nothing. */
#if USE_SYNC
# define DEPEND(dir, ...)
# define DEPEND_MULTI(dir, iters, ...)
#else
# define DEPEND(dir, ...)              depend(dir: __VA_ARGS__)
# define DEPEND_MULTI(dir, iters, ...) depend(iterator iters, dir: __VA_ARGS__)
#endif

/* Only the GPU backend needs map() clauses; expands to nothing on the host. */
#if USE_TARGET
# define MAP(...) map(__VA_ARGS__)
#else
# define MAP(...)
#endif

/* GPU-only loop clauses (num_teams / thread_limit / collapse): emitted on the
 * target backend, dropped on the host backend. */
#if USE_TARGET
# define GPU_CLAUSES(...) __VA_ARGS__
#else
# define GPU_CLAUSES(...)
#endif

#define DEFAULT_NONE default(none)

/* ---- Device data-management directives ----
 * On the host backend they expand to nothing (buffers already live in host
 * memory); on the device backend they emit the matching "#pragma omp target ..."
 * directive. Used for the one-time H2D staging and per-iteration D2H read-back. */
#if USE_TARGET
# define OMP_TARGET_ENTER_DATA(...) LU_XPRAGMA(omp target enter data __VA_ARGS__)
# define OMP_TARGET_EXIT_DATA(...)  LU_XPRAGMA(omp target exit data __VA_ARGS__)
# define OMP_TARGET_UPDATE(...)     LU_XPRAGMA(omp target update __VA_ARGS__)
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
 * afterwards. Two backends: LLVM's "#pragma omp taskgraph" and XKOMP's lambda
 * form. Without USE_TASKGRAPH the macros vanish (tasks created every iteration).
 * ------------------------------------------------------------------------- */
#if USE_TASKGRAPH && !USE_SYNC
# if USE_XKOMP
#  define TASKGRAPH_BEGIN pragma_omp_taskgraph(0, XKOMP_TASKGRAPH_FLAG_NONE, [&] (void)
#  define TASKGRAPH_END   );
# else
#  define TASKGRAPH_BEGIN LU_XPRAGMA(omp taskgraph graph_id(0))
#  define TASKGRAPH_END
# endif
#else
# define TASKGRAPH_BEGIN
# define TASKGRAPH_END
#endif

#endif /* LULESH_TASKING_H */
