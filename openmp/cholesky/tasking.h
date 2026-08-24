/*
 * tasking.h - OpenMP task / target abstraction for the tiled Cholesky
 * factorization: one source expresses two backends, selected at compile time by
 * -DUSE_TARGET.
 *
 *   USE_TARGET == 0 (default): every tile kernel (potrf/trsm/syrk/gemm) becomes
 *                    one CPU task -> "#pragma omp task depend(...)".
 *   USE_TARGET == 1: every tile kernel becomes one offloaded GPU task
 *                    -> "#pragma omp target [teams distribute parallel for]
 *                        nowait depend(...) map(...)".
 *
 * USE_SYNC switches from the asynchronous task schedule to a *synchronous* one:
 * each kernel runs to completion before the next (classic blocking "omp target"
 * with no nowait / depend / tasks). On the host backend it degenerates to a
 * serial single-thread run.
 *
 * NOTE: unlike the krylov / lulesh / llm.c apps, Cholesky is a one-shot tiled
 * factorization whose task DAG changes shape every step, so it is NOT a
 * record/replay taskgraph example -- there is intentionally no taskgraph here.
 *
 * Dependencies are tracked with single-element "token" items (one char per tile,
 * see main.c), which works identically on both backends: the tiles themselves
 * live on the device in the GPU build, so the DAG is expressed on host-side
 * tokens rather than on the device tile addresses.
 *
 * Macro model (shared with the lulesh app):
 *   OMP_TARGET_LOOP_TASK(clauses)  a parallel loop as ONE task
 *                                  GPU -> target teams distribute parallel for
 *                                  CPU -> omp task (the whole loop runs in it)
 *   OMP_TARGET_TASK(clauses)       loop-less device work / host task
 *                                  GPU -> omp target ; CPU -> omp task
 *   OMP_HOST_TASK(clauses)         always a host task
 *   DEPEND / DEPEND_MULTI          dependency clauses (vanish under USE_SYNC)
 *   MAP                            map() clauses (GPU only)
 *   GPU_CLAUSES                    extra loop clauses, e.g. collapse() (GPU only)
 */
#ifndef CHOLESKY_TASKING_H
#define CHOLESKY_TASKING_H

/* ---- Compile-time control variables (override with -D on the compiler) ---- */

#ifndef USE_TARGET          /* 0: host CPU tasks     1: GPU target tasks */
# define USE_TARGET 0
#endif

#ifndef USE_SYNC            /* 0: asynchronous tasks   1: synchronous blocking */
# define USE_SYNC 0
#endif

/* ---- Pragma stringization helpers (macros expand only through _Pragma) ---- */
#define CH_PRAGMA(...)  _Pragma(#__VA_ARGS__)
#define CH_XPRAGMA(...) CH_PRAGMA(__VA_ARGS__)

/* ----------------------------------------------------------------------------
 * Task / kernel emission macros. The SAME loop/block compiles to a host task
 * (CPU) or an offloaded target task (GPU):
 *
 *     OMP_TARGET_LOOP_TASK(DEPEND(...) MAP(...) GPU_CLAUSES(...))
 *     for (m = 0; m < ts; m++) for (n ...) { ... }   // one task / one kernel
 *
 *     OMP_TARGET_TASK(DEPEND(...) MAP(...))
 *     { ... }                                         // loop-less (e.g. potrf)
 *
 * Variables referenced in a CPU task are firstprivate by default (pointers and
 * scalars), so no explicit firstprivate is needed.
 * ------------------------------------------------------------------------- */
#if USE_SYNC

# if USE_TARGET
#  define OMP_TARGET_LOOP_TASK(...) CH_XPRAGMA(omp target teams distribute parallel for __VA_ARGS__)
#  define OMP_TARGET_TASK(...)      CH_XPRAGMA(omp target __VA_ARGS__)
# else
#  define OMP_TARGET_LOOP_TASK(...)                                              /* nothing: serial loop */
#  define OMP_TARGET_TASK(...)                                                   /* nothing: serial block */
# endif

#elif USE_TARGET

# define OMP_TARGET_LOOP_TASK(...) CH_XPRAGMA(omp target teams distribute parallel for nowait __VA_ARGS__)
# define OMP_TARGET_TASK(...)      CH_XPRAGMA(omp target nowait __VA_ARGS__)

#else

# define OMP_TARGET_LOOP_TASK(...) CH_XPRAGMA(omp task __VA_ARGS__)
# define OMP_TARGET_TASK(...)      CH_XPRAGMA(omp task __VA_ARGS__)

#endif /* USE_SYNC / USE_TARGET */

/* Host-side work (printing / timing / host-resident reductions). Real host task
 * in the asynchronous modes; runs inline in synchronous mode. */
#if USE_SYNC
# define OMP_HOST_TASK(...)                                                      /* nothing: runs inline */
#else
# define OMP_HOST_TASK(...) CH_XPRAGMA(omp task __VA_ARGS__)
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
 * In synchronous mode there are no tasks, so dependences expand to nothing --
 * program order is the schedule. */
#if USE_SYNC
# define DEPEND(dir, ...)
# define DEPEND_MULTI(dir, iters, ...)
#else
# define DEPEND(dir, ...)              depend(dir: __VA_ARGS__)
# define DEPEND_MULTI(dir, iters, ...) depend(iterator iters, dir: __VA_ARGS__)
#endif

/* Only the GPU backend needs map() clauses; expands to nothing on the host so
 * the same call site serves both (host tasks operate directly on host memory). */
#if USE_TARGET
# define MAP(...) map(__VA_ARGS__)
#else
# define MAP(...)
#endif

/* GPU-only loop clauses (e.g. collapse / num_teams / thread_limit): emitted on
 * the target backend, dropped on the host backend. */
#if USE_TARGET
# define GPU_CLAUSES(...) __VA_ARGS__
#else
# define GPU_CLAUSES(...)
#endif

#define DEFAULT_NONE default(none)

/* ---- Device data-management directives ----
 * On the host backend they expand to nothing (buffers already live in host
 * memory); on the device backend they emit the matching "#pragma omp target ..."
 * directive. Used for the one-time H2D/D2H staging of the tiles. */
#if USE_TARGET
# define OMP_TARGET_ENTER_DATA(...) CH_XPRAGMA(omp target enter data __VA_ARGS__)
# define OMP_TARGET_EXIT_DATA(...)  CH_XPRAGMA(omp target exit data __VA_ARGS__)
# define OMP_TARGET_UPDATE(...)     CH_XPRAGMA(omp target update __VA_ARGS__)
#else
# define OMP_TARGET_ENTER_DATA(...)
# define OMP_TARGET_EXIT_DATA(...)
# define OMP_TARGET_UPDATE(...)
#endif

#endif /* CHOLESKY_TASKING_H */
