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
 * TASKGRAPH_LOOP so it is recorded once and replayed on later iterations.
 * USE_TASKGRAPHLOOP then chooses how many iterations one recorded instance
 * covers: the loop's `unroll` (default) or exactly one, the A/B baseline.
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
 *   TASKWAIT                       "#pragma omp/oss taskwait", as _Pragma so it
 *                                  is usable inside a macro argument
 *   OMP_TARGET_ENTER_DATA / _EXIT_DATA / _UPDATE   device data management
 *   TASKGRAPH_LOOP / _LOOP_END     record/replay an iterative loop, `unroll`
 *                                  iterations per recorded instance
 *   TASKGRAPH_BEGIN / _END         record/replay ONE region (no loop)
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

#ifndef USE_TASKGRAPHLOOP   /* 1: TASKGRAPH_LOOP unrolls N iterations into one
                             * graph instance; 0: one instance per iteration */
# define USE_TASKGRAPHLOOP 1
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

#include <stddef.h>     /* size_t, used by the TASKGRAPH_LOOP dispatch below */

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

/* `taskwait` written with _Pragma rather than as a #pragma directive, so it may
 * appear inside a MACRO ARGUMENT -- which the epilogue of TASKGRAPH_LOOP is. A
 * `#pragma` there is not portable (C++ [cpp.pragma]: the behaviour of a directive
 * within macro arguments is undefined) and clang diagnoses it; _Pragma is an
 * operator and is well defined. In synchronous mode there are no tasks to wait
 * for -- each kernel has already completed -- so it vanishes. */
#if USE_SYNC
# define TASKWAIT
#elif USE_OMPSS
# define TASKWAIT TG_PRAGMA(oss taskwait)
#else
# define TASKWAIT TG_PRAGMA(omp taskwait)
#endif

/* ----------------------------------------------------------------------------
 * How a TASKGRAPH_LOOP epilogue may observe the instance it follows.
 *
 * TASKGRAPH_INSTANCE_DRAINED is nonzero when the instance is provably COMPLETE
 * by the time the epilogue runs: xkomp's record pass ends with an implicit
 * taskwait (xkomp_taskgraph_end -> task_dependency_graph_record_stop) and its
 * replay is submitted with COMMAND_FLAG_SYNCHRONOUS, so xkomp_taskgraph_begin /
 * _end returning means the instance has finished; USE_SYNC blocks per kernel.
 *
 * The epilogue can then take its timestamp INLINE -- exactly. Otherwise the
 * instance is still in flight and the timestamp must ride a depend-synchronized
 * host task, which must NOT block or the no-taskgraph configuration loses the
 * cross-iteration pipelining it exists to demonstrate.
 *
 * This is not cosmetic. A host task spawned after a drained instance is ready
 * immediately, but the encountering thread goes straight into the next
 * instance's synchronous replay, so the task can execute *during* it -- pushing
 * its timestamp late. The steady-state mean telescopes and is immune, but the
 * record and first-replay figures are single differences: they would absorb the
 * slip, and those are exactly the two the timing report calls out (instance 0 =
 * record, instance 1 = command-graph build + optimize + first replay).
 *
 *   EPILOGUE_TASK(clauses)  the host task, or nothing when the block may run
 *                           inline (the block itself is written once either way)
 *   EPILOGUE_NOWAIT         `nowait` on an epilogue read-back, or nothing when
 *                           drained -- an inline reader must not race an
 *                           in-flight async D2H (e.g. the -p residual).
 * ------------------------------------------------------------------------- */
#define TASKGRAPH_INSTANCE_DRAINED (USE_SYNC || (USE_TASKGRAPH && !USE_OMPSS))

#if TASKGRAPH_INSTANCE_DRAINED
# define EPILOGUE_TASK(...)                 /* nothing: the block runs inline */
# define EPILOGUE_NOWAIT                    /* blocking: already drained */
#else
# define EPILOGUE_TASK(...) OMP_HOST_TASK(__VA_ARGS__)
# define EPILOGUE_NOWAIT    NOWAIT
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
 * afterwards. Without USE_TASKGRAPH (or under USE_SYNC / USE_OMPSS, which use
 * their own schedules) the macros vanish and the tasks are simply created every
 * iteration.
 *
 * Prefer TASKGRAPH_LOOP below: this wrapper records ONE iteration per instance,
 * and a taskgraph instance carries an implicit taskgroup, so consecutive
 * iterations cannot overlap. That is the very cost TASKGRAPH_LOOP's `unroll`
 * exists to amortize. No app in this tree uses TASKGRAPH_BEGIN any more; it is
 * kept for the case of a single, non-iterated recorded region.
 * ------------------------------------------------------------------------- */
#if USE_TASKGRAPH && !USE_SYNC && !USE_OMPSS
# define TASKGRAPH_BEGIN pragma_omp_taskgraph(0, XKOMP_TASKGRAPH_FLAG_NONE, [&] (void)
# define TASKGRAPH_END   );
#else
# define TASKGRAPH_BEGIN
# define TASKGRAPH_END
#endif

/* ----------------------------------------------------------------------------
 * An iterative loop whose body is recorded once and replayed:
 *
 *     TASKGRAPH_LOOP(unroll, cond, epilogue)
 *     {
 *         ... ONE iteration: spawn the (loop-invariant) tasks ...
 *     }
 *     TASKGRAPH_LOOP_END
 *
 *   unroll   : iterations folded into one recorded graph instance. A taskgraph
 *              instance carries an implicit taskgroup, so instance i+1 cannot
 *              start before i has drained; unrolling recovers the cross-iteration
 *              overlap *inside* the instance and pays the barrier once per
 *              `unroll` iterations. Ignored unless USE_TASKGRAPH && USE_TASKGRAPHLOOP.
 *   cond     : bool(size_t done) -- loop condition, evaluated on the host between
 *              instances with the number of iterations already issued. Fixed trip
 *              count: [&](size_t d){ return d < (size_t) n; }. A convergence test
 *              may read device results written back by the previous instance (the
 *              instance boundary is a barrier), but is only evaluated every
 *              `unroll` iterations, so the loop may overshoot.
 *   epilogue : void(size_t inst, size_t done) -- host code after each instance,
 *              OUTSIDE the recorded region: per-iteration timing, progress
 *              prints, convergence bookkeeping. Spawn a task here rather than
 *              blocking, or the no-taskgraph configuration loses its pipelining.
 *
 * The body takes no iteration index on purpose: it runs only on the record pass,
 * so anything it captures is frozen into the graph and replayed verbatim. An
 * index baked into a recorded task would be wrong from the second instance on.
 * Iteration-varying state belongs in the data the graph reads/writes (a
 * device-side counter, a 1-element buffer chained through `depend`).
 *
 * Every configuration keeps the loop; only who owns it changes:
 *   taskgraph + taskgraphloop : one instance per `unroll` iterations
 *   taskgraph only            : one instance per iteration (the A/B baseline)
 *   otherwise                 : a plain loop, tasks re-created every iteration
 * ------------------------------------------------------------------------- */
/* Both taskgraph wrappers above and below are XKOMP's lambda form. LLVM's
 * `#pragma omp taskgraph` cannot serve here: a directive cannot be applied to a
 * region from inside a function template, and it has no taskgraphloop form at
 * all. Every taskgraph build in this tree already goes through xkcxx (see
 * common.mk, which hardcodes -DUSE_XKOMP=1), so this only catches a hand-rolled
 * build with the wrong toggles. */
#if USE_TASKGRAPH && !USE_SYNC && !USE_OMPSS && !USE_XKOMP
# error "USE_TASKGRAPH=1 requires USE_XKOMP=1: the taskgraph wrappers use XKOMP's lambda API, and LLVM's `#pragma omp taskgraph` has no taskgraphloop form. Build with xkcxx -DUSE_XKOMP=1."
#endif

/* Variadic so that commas inside the `cond` / `epilogue` lambda bodies (which
 * braces do not shield from the preprocessor) cannot split the argument list. */
#define TASKGRAPH_LOOP(...) tasking_taskgraph_loop(__VA_ARGS__, [&] (void)
#define TASKGRAPH_LOOP_END );

/* Dispatch for TASKGRAPH_LOOP. Every configuration groups `unroll` iterations
 * per "instance" and calls the epilogue once per group, so the iteration count,
 * the task order and the epilogue cadence are identical everywhere -- only who
 * records the group changes. That is what lets an app size its per-instance
 * arrays the same way whatever it is built with, and what keeps the reported
 * numbers comparable across -u.
 *
 * It does NOT make -u a no-op where there is no taskgraph: the epilogue, and so
 * the timing/progress task, fires once per `unroll` iterations rather than once
 * per iteration. That is a small real effect (measured ~1.7% on LULESH s=100,
 * nb=8), not a barrier being amortized -- there is no barrier to amortize there. */
template <typename Cond, typename Epilogue, typename Body>
static inline size_t
tasking_taskgraph_loop(size_t unroll, Cond cond, Epilogue epilogue, Body body)
{
    if (unroll == 0)
        unroll = 1;

#if USE_TASKGRAPH && !USE_SYNC && !USE_OMPSS && USE_TASKGRAPHLOOP
    /* the whole group is one recorded instance */
    return pragma_omp_taskgraphloop(0, XKOMP_TASKGRAPH_FLAG_NONE, unroll,
                                    cond, epilogue, body);
#else
    size_t done = 0, inst = 0;
    while (cond(done))
    {
        for (size_t u = 0 ; u < unroll ; ++u)
        {
# if USE_TASKGRAPH && !USE_SYNC && !USE_OMPSS
            /* USE_TASKGRAPHLOOP=0: one recorded instance per iteration, i.e. the
             * behaviour this construct exists to be compared against. Inlined
             * rather than routed through pragma_omp_taskgraph, whose
             * std::function parameter would type-erase (and possibly heap
             * allocate) the body on every iteration -- inside the measured
             * region, and only on this side of the A/B. */
            xkomp_taskgraph_t * tg = xkomp_taskgraph_begin(0, XKOMP_TASKGRAPH_FLAG_NONE);
            if (tg->rc == 1)
                body();
            xkomp_taskgraph_end(tg);
# else
            /* no graph: the tasks are simply re-created every iteration */
            body();
# endif
        }
        done += unroll;
        epilogue(inst++, done);
    }
    return done;
#endif
}

#endif /* OPENMP_TASKING_H */
