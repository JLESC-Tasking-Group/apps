/*
 * driver.h - shared command-line driver, statistics, and solver descriptor for
 * the Krylov solvers.
 *
 * Every solver (cg, cr, bicgstab, minres, gmres) is just an algorithm plus a
 * `const KrylovDescriptor krylov_descriptor` that this driver dispatches to. The
 * single main() (common/driver.cpp) parses the command line, builds the test
 * system, prints the banner, runs the solve, verifies the result, and reports
 * the timing statistics -- so none of that boilerplate is duplicated per solver.
 *
 * Statistics are timing-only, in the spirit of Krylov.jl's stats: the total
 * solve time plus the per-iteration wall time of iteration 0, iteration 1, and
 * the average of the remaining iterations. With -DUSE_TASKGRAPH this is exactly
 * the record / first-replay / steady-replay breakdown.
 */
#ifndef KRYLOV_DRIVER_H
#define KRYLOV_DRIVER_H

#include "spmat.h" /* SpMatrix, real_t, idx_t */

/* Every translation unit here is compiled as C++ (clang++), so this header uses
 * plain C++ linkage -- notably so that KrylovDescriptor.solve can point at the
 * solvers' functions without a C/C++ language-linkage mismatch. */

/* ---- parsed command line (superset of every solver's options) ---- */
typedef struct {
    int    N;          /* cubic grid dimension: n = N^3                        */
    int    iters;      /* iterations (or restart cycles for GMRES)             */
    int    m;          /* GMRES restart length (unused by the other solvers)   */
    int    T1, T2;     /* task granularity (vectors / SpMV sub-blocks)         */
    int    stencil;    /* 7 or 27 (stencil problems)                           */
    double conv;       /* convection strength (convection-diffusion problems)  */
    double sigma;      /* diagonal shift (MINRES indefinite test)              */
    int    print_dbg;  /* -p : per-iteration residual trace                    */
} KrylovParams;

/* ---- timing-only statistics (ala Krylov.jl) ----
 * iter_ms[k] is the wall time of iteration/restart k, recorded by the solver's
 * always-on per-iteration timing task (anchored on that iteration's residual
 * scalar, i.e. the cadence at which consecutive residuals become available --
 * one iteration in steady state). */
typedef struct {
    int     niter;     /* number of timed iterations / restarts                */
    double  total_s;   /* wall time of the whole solve loop (t1 - t0)          */
    double *iter_ms;   /* [niter] per-iteration wall time in milliseconds      */
} KrylovStats;

void krylov_stats_init  (KrylovStats *s, int niter);
void krylov_stats_free  (KrylovStats *s);
/* Print total time, iteration 0, iteration 1, and the average of iterations
 * 2..niter-1. `unit` is "iteration" or "restart"; when `taskgraph` is nonzero
 * the first two lines are annotated (record) / (1st replay). */
void krylov_stats_report(const KrylovStats *s, const char *unit, int taskgraph);

/* ---- solver descriptor: each solver translation unit provides exactly one ---- */
typedef enum { KR_SPD_STENCIL, KR_CONVDIFF, KR_STENCIL_SHIFT } kr_problem_t;

/* bits for KrylovDescriptor.opt_mask: which extra CLI options are meaningful. */
enum { OPT_STENCIL = 1u, OPT_CONV = 2u, OPT_SHIFT = 4u, OPT_MEM = 8u };

typedef struct {
    const char  *name;          /* "CG", "CR", ...  (banner + report labels)   */
    kr_problem_t problem;       /* which test matrix the driver builds         */
    unsigned     opt_mask;      /* OPT_* : extra options this solver accepts    */
    int          restarted;     /* GMRES: report unit "restart" and enable -m   */
    int          default_iters; /* default -i (seeded from MAX_ITER / RESTARTS) */
    int          default_m;     /* default -m (RESTART_M), else 0               */
    /* Run the solve: fill st (per-iteration times + st->total_s). */
    void       (*solve)(const SpMatrix *A, const real_t *b, real_t *x,
                        const KrylovParams *prm, KrylovStats *st);
} KrylovDescriptor;

/* Defined by each solver; referenced by the shared main() in driver.cpp. */
extern const KrylovDescriptor krylov_descriptor;

#endif /* KRYLOV_DRIVER_H */
