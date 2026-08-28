/*
 * cg.cpp - Jacobi-preconditioned Conjugate Gradient (CG), a clean
 * re-implementation of Krylov.jl's `cg!` using OpenMP tasking.
 *
 * The tasked/offloaded building blocks live in common/kernels.{h,cpp}; this file
 * is just the CG algorithm expressed as a task graph. One source, two backends
 * (selected at compile time by -DUSE_TARGET, see common/tasking.h). With
 * -DUSE_TASKGRAPH the per-iteration task region is recorded once and replayed;
 * CG reuses the SAME buffers every iteration, so every recorded task has
 * loop-invariant addresses -- the ideal record/replay case.
 *
 * Algorithm (preconditioned CG, x0 = 0, M = diag(A) Jacobi preconditioner):
 *     r = b;  z = M^-1 r;  p = z;  gamma = <r,z>
 *     repeat:
 *         Ap    = A*p
 *         pAp   = <p,Ap>
 *         alpha = gamma / pAp
 *         x     = x + alpha*p
 *         r     = r - alpha*Ap
 *         z     = M^-1 r
 *         g_new = <r,z>
 *         beta  = g_new / gamma
 *         p     = z + beta*p
 *         gamma = g_new
 * The reported per-iteration residual is sqrt(gamma) = ||r||_{M^-1} (as in
 * Krylov.jl); the final verification uses the true 2-norm ||b - A x||_2.
 */
#include "spmat.h"
#include "tasking.h"
#include "alloc.h"
#include "kernels.h"
#include "driver.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- Problem/solver defaults (override with -D on the compiler) ---- */
#ifndef MAX_ITER
# define MAX_ITER 50      /* fixed number of CG iterations */
#endif

/* ==========================================================================
 * The solver.
 * ========================================================================== */
static void cg_solve(const SpMatrix *A, const real_t *b, real_t *x,
                     const KrylovParams *prm, KrylovStats *st)
{
    const int     max_iter  = prm->iters;
    const int     T1        = prm->T1;
    const int     T2        = prm->T2;
    const int     print_dbg = prm->print_dbg;

    const idx_t   n       = A->n;
    const idx_t   nnz     = A->nnz;
    const idx_t  *row_ptr = A->row_ptr;
    const idx_t  *col_idx = A->col_idx;
    const real_t *val     = A->val;

    Tiling tl;
    tiling_init(&tl, n, T1, T2);

    /* Device-mapped working vectors (pinned on GPU builds). */
    real_t *r   = (real_t *) host_alloc((size_t) n * sizeof(real_t));
    real_t *p   = (real_t *) host_alloc((size_t) n * sizeof(real_t));
    real_t *Ap  = (real_t *) host_alloc((size_t) n * sizeof(real_t));
    real_t *z   = (real_t *) host_alloc((size_t) n * sizeof(real_t));
    real_t *inv = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* 1/diag(A) */

    /* Device-mapped length-1 scalar buffers. */
    real_t *gamma = (real_t *) host_alloc(sizeof(real_t));
    real_t *g_new = (real_t *) host_alloc(sizeof(real_t));
    real_t *pAp   = (real_t *) host_alloc(sizeof(real_t));
    real_t *alpha = (real_t *) host_alloc(sizeof(real_t));
    real_t *beta  = (real_t *) host_alloc(sizeof(real_t));

    /* Per-block partial dot sums (device-resident: the dot decomposes into T1
     * partial reductions into these + a finalize, on both backends). */
    real_t *part1 = (real_t *) host_alloc((size_t) tl.NTB1 * sizeof(real_t)); /* <p,Ap> */
    real_t *part2 = (real_t *) host_alloc((size_t) tl.NTB1 * sizeof(real_t)); /* <r,z>  */

    /* Host initialization: inv_diag = 1/diag(A), x = 0, r = b (since x0 = 0). */
    spmat_extract_diagonal(A, inv);
    for (idx_t i = 0; i < n; i++) inv[i] = (real_t) 1.0 / inv[i];
    for (idx_t i = 0; i < n; i++) { x[i] = (real_t) 0.0; r[i] = b[i]; }

    OMP_TARGET_ENTER_DATA(MAP(to: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], x[0:n], r[0:n])
                          MAP(alloc: p[0:n], Ap[0:n], z[0:n], gamma[0:1], g_new[0:1], pAp[0:1], alpha[0:1], beta[0:1],
                                     part1[0:tl.NTB1], part2[0:tl.NTB1]))

    const double t0 = omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {
        /* One-time setup: z = M^-1 r; p = z; gamma = <r,z>. A taskwait here is
         * fine -- it is a one-time synchronization, not per-iteration. */
        task_vmul(&tl, inv, r, z);
        task_copy(&tl, z, p);
        task_dot(&tl, r, z, part2, gamma);
        #pragma omp taskwait

        /* Reference timestamp for the first iteration's execution time. */
        double prev_ts = omp_get_wtime();

        for (int it = 0; it < max_iter; it++) {
            /* Host time spent creating (recording, then replaying) the tasks of
             * this iteration = "time to spawn all tasks". */
            const double spawn0 = print_dbg ? omp_get_wtime() : 0.0;

            /* Every iteration spawns the identical task pattern on the identical
             * buffers -> recorded once, replayed thereafter. */
            TASKGRAPH_BEGIN
            {
                task_spmv(row_ptr, col_idx, val, nnz, p, Ap, &tl);        /* Ap  = A*p       */
                task_dot_spmv(&tl, p, Ap, part1, pAp);                    /* pAp = <p,Ap>    */
                task_scalar_div(gamma, pAp, alpha);                       /* alpha = g/pAp   */
                task_axpy(&tl, alpha, (real_t) +1.0, p, x);               /* x  += alpha*p   */
                task_axpy_spmv(&tl, alpha, (real_t) -1.0, Ap, r);         /* r  -= alpha*Ap  */
                task_vmul(&tl, inv, r, z);                                /* z   = M^-1 r    */
                task_dot(&tl, r, z, part2, g_new);                        /* g_new = <r,z>   */
                task_scalar_div(g_new, gamma, beta);                      /* beta = gn/g     */
                task_xpby(&tl, z, beta, p);                               /* p = z + beta*p  */
                task_scalar_copy(g_new, gamma);                           /* gamma = g_new   */
            }
            TASKGRAPH_END

            /* Per-iteration timing (always) + optional residual print (-p). A
             * depend-synchronized host task (no taskwait): it runs after the
             * iteration's last task (gamma) and records this iteration's wall
             * time into st->iter_ms[it]. With -p it also reads back the residual
             * scalar (async D2H) and prints it. */
            const double spawn_ms = print_dbg ? (omp_get_wtime() - spawn0) * 1000.0 : 0.0;
            if (print_dbg) {
                OMP_TARGET_UPDATE(from(g_new[0:1]) NOWAIT DEPEND(inout, g_new[0]))
            }
            OMP_HOST_TASK(DEFAULT_NONE firstprivate(it, spawn_ms, g_new, gamma, print_dbg, st)
                          shared(prev_ts) DEPEND(in, g_new[0], gamma[0]))
            {
                const double now = omp_get_wtime();
                st->iter_ms[it] = (now - prev_ts) * 1000.0;
                prev_ts = now;
                if (print_dbg)
                    printf("  iter %4d   residual = %.6e   spawn = %8.3f ms   exec = %8.3f ms\n",
                           it, sqrt((double) g_new[0]), spawn_ms, st->iter_ms[it]);
            }
        }
        /* One-time end-of-solve synchronization before reading/freeing buffers. */
        #pragma omp taskwait
    }

    const double t1 = omp_get_wtime();

    OMP_TARGET_EXIT_DATA(MAP(from: x[0:n]) MAP(release: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], r[0:n],
                                      p[0:n], Ap[0:n], z[0:n], gamma[0:1], g_new[0:1], pAp[0:1], alpha[0:1], beta[0:1],
                                      part1[0:tl.NTB1], part2[0:tl.NTB1]))

    host_free(r); host_free(p); host_free(Ap); host_free(z); host_free(inv);
    host_free(gamma); host_free(g_new); host_free(pAp); host_free(alpha); host_free(beta);
    host_free(part1); host_free(part2);
    st->total_s = t1 - t0;
}

/* Theoretical FLOPs: one-time setup (z = M r, gamma = <r,z>) plus, per iteration,
 * one SpMV, three dots (<p,Ap>, <r,z> and the xpby's), two axpy, one vmul, one
 * xpby. (Scalar and copy ops are free.) */
static double cg_flops(const SpMatrix *A, const KrylovParams *prm)
{
    const double n = (double) A->n, nnz = (double) A->nnz;
    const double setup    = KR_FLOP_VMUL(n) + KR_FLOP_DOT(n);
    const double per_iter = KR_FLOP_SPMV(nnz)   /* Ap  = A p       */
                          + KR_FLOP_DOT(n)      /* pAp = <p,Ap>    */
                          + KR_FLOP_AXPY(n)     /* x  += alpha p   */
                          + KR_FLOP_AXPY(n)     /* r  -= alpha Ap  */
                          + KR_FLOP_VMUL(n)     /* z   = M r       */
                          + KR_FLOP_DOT(n)      /* g_new = <r,z>   */
                          + KR_FLOP_XPBY(n);    /* p = z + beta p  */
    return setup + (double) prm->iters * per_iter;
}

/* ==========================================================================
 * Descriptor (the shared driver in common/driver.cpp provides main()).
 * ========================================================================== */
const KrylovDescriptor krylov_descriptor = {
    /* name          */ "CG",
    /* problem       */ KR_SPD_STENCIL,
    /* opt_mask      */ OPT_STENCIL,
    /* restarted     */ 0,
    /* default_iters */ MAX_ITER,
    /* default_m     */ 0,
    /* solve         */ cg_solve,
    /* flops         */ cg_flops,
};
