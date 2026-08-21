/*
 * cr.cpp - Jacobi-preconditioned Conjugate Residual (CR), a clean
 * re-implementation of Krylov.jl's `cr!` using OpenMP tasking.
 *
 * Same infrastructure as cg.cpp: tasked/offloaded building blocks from
 * common/kernels.{h,cpp}, backend selected by -DUSE_TARGET, per-iteration task
 * region recorded/replayed with -DUSE_TASKGRAPH. CR is for SPD systems and, like
 * CG, reuses the SAME buffers every iteration (ideal for record/replay).
 *
 * Algorithm (preconditioned CR, x0 = 0, M = diag(A) Jacobi preconditioner
 * applied as z = M^-1 z = inv_diag .* z, so r is the preconditioned residual and
 * q = A*p is maintained by recurrence -> only ONE SpMV per iteration):
 *     r = M b;  Ar = A r;  rho = <r,Ar>;  p = r;  q = Ar
 *     repeat:
 *         Mq   = M q
 *         alpha = rho / <q,Mq>
 *         x    = x + alpha*p
 *         r    = r - alpha*Mq
 *         Ar   = A r
 *         rho_bar = rho;  rho = <r,Ar>;  beta = rho / rho_bar
 *         p    = r + beta*p
 *         q    = Ar + beta*q
 * The per-iteration residual reported is sqrt(|rho|) = ||r||_A; the final
 * verification uses the true 2-norm ||b - A x||_2.
 */
#include "spmat.h"
#include "tasking.h"
#include "kalloc.h"
#include "kernels.h"
#include "driver.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* ---- Problem/solver defaults (override with -D on the compiler) ---- */
#ifndef MAX_ITER
# define MAX_ITER 50      /* fixed number of CR iterations */
#endif

/* ==========================================================================
 * The solver.
 * ========================================================================== */
static void cr_solve(const SpMatrix *A, const real_t *b, real_t *x,
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
    real_t *r   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *p   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *q   = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* q = A*p (recurrence) */
    real_t *Ar  = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *Mq  = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* M^-1 q            */
    real_t *inv = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* 1/diag(A)         */

    /* Device-mapped length-1 scalar buffers. */
    real_t *rho     = (real_t *) kr_alloc(sizeof(real_t));
    real_t *rho_bar = (real_t *) kr_alloc(sizeof(real_t));
    real_t *qMq     = (real_t *) kr_alloc(sizeof(real_t));
    real_t *alpha   = (real_t *) kr_alloc(sizeof(real_t));
    real_t *beta    = (real_t *) kr_alloc(sizeof(real_t));

    /* Per-block partial dot sums (device-resident; the dot decomposes into T1
     * partial reductions into these + a finalize, on both backends). */
    real_t *part_qMq = (real_t *) kr_alloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_rho = (real_t *) kr_alloc((size_t) tl.NTB1 * sizeof(real_t));

    /* Host init: inv_diag = 1/diag(A), x = 0, r = M b = inv_diag .* b (x0 = 0). */
    spmat_extract_diagonal(A, inv);
    for (idx_t i = 0; i < n; i++) inv[i] = (real_t) 1.0 / inv[i];
    for (idx_t i = 0; i < n; i++) { x[i] = (real_t) 0.0; r[i] = inv[i] * b[i]; }

    OMP_TARGET_ENTER_DATA(map(to: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], x[0:n], r[0:n])
                          map(alloc: p[0:n], q[0:n], Ar[0:n], Mq[0:n],
                                     rho[0:1], rho_bar[0:1], qMq[0:1], alpha[0:1], beta[0:1],
                                     part_qMq[0:tl.NTB1], part_rho[0:tl.NTB1]))

    const double t0 = omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {
        /* One-time setup: Ar = A r; p = r; q = Ar; rho = <r,Ar>. */
        task_spmv(row_ptr, col_idx, val, nnz, r, Ar, &tl);
        task_copy(&tl, r, p);
        task_copy_spmv(&tl, Ar, q);
        task_dot_spmv(&tl, r, Ar, part_rho, rho);
        #pragma omp taskwait

        double prev_ts = omp_get_wtime();

        for (int it = 0; it < max_iter; it++) {
            const double spawn0 = print_dbg ? omp_get_wtime() : 0.0;

            /* Identical task pattern on identical buffers every iteration. */
            TASKGRAPH_BEGIN
            {
                task_vmul(&tl, inv, q, Mq);                                 /* Mq = M^-1 q      */
                task_dot(&tl, q, Mq, part_qMq, qMq);                       /* <q,Mq>           */
                task_scalar_div(rho, qMq, alpha);                          /* alpha = rho/<q,Mq> */
                task_axpy(&tl, alpha, (real_t) +1.0, p, x);                /* x += alpha*p     */
                task_axpy(&tl, alpha, (real_t) -1.0, Mq, r);               /* r -= alpha*Mq    */
                task_spmv(row_ptr, col_idx, val, nnz, r, Ar, &tl);        /* Ar = A r         */
                task_scalar_copy(rho, rho_bar);                            /* rho_bar = rho    */
                task_dot_spmv(&tl, r, Ar, part_rho, rho);                  /* rho = <r,Ar>     */
                task_scalar_div(rho, rho_bar, beta);                       /* beta = rho/rho_bar */
                task_xpby(&tl, r, beta, p);                                /* p = r + beta*p   */
                task_xpby_spmv(&tl, Ar, beta, q);                          /* q = Ar + beta*q  */
            }
            TASKGRAPH_END

            /* Per-iteration timing (always) + optional residual print (-p).
             * Depend-synchronized host task (no taskwait), anchored on rho (=
             * ||r||_A^2); consecutive firings bracket one iteration's work. */
            const double spawn_ms = print_dbg ? (omp_get_wtime() - spawn0) * 1000.0 : 0.0;
            if (print_dbg) {
                OMP_TARGET_UPDATE(from(rho[0:1]) NOWAIT DEPEND(inout, rho[0]))
            }
            OMP_HOST_TASK(DEFAULT_NONE firstprivate(it, spawn_ms, rho, print_dbg, st)
                          shared(prev_ts) DEPEND(in, rho[0]))
            {
                const double now = omp_get_wtime();
                st->iter_ms[it] = (now - prev_ts) * 1000.0;
                prev_ts = now;
                if (print_dbg)
                    printf("  iter %4d   residual = %.6e   spawn = %8.3f ms   exec = %8.3f ms\n",
                           it, sqrt(fabs((double) rho[0])), spawn_ms, st->iter_ms[it]);
            }
        }
        #pragma omp taskwait
    }

    const double t1 = omp_get_wtime();

    OMP_TARGET_EXIT_DATA(map(from: x[0:n])
                         map(release: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], r[0:n],
                                      p[0:n], q[0:n], Ar[0:n], Mq[0:n],
                                      rho[0:1], rho_bar[0:1], qMq[0:1], alpha[0:1], beta[0:1],
                                      part_qMq[0:tl.NTB1], part_rho[0:tl.NTB1]))

    kr_free(r); kr_free(p); kr_free(q); kr_free(Ar); kr_free(Mq); kr_free(inv);
    kr_free(rho); kr_free(rho_bar); kr_free(qMq); kr_free(alpha); kr_free(beta);
    kr_free(part_qMq); kr_free(part_rho);
    st->total_s = t1 - t0;
}

/* Theoretical FLOPs: setup (Ar = A r, rho = <r,Ar>) plus, per iteration, one
 * SpMV, two dots, two axpy, one vmul, one xpby and one xpby_spmv. */
static double cr_flops(const SpMatrix *A, const KrylovParams *prm)
{
    const double n = (double) A->n, nnz = (double) A->nnz;
    const double setup    = KR_FLOP_SPMV(nnz) + KR_FLOP_DOT(n);
    const double per_iter = KR_FLOP_VMUL(n)     /* Mq = M q        */
                          + KR_FLOP_DOT(n)      /* <q,Mq>          */
                          + KR_FLOP_AXPY(n)     /* x += alpha p    */
                          + KR_FLOP_AXPY(n)     /* r -= alpha Mq   */
                          + KR_FLOP_SPMV(nnz)   /* Ar = A r        */
                          + KR_FLOP_DOT(n)      /* rho = <r,Ar>    */
                          + KR_FLOP_XPBY(n)     /* p = r + beta p  */
                          + KR_FLOP_XPBY(n);    /* q = Ar + beta q */
    return setup + (double) prm->iters * per_iter;
}

/* ==========================================================================
 * Descriptor (the shared driver in common/driver.cpp provides main()).
 * ========================================================================== */
const KrylovDescriptor krylov_descriptor = {
    /* name          */ "CR",
    /* problem       */ KR_SPD_STENCIL,
    /* opt_mask      */ OPT_STENCIL,
    /* restarted     */ 0,
    /* default_iters */ MAX_ITER,
    /* default_m     */ 0,
    /* solve         */ cr_solve,
    /* flops         */ cr_flops,
};
