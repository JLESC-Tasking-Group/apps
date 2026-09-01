/*
 * bicgstab.cpp - Jacobi-preconditioned BiCGSTAB, a clean re-implementation of
 * Krylov.jl's `bicgstab!` using OpenMP tasking. BiCGSTAB targets NONSYMMETRIC
 * systems, so it is run on the convection-diffusion matrix (spmat_generate_convdiff).
 *
 * Same infrastructure as the other solvers: tasked/offloaded building blocks in
 * common/kernels.{h,cpp}, backend selected by -DUSE_TARGET, per-iteration task
 * region recorded/replayed with -DUSE_TASKGRAPH.
 *
 * Algorithm (left-Jacobi preconditioning M = diag(A)^-1 applied as inv_diag .* ;
 * N = I; two matvecs per iteration; r is the true residual b - A x):
 *     r = b; c = r; p = r; next_rho = <c,r>
 *     repeat:
 *         rho = next_rho
 *         q = A p;   v = M q;   alpha = rho / <c,v>
 *         s = r - alpha*v;      x += alpha*p
 *         d = A s;   t = M d;   omega = <t,s> / <t,t>
 *         x += omega*s;         r = s - omega*t
 *         next_rho = <c,r>;     beta = (next_rho/rho)*(alpha/omega)
 *         p = p - omega*v;      p = r + beta*p
 * The per-iteration residual reported is sqrt(<r,r>) = ||b - A x||_2.
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
# define MAX_ITER 50
#endif

/* ==========================================================================
 * The solver.
 * ========================================================================== */
static void bicgstab_solve(const SpMatrix *A, const real_t *b, real_t *x,
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
    tiling_init(&tl, A, T1, T2);

    /* Device-mapped working vectors (pinned on GPU builds). */
    real_t *r   = (real_t *) host_alloc((size_t) n * sizeof(real_t));
    real_t *c   = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* shadow residual r0 */
    real_t *p   = (real_t *) host_alloc((size_t) n * sizeof(real_t));
    real_t *q   = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* A p                */
    real_t *v   = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* M q                */
    real_t *s   = (real_t *) host_alloc((size_t) n * sizeof(real_t));
    real_t *d   = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* A s                */
    real_t *t   = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* M d                */
    real_t *inv = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* 1/diag(A)          */

    /* Device-mapped length-1 scalar buffers. */
    real_t *rho      = (real_t *) host_alloc(sizeof(real_t));
    real_t *next_rho = (real_t *) host_alloc(sizeof(real_t));
    real_t *cv       = (real_t *) host_alloc(sizeof(real_t));
    real_t *ts       = (real_t *) host_alloc(sizeof(real_t));
    real_t *tt       = (real_t *) host_alloc(sizeof(real_t));
    real_t *alpha    = (real_t *) host_alloc(sizeof(real_t));
    real_t *omega    = (real_t *) host_alloc(sizeof(real_t));
    real_t *beta     = (real_t *) host_alloc(sizeof(real_t));
    real_t *rr       = (real_t *) host_alloc(sizeof(real_t)); /* <r,r> for the residual */

    /* Per-block partial dot sums (device-resident; the dot decomposes into T1
     * partial reductions into these + a finalize, on both backends). */
    real_t *part_cv = (real_t *) host_alloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_ts = (real_t *) host_alloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_tt = (real_t *) host_alloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_nr = (real_t *) host_alloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_rr = (real_t *) host_alloc((size_t) tl.NTB1 * sizeof(real_t));

    /* Host init: inv_diag = 1/diag(A), x = 0, r = b (true residual, x0 = 0). */
    spmat_extract_diagonal(A, inv);
    for (idx_t i = 0; i < n; i++) inv[i] = (real_t) 1.0 / inv[i];
    for (idx_t i = 0; i < n; i++) { x[i] = (real_t) 0.0; r[i] = b[i]; }

    OMP_TARGET_ENTER_DATA(map(to: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], x[0:n], r[0:n])
                          map(alloc: c[0:n], p[0:n], q[0:n], v[0:n], s[0:n], d[0:n], t[0:n],
                                     rho[0:1], next_rho[0:1], cv[0:1], ts[0:1], tt[0:1],
                                     alpha[0:1], omega[0:1], beta[0:1], rr[0:1],
                                     part_cv[0:tl.NTB1], part_ts[0:tl.NTB1], part_tt[0:tl.NTB1],
                                     part_nr[0:tl.NTB1], part_rr[0:tl.NTB1]))

    const double t0 = omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {
        /* One-time setup: c = r; p = r; next_rho = <c,r>. */
        task_copy(&tl, r, c);
        task_copy(&tl, r, p);
        task_dot(&tl, c, r, part_nr, next_rho);
        #pragma omp taskwait

        double prev_ts = omp_get_wtime();

        for (int it = 0; it < max_iter; it++) {
            const double spawn0 = print_dbg ? omp_get_wtime() : 0.0;

            TASKGRAPH_BEGIN
            {
                task_scalar_copy(next_rho, rho);                          /* rho = next_rho    */
                task_spmv(row_ptr, col_idx, val, nnz, p, q, &tl);        /* q = A p           */
                task_vmul_spmv(&tl, inv, q, v);                          /* v = M q           */
                task_dot(&tl, c, v, part_cv, cv);                        /* <c,v>             */
                task_scalar_div(rho, cv, alpha);                         /* alpha = rho/<c,v> */
                task_copy(&tl, r, s);                                    /* s = r            */
                task_axpy(&tl, alpha, (real_t) -1.0, v, s);              /* s -= alpha*v     */
                task_axpy(&tl, alpha, (real_t) +1.0, p, x);              /* x += alpha*p     */
                task_spmv(row_ptr, col_idx, val, nnz, s, d, &tl);       /* d = A s          */
                task_vmul_spmv(&tl, inv, d, t);                         /* t = M d          */
                task_dot(&tl, t, s, part_ts, ts);                       /* <t,s>            */
                task_dot(&tl, t, t, part_tt, tt);                       /* <t,t>            */
                task_scalar_div(ts, tt, omega);                         /* omega = <t,s>/<t,t> */
                task_axpy(&tl, omega, (real_t) +1.0, s, x);             /* x += omega*s     */
                task_copy(&tl, s, r);                                   /* r = s            */
                task_axpy(&tl, omega, (real_t) -1.0, t, r);             /* r -= omega*t     */
                task_dot(&tl, c, r, part_nr, next_rho);                 /* next_rho = <c,r> */
                task_dot(&tl, r, r, part_rr, rr);                       /* <r,r> (residual) */
                /* beta = (next_rho/rho) * (alpha/omega) -- a compound scalar task. */
                OMP_TARGET_TASK(DEFAULT_NONE
                                DEPEND(in, next_rho[0], rho[0], alpha[0], omega[0]) DEPEND(out, beta[0])
                                MAP(present: next_rho[0:1], rho[0:1], alpha[0:1], omega[0:1], beta[0:1])
                                SHARED(next_rho, rho, alpha, omega, beta))
                {
                    beta[0] = (next_rho[0] / rho[0]) * (alpha[0] / omega[0]);
                }
                task_axpy(&tl, omega, (real_t) -1.0, v, p);             /* p -= omega*v     */
                task_xpby(&tl, r, beta, p);                             /* p = r + beta*p   */
            }
            TASKGRAPH_END

            /* Per-iteration timing (always) + optional residual print (-p).
             * Depend-synchronized host task (no taskwait), anchored on rr =
             * <r,r> = ||b - A x||_2^2. */
            const double spawn_ms = print_dbg ? (omp_get_wtime() - spawn0) * 1000.0 : 0.0;
            if (print_dbg) {
                OMP_TARGET_UPDATE(from(rr[0:1]) NOWAIT DEPEND(inout, rr[0]))
            }
            OMP_HOST_TASK(DEFAULT_NONE firstprivate(it, spawn_ms, rr, print_dbg, st)
                          shared(prev_ts) DEPEND(in, rr[0]))
            {
                const double now = omp_get_wtime();
                st->iter_ms[it] = (now - prev_ts) * 1000.0;
                prev_ts = now;
                if (print_dbg)
                    printf("  iter %4d   residual = %.6e   spawn = %8.3f ms   exec = %8.3f ms\n",
                           it, sqrt(fabs((double) rr[0])), spawn_ms, st->iter_ms[it]);
            }
        }
        #pragma omp taskwait
    }

    const double t1 = omp_get_wtime();

    OMP_TARGET_EXIT_DATA(map(from: x[0:n])
                         map(release: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], r[0:n],
                                      c[0:n], p[0:n], q[0:n], v[0:n], s[0:n], d[0:n], t[0:n],
                                      rho[0:1], next_rho[0:1], cv[0:1], ts[0:1], tt[0:1],
                                      alpha[0:1], omega[0:1], beta[0:1], rr[0:1],
                                      part_cv[0:tl.NTB1], part_ts[0:tl.NTB1], part_tt[0:tl.NTB1],
                                      part_nr[0:tl.NTB1], part_rr[0:tl.NTB1]))

    host_free(r); host_free(c); host_free(p); host_free(q); host_free(v);
    host_free(s); host_free(d); host_free(t); host_free(inv);
    host_free(rho); host_free(next_rho); host_free(cv); host_free(ts); host_free(tt);
    host_free(alpha); host_free(omega); host_free(beta); host_free(rr);
    host_free(part_cv); host_free(part_ts); host_free(part_tt); host_free(part_nr); host_free(part_rr);
    tiling_fini(&tl);
    st->total_s = t1 - t0;
}

/* Theoretical FLOPs: setup (next_rho = <c,r>) plus, per iteration, two SpMVs,
 * two vmul (v = M q, t = M d), five dots (<c,v>,<t,s>,<t,t>,<c,r>,<r,r>), five
 * axpy and one xpby. */
static double bicgstab_flops(const SpMatrix *A, const KrylovParams *prm)
{
    const double n = (double) A->n, nnz = (double) A->nnz;
    const double setup    = KR_FLOP_DOT(n);
    const double per_iter = 2.0 * KR_FLOP_SPMV(nnz)   /* q = A p, d = A s        */
                          + 2.0 * KR_FLOP_VMUL(n)     /* v = M q, t = M d        */
                          + 5.0 * KR_FLOP_DOT(n)      /* cv, ts, tt, next_rho, rr */
                          + 5.0 * KR_FLOP_AXPY(n)     /* s, x, x, r, p updates    */
                          + KR_FLOP_XPBY(n);          /* p = r + beta p           */
    return setup + (double) prm->iters * per_iter;
}

/* ==========================================================================
 * Descriptor (the shared driver in common/driver.cpp provides main()).
 * ========================================================================== */
const KrylovDescriptor krylov_descriptor = {
    /* name          */ "BiCGSTAB",
    /* problem       */ KR_CONVDIFF,
    /* opt_mask      */ OPT_CONV,
    /* restarted     */ 0,
    /* default_iters */ MAX_ITER,
    /* default_m     */ 0,
    /* solve         */ bicgstab_solve,
    /* flops         */ bicgstab_flops,
};
