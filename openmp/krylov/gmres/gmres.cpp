/*
 * gmres.cpp - Jacobi-preconditioned restarted GMRES(m), a clean re-implementation
 * of Krylov.jl's `gmres!` using OpenMP tasking. GMRES targets NONSYMMETRIC
 * systems, so it runs on the convection-diffusion matrix.
 *
 * Structure (left-Jacobi preconditioning M = diag(A)^-1 applied as inv_diag .* ;
 * N = I; restarted every m steps):
 *   for each restart:
 *       r0 = M (b - A x);  beta = ||r0||;  V[0] = r0 / beta      (setup)
 *       for j = 0 .. m-1:                                        (Arnoldi + MGS)
 *           w = A V[j];  q = M w
 *           for i = 0 .. j:  H[i,j] = <V[i],q>;  q -= H[i,j] V[i]
 *           H[j+1,j] = ||q||;  V[j+1] = q / H[j+1,j]
 *       solve  min || beta e1 - H y ||  (Givens QR + back-sub, on the host)
 *       x += sum_i y[i] V[i]
 *
 * The vector work (SpMV, modified Gram-Schmidt dots/axpys, norms, basis
 * normalization, and the x update) is tasked/offloaded via common/kernels; the
 * small (m+1) x m least-squares is done on the host once per restart. The inner
 * Arnoldi loop is the taskgraph unit: it spawns the identical task pattern (same
 * V/H buffers) every restart, so it is recorded once and replayed.
 *
 * Note: unlike the short-recurrence solvers, GMRES needs one host synchronization
 * per restart (to run the least-squares); this is infrequent (every m steps).
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
#include <string.h>

#ifndef RESTARTS
# define RESTARTS 10       /* number of restart cycles */
#endif
#ifndef RESTART_M
# define RESTART_M 30      /* Krylov subspace size (restart length) */
#endif

/* Host least-squares: solve min ||beta*e1 - H y|| for an (m+1) x m upper-
 * Hessenberg H (column-major, leading dim ld = m+1) via Givens QR + back
 * substitution. H is overwritten. Returns the residual estimate |g[m]|. */
static double gmres_least_squares(real_t *H, int ld, int m, real_t beta, real_t *y)
{
    double *g  = (double *) malloc((size_t)(m + 1) * sizeof(double));
    double *cs = (double *) malloc((size_t) m * sizeof(double));
    double *sn = (double *) malloc((size_t) m * sizeof(double));
    for (int i = 0; i <= m; i++) g[i] = 0.0;
    g[0] = (double) beta;

    for (int j = 0; j < m; j++) {
        /* apply previous rotations to column j */
        for (int i = 0; i < j; i++) {
            const double h1 = (double) H[i + j * ld];
            const double h2 = (double) H[(i + 1) + j * ld];
            H[i + j * ld]       = (real_t) (cs[i] * h1 + sn[i] * h2);
            H[(i + 1) + j * ld] = (real_t) (-sn[i] * h1 + cs[i] * h2);
        }
        /* new rotation zeroing H[j+1,j] */
        const double a = (double) H[j + j * ld];
        const double bb = (double) H[(j + 1) + j * ld];
        const double d = hypot(a, bb);
        if (d == 0.0) { cs[j] = 1.0; sn[j] = 0.0; }
        else          { cs[j] = a / d; sn[j] = bb / d; }
        H[j + j * ld]       = (real_t) (cs[j] * a + sn[j] * bb);
        H[(j + 1) + j * ld] = (real_t) 0.0;
        /* apply to the rhs g */
        const double g1 = g[j], g2 = g[j + 1];
        g[j]     = cs[j] * g1 + sn[j] * g2;
        g[j + 1] = -sn[j] * g1 + cs[j] * g2;
    }

    /* back substitution: R y = g */
    for (int i = m - 1; i >= 0; i--) {
        double sum = g[i];
        for (int k = i + 1; k < m; k++) sum -= (double) H[i + k * ld] * (double) y[k];
        const double rii = (double) H[i + i * ld];
        y[i] = (real_t) ((rii != 0.0) ? sum / rii : 0.0);
    }

    const double resid = fabs(g[m]);
    free(g); free(cs); free(sn);
    return resid;
}

/* ==========================================================================
 * The solver.
 * ========================================================================== */
static void gmres_solve(const SpMatrix *A, const real_t *b, real_t *x,
                        const KrylovParams *prm, KrylovStats *st)
{
    const int     nrestart  = prm->iters;
    const int     m         = prm->m;
    const int     T1        = prm->T1;
    const int     T2        = prm->T2;
    const int     print_dbg = prm->print_dbg;

    const idx_t   n       = A->n;
    const idx_t   nnz     = A->nnz;
    const idx_t  *row_ptr = A->row_ptr;
    const idx_t  *col_idx = A->col_idx;
    const real_t *val     = A->val;
    const int     ld      = m + 1; /* leading dim of H */

    Tiling tl;
    tiling_init(&tl, A, T1, T2);

    /* Device-mapped vectors: basis V[0..m], plus work vectors. */
    real_t *Vd  = (real_t *) host_alloc((size_t)(m + 1) * (size_t) n * sizeof(real_t));
    real_t *w   = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* A v (SpMV output)  */
    real_t *q   = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* M w, MGS work      */
    real_t *res = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* b - A x            */
    real_t *bd  = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* device copy of b   */
    real_t *inv = (real_t *) host_alloc((size_t) n * sizeof(real_t)); /* 1/diag(A)          */

    /* Device-mapped scalars and small arrays. */
    real_t *H     = (real_t *) host_alloc((size_t) ld * (size_t) m * sizeof(real_t)); /* Hessenberg */
    real_t *y     = (real_t *) host_alloc((size_t) m * sizeof(real_t));               /* LS solution */
    real_t *one   = (real_t *) host_alloc(sizeof(real_t));
    real_t *beta  = (real_t *) host_alloc(sizeof(real_t)); /* ||r0|| (also <q,q> then sqrt) */
    real_t *ibeta = (real_t *) host_alloc(sizeof(real_t)); /* 1/beta  */
    real_t *hh    = (real_t *) host_alloc(sizeof(real_t)); /* <q,q>   */
    real_t *ih    = (real_t *) host_alloc(sizeof(real_t)); /* 1/||q|| */
    real_t *part  = (real_t *) host_alloc((size_t) tl.NTB1 * sizeof(real_t)); /* per-block dot partials (device-resident) */
    real_t *Hhost = (real_t *) malloc((size_t) ld * (size_t) m * sizeof(real_t)); /* host H (least-squares) */

    /* Host init: inv_diag, x = 0, b on device, H = 0, one = 1. */
    spmat_extract_diagonal(A, inv);
    for (idx_t i = 0; i < n; i++) inv[i] = (real_t) 1.0 / inv[i];
    for (idx_t i = 0; i < n; i++) { x[i] = (real_t) 0.0; bd[i] = b[i]; }
    for (idx_t i = 0; i < (idx_t) ld * m; i++) H[i] = (real_t) 0.0;
    one[0] = (real_t) 1.0;

    OMP_TARGET_ENTER_DATA(map(to: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], x[0:n], bd[0:n],
                                  H[0:ld * m], one[0:1])
                          map(alloc: Vd[0:(m + 1) * n], w[0:n], q[0:n], res[0:n],
                                     y[0:m], beta[0:1], ibeta[0:1], hh[0:1], ih[0:1],
                                     part[0:tl.NTB1]))

    const double t0 = omp_get_wtime();
    double prev_ts = t0;

    #pragma omp parallel
    #pragma omp single
    {
        for (int r = 0; r < nrestart; r++) {
            const double spawn0 = print_dbg ? omp_get_wtime() : 0.0;

            /* --- Arnoldi restart cycle (recorded once, replayed) --- */
            TASKGRAPH_BEGIN
            {
                /* Setup: r0 = M(b - A x); beta = ||r0||; V[0] = r0/beta. */
                task_spmv(row_ptr, col_idx, val, nnz, x, w, &tl);          /* w = A x        */
                task_copy(&tl, bd, res);                                   /* res = b        */
                task_axpy_spmv(&tl, one, (real_t) -1.0, w, res);           /* res -= A x     */
                task_vmul(&tl, inv, res, q);                              /* q = M res      */
                task_dot(&tl, q, q, part, beta);                         /* beta = <q,q>   */
                OMP_TARGET_TASK(DEFAULT_NONE MAP(present: beta[0:1], ibeta[0:1]) SHARED(beta, ibeta))
                {
                    beta[0]  = sqrt(beta[0]);
                    ibeta[0] = (real_t) 1.0 / beta[0];
                }
                task_scal_copy(&tl, ibeta, q, Vd /* V[0] */);             /* V[0] = q/beta  */

                /* Arnoldi + modified Gram-Schmidt. */
                for (int j = 0; j < m; j++) {
                    real_t *Vj = Vd + (size_t) j * n;
                    task_spmv(row_ptr, col_idx, val, nnz, Vj, w, &tl);       /* w = A V[j] */
                    task_vmul_spmv(&tl, inv, w, q);                          /* q = M w    */
                    for (int i = 0; i <= j; i++) {
                        real_t *Vi = Vd + (size_t) i * n;
                        task_dot(&tl, Vi, q, part, H + (i + j * ld));         /* H[i,j] = <V[i],q> */
                        task_axpy(&tl, H + (i + j * ld), (real_t) -1.0, Vi, q); /* q -= H[i,j] V[i] */
                    }
                    task_dot(&tl, q, q, part, hh);                          /* hh = <q,q> */
                    OMP_TARGET_TASK(DEFAULT_NONE firstprivate(j, ld)
                                    DEPEND(in, hh[0]) DEPEND(out, ih[0], H[(j + 1) + j * ld])
                                    MAP(present: hh[0:1], ih[0:1], H[0:ld * m])
                                    SHARED(hh, ih, H))
                    {
                        const real_t hn = sqrt(hh[0]);
                        H[(j + 1) + j * ld] = hn;
                        ih[0] = (hn != (real_t) 0.0) ? (real_t) 1.0 / hn : (real_t) 0.0;
                    }
                    if (j < m - 1) {
                        real_t *Vjp1 = Vd + (size_t)(j + 1) * n;
                        task_scal_copy(&tl, ih, q, Vjp1);                   /* V[j+1] = q/||q|| */
                    }
                }
            }
            TASKGRAPH_END

            /* Host time to record/replay the restart's task graph. */
            const double spawn_ms = print_dbg ? (omp_get_wtime() - spawn0) * 1000.0 : 0.0;

            #pragma omp taskwait
            OMP_TARGET_UPDATE(from(H[0:ld * m]) from(beta[0:1]))

            /* Host least-squares min ||beta e1 - H y||, then push y to device. */
            memcpy(Hhost, H, (size_t) ld * m * sizeof(real_t));
            const double resid = gmres_least_squares(Hhost, ld, m, beta[0], y);
            OMP_TARGET_UPDATE(to(y[0:m]))

            /* x += sum_i y[i] V[i]. */
            for (int i = 0; i < m; i++)
                task_axpy(&tl, y + i, (real_t) +1.0, Vd + (size_t) i * n, x);
            #pragma omp taskwait

            /* Per-restart timing (always; GMRES already taskwaits per restart, so
             * this is a plain host measurement) + optional residual print (-p). */
            const double now = omp_get_wtime();
            st->iter_ms[r] = (now - prev_ts) * 1000.0;
            if (print_dbg)
                printf("  restart %4d   residual = %.6e   spawn = %8.3f ms   exec = %8.3f ms\n",
                       r, resid, spawn_ms, st->iter_ms[r]);
            prev_ts = now;
        }
    }

    const double t1 = omp_get_wtime();

    OMP_TARGET_EXIT_DATA(map(from: x[0:n])
                         map(release: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], bd[0:n],
                                      Vd[0:(m + 1) * n], w[0:n], q[0:n], res[0:n], H[0:ld * m], y[0:m],
                                      one[0:1], beta[0:1], ibeta[0:1], hh[0:1], ih[0:1],
                                      part[0:tl.NTB1]))

    host_free(Vd); host_free(w); host_free(q); host_free(res); host_free(bd); host_free(inv);
    host_free(H); host_free(y); host_free(one); host_free(beta); host_free(ibeta); host_free(hh); host_free(ih);
    host_free(part); free(Hhost);
    tiling_fini(&tl);
    st->total_s = t1 - t0;
}

/* Theoretical FLOPs per restart: setup (r0 = M(b - A x), beta, V[0]) + the
 * Arnoldi/MGS loop (m SpMVs, and O(n*m^2) dots/axpys from modified Gram-Schmidt)
 * + the x-update (m axpys). The (m+1)x m host least-squares is O(m^2), negligible
 * against the O(n*m^2) vector work, and omitted. */
static double gmres_flops(const SpMatrix *A, const KrylovParams *prm)
{
    const double n = (double) A->n, nnz = (double) A->nnz;
    const int    m = prm->m;

    double per_restart = KR_FLOP_SPMV(nnz)   /* w = A x            */
                       + KR_FLOP_AXPY(n)     /* res -= A x         */
                       + KR_FLOP_VMUL(n)     /* q = M res          */
                       + KR_FLOP_DOT(n)      /* beta = <q,q>       */
                       + KR_FLOP_SCAL(n);    /* V[0] = q / beta    */
    for (int j = 0; j < m; j++) {
        per_restart += KR_FLOP_SPMV(nnz)     /* w = A V[j]         */
                     + KR_FLOP_VMUL(n)       /* q = M w            */
                     + KR_FLOP_DOT(n);       /* hh = <q,q>         */
        per_restart += (double)(j + 1) * (KR_FLOP_DOT(n) + KR_FLOP_AXPY(n)); /* MGS */
        if (j < m - 1) per_restart += KR_FLOP_SCAL(n);            /* V[j+1] = q/||q|| */
    }
    per_restart += (double) m * KR_FLOP_AXPY(n);                  /* x += sum y[i] V[i] */

    return (double) prm->iters * per_restart;
}

/* ==========================================================================
 * Descriptor (the shared driver in common/driver.cpp provides main()).
 * ========================================================================== */
const KrylovDescriptor krylov_descriptor = {
    /* name          */ "GMRES",
    /* problem       */ KR_CONVDIFF,
    /* opt_mask      */ OPT_CONV | OPT_MEM,
    /* restarted     */ 1,
    /* default_iters */ RESTARTS,
    /* default_m     */ RESTART_M,
    /* solve         */ gmres_solve,
    /* flops         */ gmres_flops,
};
