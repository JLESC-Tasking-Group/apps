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

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Problem/solver defaults (override with -D on the compiler) ---- */
#ifndef GRID_N
# define GRID_N 64        /* cubic grid dimension: n = GRID_N^3 unknowns */
#endif
#ifndef MAX_ITER
# define MAX_ITER 50      /* fixed number of CR iterations */
#endif

/* ==========================================================================
 * The solver.
 * ========================================================================== */
static double cr_solve(const SpMatrix *A, const real_t *b, real_t *x,
                       int max_iter, int T1, int T2, int print_dbg)
{
    const idx_t   n       = A->n;
    const idx_t   nnz     = A->nnz;
    const idx_t  *row_ptr = A->row_ptr;
    const idx_t  *col_idx = A->col_idx;
    const real_t *val     = A->val;

    Tiling tl;
    tiling_init(&tl, n, T1, T2);
    char *ar_tok = spmv_tokens_alloc(&tl); /* dependency tokens for Ar = A*r */

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

    /* Host-only partial sums for the CPU dot reductions (unused on GPU). */
    real_t *part_qMq = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_rho = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t));

    /* Host init: inv_diag = 1/diag(A), x = 0, r = M b = inv_diag .* b (x0 = 0). */
    spmat_extract_diagonal(A, inv);
    for (idx_t i = 0; i < n; i++) inv[i] = (real_t) 1.0 / inv[i];
    for (idx_t i = 0; i < n; i++) { x[i] = (real_t) 0.0; r[i] = inv[i] * b[i]; }

    OMP_TARGET_ENTER_DATA(map(to: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], x[0:n], r[0:n])
                          map(alloc: p[0:n], q[0:n], Ar[0:n], Mq[0:n],
                                     rho[0:1], rho_bar[0:1], qMq[0:1], alpha[0:1], beta[0:1]))

    const double t0 = omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {
        /* One-time setup: Ar = A r; p = r; q = Ar; rho = <r,Ar>. */
        task_spmv(row_ptr, col_idx, val, nnz, r, Ar, ar_tok, &tl);
        task_copy(&tl, r, p);
        task_copy_spmv(&tl, Ar, ar_tok, q);
        task_dot_spmv(&tl, r, Ar, ar_tok, part_rho, rho);
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
                task_spmv(row_ptr, col_idx, val, nnz, r, Ar, ar_tok, &tl); /* Ar = A r         */
                task_scalar_copy(rho, rho_bar);                            /* rho_bar = rho    */
                task_dot_spmv(&tl, r, Ar, ar_tok, part_rho, rho);          /* rho = <r,Ar>     */
                task_scalar_div(rho, rho_bar, beta);                       /* beta = rho/rho_bar */
                task_xpby(&tl, r, beta, p);                                /* p = r + beta*p   */
                task_xpby_spmv(&tl, Ar, ar_tok, beta, q);                  /* q = Ar + beta*q  */
            }
            TASKGRAPH_END

            if (print_dbg) {
                const double spawn_ms = (omp_get_wtime() - spawn0) * 1000.0;
                /* Async D2H of rho (= ||r||_A^2), ordered via its token. */
                OMP_TARGET_UPDATE(from(rho[0:1]) nowait DEPEND(inout, rho[0]))
                /* Depend-synchronized debug print (no taskwait); anchored on rho
                 * (computed after the SpMV), so consecutive prints bracket exactly
                 * one iteration's work. */
                OMP_HOST_TASK(firstprivate(it, spawn_ms) shared(prev_ts) DEPEND(in, rho[0]))
                {
                    const double now     = omp_get_wtime();
                    const double exec_ms = (now - prev_ts) * 1000.0;
                    prev_ts = now;
                    printf("  iter %4d   residual = %.6e   spawn = %8.3f ms   exec = %8.3f ms\n",
                           it, sqrt(fabs((double) rho[0])), spawn_ms, exec_ms);
                }
            }
        }
        #pragma omp taskwait
    }

    const double t1 = omp_get_wtime();

    OMP_TARGET_EXIT_DATA(map(from: x[0:n])
                         map(release: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], r[0:n],
                                      p[0:n], q[0:n], Ar[0:n], Mq[0:n],
                                      rho[0:1], rho_bar[0:1], qMq[0:1], alpha[0:1], beta[0:1]))

    kr_free(r); kr_free(p); kr_free(q); kr_free(Ar); kr_free(Mq); kr_free(inv);
    kr_free(rho); kr_free(rho_bar); kr_free(qMq); kr_free(alpha); kr_free(beta);
    free(part_qMq); free(part_rho);
    spmv_tokens_free(ar_tok);
    return t1 - t0;
}

/* ==========================================================================
 * Driver.
 * ========================================================================== */
static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-n N] [-i ITER] [-t T1] [-s T2] [-S {7|27}] [-p]\n"
            "  -n N       cubic grid: solve an N*N*N system   (default %d)\n"
            "  -i ITER    fixed number of CR iterations       (default %d)\n"
            "  -t T1      number of tasks per vector op        (default: omp threads)\n"
            "  -s T2      number of SpMV sub-tasks per block   (default: omp threads)\n"
            "  -S STENCIL 7- or 27-point 3-D stencil           (default 27)\n"
            "  -p         print the residual at each iteration\n",
            prog, GRID_N, MAX_ITER);
}

int main(int argc, char **argv)
{
    int N = GRID_N, max_iter = MAX_ITER, stencil = SPMAT_STENCIL_27PT;
    int T1 = 0, T2 = 0, print_dbg = 0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-n") && i + 1 < argc) N        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) max_iter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) T1       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) T2       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) stencil  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-p"))                 print_dbg = 1;
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (T1 <= 0) T1 = omp_get_max_threads();
    if (T2 <= 0) T2 = omp_get_max_threads();
    if (stencil != SPMAT_STENCIL_7PT && stencil != SPMAT_STENCIL_27PT) stencil = SPMAT_STENCIL_27PT;

    /* Build the SPD test system (exact solution is all-ones). */
    SpMatrix A;
    real_t *b = NULL, *xexact = NULL;
    spmat_generate_stencil(&A, N, N, N, stencil, &b, &xexact);

    printf("Krylov CR (Jacobi-preconditioned)\n");
    printf("  backend    : %s\n", USE_TARGET ? "GPU (omp target, pinned host mem)" : "CPU (omp task, malloc)");
    printf("  taskgraph  : %s\n", USE_TASKGRAPH ? "on" : "off");
    printf("  grid       : %d x %d x %d  (n = %d, nnz = %d, %d-pt stencil)\n",
           N, N, N, A.n, A.nnz, stencil);
    printf("  iterations : %d\n", max_iter);
    printf("  tasks      : T1 = %d (vectors), T2 = %d (SpMV sub-blocks)\n", T1, T2);
    printf("  omp threads: %d\n", omp_get_max_threads());

    real_t *x = (real_t *) kr_alloc((size_t) A.n * sizeof(real_t));

    const double solve_time = cr_solve(&A, b, x, max_iter, T1, T2, print_dbg);

    /* Verification: true residual ||b - A x||_2 and error ||x - xexact||_2. */
    real_t *Ax = (real_t *) malloc((size_t) A.n * sizeof(real_t));
    spmat_spmv(&A, x, Ax);
    double res2 = 0.0, b2 = 0.0, err2 = 0.0, xe2 = 0.0;
    for (idx_t i = 0; i < A.n; i++) {
        const double ri = (double) b[i] - (double) Ax[i];
        const double ei = (double) x[i] - (double) xexact[i];
        res2 += ri * ri; b2 += (double) b[i] * (double) b[i];
        err2 += ei * ei; xe2 += (double) xexact[i] * (double) xexact[i];
    }
    printf("Results\n");
    printf("  solve time            : %.4f s\n", solve_time);
    printf("  relative residual     : %.6e   (||b-Ax|| / ||b||)\n", sqrt(res2 / b2));
    printf("  relative error        : %.6e   (||x-xexact|| / ||xexact||)\n", sqrt(err2 / xe2));

    free(Ax); kr_free(x); free(b); free(xexact);
    spmat_free(&A);
    return 0;
}
