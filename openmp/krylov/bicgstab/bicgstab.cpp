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
#include "kalloc.h"
#include "kernels.h"

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Problem/solver defaults (override with -D on the compiler) ---- */
#ifndef GRID_N
# define GRID_N 64
#endif
#ifndef MAX_ITER
# define MAX_ITER 50
#endif
#ifndef CONV_COEFF
# define CONV_COEFF 1.0   /* convection strength of the test matrix (0 => symmetric) */
#endif

/* ==========================================================================
 * The solver.
 * ========================================================================== */
static double bicgstab_solve(const SpMatrix *A, const real_t *b, real_t *x,
                             int max_iter, int T1, int T2, int print_dbg)
{
    const idx_t   n       = A->n;
    const idx_t   nnz     = A->nnz;
    const idx_t  *row_ptr = A->row_ptr;
    const idx_t  *col_idx = A->col_idx;
    const real_t *val     = A->val;

    Tiling tl;
    tiling_init(&tl, n, T1, T2);
    char *q_tok = spmv_tokens_alloc(&tl); /* tokens for q = A p */
    char *d_tok = spmv_tokens_alloc(&tl); /* tokens for d = A s */

    /* Device-mapped working vectors (pinned on GPU builds). */
    real_t *r   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *c   = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* shadow residual r0 */
    real_t *p   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *q   = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* A p                */
    real_t *v   = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* M q                */
    real_t *s   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *d   = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* A s                */
    real_t *t   = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* M d                */
    real_t *inv = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* 1/diag(A)          */

    /* Device-mapped length-1 scalar buffers. */
    real_t *rho      = (real_t *) kr_alloc(sizeof(real_t));
    real_t *next_rho = (real_t *) kr_alloc(sizeof(real_t));
    real_t *cv       = (real_t *) kr_alloc(sizeof(real_t));
    real_t *ts       = (real_t *) kr_alloc(sizeof(real_t));
    real_t *tt       = (real_t *) kr_alloc(sizeof(real_t));
    real_t *alpha    = (real_t *) kr_alloc(sizeof(real_t));
    real_t *omega    = (real_t *) kr_alloc(sizeof(real_t));
    real_t *beta     = (real_t *) kr_alloc(sizeof(real_t));
    real_t *rr       = (real_t *) kr_alloc(sizeof(real_t)); /* <r,r> for the residual */

    /* Host-only partial sums for the CPU dot reductions (unused on GPU). */
    real_t *part_cv = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_ts = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_tt = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_nr = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_rr = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t));

    /* Host init: inv_diag = 1/diag(A), x = 0, r = b (true residual, x0 = 0). */
    spmat_extract_diagonal(A, inv);
    for (idx_t i = 0; i < n; i++) inv[i] = (real_t) 1.0 / inv[i];
    for (idx_t i = 0; i < n; i++) { x[i] = (real_t) 0.0; r[i] = b[i]; }

    OMP_TARGET_ENTER_DATA(map(to: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], x[0:n], r[0:n])
                          map(alloc: c[0:n], p[0:n], q[0:n], v[0:n], s[0:n], d[0:n], t[0:n],
                                     rho[0:1], next_rho[0:1], cv[0:1], ts[0:1], tt[0:1],
                                     alpha[0:1], omega[0:1], beta[0:1], rr[0:1]))

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
                task_spmv(row_ptr, col_idx, val, nnz, p, q, q_tok, &tl);  /* q = A p           */
                task_vmul_spmv(&tl, inv, q, q_tok, v);                    /* v = M q           */
                task_dot(&tl, c, v, part_cv, cv);                        /* <c,v>             */
                task_scalar_div(rho, cv, alpha);                         /* alpha = rho/<c,v> */
                task_copy(&tl, r, s);                                    /* s = r            */
                task_axpy(&tl, alpha, (real_t) -1.0, v, s);              /* s -= alpha*v     */
                task_axpy(&tl, alpha, (real_t) +1.0, p, x);              /* x += alpha*p     */
                task_spmv(row_ptr, col_idx, val, nnz, s, d, d_tok, &tl); /* d = A s          */
                task_vmul_spmv(&tl, inv, d, d_tok, t);                   /* t = M d          */
                task_dot(&tl, t, s, part_ts, ts);                       /* <t,s>            */
                task_dot(&tl, t, t, part_tt, tt);                       /* <t,t>            */
                task_scalar_div(ts, tt, omega);                         /* omega = <t,s>/<t,t> */
                task_axpy(&tl, omega, (real_t) +1.0, s, x);             /* x += omega*s     */
                task_copy(&tl, s, r);                                   /* r = s            */
                task_axpy(&tl, omega, (real_t) -1.0, t, r);             /* r -= omega*t     */
                task_dot(&tl, c, r, part_nr, next_rho);                 /* next_rho = <c,r> */
                task_dot(&tl, r, r, part_rr, rr);                       /* <r,r> (residual) */
                /* beta = (next_rho/rho) * (alpha/omega) -- a compound scalar task. */
                OMP_TARGET_TASK(DEPEND(in, next_rho[0], rho[0], alpha[0], omega[0]) DEPEND(out, beta[0])
                                MAP(present, alloc: next_rho[0:1], rho[0:1], alpha[0:1], omega[0:1], beta[0:1]))
                {
                    beta[0] = (next_rho[0] / rho[0]) * (alpha[0] / omega[0]);
                }
                task_axpy(&tl, omega, (real_t) -1.0, v, p);             /* p -= omega*v     */
                task_xpby(&tl, r, beta, p);                             /* p = r + beta*p   */
            }
            TASKGRAPH_END

            if (print_dbg) {
                const double spawn_ms = (omp_get_wtime() - spawn0) * 1000.0;
                OMP_TARGET_UPDATE(from(rr[0:1]) nowait DEPEND(inout, rr[0]))
                OMP_HOST_TASK(firstprivate(it, spawn_ms) shared(prev_ts) DEPEND(in, rr[0]))
                {
                    const double now     = omp_get_wtime();
                    const double exec_ms = (now - prev_ts) * 1000.0;
                    prev_ts = now;
                    printf("  iter %4d   residual = %.6e   spawn = %8.3f ms   exec = %8.3f ms\n",
                           it, sqrt(fabs((double) rr[0])), spawn_ms, exec_ms);
                }
            }
        }
        #pragma omp taskwait
    }

    const double t1 = omp_get_wtime();

    OMP_TARGET_EXIT_DATA(map(from: x[0:n])
                         map(release: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], r[0:n],
                                      c[0:n], p[0:n], q[0:n], v[0:n], s[0:n], d[0:n], t[0:n],
                                      rho[0:1], next_rho[0:1], cv[0:1], ts[0:1], tt[0:1],
                                      alpha[0:1], omega[0:1], beta[0:1], rr[0:1]))

    kr_free(r); kr_free(c); kr_free(p); kr_free(q); kr_free(v);
    kr_free(s); kr_free(d); kr_free(t); kr_free(inv);
    kr_free(rho); kr_free(next_rho); kr_free(cv); kr_free(ts); kr_free(tt);
    kr_free(alpha); kr_free(omega); kr_free(beta); kr_free(rr);
    free(part_cv); free(part_ts); free(part_tt); free(part_nr); free(part_rr);
    spmv_tokens_free(q_tok);
    spmv_tokens_free(d_tok);
    return t1 - t0;
}

/* ==========================================================================
 * Driver.
 * ========================================================================== */
static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-n N] [-i ITER] [-t T1] [-s T2] [-c CONV] [-p]\n"
            "  -n N       cubic grid: solve an N*N*N system   (default %d)\n"
            "  -i ITER    fixed number of BiCGSTAB iterations (default %d)\n"
            "  -t T1      number of tasks per vector op        (default: omp threads)\n"
            "  -s T2      number of SpMV sub-tasks per block   (default: omp threads)\n"
            "  -c CONV    convection strength (0 => symmetric) (default %.2f)\n"
            "  -p         print the residual at each iteration\n",
            prog, GRID_N, MAX_ITER, (double) CONV_COEFF);
}

int main(int argc, char **argv)
{
    int    N = GRID_N, max_iter = MAX_ITER, T1 = 0, T2 = 0, print_dbg = 0;
    double conv = CONV_COEFF;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-n") && i + 1 < argc) N        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) max_iter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) T1       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) T2       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-c") && i + 1 < argc) conv     = atof(argv[++i]);
        else if (!strcmp(argv[i], "-p"))                 print_dbg = 1;
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (T1 <= 0) T1 = omp_get_max_threads();
    if (T2 <= 0) T2 = omp_get_max_threads();

    /* Build the nonsymmetric convection-diffusion system (exact solution = 1). */
    SpMatrix A;
    real_t *b = NULL, *xexact = NULL;
    spmat_generate_convdiff(&A, N, N, N, (real_t) conv, &b, &xexact);

    printf("Krylov BiCGSTAB (Jacobi-preconditioned)\n");
    printf("  backend    : %s\n", USE_TARGET ? "GPU (omp target, pinned host mem)" : "CPU (omp task, malloc)");
    printf("  taskgraph  : %s\n", USE_TASKGRAPH ? "on" : "off");
    printf("  matrix     : convection-diffusion (conv = %.2f, %s)\n",
           conv, conv == 0.0 ? "symmetric" : "nonsymmetric");
    printf("  grid       : %d x %d x %d  (n = %d, nnz = %d)\n", N, N, N, A.n, A.nnz);
    printf("  iterations : %d\n", max_iter);
    printf("  tasks      : T1 = %d (vectors), T2 = %d (SpMV sub-blocks)\n", T1, T2);
    printf("  omp threads: %d\n", omp_get_max_threads());

    real_t *x = (real_t *) kr_alloc((size_t) A.n * sizeof(real_t));

    const double solve_time = bicgstab_solve(&A, b, x, max_iter, T1, T2, print_dbg);

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
