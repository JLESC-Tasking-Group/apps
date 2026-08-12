/*
 * minres.cpp - Jacobi-preconditioned MINRES, a clean re-implementation of
 * Krylov.jl's `minres!` using OpenMP tasking. MINRES solves SYMMETRIC (possibly
 * INDEFINITE) systems via Lanczos tridiagonalization + Givens rotations, so it
 * can be run on the SPD stencil or on a diagonally shifted (indefinite) one
 * (see -g below).
 *
 * Same infrastructure as the other solvers (common/kernels.{h,cpp}, -DUSE_TARGET,
 * -DUSE_TASKGRAPH). MINRES has substantial scalar (Givens) bookkeeping; it is
 * consolidated into three small scalar tasks per iteration (host tasks on CPU,
 * single-thread device tasks on GPU) so the scalars stay device-resident and the
 * per-iteration task graph is identical across iterations (record/replay-safe).
 *
 * One SpMV + two dots + a handful of axpy/scal per iteration. The reported
 * residual is phibar (MINRES tracks ||r|| = phibar by recurrence, no extra dot).
 * The w search-direction recurrence uses three fixed buffers (wm2, wm1, wcur)
 * rotated by copies instead of a pointer swap, so all task addresses are
 * loop-invariant.
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

#ifndef GRID_N
# define GRID_N 64
#endif
#ifndef MAX_ITER
# define MAX_ITER 50
#endif

#define GAMMA_FLOOR ((real_t) 1e-300) /* avoid division by zero in the rotation */

/* ==========================================================================
 * The solver.
 * ========================================================================== */
static double minres_solve(const SpMatrix *A, const real_t *b, real_t *x,
                           int max_iter, int T1, int T2, int print_dbg)
{
    const idx_t   n       = A->n;
    const idx_t   nnz     = A->nnz;
    const idx_t  *row_ptr = A->row_ptr;
    const idx_t  *col_idx = A->col_idx;
    const real_t *val     = A->val;

    Tiling tl;
    tiling_init(&tl, n, T1, T2);
    char *av_tok = spmv_tokens_alloc(&tl); /* tokens for Av = A v */

    /* Device-mapped vectors. */
    real_t *v    = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* Lanczos vector (M^-1 r2) */
    real_t *Av   = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* A v (SpMV output)        */
    real_t *y    = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* working Lanczos residual */
    real_t *r1   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *r2   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *wcur = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* w_k                      */
    real_t *wm1  = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* w_{k-1}                  */
    real_t *wm2  = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* w_{k-2}                  */
    real_t *inv  = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* 1/diag(A)                */

    /* Device-mapped scalars: persistent Lanczos/Givens state + per-iter temps. */
    real_t *beta      = (real_t *) kr_alloc(sizeof(real_t));
    real_t *oldb      = (real_t *) kr_alloc(sizeof(real_t));
    real_t *dbar      = (real_t *) kr_alloc(sizeof(real_t));
    real_t *eps       = (real_t *) kr_alloc(sizeof(real_t));
    real_t *cs        = (real_t *) kr_alloc(sizeof(real_t));
    real_t *sn        = (real_t *) kr_alloc(sizeof(real_t));
    real_t *phibar    = (real_t *) kr_alloc(sizeof(real_t));
    real_t *inv_beta  = (real_t *) kr_alloc(sizeof(real_t));
    real_t *bob       = (real_t *) kr_alloc(sizeof(real_t)); /* beta/oldbeta */
    real_t *vy        = (real_t *) kr_alloc(sizeof(real_t));
    real_t *alpha     = (real_t *) kr_alloc(sizeof(real_t));
    real_t *delta     = (real_t *) kr_alloc(sizeof(real_t));
    real_t *aob       = (real_t *) kr_alloc(sizeof(real_t)); /* alpha/beta */
    real_t *r2v       = (real_t *) kr_alloc(sizeof(real_t));
    real_t *phi       = (real_t *) kr_alloc(sizeof(real_t));
    real_t *gamma_inv = (real_t *) kr_alloc(sizeof(real_t));

    real_t *part_vy  = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t));
    real_t *part_r2v = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t));

    /* Host init: inv_diag, x = 0, r1 = r2 = b, v = M^-1 b = inv .* b, w's = 0. */
    spmat_extract_diagonal(A, inv);
    for (idx_t i = 0; i < n; i++) inv[i] = (real_t) 1.0 / inv[i];
    for (idx_t i = 0; i < n; i++) {
        x[i] = (real_t) 0.0;
        r1[i] = b[i]; r2[i] = b[i];
        v[i] = inv[i] * b[i];
        wm1[i] = (real_t) 0.0; wm2[i] = (real_t) 0.0;
    }

    OMP_TARGET_ENTER_DATA(map(to: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n],
                                  x[0:n], r1[0:n], r2[0:n], v[0:n], wm1[0:n], wm2[0:n])
                          map(alloc: Av[0:n], y[0:n], wcur[0:n],
                                     beta[0:1], oldb[0:1], dbar[0:1], eps[0:1], cs[0:1], sn[0:1], phibar[0:1],
                                     inv_beta[0:1], bob[0:1], vy[0:1], alpha[0:1], delta[0:1], aob[0:1],
                                     r2v[0:1], phi[0:1], gamma_inv[0:1]))

    const double t0 = omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {
        /* One-time setup: beta = sqrt(<r1,v>); init the Givens state. */
        task_dot(&tl, r1, v, part_vy, beta); /* beta <- <r1, v> (= beta_1^2) */
        OMP_TARGET_TASK(DEPEND(inout, beta[0]) DEPEND(out, phibar[0], cs[0], sn[0], dbar[0], eps[0], oldb[0])
                        MAP(present, alloc: beta[0:1], phibar[0:1], cs[0:1], sn[0:1], dbar[0:1], eps[0:1], oldb[0:1]))
        {
            beta[0]   = sqrt(beta[0]);
            phibar[0] = beta[0];
            cs[0]     = (real_t) -1.0;
            sn[0]     = (real_t) 0.0;
            dbar[0]   = (real_t) 0.0;
            eps[0]    = (real_t) 0.0;
            oldb[0]   = (real_t) 0.0;
        }
        #pragma omp taskwait

        double prev_ts = omp_get_wtime();

        for (int it = 0; it < max_iter; it++) {
            const double spawn0 = print_dbg ? omp_get_wtime() : 0.0;

            TASKGRAPH_BEGIN
            {
                /* --- scalars: inv_beta = 1/beta, bob = beta/oldbeta (0 first iter) --- */
                OMP_TARGET_TASK(DEPEND(in, beta[0], oldb[0]) DEPEND(out, inv_beta[0], bob[0])
                                MAP(present, alloc: beta[0:1], oldb[0:1], inv_beta[0:1], bob[0:1]))
                {
                    inv_beta[0] = (real_t) 1.0 / beta[0];
                    bob[0]      = (oldb[0] == (real_t) 0.0) ? (real_t) 0.0 : beta[0] / oldb[0];
                }

                /* --- Lanczos: y = (A v)/beta - (beta/oldbeta) r1 - (alpha/beta) r2 --- */
                task_spmv(row_ptr, col_idx, val, nnz, v, Av, av_tok, &tl); /* Av = A v         */
                task_copy_spmv(&tl, Av, av_tok, y);                        /* y = Av (retokenize) */
                task_scal(&tl, inv_beta, y);                              /* y /= beta        */
                task_axpy(&tl, bob, (real_t) -1.0, r1, y);               /* y -= (beta/oldb) r1 */
                task_dot(&tl, v, y, part_vy, vy);                        /* <v,y>            */

                /* --- scalars: alpha, delta (for w), alpha/beta --- */
                OMP_TARGET_TASK(DEPEND(in, vy[0], beta[0], cs[0], sn[0], dbar[0]) DEPEND(out, alpha[0], delta[0], aob[0])
                                MAP(present, alloc: vy[0:1], beta[0:1], cs[0:1], sn[0:1], dbar[0:1], alpha[0:1], delta[0:1], aob[0:1]))
                {
                    alpha[0] = vy[0] / beta[0];
                    delta[0] = cs[0] * dbar[0] + sn[0] * alpha[0];
                    aob[0]   = alpha[0] / beta[0];
                }
                task_axpy(&tl, aob, (real_t) -1.0, r2, y);               /* y -= (alpha/beta) r2 */

                /* --- w recurrence: wcur = v/beta - delta*wm1 - eps*wm2 --- */
                task_scal_copy(&tl, inv_beta, v, wcur);                   /* wcur = v/beta    */
                task_axpy(&tl, delta, (real_t) -1.0, wm1, wcur);         /* wcur -= delta*wm1 */
                task_axpy(&tl, eps,   (real_t) -1.0, wm2, wcur);         /* wcur -= eps*wm2  */

                /* --- advance Lanczos vectors: r1=r2, r2=y, v=M^-1 r2 --- */
                task_copy(&tl, r2, r1);                                   /* r1 = r2          */
                task_copy(&tl, y, r2);                                    /* r2 = y           */
                task_vmul(&tl, inv, r2, v);                              /* v = M^-1 r2      */
                task_dot(&tl, r2, v, part_r2v, r2v);                    /* <r2,v> (= beta^2) */

                /* --- scalars: new beta + plane rotation; phi, 1/gamma --- */
                OMP_TARGET_TASK(DEPEND(in, r2v[0], alpha[0])
                                DEPEND(inout, beta[0], cs[0], sn[0], dbar[0], phibar[0])
                                DEPEND(out, oldb[0], eps[0], phi[0], gamma_inv[0])
                                MAP(present, alloc: r2v[0:1], alpha[0:1], beta[0:1], cs[0:1], sn[0:1],
                                                    dbar[0:1], phibar[0:1], oldb[0:1], eps[0:1], phi[0:1], gamma_inv[0:1]))
                {
                    const real_t b_new  = sqrt(r2v[0]);
                    const real_t gbar   = sn[0] * dbar[0] - cs[0] * alpha[0];
                    const real_t eps_n  = sn[0] * b_new;
                    const real_t dbar_n = -cs[0] * b_new;
                    real_t g = sqrt(gbar * gbar + b_new * b_new);
                    if (g < GAMMA_FLOOR) g = GAMMA_FLOOR;
                    const real_t cs_n   = gbar / g;
                    const real_t sn_n   = b_new / g;
                    phi[0]       = cs_n * phibar[0];
                    const real_t pb_n   = sn_n * phibar[0];
                    /* commit persistent state (all old values already read above) */
                    oldb[0]      = beta[0];
                    beta[0]      = b_new;
                    eps[0]       = eps_n;
                    dbar[0]      = dbar_n;
                    cs[0]        = cs_n;
                    sn[0]        = sn_n;
                    phibar[0]    = pb_n;
                    gamma_inv[0] = (real_t) 1.0 / g;
                }

                /* --- finish w, update x, rotate the w buffers --- */
                task_scal(&tl, gamma_inv, wcur);                         /* wcur /= gamma    */
                task_axpy(&tl, phi, (real_t) +1.0, wcur, x);            /* x += phi*wcur    */
                task_copy(&tl, wm1, wm2);                                /* wm2 = wm1        */
                task_copy(&tl, wcur, wm1);                               /* wm1 = wcur       */
            }
            TASKGRAPH_END

            if (print_dbg) {
                const double spawn_ms = (omp_get_wtime() - spawn0) * 1000.0;
                OMP_TARGET_UPDATE(from(phibar[0:1]) nowait DEPEND(inout, phibar[0]))
                OMP_HOST_TASK(firstprivate(it, spawn_ms) shared(prev_ts) DEPEND(in, phibar[0]))
                {
                    const double now     = omp_get_wtime();
                    const double exec_ms = (now - prev_ts) * 1000.0;
                    prev_ts = now;
                    printf("  iter %4d   residual = %.6e   spawn = %8.3f ms   exec = %8.3f ms\n",
                           it, fabs((double) phibar[0]), spawn_ms, exec_ms);
                }
            }
        }
        #pragma omp taskwait
    }

    const double t1 = omp_get_wtime();

    OMP_TARGET_EXIT_DATA(map(from: x[0:n])
                         map(release: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n],
                                      r1[0:n], r2[0:n], v[0:n], Av[0:n], y[0:n], wcur[0:n], wm1[0:n], wm2[0:n],
                                      beta[0:1], oldb[0:1], dbar[0:1], eps[0:1], cs[0:1], sn[0:1], phibar[0:1],
                                      inv_beta[0:1], bob[0:1], vy[0:1], alpha[0:1], delta[0:1], aob[0:1],
                                      r2v[0:1], phi[0:1], gamma_inv[0:1]))

    kr_free(v); kr_free(Av); kr_free(y); kr_free(r1); kr_free(r2);
    kr_free(wcur); kr_free(wm1); kr_free(wm2); kr_free(inv);
    kr_free(beta); kr_free(oldb); kr_free(dbar); kr_free(eps); kr_free(cs); kr_free(sn); kr_free(phibar);
    kr_free(inv_beta); kr_free(bob); kr_free(vy); kr_free(alpha); kr_free(delta); kr_free(aob);
    kr_free(r2v); kr_free(phi); kr_free(gamma_inv);
    free(part_vy); free(part_r2v);
    spmv_tokens_free(av_tok);
    return t1 - t0;
}

/* ==========================================================================
 * Driver.
 * ========================================================================== */
static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [-n N] [-i ITER] [-t T1] [-s T2] [-S {7|27}] [-g SIGMA] [-p]\n"
            "  -n N       cubic grid: solve an N*N*N system   (default %d)\n"
            "  -i ITER    fixed number of MINRES iterations   (default %d)\n"
            "  -t T1      number of tasks per vector op        (default: omp threads)\n"
            "  -s T2      number of SpMV sub-tasks per block   (default: omp threads)\n"
            "  -S STENCIL 7- or 27-point 3-D stencil           (default 27)\n"
            "  -g SIGMA   shift the diagonal by -SIGMA (indefinite test) (default 0)\n"
            "  -p         print the residual at each iteration\n",
            prog, GRID_N, MAX_ITER);
}

int main(int argc, char **argv)
{
    int    N = GRID_N, max_iter = MAX_ITER, stencil = SPMAT_STENCIL_27PT;
    int    T1 = 0, T2 = 0, print_dbg = 0;
    double sigma = 0.0;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "-n") && i + 1 < argc) N        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-i") && i + 1 < argc) max_iter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) T1       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) T2       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) stencil  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-g") && i + 1 < argc) sigma    = atof(argv[++i]);
        else if (!strcmp(argv[i], "-p"))                 print_dbg = 1;
        else if (!strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); usage(argv[0]); return 1; }
    }
    if (T1 <= 0) T1 = omp_get_max_threads();
    if (T2 <= 0) T2 = omp_get_max_threads();
    if (stencil != SPMAT_STENCIL_7PT && stencil != SPMAT_STENCIL_27PT) stencil = SPMAT_STENCIL_27PT;

    /* Build the symmetric test system; optionally shift the diagonal to make it
     * indefinite (keeping the all-ones exact solution by also shifting b). */
    SpMatrix A;
    real_t *b = NULL, *xexact = NULL;
    spmat_generate_stencil(&A, N, N, N, stencil, &b, &xexact);
    if (sigma != 0.0) {
        spmat_shift_diagonal(&A, (real_t) sigma);
        for (idx_t i = 0; i < A.n; i++) b[i] -= (real_t) sigma; /* keep xexact = 1 */
    }

    printf("Krylov MINRES (Jacobi-preconditioned)\n");
    printf("  backend    : %s\n", USE_TARGET ? "GPU (omp target, pinned host mem)" : "CPU (omp task, malloc)");
    printf("  taskgraph  : %s\n", USE_TASKGRAPH ? "on" : "off");
    printf("  matrix     : %d-pt stencil, diagonal shift sigma = %.3f (%s)\n",
           stencil, sigma, sigma == 0.0 ? "SPD" : "indefinite");
    printf("  grid       : %d x %d x %d  (n = %d, nnz = %d)\n", N, N, N, A.n, A.nnz);
    printf("  iterations : %d\n", max_iter);
    printf("  tasks      : T1 = %d (vectors), T2 = %d (SpMV sub-blocks)\n", T1, T2);
    printf("  omp threads: %d\n", omp_get_max_threads());

    real_t *x = (real_t *) kr_alloc((size_t) A.n * sizeof(real_t));

    const double solve_time = minres_solve(&A, b, x, max_iter, T1, T2, print_dbg);

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
