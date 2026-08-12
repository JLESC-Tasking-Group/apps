/*
 * cg.cpp - Jacobi-preconditioned Conjugate Gradient (CG), a clean
 * re-implementation of Krylov.jl's `cg!` using OpenMP tasking.
 *
 * One source, two backends (selected at compile time by -DUSE_TARGET, see
 * common/tasking.h):
 *   - CPU: each vector op is split into tasks (#pragma omp task depend(...));
 *   - GPU: each vector op is one offloaded task (#pragma omp target ... nowait
 *          depend(...) map(...)).
 *
 * With -DUSE_TASKGRAPH the per-iteration task region is recorded once and
 * replayed. CG reuses the SAME buffers every iteration, so every recorded task
 * has loop-invariant addresses and dependencies -- the ideal record/replay case.
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
# define MAX_ITER 50      /* fixed number of CG iterations */
#endif

#define KR_MIN(a, b) ((a) < (b) ? (a) : (b))

/* Number of rows in a task block: 1 per GPU thread, else ceil(n / ntasks). */
static inline idx_t block_size(idx_t n, int ntasks)
{
    return USE_TARGET ? (idx_t) 1 : (idx_t) ((n + ntasks - 1) / ntasks);
}

/* ==========================================================================
 * Tasked / offloaded kernels. Each creates the tasks for one vector operation.
 * On CPU the outer OMP_TARGET_LOOP_TASK vanishes and OMP_TASK spawns one task
 * per row-block; on GPU it is the reverse (one target parallel-for, bs == 1).
 * ========================================================================== */

/* Ap = A*p  (CSR SpMV, split into T2 row-blocks). SpMV reads scattered x, so
 * the input dependency is on the whole vector; the output is per-block. */
static void task_spmv(const idx_t *row_ptr, const idx_t *col_idx, const real_t *val,
                      idx_t n, idx_t nnz, const real_t *x, real_t *y, int T2)
{
    (void) nnz; /* used only in the GPU map() clause */
    const idx_t bs = block_size(n, T2);
    OMP_TARGET_LOOP_TASK(DEPEND(in, x[0:n]) DEPEND(out, y[0:n])
                         MAP(present, alloc: x[0:n], y[0:n], val[0:nnz], col_idx[0:nnz], row_ptr[0:n + 1]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n) DEPEND(in, x[0:n]) DEPEND(out, y[blk:bs]))
        {
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) {
                real_t sum = (real_t) 0.0;
                for (idx_t k = row_ptr[i]; k < row_ptr[i + 1]; k++)
                    sum += val[k] * x[col_idx[k]];
                y[i] = sum;
            }
        }
    }
}

/* y = x  (copy). */
static void task_copy(idx_t n, int T1, const real_t *x, real_t *y)
{
    const idx_t bs = block_size(n, T1);
    OMP_TARGET_LOOP_TASK(DEPEND(in, x[0:n]) DEPEND(out, y[0:n]) MAP(present, alloc: x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n) DEPEND(in, x[blk:bs]) DEPEND(out, y[blk:bs]))
        {
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = x[i];
        }
    }
}

/* y = d .* x  (elementwise product; the Jacobi apply z = inv_diag .* r). */
static void task_vmul(idx_t n, int T1, const real_t *d, const real_t *x, real_t *y)
{
    const idx_t bs = block_size(n, T1);
    OMP_TARGET_LOOP_TASK(DEPEND(in, d[0:n], x[0:n]) DEPEND(out, y[0:n]) MAP(present, alloc: d[0:n], x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n) DEPEND(in, d[blk:bs], x[blk:bs]) DEPEND(out, y[blk:bs]))
        {
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = d[i] * x[i];
        }
    }
}

/* y = y + sign*s*x  (axpy; s is a length-1 scalar buffer, sign is +1 or -1). */
static void task_axpy(idx_t n, int T1, const real_t *s, real_t sign, const real_t *x, real_t *y)
{
    const idx_t bs = block_size(n, T1);
    OMP_TARGET_LOOP_TASK(DEPEND(in, s[0:1], x[0:n]) DEPEND(inout, y[0:n]) MAP(present, alloc: s[0:1], x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n, sign) DEPEND(in, s[0:1], x[blk:bs]) DEPEND(inout, y[blk:bs]))
        {
            const real_t c = sign * s[0];
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] += c * x[i];
        }
    }
}

/* y = x + s*y  (axpby with the scale on y; the direction update p = z + beta*p). */
static void task_xpby(idx_t n, int T1, const real_t *x, const real_t *s, real_t *y)
{
    const idx_t bs = block_size(n, T1);
    OMP_TARGET_LOOP_TASK(DEPEND(in, x[0:n], s[0:1]) DEPEND(inout, y[0:n]) MAP(present, alloc: x[0:n], s[0:1], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n) DEPEND(in, x[blk:bs], s[0:1]) DEPEND(inout, y[blk:bs]))
        {
            const real_t c = s[0];
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = x[i] + c * y[i];
        }
    }
}

/* result = <a,b>  (dot product). The reduction genuinely differs per backend:
 *   GPU: one target reduction into the device scalar result[0];
 *   CPU: T1 per-block partial sums (part[]) then one gather task. */
#if USE_TARGET
static void task_dot(idx_t n, int T1, const real_t *a, const real_t *b, real_t *part, real_t *result)
{
    (void) T1;
    (void) part;
    /* result[0] must be zeroed first: the reduction combines into the original. */
    OMP_TARGET_TASK(DEPEND(out, result[0:1]) MAP(present, alloc: result[0:1]))
    {
        result[0] = (real_t) 0.0;
    }
    KR_XPRAGMA(omp target teams distribute parallel for REPLAYABLE_CLAUSE nowait
               map(present, alloc: a[0:n], b[0:n], result[0:1])
               reduction(+: result[0]) depend(in: a[0:n], b[0:n]) depend(inout: result[0:1]))
    for (idx_t i = 0; i < n; i++)
        result[0] += a[i] * b[i];
}
#else
static void task_dot(idx_t n, int T1, const real_t *a, const real_t *b, real_t *part, real_t *result)
{
    const idx_t bs   = block_size(n, T1);
    const int   nblk = (int) ((n + bs - 1) / bs);
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n) DEPEND(in, a[blk:bs], b[blk:bs]) DEPEND(out, part[blk / bs:1]))
        {
            const idx_t end = KR_MIN(blk + bs, n);
            real_t s = (real_t) 0.0;
            for (idx_t i = blk; i < end; i++) s += a[i] * b[i];
            part[blk / bs] = s;
        }
    }
    OMP_TASK(firstprivate(nblk) DEPEND_MULTI(in, (t=0:nblk), part[t]) DEPEND(out, result[0:1]))
    {
        real_t s = (real_t) 0.0;
        for (int t = 0; t < nblk; t++) s += part[t];
        result[0] = s;
    }
}
#endif

/* c = a / b  (tiny scalar task). */
static void task_scalar_div(const real_t *a, const real_t *b, real_t *c)
{
    OMP_TARGET_TASK(DEPEND(in, a[0:1], b[0:1]) DEPEND(out, c[0:1]) MAP(present, alloc: a[0:1], b[0:1], c[0:1]))
    {
        c[0] = a[0] / b[0];
    }
}

/* b = a  (tiny scalar copy). */
static void task_scalar_copy(const real_t *a, real_t *b)
{
    OMP_TARGET_TASK(DEPEND(in, a[0:1]) DEPEND(out, b[0:1]) MAP(present, alloc: a[0:1], b[0:1]))
    {
        b[0] = a[0];
    }
}

/* ==========================================================================
 * The solver.
 * ========================================================================== */
static double cg_solve(const SpMatrix *A, const real_t *b, real_t *x,
                       int max_iter, int T1, int T2, int print_dbg)
{
    const idx_t   n       = A->n;
    const idx_t   nnz     = A->nnz;
    const idx_t  *row_ptr = A->row_ptr;
    const idx_t  *col_idx = A->col_idx;
    const real_t *val     = A->val;

    /* Working vectors. */
    real_t *r   = (real_t *) malloc((size_t) n * sizeof(real_t));
    real_t *p   = (real_t *) malloc((size_t) n * sizeof(real_t));
    real_t *Ap  = (real_t *) malloc((size_t) n * sizeof(real_t));
    real_t *z   = (real_t *) malloc((size_t) n * sizeof(real_t));
    real_t *inv = (real_t *) malloc((size_t) n * sizeof(real_t)); /* 1/diag(A) */

    /* Length-1 scalar buffers (device-resident under USE_TARGET). */
    real_t *gamma = (real_t *) malloc(sizeof(real_t));
    real_t *g_new = (real_t *) malloc(sizeof(real_t));
    real_t *pAp   = (real_t *) malloc(sizeof(real_t));
    real_t *alpha = (real_t *) malloc(sizeof(real_t));
    real_t *beta  = (real_t *) malloc(sizeof(real_t));

    /* Per-block partial sums for the CPU dot reduction (unused on GPU). */
    real_t *part1 = (real_t *) malloc((size_t) T1 * sizeof(real_t)); /* for <p,Ap>  */
    real_t *part2 = (real_t *) malloc((size_t) T1 * sizeof(real_t)); /* for <r,z>   */

    /* Host initialization: inv_diag = 1/diag(A), x = 0, r = b (since x0 = 0). */
    spmat_extract_diagonal(A, inv);
    for (idx_t i = 0; i < n; i++) inv[i] = (real_t) 1.0 / inv[i];
    for (idx_t i = 0; i < n; i++) { x[i] = (real_t) 0.0; r[i] = b[i]; }

#if USE_TARGET
    #pragma omp target enter data map(to: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], x[0:n], r[0:n]) \
        map(alloc: p[0:n], Ap[0:n], z[0:n], gamma[0:1], g_new[0:1], pAp[0:1], alpha[0:1], beta[0:1])
#endif

    const double t0 = omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {
        /* One-time setup (outside the taskgraph region): z = M^-1 r; p = z;
         * gamma = <r,z>. The taskwait makes z/p/gamma ready before the first
         * taskgraph record, keeping the recorded region self-contained. */
        task_vmul(n, T1, inv, r, z);
        task_copy(n, T1, z, p);
        task_dot(n, T1, r, z, part2, gamma);
        #pragma omp taskwait

        for (int it = 0; it < max_iter; it++) {
            /* Every iteration spawns the identical task pattern on the identical
             * buffers -> recorded once, replayed thereafter. */
            TASKGRAPH_BEGIN
            {
                task_spmv(row_ptr, col_idx, val, n, nnz, p, Ap, T2); /* Ap  = A*p       */
                task_dot(n, T1, p, Ap, part1, pAp);                  /* pAp = <p,Ap>    */
                task_scalar_div(gamma, pAp, alpha);                  /* alpha = g/pAp   */
                task_axpy(n, T1, alpha, (real_t) +1.0, p, x);        /* x  += alpha*p   */
                task_axpy(n, T1, alpha, (real_t) -1.0, Ap, r);       /* r  -= alpha*Ap  */
                task_vmul(n, T1, inv, r, z);                         /* z   = M^-1 r    */
                task_dot(n, T1, r, z, part2, g_new);                 /* g_new = <r,z>   */
                task_scalar_div(g_new, gamma, beta);                 /* beta = gn/g     */
                task_xpby(n, T1, z, beta, p);                        /* p = z + beta*p  */
                task_scalar_copy(g_new, gamma);                      /* gamma = g_new   */
            }
            TASKGRAPH_END

            if (print_dbg) {
                #pragma omp taskwait
#if USE_TARGET
                #pragma omp target update from(g_new[0:1])
#endif
                printf("  iter %4d   residual = %.6e\n", it, sqrt((double) g_new[0]));
            }
        }
        #pragma omp taskwait
    }

    const double t1 = omp_get_wtime();

#if USE_TARGET
    #pragma omp target exit data map(from: x[0:n]) \
        map(release: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], r[0:n], \
                     p[0:n], Ap[0:n], z[0:n], gamma[0:1], g_new[0:1], pAp[0:1], alpha[0:1], beta[0:1])
#endif

    free(r); free(p); free(Ap); free(z); free(inv);
    free(gamma); free(g_new); free(pAp); free(alpha); free(beta);
    free(part1); free(part2);
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
            "  -i ITER    fixed number of CG iterations       (default %d)\n"
            "  -t T1      number of tasks per vector op        (default: omp threads)\n"
            "  -s T2      number of tasks for the SpMV         (default: omp threads)\n"
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

    printf("Krylov CG (Jacobi-preconditioned)\n");
    printf("  backend    : %s\n", USE_TARGET ? "GPU (omp target)" : "CPU (omp task)");
    printf("  taskgraph  : %s\n", USE_TASKGRAPH ? "on" : "off");
    printf("  grid       : %d x %d x %d  (n = %d, nnz = %d, %d-pt stencil)\n",
           N, N, N, A.n, A.nnz, stencil);
    printf("  iterations : %d\n", max_iter);
    printf("  tasks      : T1 = %d (vectors), T2 = %d (SpMV)\n", T1, T2);
    printf("  omp threads: %d\n", omp_get_max_threads());

    real_t *x = (real_t *) malloc((size_t) A.n * sizeof(real_t));

    const double solve_time = cg_solve(&A, b, x, max_iter, T1, T2, print_dbg);

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

    free(Ax); free(x); free(b); free(xexact);
    spmat_free(&A);
    return 0;
}
