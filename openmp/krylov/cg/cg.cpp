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
 * Task granularity (host backend), following HPCCG:
 *   - a vector operation of length n is split into T1 tasks (block size BS);
 *   - the SpMV is split into T1 x T2 tasks (each T1 block into T2 sub-blocks),
 *     so its output chunks carry their own dependencies.
 *
 * Dependencies use SINGLE-ELEMENT tokens at chunk-aligned addresses (never array
 * sections): OpenMP only matches depend list items that are *identical or
 * disjoint*, so x[i] and x[j!=i] are independent tokens and x[0:n] would be
 * identical to x[0]. Vector chunks are keyed by the data address at the block
 * start (v[t1*BS]); the SpMV sub-block chunks are keyed by a dedicated token
 * array `aptok` (like cholesky's dep array) so every token is in-bounds and the
 * T1 consumers of Ap can wait on all T2 sub-blocks via a depend iterator.
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
#include "kalloc.h"

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

/* ---- Task tiling (see the file header). ---- */
typedef struct {
    idx_t n;
    int   T1;      /* number of tasks per vector op                          */
    int   T2;      /* number of SpMV sub-tasks per T1 block (T1*T2 total)     */
    idx_t BS;      /* rows per vector block  = ceil(n / T1)                   */
    idx_t SBS;     /* rows per SpMV sub-block = ceil(BS / T2)                 */
    int   NTB1;    /* actual number of vector blocks = ceil(n / BS) (<= T1)   */
    char *aptok;   /* [NTB1*T2] dependency tokens for SpMV output sub-blocks  */
} Tiling;

static void tiling_init(Tiling *tl, idx_t n, int T1, int T2)
{
    tl->n    = n;
    tl->T1   = T1;
    tl->T2   = T2;
    tl->BS   = (n + T1 - 1) / T1;
    tl->NTB1 = (int) ((n + tl->BS - 1) / tl->BS);
    tl->SBS  = (tl->BS + T2 - 1) / T2;
#if USE_TARGET
    tl->aptok = NULL; /* GPU: one task per op, so no per-chunk tokens needed */
#else
    tl->aptok = (char *) malloc((size_t) tl->NTB1 * (size_t) T2);
#endif
}

static void tiling_free(Tiling *tl)
{
    free(tl->aptok);
    tl->aptok = NULL;
}

/* ==========================================================================
 * Tasked / offloaded kernels.
 * ========================================================================== */

#if USE_TARGET
/* result = <a,b> on the device: zero the scalar, then a target reduction into
 * it (an OpenMP 5.x array-element reduction on the mapped scalar result[0]). */
static void gpu_dot(idx_t n, const real_t *a, const real_t *b, real_t *result)
{
    OMP_TARGET_TASK(DEPEND(out, result[0]) MAP(present, alloc: result[0:1]))
    {
        result[0] = (real_t) 0.0;
    }
    OMP_TARGET_LOOP_TASK(reduction(+: result[0]) DEPEND(in, a[0], b[0]) DEPEND(inout, result[0])
                         MAP(present, alloc: a[0:n], b[0:n], result[0:1]))
    for (idx_t i = 0; i < n; i++)
        result[0] += a[i] * b[i];
}
#else
/* Gather the T1 partial sums into the final dot result (one task). */
static void task_dot_finalize(const Tiling *tl, const real_t *part, real_t *result)
{
    const int NTB1 = tl->NTB1;
    OMP_TASK(firstprivate(NTB1) DEPEND_MULTI(in, (t1=0:NTB1), part[t1]) DEPEND(out, result[0]))
    {
        real_t s = (real_t) 0.0;
        for (int t1 = 0; t1 < NTB1; t1++) s += part[t1];
        result[0] = s;
    }
}
#endif

/* Ap = A*p  (CSR SpMV). Host: T1*T2 sub-block tasks, each reading all p blocks
 * (stencil SpMV gathers scattered p) and writing its Ap sub-block (token aptok).
 * Device: one target parallel-for, one thread per row. */
static void task_spmv(const idx_t *row_ptr, const idx_t *col_idx, const real_t *val,
                      idx_t nnz, const real_t *p, real_t *Ap, const Tiling *tl)
{
    const idx_t n = tl->n;
    (void) nnz; /* used only in the GPU map() clause */
#if USE_TARGET
    OMP_TARGET_LOOP_TASK(DEPEND(in, p[0]) DEPEND(out, Ap[0])
                         MAP(present, alloc: p[0:n], Ap[0:n], val[0:nnz], col_idx[0:nnz], row_ptr[0:n + 1]))
    for (idx_t i = 0; i < n; i++) {
        real_t sum = (real_t) 0.0;
        for (idx_t k = row_ptr[i]; k < row_ptr[i + 1]; k++)
            sum += val[k] * p[col_idx[k]];
        Ap[i] = sum;
    }
#else
    const idx_t BS = tl->BS, SBS = tl->SBS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    char       *aptok = tl->aptok;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t bbegin = (idx_t) t1 * BS;
        const idx_t bend   = KR_MIN(bbegin + BS, n);
        for (int t2 = 0; t2 < T2; t2++) {
            const idx_t begin = bbegin + (idx_t) t2 * SBS;
            const idx_t end   = KR_MIN(begin + SBS, bend);
            OMP_TASK(firstprivate(begin, end) DEPEND_MULTI(in, (b=0:NTB1), p[b * BS]) DEPEND(out, aptok[t1 * T2 + t2]))
            {
                for (idx_t i = begin; i < end; i++) {
                    real_t sum = (real_t) 0.0;
                    for (idx_t k = row_ptr[i]; k < row_ptr[i + 1]; k++)
                        sum += val[k] * p[col_idx[k]];
                    Ap[i] = sum;
                }
            }
        }
    }
#endif
}

/* y = x  (copy). Vector op: shared loop nest, single-element block tokens. */
static void task_copy(const Tiling *tl, const real_t *x, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, x[0]) DEPEND(out, y[0]) MAP(present, alloc: x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n) DEPEND(in, x[blk]) DEPEND(out, y[blk]))
        {
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = x[i];
        }
    }
}

/* y = d .* x  (elementwise product; the Jacobi apply z = inv_diag .* r). */
static void task_vmul(const Tiling *tl, const real_t *d, const real_t *x, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, d[0], x[0]) DEPEND(out, y[0]) MAP(present, alloc: d[0:n], x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n) DEPEND(in, d[blk], x[blk]) DEPEND(out, y[blk]))
        {
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = d[i] * x[i];
        }
    }
}

/* y = y + sign*s*x  (axpy; both operands are T1-tiled vectors). */
static void task_axpy(const Tiling *tl, const real_t *s, real_t sign, const real_t *x, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, s[0], x[0]) DEPEND(inout, y[0]) MAP(present, alloc: s[0:1], x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n, sign) DEPEND(in, s[0], x[blk]) DEPEND(inout, y[blk]))
        {
            const real_t c = sign * s[0];
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] += c * x[i];
        }
    }
}

/* y = y + sign*s*Ap  (axpy consuming Ap; the r -= alpha*Ap update). Each T1
 * block waits on all T2 SpMV sub-block tokens of that block via an iterator. */
static void task_axpy_Ap(const Tiling *tl, const real_t *s, real_t sign, const real_t *Ap, real_t *y)
{
    const idx_t n = tl->n;
#if USE_TARGET
    OMP_TARGET_LOOP_TASK(DEPEND(in, s[0], Ap[0]) DEPEND(inout, y[0]) MAP(present, alloc: s[0:1], Ap[0:n], y[0:n]))
    for (idx_t i = 0; i < n; i++) y[i] += sign * s[0] * Ap[i];
#else
    const idx_t BS = tl->BS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    char       *aptok = tl->aptok;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        OMP_TASK(firstprivate(begin, end, sign) DEPEND(in, s[0]) DEPEND_MULTI(in, (t2=0:T2), aptok[t1 * T2 + t2]) DEPEND(inout, y[begin]))
        {
            const real_t c = sign * s[0];
            for (idx_t i = begin; i < end; i++) y[i] += c * Ap[i];
        }
    }
#endif
}

/* y = x + s*y  (axpby with the scale on y; the direction update p = z + beta*p). */
static void task_xpby(const Tiling *tl, const real_t *x, const real_t *s, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, x[0], s[0]) DEPEND(inout, y[0]) MAP(present, alloc: x[0:n], s[0:1], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(firstprivate(blk, bs, n) DEPEND(in, x[blk], s[0]) DEPEND(inout, y[blk]))
        {
            const real_t c = s[0];
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = x[i] + c * y[i];
        }
    }
}

/* result = <a,b>  (both operands are T1-tiled vectors). */
static void task_dot(const Tiling *tl, const real_t *a, const real_t *b, real_t *part, real_t *result)
{
    const idx_t n = tl->n;
#if USE_TARGET
    (void) part;
    gpu_dot(n, a, b, result);
#else
    const idx_t BS = tl->BS;
    const int   NTB1 = tl->NTB1;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        OMP_TASK(firstprivate(begin, end, t1) DEPEND(in, a[begin], b[begin]) DEPEND(out, part[t1]))
        {
            real_t s = (real_t) 0.0;
            for (idx_t i = begin; i < end; i++) s += a[i] * b[i];
            part[t1] = s;
        }
    }
    task_dot_finalize(tl, part, result);
#endif
}

/* result = <p,Ap>  (Ap operand; each T1 block waits on its T2 sub-blocks). */
static void task_dot_Ap(const Tiling *tl, const real_t *p, const real_t *Ap, real_t *part, real_t *result)
{
    const idx_t n = tl->n;
#if USE_TARGET
    (void) part;
    gpu_dot(n, p, Ap, result);
#else
    const idx_t BS = tl->BS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    char       *aptok = tl->aptok;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        OMP_TASK(firstprivate(begin, end, t1) DEPEND(in, p[begin]) DEPEND_MULTI(in, (t2=0:T2), aptok[t1 * T2 + t2]) DEPEND(out, part[t1]))
        {
            real_t s = (real_t) 0.0;
            for (idx_t i = begin; i < end; i++) s += p[i] * Ap[i];
            part[t1] = s;
        }
    }
    task_dot_finalize(tl, part, result);
#endif
}

/* c = a / b  (tiny scalar task). */
static void task_scalar_div(const real_t *a, const real_t *b, real_t *c)
{
    OMP_TARGET_TASK(DEPEND(in, a[0], b[0]) DEPEND(out, c[0]) MAP(present, alloc: a[0:1], b[0:1], c[0:1]))
    {
        c[0] = a[0] / b[0];
    }
}

/* b = a  (tiny scalar copy). */
static void task_scalar_copy(const real_t *a, real_t *b)
{
    OMP_TARGET_TASK(DEPEND(in, a[0]) DEPEND(out, b[0]) MAP(present, alloc: a[0:1], b[0:1]))
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

    Tiling tl;
    tiling_init(&tl, n, T1, T2);

    /* Device-mapped working vectors (pinned on GPU builds). */
    real_t *r   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *p   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *Ap  = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *z   = (real_t *) kr_alloc((size_t) n * sizeof(real_t));
    real_t *inv = (real_t *) kr_alloc((size_t) n * sizeof(real_t)); /* 1/diag(A) */

    /* Device-mapped length-1 scalar buffers. */
    real_t *gamma = (real_t *) kr_alloc(sizeof(real_t));
    real_t *g_new = (real_t *) kr_alloc(sizeof(real_t));
    real_t *pAp   = (real_t *) kr_alloc(sizeof(real_t));
    real_t *alpha = (real_t *) kr_alloc(sizeof(real_t));
    real_t *beta  = (real_t *) kr_alloc(sizeof(real_t));

    /* Host-only partial sums for the CPU dot reduction (unused on GPU). */
    real_t *part1 = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t)); /* <p,Ap> */
    real_t *part2 = (real_t *) malloc((size_t) tl.NTB1 * sizeof(real_t)); /* <r,z>  */

    /* Host initialization: inv_diag = 1/diag(A), x = 0, r = b (since x0 = 0). */
    spmat_extract_diagonal(A, inv);
    for (idx_t i = 0; i < n; i++) inv[i] = (real_t) 1.0 / inv[i];
    for (idx_t i = 0; i < n; i++) { x[i] = (real_t) 0.0; r[i] = b[i]; }

    OMP_TARGET_ENTER_DATA(map(to: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], x[0:n], r[0:n])
                          map(alloc: p[0:n], Ap[0:n], z[0:n], gamma[0:1], g_new[0:1], pAp[0:1], alpha[0:1], beta[0:1]))

    const double t0 = omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {
        /* One-time setup: z = M^-1 r; p = z; gamma = <r,z>. A taskwait here is
         * fine -- it is a one-time synchronization, not per-iteration -- and it
         * makes z/p/gamma ready before the first taskgraph record. */
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
                task_spmv(row_ptr, col_idx, val, nnz, p, Ap, &tl); /* Ap  = A*p       */
                task_dot_Ap(&tl, p, Ap, part1, pAp);               /* pAp = <p,Ap>    */
                task_scalar_div(gamma, pAp, alpha);                /* alpha = g/pAp   */
                task_axpy(&tl, alpha, (real_t) +1.0, p, x);        /* x  += alpha*p   */
                task_axpy_Ap(&tl, alpha, (real_t) -1.0, Ap, r);    /* r  -= alpha*Ap  */
                task_vmul(&tl, inv, r, z);                         /* z   = M^-1 r    */
                task_dot(&tl, r, z, part2, g_new);                 /* g_new = <r,z>   */
                task_scalar_div(g_new, gamma, beta);               /* beta = gn/g     */
                task_xpby(&tl, z, beta, p);                        /* p = z + beta*p  */
                task_scalar_copy(g_new, gamma);                    /* gamma = g_new   */
            }
            TASKGRAPH_END

            if (print_dbg) {
                const double spawn_ms = (omp_get_wtime() - spawn0) * 1000.0;
                /* Async D2H of the residual scalar, ordered via the g_new token
                 * (expands to nothing on the host backend). */
                OMP_TARGET_UPDATE(from(g_new[0:1]) nowait DEPEND(inout, g_new[0]))
                /* Debug print as a task synchronized by dependencies (no taskwait):
                 * it runs after the iteration's last task (gamma) and records the
                 * completion time to derive each iteration's execution time. The
                 * WAR on g_new forces the next iteration's dot to wait for it, so
                 * the debug tasks stay ordered and prev_ts is race-free. */
                OMP_HOST_TASK(firstprivate(it, spawn_ms) shared(prev_ts) DEPEND(in, g_new[0], gamma[0]))
                {
                    const double now     = omp_get_wtime();
                    const double exec_ms = (now - prev_ts) * 1000.0;
                    prev_ts = now;
                    printf("  iter %4d   residual = %.6e   spawn = %8.3f ms   exec = %8.3f ms\n",
                           it, sqrt((double) g_new[0]), spawn_ms, exec_ms);
                }
            }
        }
        /* One-time end-of-solve synchronization before reading/freeing buffers. */
        #pragma omp taskwait
    }

    const double t1 = omp_get_wtime();

    OMP_TARGET_EXIT_DATA(map(from: x[0:n])
                         map(release: row_ptr[0:n + 1], col_idx[0:nnz], val[0:nnz], inv[0:n], r[0:n],
                                      p[0:n], Ap[0:n], z[0:n], gamma[0:1], g_new[0:1], pAp[0:1], alpha[0:1], beta[0:1]))

    kr_free(r); kr_free(p); kr_free(Ap); kr_free(z); kr_free(inv);
    kr_free(gamma); kr_free(g_new); kr_free(pAp); kr_free(alpha); kr_free(beta);
    free(part1); free(part2);
    tiling_free(&tl);
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

    printf("Krylov CG (Jacobi-preconditioned)\n");
    printf("  backend    : %s\n", USE_TARGET ? "GPU (omp target, pinned host mem)" : "CPU (omp task, malloc)");
    printf("  taskgraph  : %s\n", USE_TASKGRAPH ? "on" : "off");
    printf("  grid       : %d x %d x %d  (n = %d, nnz = %d, %d-pt stencil)\n",
           N, N, N, A.n, A.nnz, stencil);
    printf("  iterations : %d\n", max_iter);
    printf("  tasks      : T1 = %d (vectors), T2 = %d (SpMV sub-blocks)\n", T1, T2);
    printf("  omp threads: %d\n", omp_get_max_threads());

    real_t *x = (real_t *) kr_alloc((size_t) A.n * sizeof(real_t));

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

    free(Ax); kr_free(x); free(b); free(xexact);
    spmat_free(&A);
    return 0;
}
