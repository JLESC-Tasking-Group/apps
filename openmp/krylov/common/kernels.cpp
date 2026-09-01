/*
 * kernels.cpp - implementation of the shared tasked/offloaded kernels.
 *
 * Data-sharing is explicit everywhere: host tasks use default(none) and list
 * every referenced variable (firstprivate for captured pointers/scalars; the
 * pointed-to memory is shared through the firstprivate pointer copy). Device
 * target constructs provide the data through map(present:) and rely on the
 * implicit firstprivate of scalars/pointers (explicit firstprivate on a target
 * *loop* construct trips a clang codegen assertion, so it is omitted there).
 *
 * Both backends tile every operation identically (see OMP_TILE in tasking.h): a
 * length-n vector op is split into T1 tiles (the SpMV into T1*T2 sub-tiles), and
 * each tile becomes one host task (CPU) or one offloaded parallel-for over the
 * tile's sub-range (GPU). Task granularity therefore follows -t/-s on BOTH
 * backends; T1 = T2 = 1 yields one tile == one kernel per op (the coarse
 * single-task-per-op schedule).
 */
#include "kernels.h"
#include "tasking.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KR_MIN(a, b) ((a) < (b) ? (a) : (b))

int tiling_nsub(const Tiling *tl, int t1)
{
    const idx_t bbegin = (idx_t) t1 * tl->BS;
    const idx_t bend   = KR_MIN(bbegin + tl->BS, tl->n);
    const idx_t rows   = (bend > bbegin) ? (bend - bbegin) : 0;
    int ns = (int) ((rows + tl->SBS - 1) / tl->SBS);
    if (ns > tl->T2) ns = tl->T2;
    if (ns < 1)      ns = 1;   /* always emit one (possibly empty) sub-block */
    return ns;
}

/* Precompute, for every SpMV sub-block, the set of x blocks it actually reads.
 * Mirrors the reference HPCCG implementation (__HPC_sparsemv_deps_init): walk the
 * sub-block's rows, map each column index to its block start (col/BS*BS) and keep
 * the distinct ones. `stamp` marks the blocks already collected for the current
 * sub-block, so each sub-block costs O(its nnz) and the whole build is O(nnz). */
static void tiling_build_spmv_deps(Tiling *tl, const SpMatrix *A)
{
    const idx_t BS = tl->BS, SBS = tl->SBS, n = tl->n;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;

    /* These allocations must not fail silently: an empty dependency set would
     * make the SpMV tasks depend on nothing and silently produce wrong results. */
    tl->spmv_deps = (SpMVDeps *) calloc((size_t) NTB1 * T2, sizeof(SpMVDeps));
    int   *stamp  = (int *)   malloc((size_t) NTB1 * sizeof(int));
    idx_t *tmp    = (idx_t *) malloc((size_t) NTB1 * sizeof(idx_t));
    if (!tl->spmv_deps || !stamp || !tmp) {
        fprintf(stderr, "krylov: out of memory building the SpMV dependency table "
                        "(NTB1=%d, T2=%d)\n", NTB1, T2);
        exit(EXIT_FAILURE);
    }
    for (int b = 0; b < NTB1; b++) stamp[b] = -1;

    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t bbegin = (idx_t) t1 * BS;
        const idx_t bend   = KR_MIN(bbegin + BS, n);
        const int   ns     = tiling_nsub(tl, t1);
        for (int t2 = 0; t2 < ns; t2++) {
            const int   slot  = t1 * T2 + t2;
            const idx_t begin = bbegin + (idx_t) t2 * SBS;
            const idx_t end   = KR_MIN(begin + SBS, bend);

            int cnt = 0;
            for (idx_t i = begin; i < end; i++) {
                for (idx_t k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
                    const int b = (int) (A->col_idx[k] / BS);
                    if (b < 0 || b >= NTB1 || stamp[b] == slot) continue;
                    stamp[b] = slot;
                    tmp[cnt++] = (idx_t) b * BS;
                }
            }

            SpMVDeps *d = &tl->spmv_deps[slot];
            d->size = cnt;
            if (cnt) {
                d->indices = (idx_t *) malloc((size_t) cnt * sizeof(idx_t));
                if (!d->indices) {
                    fprintf(stderr, "krylov: out of memory building the SpMV "
                                    "dependency table (slot %d, %d blocks)\n", slot, cnt);
                    exit(EXIT_FAILURE);
                }
                memcpy(d->indices, tmp, (size_t) cnt * sizeof(idx_t));
            }
        }
    }
    free(stamp);
    free(tmp);
}

void tiling_init(Tiling *tl, const SpMatrix *A, int T1, int T2)
{
    const idx_t n = A->n;
    tl->n    = n;
    tl->T1   = T1;
    tl->T2   = T2;
    tl->BS   = (n + T1 - 1) / T1;
    tl->NTB1 = (int) ((n + tl->BS - 1) / tl->BS);
    tl->SBS  = (tl->BS + T2 - 1) / T2;
    tl->spmv_deps = NULL;
    tiling_build_spmv_deps(tl, A);
}

void tiling_fini(Tiling *tl)
{
    if (!tl->spmv_deps) return;
    for (int i = 0; i < tl->NTB1 * tl->T2; i++) free(tl->spmv_deps[i].indices);
    free(tl->spmv_deps);
    tl->spmv_deps = NULL;
}

/* ==========================================================================
 * SpMV.
 *
 * Each sub-block (t1, t2) is one task that:
 *   - reads exactly the x blocks it needs -- the precomputed tl->spmv_deps set,
 *     iterated with a depend iterator. Depending on all NTB1 blocks instead would
 *     make every SpMV sub-task a reader of every x chunk, so the next writer of x
 *     (e.g. CG's p = z + beta*p) would gain NTB1*T2 anti-dependence predecessors
 *     and the per-iteration graph would grow as O(NTB1^2 * T2).
 *   - writes its OWN output token y[begin] (begin = t1*BS + t2*SBS). The sub-blocks
 *     write disjoint rows and their tokens are disjoint, so they are independent
 *     without needing a shared `inoutset` set; consumers of the tile join over the
 *     tokens (see the *_spmv variants below).
 *
 * Only the tile's non-empty sub-blocks are emitted (tiling_nsub): with a ragged
 * last tile, t2*SBS can run past the tile, and the token y[begin] of such an empty
 * sub-block would be an out-of-range address.
 * ========================================================================== */
void task_spmv(const idx_t *row_ptr, const idx_t *col_idx, const real_t *val,
               idx_t nnz, const real_t *x, real_t *y, const Tiling *tl)
{
    const idx_t n = tl->n, BS = tl->BS, SBS = tl->SBS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    (void) nnz; /* used only in the GPU map() clause */
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t bbegin = (idx_t) t1 * BS;
        const idx_t bend   = KR_MIN(bbegin + BS, n);
        const int   ns     = tiling_nsub(tl, t1);
        for (int t2 = 0; t2 < ns; t2++) {
            const idx_t begin = bbegin + (idx_t) t2 * SBS;
            const idx_t end   = KR_MIN(begin + SBS, bend);
            const SpMVDeps *d = &tl->spmv_deps[t1 * T2 + t2];
            OMP_TILE(DEPEND_MULTI(in, (i=0:d->size), x[d->indices[i]]) DEPEND(out, y[begin]),
                     MAP(present: x[0:n], y[0:n], val[0:nnz], col_idx[0:nnz], row_ptr[0:n + 1]),
                     firstprivate(row_ptr, col_idx, val, x, y, begin, end))
            for (idx_t i = begin; i < end; i++) {
                real_t sum = (real_t) 0.0;
                for (idx_t k = row_ptr[i]; k < row_ptr[i + 1]; k++)
                    sum += val[k] * x[col_idx[k]];
                y[i] = sum;
            }
        }
    }
}

/* ==========================================================================
 * Vector ops (all operands T1-tiled).
 * ========================================================================== */
void task_copy(const Tiling *tl, const real_t *x, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS;
    for (idx_t blk = 0; blk < n; blk += BS) {
        const idx_t begin = blk, end = KR_MIN(blk + BS, n);
        OMP_TILE(DEPEND(in, x[begin]) DEPEND(out, y[begin]),
                 MAP(present: x[0:n], y[0:n]),
                 firstprivate(x, y, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] = x[i];
    }
}

void task_vmul(const Tiling *tl, const real_t *d, const real_t *x, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS;
    for (idx_t blk = 0; blk < n; blk += BS) {
        const idx_t begin = blk, end = KR_MIN(blk + BS, n);
        OMP_TILE(DEPEND(in, d[begin], x[begin]) DEPEND(out, y[begin]),
                 MAP(present: d[0:n], x[0:n], y[0:n]),
                 firstprivate(d, x, y, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] = d[i] * x[i];
    }
}

void task_scal(const Tiling *tl, const real_t *s, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS;
    for (idx_t blk = 0; blk < n; blk += BS) {
        const idx_t begin = blk, end = KR_MIN(blk + BS, n);
        OMP_TILE(DEPEND(in, s[0]) DEPEND(inout, y[begin]),
                 MAP(present: s[0:1], y[0:n]),
                 firstprivate(s, y, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] = s[0] * y[i];
    }
}

void task_scal_copy(const Tiling *tl, const real_t *s, const real_t *x, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS;
    for (idx_t blk = 0; blk < n; blk += BS) {
        const idx_t begin = blk, end = KR_MIN(blk + BS, n);
        OMP_TILE(DEPEND(in, s[0], x[begin]) DEPEND(out, y[begin]),
                 MAP(present: s[0:1], x[0:n], y[0:n]),
                 firstprivate(s, x, y, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] = s[0] * x[i];
    }
}

void task_axpy(const Tiling *tl, const real_t *s, real_t sign, const real_t *x, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS;
    for (idx_t blk = 0; blk < n; blk += BS) {
        const idx_t begin = blk, end = KR_MIN(blk + BS, n);
        OMP_TILE(DEPEND(in, s[0], x[begin]) DEPEND(inout, y[begin]),
                 MAP(present: s[0:1], x[0:n], y[0:n]),
                 firstprivate(s, x, y, sign, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] += sign * s[0] * x[i];
    }
}

void task_xpby(const Tiling *tl, const real_t *x, const real_t *s, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS;
    for (idx_t blk = 0; blk < n; blk += BS) {
        const idx_t begin = blk, end = KR_MIN(blk + BS, n);
        OMP_TILE(DEPEND(in, x[begin], s[0]) DEPEND(inout, y[begin]),
                 MAP(present: x[0:n], s[0:1], y[0:n]),
                 firstprivate(x, s, y, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] = x[i] + s[0] * y[i];
    }
}

/* ==========================================================================
 * Dot product: T1 partial reductions into part[] + a finalize sum.
 *   GPU: each partial is a teams reduction into the device-resident part[t1];
 *        the coarse NTB1 == 1 case is the single zero+reduce dot (identical
 *        schedule to -t 1).
 *   CPU: each partial is a plain task writing part[t1]; dot_finalize sums them.
 * ========================================================================== */
#if USE_TARGET
static void gpu_dot_zero(const Tiling *tl, real_t *part)
{
    const int NTB1 = tl->NTB1;
    OMP_TARGET_TASK(DEPEND_MULTI(out, (t1=0:NTB1), part[t1]) MAP(present: part[0:NTB1]))
    {
        for (int t1 = 0; t1 < NTB1; t1++) part[t1] = (real_t) 0.0;
    }
}

static void gpu_dot_finalize(const Tiling *tl, const real_t *part, real_t *result)
{
    const int NTB1 = tl->NTB1;
    OMP_TARGET_TASK(DEPEND_MULTI(in, (t1=0:NTB1), part[t1]) DEPEND(out, result[0])
                    MAP(present: part[0:NTB1], result[0:1]))
    {
        real_t s = (real_t) 0.0;
        for (int t1 = 0; t1 < NTB1; t1++) s += part[t1];
        result[0] = s;
    }
}
#else
/* Gather the T1 partial sums into the final dot result (one host task). */
static void dot_finalize(const Tiling *tl, const real_t *part, real_t *result)
{
    const int NTB1 = tl->NTB1;
    OMP_TASK(DEFAULT_NONE firstprivate(part, result, NTB1) DEPEND_MULTI(in, (t1=0:NTB1), part[t1]) DEPEND(out, result[0]))
    {
        real_t s = (real_t) 0.0;
        for (int t1 = 0; t1 < NTB1; t1++) s += part[t1];
        result[0] = s;
    }
}
#endif

void task_dot(const Tiling *tl, const real_t *a, const real_t *b, real_t *part, real_t *result)
{
    const idx_t n = tl->n, BS = tl->BS;
    const int   NTB1 = tl->NTB1;
#if USE_TARGET
    if (NTB1 == 1) { /* coarse: single zero + reduction (== -t 1) */
        OMP_TARGET_TASK(DEFAULT_NONE DEPEND(out, result[0]) MAP(present: result[0:1]))
        {
            result[0] = (real_t) 0.0;
        }
        OMP_TARGET_LOOP_TASK(reduction(+: result[0]) DEPEND(in, a[0], b[0]) DEPEND(inout, result[0])
                             MAP(present: a[0:n], b[0:n], result[0:1]))
        for (idx_t i = 0; i < n; i++) result[0] += a[i] * b[i];
        return;
    }
    gpu_dot_zero(tl, part);
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS, end = KR_MIN(begin + BS, n);
        OMP_TARGET_LOOP_TASK(reduction(+: part[t1]) DEPEND(in, a[begin], b[begin]) DEPEND(inout, part[t1])
                             MAP(present: a[0:n], b[0:n], part[t1:1]))
        for (idx_t i = begin; i < end; i++) part[t1] += a[i] * b[i];
    }
    gpu_dot_finalize(tl, part, result);
#else
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        OMP_TASK(DEFAULT_NONE firstprivate(a, b, part, begin, end, t1) DEPEND(in, a[begin], b[begin]) DEPEND(out, part[t1]))
        {
            real_t s = (real_t) 0.0;
            for (idx_t i = begin; i < end; i++) s += a[i] * b[i];
            part[t1] = s;
        }
    }
    dot_finalize(tl, part, result);
#endif
}

/* ==========================================================================
 * Variants consuming a fresh SpMV-output vector `ys`: each T1 block waits on the
 * output tokens of ALL non-empty sub-blocks of its tile -- ys[begin + k*SBS] for
 * k in [0, tiling_nsub(t1)) -- i.e. exactly the addresses task_spmv declared `out`.
 * ========================================================================== */
void task_copy_spmv(const Tiling *tl, const real_t *ys, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS, SBS = tl->SBS;
    const int   NTB1 = tl->NTB1;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS, end = KR_MIN(begin + BS, n);
        const int   ns = tiling_nsub(tl, t1);
        OMP_TILE(DEPEND_MULTI(in, (k=0:ns), ys[begin + k * SBS]) DEPEND(out, y[begin]),
                 MAP(present: ys[0:n], y[0:n]),
                 firstprivate(ys, y, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] = ys[i];
    }
}

void task_vmul_spmv(const Tiling *tl, const real_t *d, const real_t *ys, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS, SBS = tl->SBS;
    const int   NTB1 = tl->NTB1;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS, end = KR_MIN(begin + BS, n);
        const int   ns = tiling_nsub(tl, t1);
        OMP_TILE(DEPEND(in, d[begin]) DEPEND_MULTI(in, (k=0:ns), ys[begin + k * SBS]) DEPEND(out, y[begin]),
                 MAP(present: d[0:n], ys[0:n], y[0:n]),
                 firstprivate(d, ys, y, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] = d[i] * ys[i];
    }
}

void task_axpy_spmv(const Tiling *tl, const real_t *s, real_t sign, const real_t *ys, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS, SBS = tl->SBS;
    const int   NTB1 = tl->NTB1;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS, end = KR_MIN(begin + BS, n);
        const int   ns = tiling_nsub(tl, t1);
        OMP_TILE(DEPEND(in, s[0]) DEPEND_MULTI(in, (k=0:ns), ys[begin + k * SBS]) DEPEND(inout, y[begin]),
                 MAP(present: s[0:1], ys[0:n], y[0:n]),
                 firstprivate(s, ys, y, sign, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] += sign * s[0] * ys[i];
    }
}

void task_xpby_spmv(const Tiling *tl, const real_t *ys, const real_t *s, real_t *y)
{
    const idx_t n = tl->n, BS = tl->BS, SBS = tl->SBS;
    const int   NTB1 = tl->NTB1;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS, end = KR_MIN(begin + BS, n);
        const int   ns = tiling_nsub(tl, t1);
        OMP_TILE(DEPEND(in, s[0]) DEPEND_MULTI(in, (k=0:ns), ys[begin + k * SBS]) DEPEND(inout, y[begin]),
                 MAP(present: ys[0:n], s[0:1], y[0:n]),
                 firstprivate(ys, s, y, begin, end))
        for (idx_t i = begin; i < end; i++) y[i] = ys[i] + s[0] * y[i];
    }
}

void task_dot_spmv(const Tiling *tl, const real_t *a, const real_t *ys, real_t *part, real_t *result)
{
    const idx_t n = tl->n, BS = tl->BS, SBS = tl->SBS;
    const int   NTB1 = tl->NTB1;
#if USE_TARGET
    if (NTB1 == 1) { /* coarse: single zero + reduction (== -t 1) */
        const int ns = tiling_nsub(tl, 0);
        OMP_TARGET_TASK(DEFAULT_NONE DEPEND(out, result[0]) MAP(present: result[0:1]))
        {
            result[0] = (real_t) 0.0;
        }
        OMP_TARGET_LOOP_TASK(reduction(+: result[0]) DEPEND(in, a[0])
                             DEPEND_MULTI(in, (k=0:ns), ys[k * SBS]) DEPEND(inout, result[0])
                             MAP(present: a[0:n], ys[0:n], result[0:1]))
        for (idx_t i = 0; i < n; i++) result[0] += a[i] * ys[i];
        return;
    }
    gpu_dot_zero(tl, part);
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS, end = KR_MIN(begin + BS, n);
        const int   ns = tiling_nsub(tl, t1);
        OMP_TARGET_LOOP_TASK(reduction(+: part[t1]) DEPEND(in, a[begin])
                             DEPEND_MULTI(in, (k=0:ns), ys[begin + k * SBS]) DEPEND(inout, part[t1])
                             MAP(present: a[0:n], ys[0:n], part[t1:1]))
        for (idx_t i = begin; i < end; i++) part[t1] += a[i] * ys[i];
    }
    gpu_dot_finalize(tl, part, result);
#else
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        const int   ns    = tiling_nsub(tl, t1);
        OMP_TASK(DEFAULT_NONE firstprivate(a, ys, part, begin, end, t1)
                 DEPEND(in, a[begin]) DEPEND_MULTI(in, (k=0:ns), ys[begin + k * SBS])
                 DEPEND(out, part[t1]))
        {
            real_t s = (real_t) 0.0;
            for (idx_t i = begin; i < end; i++) s += a[i] * ys[i];
            part[t1] = s;
        }
    }
    dot_finalize(tl, part, result);
#endif
}

/* ==========================================================================
 * Scalar (length-1) ops.
 * ========================================================================== */
void task_scalar_div(const real_t *a, const real_t *b, real_t *c)
{
    OMP_TARGET_TASK(DEPEND(in, a[0], b[0]) DEPEND(out, c[0]) MAP(present: a[0:1], b[0:1], c[0:1]))
    {
        c[0] = a[0] / b[0];
    }
}

void task_scalar_copy(const real_t *a, real_t *b)
{
    OMP_TARGET_TASK(DEPEND(in, a[0]) DEPEND(out, b[0]) MAP(present: a[0:1], b[0:1]))
    {
        b[0] = a[0];
    }
}
