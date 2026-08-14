/*
 * kernels.cpp - implementation of the shared tasked/offloaded kernels.
 *
 * Data-sharing is explicit everywhere: host tasks use default(none) and list
 * every referenced variable (firstprivate for captured pointers/scalars; the
 * pointed-to memory is shared through the firstprivate pointer copy). Device
 * target constructs list scalars firstprivate and provide the data through
 * map(present:). See DEFAULT_NONE in common/tasking.h.
 */
#include "kernels.h"
#include "tasking.h"

#include <stdlib.h>

#define KR_MIN(a, b) ((a) < (b) ? (a) : (b))

void tiling_init(Tiling *tl, idx_t n, int T1, int T2)
{
    tl->n    = n;
    tl->T1   = T1;
    tl->T2   = T2;
    tl->BS   = (n + T1 - 1) / T1;
    tl->NTB1 = (int) ((n + tl->BS - 1) / tl->BS);
    tl->SBS  = (tl->BS + T2 - 1) / T2;
}

char *spmv_tokens_alloc(const Tiling *tl)
{
#if USE_TARGET
    (void) tl;
    return NULL; /* GPU: one task per op, no per-chunk tokens needed */
#else
    return (char *) malloc((size_t) tl->NTB1 * (size_t) tl->T2);
#endif
}

void spmv_tokens_free(char *tok)
{
    free(tok); /* free(NULL) is a no-op on GPU builds */
}

/* ==========================================================================
 * Internal reduction helpers.
 * ========================================================================== */
#if USE_TARGET
/* result = <a,b> on the device: zero the scalar, then a target reduction into
 * it (an OpenMP 5.x array-element reduction on the mapped scalar result[0]). */
static void gpu_dot(idx_t n, const real_t *a, const real_t *b, real_t *result)
{
    OMP_TARGET_TASK(DEFAULT_NONE DEPEND(out, result[0]) MAP(present: result[0:1]))
    {
        result[0] = (real_t) 0.0;
    }
    OMP_TARGET_LOOP_TASK(reduction(+: result[0]) DEPEND(in, a[0], b[0]) DEPEND(inout, result[0])
                         MAP(present: a[0:n], b[0:n], result[0:1]))
    for (idx_t i = 0; i < n; i++)
        result[0] += a[i] * b[i];
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

/* ==========================================================================
 * SpMV.
 * ========================================================================== */
void task_spmv(const idx_t *row_ptr, const idx_t *col_idx, const real_t *val,
               idx_t nnz, const real_t *x, real_t *y, char *y_tok, const Tiling *tl)
{
    const idx_t n = tl->n;
    (void) nnz;    /* used only in the GPU map() clause */
    (void) y_tok;  /* used only in the CPU per-chunk depend */
#if USE_TARGET
    OMP_TARGET_LOOP_TASK(DEPEND(in, x[0]) DEPEND(out, y[0])
                         MAP(present: x[0:n], y[0:n], val[0:nnz], col_idx[0:nnz], row_ptr[0:n + 1]))
    for (idx_t i = 0; i < n; i++) {
        real_t sum = (real_t) 0.0;
        for (idx_t k = row_ptr[i]; k < row_ptr[i + 1]; k++)
            sum += val[k] * x[col_idx[k]];
        y[i] = sum;
    }
#else
    const idx_t BS = tl->BS, SBS = tl->SBS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t bbegin = (idx_t) t1 * BS;
        const idx_t bend   = KR_MIN(bbegin + BS, n);
        for (int t2 = 0; t2 < T2; t2++) {
            const idx_t begin = bbegin + (idx_t) t2 * SBS;
            const idx_t end   = KR_MIN(begin + SBS, bend);
            OMP_TASK(DEFAULT_NONE firstprivate(row_ptr, col_idx, val, x, y, y_tok, begin, end, t1, t2, BS, T2, NTB1)
                     DEPEND_MULTI(in, (b=0:NTB1), x[b * BS]) DEPEND(out, y_tok[t1 * T2 + t2]))
            {
                for (idx_t i = begin; i < end; i++) {
                    real_t sum = (real_t) 0.0;
                    for (idx_t k = row_ptr[i]; k < row_ptr[i + 1]; k++)
                        sum += val[k] * x[col_idx[k]];
                    y[i] = sum;
                }
            }
        }
    }
#endif
}

/* ==========================================================================
 * Vector ops (all operands T1-tiled).
 * ========================================================================== */
void task_copy(const Tiling *tl, const real_t *x, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, x[0]) DEPEND(out, y[0]) MAP(present: x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(DEFAULT_NONE firstprivate(x, y, blk, bs, n) DEPEND(in, x[blk]) DEPEND(out, y[blk]))
        {
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = x[i];
        }
    }
}

void task_vmul(const Tiling *tl, const real_t *d, const real_t *x, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, d[0], x[0]) DEPEND(out, y[0]) MAP(present: d[0:n], x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(DEFAULT_NONE firstprivate(d, x, y, blk, bs, n) DEPEND(in, d[blk], x[blk]) DEPEND(out, y[blk]))
        {
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = d[i] * x[i];
        }
    }
}

void task_scal(const Tiling *tl, const real_t *s, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, s[0]) DEPEND(inout, y[0]) MAP(present: s[0:1], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(DEFAULT_NONE firstprivate(s, y, blk, bs, n) DEPEND(in, s[0]) DEPEND(inout, y[blk]))
        {
            const real_t c = s[0];
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = c * y[i];
        }
    }
}

void task_scal_copy(const Tiling *tl, const real_t *s, const real_t *x, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, s[0], x[0]) DEPEND(out, y[0]) MAP(present: s[0:1], x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(DEFAULT_NONE firstprivate(s, x, y, blk, bs, n) DEPEND(in, s[0], x[blk]) DEPEND(out, y[blk]))
        {
            const real_t c = s[0];
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = c * x[i];
        }
    }
}

void task_axpy(const Tiling *tl, const real_t *s, real_t sign, const real_t *x, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, s[0], x[0]) DEPEND(inout, y[0]) MAP(present: s[0:1], x[0:n], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(DEFAULT_NONE firstprivate(s, x, y, sign, blk, bs, n) DEPEND(in, s[0], x[blk]) DEPEND(inout, y[blk]))
        {
            const real_t c = sign * s[0];
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] += c * x[i];
        }
    }
}

void task_xpby(const Tiling *tl, const real_t *x, const real_t *s, real_t *y)
{
    const idx_t n  = tl->n;
    const idx_t bs = USE_TARGET ? (idx_t) 1 : tl->BS;
    OMP_TARGET_LOOP_TASK(DEPEND(in, x[0], s[0]) DEPEND(inout, y[0]) MAP(present: x[0:n], s[0:1], y[0:n]))
    for (idx_t blk = 0; blk < n; blk += bs) {
        OMP_TASK(DEFAULT_NONE firstprivate(x, s, y, blk, bs, n) DEPEND(in, x[blk], s[0]) DEPEND(inout, y[blk]))
        {
            const real_t c = s[0];
            const idx_t end = KR_MIN(blk + bs, n);
            for (idx_t i = blk; i < end; i++) y[i] = x[i] + c * y[i];
        }
    }
}

void task_dot(const Tiling *tl, const real_t *a, const real_t *b, real_t *part, real_t *result)
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
 * Variants consuming a fresh SpMV-output vector `ys` (tokens ys_tok). On the
 * host, each T1 block waits on all T2 SpMV sub-block tokens of that block via a
 * depend iterator; on the device it is a single target task.
 * ========================================================================== */
void task_copy_spmv(const Tiling *tl, const real_t *ys, char *ys_tok, real_t *y)
{
    const idx_t n = tl->n;
    (void) ys_tok;
#if USE_TARGET
    OMP_TARGET_LOOP_TASK(DEPEND(in, ys[0]) DEPEND(out, y[0]) MAP(present: ys[0:n], y[0:n]))
    for (idx_t i = 0; i < n; i++) y[i] = ys[i];
#else
    const idx_t BS = tl->BS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        OMP_TASK(DEFAULT_NONE firstprivate(ys, y, ys_tok, begin, end, t1, T2)
                 DEPEND_MULTI(in, (t2=0:T2), ys_tok[t1 * T2 + t2]) DEPEND(out, y[begin]))
        {
            for (idx_t i = begin; i < end; i++) y[i] = ys[i];
        }
    }
#endif
}

void task_vmul_spmv(const Tiling *tl, const real_t *d, const real_t *ys, char *ys_tok, real_t *y)
{
    const idx_t n = tl->n;
    (void) ys_tok;
#if USE_TARGET
    OMP_TARGET_LOOP_TASK(DEPEND(in, d[0], ys[0]) DEPEND(out, y[0]) MAP(present: d[0:n], ys[0:n], y[0:n]))
    for (idx_t i = 0; i < n; i++) y[i] = d[i] * ys[i];
#else
    const idx_t BS = tl->BS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        OMP_TASK(DEFAULT_NONE firstprivate(d, ys, y, ys_tok, begin, end, t1, T2)
                 DEPEND(in, d[begin]) DEPEND_MULTI(in, (t2=0:T2), ys_tok[t1 * T2 + t2]) DEPEND(out, y[begin]))
        {
            for (idx_t i = begin; i < end; i++) y[i] = d[i] * ys[i];
        }
    }
#endif
}

void task_axpy_spmv(const Tiling *tl, const real_t *s, real_t sign, const real_t *ys, char *ys_tok, real_t *y)
{
    const idx_t n = tl->n;
    (void) ys_tok;
#if USE_TARGET
    OMP_TARGET_LOOP_TASK(DEPEND(in, s[0], ys[0]) DEPEND(inout, y[0]) MAP(present: s[0:1], ys[0:n], y[0:n]))
    for (idx_t i = 0; i < n; i++) y[i] += sign * s[0] * ys[i];
#else
    const idx_t BS = tl->BS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        OMP_TASK(DEFAULT_NONE firstprivate(s, ys, y, sign, ys_tok, begin, end, t1, T2)
                 DEPEND(in, s[0]) DEPEND_MULTI(in, (t2=0:T2), ys_tok[t1 * T2 + t2]) DEPEND(inout, y[begin]))
        {
            const real_t c = sign * s[0];
            for (idx_t i = begin; i < end; i++) y[i] += c * ys[i];
        }
    }
#endif
}

void task_xpby_spmv(const Tiling *tl, const real_t *ys, char *ys_tok, const real_t *s, real_t *y)
{
    const idx_t n = tl->n;
    (void) ys_tok;
#if USE_TARGET
    OMP_TARGET_LOOP_TASK(DEPEND(in, ys[0], s[0]) DEPEND(inout, y[0]) MAP(present: ys[0:n], s[0:1], y[0:n]))
    for (idx_t i = 0; i < n; i++) y[i] = ys[i] + s[0] * y[i];
#else
    const idx_t BS = tl->BS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        OMP_TASK(DEFAULT_NONE firstprivate(ys, s, y, ys_tok, begin, end, t1, T2)
                 DEPEND(in, s[0]) DEPEND_MULTI(in, (t2=0:T2), ys_tok[t1 * T2 + t2]) DEPEND(inout, y[begin]))
        {
            const real_t c = s[0];
            for (idx_t i = begin; i < end; i++) y[i] = ys[i] + c * y[i];
        }
    }
#endif
}

void task_dot_spmv(const Tiling *tl, const real_t *a, const real_t *ys, char *ys_tok, real_t *part, real_t *result)
{
    const idx_t n = tl->n;
    (void) ys_tok;
#if USE_TARGET
    (void) part;
    gpu_dot(n, a, ys, result);
#else
    const idx_t BS = tl->BS;
    const int   NTB1 = tl->NTB1, T2 = tl->T2;
    for (int t1 = 0; t1 < NTB1; t1++) {
        const idx_t begin = (idx_t) t1 * BS;
        const idx_t end   = KR_MIN(begin + BS, n);
        OMP_TASK(DEFAULT_NONE firstprivate(a, ys, part, ys_tok, begin, end, t1, T2)
                 DEPEND(in, a[begin]) DEPEND_MULTI(in, (t2=0:T2), ys_tok[t1 * T2 + t2]) DEPEND(out, part[t1]))
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
    OMP_TARGET_TASK(DEFAULT_NONE firstprivate(a, b, c), DEPEND(in, a[0], b[0]) DEPEND(out, c[0]) MAP(present: a[0:1], b[0:1], c[0:1]))
    {
        c[0] = a[0] / b[0];
    }
}

void task_scalar_copy(const real_t *a, real_t *b)
{
    OMP_TARGET_TASK(DEFAULT_NONE firstprivate(a, b) DEPEND(in, a[0]) DEPEND(out, b[0]) MAP(present: a[0:1], b[0:1]))
    {
        b[0] = a[0];
    }
}
