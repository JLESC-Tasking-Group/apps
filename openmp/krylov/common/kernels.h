/*
 * kernels.h - shared tasked/offloaded BLAS-1 + SpMV kernels for the Krylov
 * solvers. Every solver (CG, CR, BiCGSTAB, MINRES, GMRES, ...) is built from
 * these building blocks, so they live here rather than in any one solver.
 *
 * Backend selection (CPU tasks vs GPU target tasks) and the taskgraph handling
 * are entirely inside common/tasking.h; a solver just calls these functions.
 *
 * Task granularity (host backend), following HPCCG:
 *   - a length-n vector op is split into T1 tasks (block size BS = ceil(n/T1));
 *   - the SpMV is split into T1*T2 sub-block tasks (SBS = ceil(BS/T2) rows each)
 *     so its output chunks carry independent dependencies.
 *
 * Dependencies use SINGLE-ELEMENT depend items at chunk-aligned addresses (never
 * array sections): OpenMP only matches depend items that are *identical or
 * disjoint*. Vector chunks are keyed by the data address at the block start
 * (v[t1*BS]). An SpMV output vector is keyed the same way, on its own addresses:
 * the T2 sub-blocks of one output tile all use `inoutset` on the tile's first
 * address (v[t1*BS]) so they run concurrently (they write disjoint rows), while
 * a consumer waits on the whole set with a single `in` on that address.
 */
#ifndef KRYLOV_KERNELS_H
#define KRYLOV_KERNELS_H

#include "spmat.h" /* real_t, idx_t */

/* Task tiling parameters (granularity only). */
typedef struct {
    idx_t n;
    int   T1;      /* number of tasks per vector op                          */
    int   T2;      /* number of SpMV sub-tasks per T1 block (T1*T2 total)     */
    idx_t BS;      /* rows per vector block   = ceil(n / T1)                  */
    idx_t SBS;     /* rows per SpMV sub-block = ceil(BS / T2)                 */
    int   NTB1;    /* actual number of vector blocks = ceil(n / BS) (<= T1)   */
} Tiling;

#ifdef __cplusplus
extern "C" {
#endif

void tiling_init(Tiling *tl, idx_t n, int T1, int T2);

/* y = A*x   (CSR SpMV). The T2 sub-blocks of each output tile are `inoutset` on
 * the tile's first address, so consumers depend (in) on y at that address. */
void task_spmv(const idx_t *row_ptr, const idx_t *col_idx, const real_t *val,
               idx_t nnz, const real_t *x, real_t *y, const Tiling *tl);

/* --- vector ops where every operand is a T1-tiled vector --- */
void task_copy     (const Tiling *tl, const real_t *x, real_t *y);                     /* y = x        */
void task_vmul     (const Tiling *tl, const real_t *d, const real_t *x, real_t *y);    /* y = d .* x   */
void task_scal     (const Tiling *tl, const real_t *s, real_t *y);                     /* y = s * y    */
void task_scal_copy(const Tiling *tl, const real_t *s, const real_t *x, real_t *y);    /* y = s * x    */
void task_axpy     (const Tiling *tl, const real_t *s, real_t sign, const real_t *x, real_t *y); /* y += sign*s*x */
void task_xpby     (const Tiling *tl, const real_t *x, const real_t *s, real_t *y);    /* y = x + s*y  */
void task_dot      (const Tiling *tl, const real_t *a, const real_t *b, real_t *part, real_t *result); /* result = <a,b> */

/* --- variants consuming a fresh SpMV-output vector `ys` --- */
void task_copy_spmv(const Tiling *tl, const real_t *ys, real_t *y);                  /* y = ys      */
void task_vmul_spmv(const Tiling *tl, const real_t *d, const real_t *ys, real_t *y); /* y = d .* ys */
void task_axpy_spmv(const Tiling *tl, const real_t *s, real_t sign, const real_t *ys, real_t *y); /* y += sign*s*ys */
void task_xpby_spmv(const Tiling *tl, const real_t *ys, const real_t *s, real_t *y); /* y = ys + s*y */
void task_dot_spmv (const Tiling *tl, const real_t *a, const real_t *ys, real_t *part, real_t *result); /* result = <a,ys> */

/* --- scalar (length-1 buffer) ops --- */
void task_scalar_div (const real_t *a, const real_t *b, real_t *c); /* c = a / b */
void task_scalar_copy(const real_t *a, real_t *b);                  /* b = a     */

#ifdef __cplusplus
}
#endif

#endif /* KRYLOV_KERNELS_H */
