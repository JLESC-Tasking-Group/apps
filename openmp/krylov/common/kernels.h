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
 * (v[t1*BS]). An SpMV *output* is finer-grained: each of the T2 sub-blocks owns
 * its OWN token at its own start address (v[t1*BS + t2*SBS]) and declares it
 * `out`, so the sub-blocks are independent because their tokens are disjoint --
 * not because of a shared concurrent set. A consumer of the whole tile joins
 * over those tokens with a depend iterator over the tile's non-empty sub-blocks
 * (tiling_nsub). This mirrors the reference HPCCG task implementation and keeps
 * every access strictly in/out (no `inoutset`).
 *
 * The SpMV *input* dependency is likewise exact rather than conservative: for
 * each sub-block we precompute, from the matrix column indices, the set of x
 * blocks it actually reads (Tiling::spmv_deps). Depending on all NTB1 blocks
 * instead would make every SpMV sub-task a reader of every x chunk, which turns
 * the next writer of x into a node with NTB1*T2 predecessors and inflates the
 * graph to O(NTB1^2 * T2) edges.
 */
#ifndef KRYLOV_KERNELS_H
#define KRYLOV_KERNELS_H

#include "spmat.h" /* real_t, idx_t, SpMatrix */

/* The x blocks that one SpMV sub-block reads (precomputed from col_idx). */
typedef struct {
    int    size;      /* number of distinct x blocks read                     */
    idx_t *indices;   /* [size] their block-start offsets (b * BS)            */
} SpMVDeps;

/* Task tiling parameters (granularity) + the precomputed SpMV dependencies. */
typedef struct {
    idx_t n;
    int   T1;      /* number of tasks per vector op                          */
    int   T2;      /* number of SpMV sub-tasks per T1 block (T1*T2 total)     */
    idx_t BS;      /* rows per vector block   = ceil(n / T1)                  */
    idx_t SBS;     /* rows per SpMV sub-block = ceil(BS / T2)                 */
    int   NTB1;    /* actual number of vector blocks = ceil(n / BS) (<= T1)   */
    SpMVDeps *spmv_deps; /* [NTB1*T2]; (t1*T2+t2) valid for t2 < tiling_nsub  */
} Tiling;

#ifdef __cplusplus
extern "C" {
#endif

/* Build the tiling and precompute the SpMV input dependencies from A. */
void tiling_init(Tiling *tl, const SpMatrix *A, int T1, int T2);

/* Release the precomputed dependencies (call once per solve, after the tasks). */
void tiling_fini(Tiling *tl);

/* Number of NON-EMPTY sub-blocks of tile t1: min(T2, ceil(tile_rows / SBS)).
 * A ragged last tile has fewer than T2 of them; producer and consumers must use
 * the same count so their token sets match exactly. */
int  tiling_nsub(const Tiling *tl, int t1);

/* y = A*x   (CSR SpMV). Each sub-block declares `out` on its own start address
 * y[t1*BS + t2*SBS]; consumers join over those tokens (see the *_spmv variants). */
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

/* --- variants consuming a fresh SpMV-output vector `ys` ---
 * Each waits on ALL sub-block tokens of its tile (iterator over tiling_nsub). */
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
