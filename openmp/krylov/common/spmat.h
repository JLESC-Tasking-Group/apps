/*
 * spmat.h - a small sparse-matrix library for the Krylov solvers.
 *
 * Provides a clean CSR (Compressed Sparse Row) matrix type, a generator for
 * SPD 3-D finite-difference stencil matrices, a serial reference SpMV and a
 * diagonal extractor. It is deliberately backend-agnostic (plain host code, no
 * OpenMP) so every solver (CG now, CR / MINRES / ... later) can reuse it.
 */
#ifndef KRYLOV_SPMAT_H
#define KRYLOV_SPMAT_H

#include <stddef.h>

/* Index and value types shared by all solvers (change here to retune). */
typedef int    idx_t;
typedef double real_t;

/* Stencil kinds for the 3-D finite-difference matrix generator. */
enum { SPMAT_STENCIL_7PT = 7, SPMAT_STENCIL_27PT = 27 };

/*
 * Sparse matrix in Compressed Sparse Row (CSR) format (square, n x n).
 *
 *   row i occupies indices [row_ptr[i], row_ptr[i+1]) of col_idx[] / val[]:
 *       columns  col_idx[row_ptr[i] .. row_ptr[i+1])
 *       values   val    [row_ptr[i] .. row_ptr[i+1])
 *
 * Columns within a row are not necessarily sorted (SpMV does not require it).
 */
typedef struct {
    idx_t   n;        /* number of rows == number of columns             */
    idx_t   nnz;      /* number of stored nonzeros                        */
    idx_t  *row_ptr;  /* [n + 1] CSR row offsets (row_ptr[0]=0, [n]=nnz)  */
    idx_t  *col_idx;  /* [nnz]   column index of each nonzero             */
    real_t *val;      /* [nnz]   value of each nonzero                    */
} SpMatrix;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Build an SPD matrix from a 3-D finite-difference stencil on an nx x ny x nz
 * grid (n = nx*ny*nz unknowns).
 *
 * The diagonal equals the stencil size (7 or 27) and off-diagonals equal -1,
 * so A is symmetric and strictly diagonally dominant, hence SPD. The RHS b is
 * set to the row sums of A, which makes the all-ones vector the exact solution
 * (a convenient built-in correctness check for the solvers).
 *
 *   A       : output matrix; CSR arrays are device-mapped so they are allocated
 *             with kr_alloc (pinned on GPU builds) -- free them with spmat_free
 *   stencil : SPMAT_STENCIL_7PT or SPMAT_STENCIL_27PT
 *   b       : if non-NULL, *b is malloc'd [n] and filled with the RHS (host-only;
 *             free with free())
 *   xexact  : if non-NULL, *xexact is malloc'd [n], all ones (host-only; free())
 */
void spmat_generate_stencil(SpMatrix *A, idx_t nx, idx_t ny, idx_t nz,
                            int stencil, real_t **b, real_t **xexact);

/*
 * Build a NONSYMMETRIC convection-diffusion matrix on an nx x ny x nz grid:
 *   L = -Laplacian + I + conv * (central-difference first derivatives).
 * The diagonal is 7 and the six face-neighbor off-diagonals are -1 +/- conv/2
 * (asymmetric when conv != 0; conv = 0 reproduces the symmetric 7-point matrix).
 * It is strictly diagonally dominant for |conv| < 2. As with the stencil
 * generator, b is set to the row sums so the exact solution is all-ones -- this
 * holds for any A since (A*1)_i is the row sum, symmetric or not.
 *
 *   conv : convection strength (0 => symmetric); a good default is 1.0
 *   b, xexact : as in spmat_generate_stencil (host-only, free with free())
 */
void spmat_generate_convdiff(SpMatrix *A, idx_t nx, idx_t ny, idx_t nz,
                             real_t conv, real_t **b, real_t **xexact);

/* y = A*x  (serial reference SpMV; host only, used for verification). */
void spmat_spmv(const SpMatrix *A, const real_t *x, real_t *y);

/* diag[i] = A(i,i)  (extract the main diagonal; diag must hold n reals). */
void spmat_extract_diagonal(const SpMatrix *A, real_t *diag);

/* A(i,i) -= sigma for every row (shift the diagonal). Handy to turn the SPD
 * stencil matrix into a symmetric INDEFINITE one for MINRES. To keep the
 * all-ones exact solution, the caller should also do b[i] -= sigma. */
void spmat_shift_diagonal(SpMatrix *A, real_t sigma);

/* Release all memory owned by A and reset its fields. */
void spmat_free(SpMatrix *A);

#ifdef __cplusplus
}
#endif

#endif /* KRYLOV_SPMAT_H */
