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
 *   A       : output matrix; its arrays are allocated here (free w/ spmat_free)
 *   stencil : SPMAT_STENCIL_7PT or SPMAT_STENCIL_27PT
 *   b       : if non-NULL, *b is allocated [n] and filled with the RHS
 *   xexact  : if non-NULL, *xexact is allocated [n] and filled with all ones
 */
void spmat_generate_stencil(SpMatrix *A, idx_t nx, idx_t ny, idx_t nz,
                            int stencil, real_t **b, real_t **xexact);

/* y = A*x  (serial reference SpMV; host only, used for verification). */
void spmat_spmv(const SpMatrix *A, const real_t *x, real_t *y);

/* diag[i] = A(i,i)  (extract the main diagonal; diag must hold n reals). */
void spmat_extract_diagonal(const SpMatrix *A, real_t *diag);

/* Release all memory owned by A and reset its fields. */
void spmat_free(SpMatrix *A);

#ifdef __cplusplus
}
#endif

#endif /* KRYLOV_SPMAT_H */
