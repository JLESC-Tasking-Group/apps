/*
 * spmat.cpp - implementation of the sparse-matrix library (see spmat.h).
 */
#include "spmat.h"

#include <stdio.h>
#include <stdlib.h>

/* Fatal allocation helper: never returns NULL. */
static void *xmalloc(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "spmat: out of memory (%zu bytes)\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

void spmat_generate_stencil(SpMatrix *A, idx_t nx, idx_t ny, idx_t nz,
                            int stencil, real_t **b, real_t **xexact)
{
    const idx_t  n        = nx * ny * nz;
    const int    use7     = (stencil == SPMAT_STENCIL_7PT);
    const real_t diag_val = (real_t) stencil;      /* 7 or 27 on the diagonal */
    const idx_t  max_nnz  = (idx_t) stencil * n;   /* upper bound (interior rows) */

    A->n       = n;
    A->row_ptr = (idx_t  *) xmalloc((size_t)(n + 1) * sizeof(idx_t));
    A->col_idx = (idx_t  *) xmalloc((size_t) max_nnz * sizeof(idx_t));
    A->val     = (real_t *) xmalloc((size_t) max_nnz * sizeof(real_t));

    real_t *bb = b      ? (real_t *) xmalloc((size_t) n * sizeof(real_t)) : NULL;
    real_t *xx = xexact ? (real_t *) xmalloc((size_t) n * sizeof(real_t)) : NULL;

    idx_t k = 0;                 /* running nonzero counter */
    A->row_ptr[0] = 0;

    for (idx_t iz = 0; iz < nz; iz++) {
        for (idx_t iy = 0; iy < ny; iy++) {
            for (idx_t ix = 0; ix < nx; ix++) {
                const idx_t row = (iz * ny + iy) * nx + ix;
                idx_t noffdiag = 0;

                /* Scan the 3x3x3 neighborhood; skip out-of-grid neighbors. */
                for (int sz = -1; sz <= 1; sz++) {
                    const idx_t jz = iz + sz;
                    if (jz < 0 || jz >= nz) continue;
                    for (int sy = -1; sy <= 1; sy++) {
                        const idx_t jy = iy + sy;
                        if (jy < 0 || jy >= ny) continue;
                        for (int sx = -1; sx <= 1; sx++) {
                            const idx_t jx = ix + sx;
                            if (jx < 0 || jx >= nx) continue;
                            /* 7-point keeps only axis neighbors (|offset| <= 1). */
                            if (use7 && (sx * sx + sy * sy + sz * sz > 1)) continue;

                            const idx_t col = (jz * ny + jy) * nx + jx;
                            A->col_idx[k] = col;
                            if (col == row) {
                                A->val[k] = diag_val;
                            } else {
                                A->val[k] = (real_t) -1.0;
                                noffdiag++;
                            }
                            k++;
                        }
                    }
                }

                A->row_ptr[row + 1] = k;
                if (bb) bb[row] = diag_val - (real_t) noffdiag; /* = row sum of A */
                if (xx) xx[row] = (real_t) 1.0;                 /* exact solution */
            }
        }
    }

    A->nnz = k;
    if (b)      *b      = bb;
    if (xexact) *xexact = xx;
}

void spmat_spmv(const SpMatrix *A, const real_t *x, real_t *y)
{
    for (idx_t i = 0; i < A->n; i++) {
        real_t sum = (real_t) 0.0;
        for (idx_t k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++)
            sum += A->val[k] * x[A->col_idx[k]];
        y[i] = sum;
    }
}

void spmat_extract_diagonal(const SpMatrix *A, real_t *diag)
{
    for (idx_t i = 0; i < A->n; i++) {
        diag[i] = (real_t) 0.0;
        for (idx_t k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
            if (A->col_idx[k] == i) { diag[i] = A->val[k]; break; }
        }
    }
}

void spmat_free(SpMatrix *A)
{
    free(A->row_ptr);
    free(A->col_idx);
    free(A->val);
    A->row_ptr = NULL;
    A->col_idx = NULL;
    A->val     = NULL;
    A->n = A->nnz = 0;
}
