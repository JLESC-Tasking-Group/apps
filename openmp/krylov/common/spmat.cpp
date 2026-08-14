/*
 * spmat.cpp - implementation of the sparse-matrix library (see spmat.h).
 */
#include "spmat.h"
#include "kalloc.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

/* Fatal allocation helper for device-mapped arrays (pinned on GPU builds). */
static void *xalloc_dev(size_t bytes)
{
    void *p = kr_alloc(bytes);
    if (!p) {
        fprintf(stderr, "spmat: out of memory (%zu bytes)\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

/* Fatal allocation helper for host-only arrays. */
static void *xalloc_host(size_t bytes)
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

    /* CSR arrays are mapped onto the device -> allocate them pinned (kr_alloc). */
    A->n       = n;
    A->row_ptr = (idx_t  *) xalloc_dev((size_t)(n + 1) * sizeof(idx_t));
    A->col_idx = (idx_t  *) xalloc_dev((size_t) max_nnz * sizeof(idx_t));
    A->val     = (real_t *) xalloc_dev((size_t) max_nnz * sizeof(real_t));

    /* b / xexact are host-only (b seeds the residual, xexact verifies). */
    real_t *bb = b      ? (real_t *) xalloc_host((size_t) n * sizeof(real_t)) : NULL;
    real_t *xx = xexact ? (real_t *) xalloc_host((size_t) n * sizeof(real_t)) : NULL;

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

void spmat_generate_convdiff(SpMatrix *A, idx_t nx, idx_t ny, idx_t nz,
                             real_t conv, real_t **b, real_t **xexact)
{
    const idx_t  n        = nx * ny * nz;
    const real_t diag_val = (real_t) 7.0;         /* 6 (-Laplacian) + 1 (reaction) */
    const real_t half     = (real_t) 0.5 * conv;  /* central-difference convection  */
    const idx_t  max_nnz  = (idx_t) 7 * n;        /* 7-point stencil                */

    /* Six face-neighbor offsets and their (asymmetric) coefficients. */
    const int    off[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
    const real_t coef[6]   = {(real_t) -1.0 - half, (real_t) -1.0 + half,
                              (real_t) -1.0 - half, (real_t) -1.0 + half,
                              (real_t) -1.0 - half, (real_t) -1.0 + half};

    A->n       = n;
    A->row_ptr = (idx_t  *) xalloc_dev((size_t)(n + 1) * sizeof(idx_t));
    A->col_idx = (idx_t  *) xalloc_dev((size_t) max_nnz * sizeof(idx_t));
    A->val     = (real_t *) xalloc_dev((size_t) max_nnz * sizeof(real_t));

    real_t *bb = b      ? (real_t *) xalloc_host((size_t) n * sizeof(real_t)) : NULL;
    real_t *xx = xexact ? (real_t *) xalloc_host((size_t) n * sizeof(real_t)) : NULL;

    idx_t k = 0;
    A->row_ptr[0] = 0;

    for (idx_t iz = 0; iz < nz; iz++) {
        for (idx_t iy = 0; iy < ny; iy++) {
            for (idx_t ix = 0; ix < nx; ix++) {
                const idx_t row = (iz * ny + iy) * nx + ix;
                real_t rowsum = (real_t) 0.0;

                for (int m = 0; m < 6; m++) {
                    const idx_t jx = ix + off[m][0];
                    const idx_t jy = iy + off[m][1];
                    const idx_t jz = iz + off[m][2];
                    if (jx < 0 || jx >= nx || jy < 0 || jy >= ny || jz < 0 || jz >= nz) continue;
                    A->col_idx[k] = (jz * ny + jy) * nx + jx;
                    A->val[k]     = coef[m];
                    rowsum       += coef[m];
                    k++;
                }
                A->col_idx[k] = row;        /* diagonal (unsorted within row is fine) */
                A->val[k]     = diag_val;
                rowsum       += diag_val;
                k++;

                A->row_ptr[row + 1] = k;
                if (bb) bb[row] = rowsum;             /* = row sum of A */
                if (xx) xx[row] = (real_t) 1.0;       /* exact solution */
            }
        }
    }

    A->nnz = k;
    if (b)      *b      = bb;
    if (xexact) *xexact = xx;
}

/* ==========================================================================
 * Matrix Market (.mtx) importer (SuiteSparse Matrix Collection).
 * ========================================================================== */

/* Fatal error helper for the loader. */
static void mm_fatal(const char *path, const char *msg)
{
    fprintf(stderr, "spmat: Matrix Market load '%s': %s\n", path, msg);
    exit(EXIT_FAILURE);
}

/* Case-insensitive string equality (MatrixMarket keywords are case-insensitive). */
static int mm_ieq(const char *a, const char *b)
{
    while (*a && *b) {
        if (tolower((unsigned char) *a) != tolower((unsigned char) *b)) return 0;
        a++; b++;
    }
    return *a == '\0' && *b == '\0';
}

/* Read the next line that is neither a comment ('%') nor blank. 1 on success. */
static int mm_next_data_line(FILE *f, char *buf, size_t bufsz)
{
    while (fgets(buf, (int) bufsz, f)) {
        const char *p = buf;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '%' || *p == '\0' || *p == '\n' || *p == '\r') continue;
        return 1;
    }
    return 0;
}

void spmat_load_matrixmarket(SpMatrix *A, const char *path,
                             real_t **b, real_t **xexact)
{
    FILE *f = fopen(path, "r");
    if (!f) mm_fatal(path, "cannot open file");

    char line[512];

    /* --- banner: %%MatrixMarket matrix coordinate <field> <symmetry> --- */
    if (!fgets(line, sizeof line, f)) mm_fatal(path, "empty file (no header)");
    char banner[64] = "", obj[64] = "", fmt[64] = "", field[64] = "", sym[64] = "";
    if (sscanf(line, "%63s %63s %63s %63s %63s", banner, obj, fmt, field, sym) < 5 ||
        !mm_ieq(banner, "%%MatrixMarket"))
        mm_fatal(path, "not a MatrixMarket file (bad banner)");
    if (!mm_ieq(obj, "matrix"))
        mm_fatal(path, "object is not 'matrix'");
    if (!mm_ieq(fmt, "coordinate"))
        mm_fatal(path, "unsupported format (expected 'coordinate'; 'array'/dense not supported)");

    const int is_pattern = mm_ieq(field, "pattern");
    const int is_integer = mm_ieq(field, "integer");
    if (!is_pattern && !is_integer && !mm_ieq(field, "real"))
        mm_fatal(path, "unsupported field (expected real/integer/pattern; complex not supported)");

    const int is_sym  = mm_ieq(sym, "symmetric");
    const int is_skew = mm_ieq(sym, "skew-symmetric");
    const int is_gen  = mm_ieq(sym, "general");
    if (!is_sym && !is_skew && !is_gen)
        mm_fatal(path, "unsupported symmetry (expected general/symmetric/skew-symmetric; hermitian not supported)");
    const int mirror = is_sym || is_skew;

    /* --- size line: nrows ncols nnz --- */
    if (!mm_next_data_line(f, line, sizeof line)) mm_fatal(path, "missing size line");
    long nr = 0, nc = 0, snnz = 0;
    if (sscanf(line, "%ld %ld %ld", &nr, &nc, &snnz) != 3)
        mm_fatal(path, "malformed size line");
    if (nr != nc)              mm_fatal(path, "matrix is not square");
    if (nr <= 0 || snnz < 0)   mm_fatal(path, "invalid dimensions");
    const idx_t n = (idx_t) nr;

    /* --- read stored entries into a temporary COO (1-based -> 0-based) --- */
    const size_t tcap = (size_t)(snnz > 0 ? snnz : 1);
    long   *Ti = (long   *) xalloc_host(tcap * sizeof(long));
    long   *Tj = (long   *) xalloc_host(tcap * sizeof(long));
    real_t *Tv = (real_t *) xalloc_host(tcap * sizeof(real_t));

    long ndiag = 0;
    for (long e = 0; e < snnz; e++) {
        if (!mm_next_data_line(f, line, sizeof line))
            mm_fatal(path, "truncated file (fewer entries than declared in the size line)");
        long i = 0, j = 0; double v = 1.0;
        const int got = is_pattern ? sscanf(line, "%ld %ld", &i, &j)
                                   : sscanf(line, "%ld %ld %lg", &i, &j, &v);
        if (got < (is_pattern ? 2 : 3)) mm_fatal(path, "malformed entry line");
        if (i < 1 || i > nr || j < 1 || j > nc) mm_fatal(path, "entry index out of range");
        Ti[e] = i - 1; Tj[e] = j - 1; Tv[e] = (real_t) v;
        if (Ti[e] == Tj[e]) ndiag++;
    }
    fclose(f);

    /* --- expanded nnz (mirror the off-diagonal of a (skew-)symmetric matrix) --- */
    const long full = mirror ? (2 * snnz - ndiag) : snnz;
    if (full > 0x7fffffffL)   /* idx_t is 32-bit */
        mm_fatal(path, "too many nonzeros for 32-bit indexing (idx_t)");

    /* --- allocate CSR (device-mapped, like the generators) --- */
    A->n       = n;
    A->nnz     = (idx_t) full;
    A->row_ptr = (idx_t  *) xalloc_dev((size_t)(n + 1) * sizeof(idx_t));
    A->col_idx = (idx_t  *) xalloc_dev((size_t)(full > 0 ? full : 1) * sizeof(idx_t));
    A->val     = (real_t *) xalloc_dev((size_t)(full > 0 ? full : 1) * sizeof(real_t));

    /* --- COO -> CSR: count per row, prefix sum, scatter --- */
    for (idx_t i = 0; i <= n; i++) A->row_ptr[i] = 0;
    for (long e = 0; e < snnz; e++) {
        A->row_ptr[Ti[e] + 1]++;
        if (mirror && Ti[e] != Tj[e]) A->row_ptr[Tj[e] + 1]++;
    }
    for (idx_t i = 0; i < n; i++) A->row_ptr[i + 1] += A->row_ptr[i];

    idx_t *cur = (idx_t *) xalloc_host((size_t) n * sizeof(idx_t));
    for (idx_t i = 0; i < n; i++) cur[i] = A->row_ptr[i];
    for (long e = 0; e < snnz; e++) {
        const idx_t  i = (idx_t) Ti[e], j = (idx_t) Tj[e];
        const real_t v = Tv[e];
        const idx_t  p = cur[i]++;
        A->col_idx[p] = j; A->val[p] = v;
        if (mirror && i != j) {
            const idx_t q = cur[j]++;
            A->col_idx[q] = i; A->val[q] = is_skew ? -v : v;
        }
    }
    free(cur); free(Ti); free(Tj); free(Tv);

    /* --- b = A*1 (row sums) and xexact = 1, as for the generators --- */
    real_t *bb = b      ? (real_t *) xalloc_host((size_t) n * sizeof(real_t)) : NULL;
    real_t *xx = xexact ? (real_t *) xalloc_host((size_t) n * sizeof(real_t)) : NULL;
    long nzero_diag = 0;
    for (idx_t i = 0; i < n; i++) {
        real_t rowsum = (real_t) 0.0, diag = (real_t) 0.0;
        for (idx_t k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
            rowsum += A->val[k];
            if (A->col_idx[k] == i) diag = A->val[k];
        }
        if (bb) bb[i] = rowsum;
        if (xx) xx[i] = (real_t) 1.0;
        if (diag == (real_t) 0.0) nzero_diag++;
    }
    if (b)      *b      = bb;
    if (xexact) *xexact = xx;

    /* --- report the imported matrix --- */
    printf("Matrix Market import: %s\n", path);
    printf("  format      : coordinate %s %s\n",
           is_pattern ? "pattern" : (is_integer ? "integer" : "real"),
           is_gen ? "general" : (is_skew ? "skew-symmetric" : "symmetric"));
    printf("  nrows       : %ld\n", nr);
    printf("  ncols       : %ld\n", nc);
    printf("  stored nnz  : %ld\n", snnz);
    printf("  full nnz    : %ld%s\n", full, mirror ? "   (triangle expanded)" : "");
    printf("  nnz/row avg : %.1f\n", n ? (double) full / (double) n : 0.0);
    if (nzero_diag)
        printf("  WARNING     : %ld row(s) have a zero diagonal "
               "(Jacobi preconditioner will divide by zero)\n", nzero_diag);
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

void spmat_shift_diagonal(SpMatrix *A, real_t sigma)
{
    for (idx_t i = 0; i < A->n; i++)
        for (idx_t k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++)
            if (A->col_idx[k] == i) { A->val[k] -= sigma; break; }
}

void spmat_free(SpMatrix *A)
{
    kr_free(A->row_ptr);
    kr_free(A->col_idx);
    kr_free(A->val);
    A->row_ptr = NULL;
    A->col_idx = NULL;
    A->val     = NULL;
    A->n = A->nnz = 0;
}
