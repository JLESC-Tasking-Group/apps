/*
 * kernels.h - hand-written tiled Cholesky kernels (potrf/trsm/syrk/gemm), one
 * task each. The SAME source compiles to a CPU task or a GPU target task,
 * selected by -DUSE_TARGET (see tasking.h). No external BLAS/LAPACK dependency:
 * the arithmetic is written out so it offloads to the device unchanged.
 *
 * Every tile is ts x ts, row-major, addressed as A[m*ts + n]. The factorization
 * is left-looking on the lower triangle. Dependencies are expressed on the
 * per-tile "token" chars passed in (dep_*), which works on both backends.
 *
 *   potrf : A_kk = chol(A_kk)                         (serial; one target task)
 *   trsm  : A_ik = A_ik * inv(tril(A_kk))^T           (parallel over tile rows)
 *   syrk  : A_ii = A_ii - A_ik * A_ik^T               (lower triangle)
 *   gemm  : A_ij = A_ij - A_ik * A_jk^T
 */
#ifndef CHOLESKY_KERNELS_H
#define CHOLESKY_KERNELS_H

#include "tasking.h"
#include <math.h>

/* A_kk = Cholesky(A_kk). Inherently serial (loop-carried), so it is a single
 * (loop-less) target task on the GPU / one host task on the CPU. */
static void cholesky_potrf(double *A_kk, int ts, char *dep_kk)
{
    OMP_TARGET_TASK(DEPEND(inout, *dep_kk) MAP(present, alloc: A_kk[0:ts*ts]))
    {
        for (int i = 0; i < ts; i++) {
            for (int j = 0; j < i; j++) {
                double sum = 0.0;
                for (int m = 0; m < j; m++)
                    sum += A_kk[i * ts + m] * A_kk[j * ts + m];
                A_kk[i * ts + j] = (A_kk[i * ts + j] - sum) / A_kk[j * ts + j];
            }
            double sum = 0.0;
            for (int m = 0; m < i; m++)
                sum += A_kk[i * ts + m] * A_kk[i * ts + m];
            A_kk[i * ts + i] = sqrt(A_kk[i * ts + i] - sum);
        }
    }
}

/* A_ik = A_ik * inv(tril(A_kk))^T (triangular solve of the off-diagonal tile). */
static void cholesky_trsm(double *A_kk, double *A_ik, int ts, char *dep_kk, char *dep_ik)
{
    OMP_TARGET_LOOP_TASK(DEPEND(in, *dep_kk) DEPEND(inout, *dep_ik)
                         MAP(present, alloc: A_kk[0:ts*ts], A_ik[0:ts*ts]) GPU_CLAUSES(collapse(2)))
    for (int m = 0; m < ts; m++) {
        for (int n = 0; n < ts; n++) {
            double sum = 0.0;
            for (int p = 0; p < n; p++)
                sum += A_ik[m * ts + p] * A_kk[n * ts + p];
            A_ik[m * ts + n] = (A_ik[m * ts + n] - sum) / A_kk[n * ts + n];
        }
    }
}

/* A_ii = A_ii - A_ik * A_ik^T (symmetric rank-k update of a diagonal tile). */
static void cholesky_syrk(double *A_ik, double *A_ii, int ts, char *dep_ik, char *dep_ii)
{
    OMP_TARGET_LOOP_TASK(DEPEND(in, *dep_ik) DEPEND(inout, *dep_ii)
                         MAP(present, alloc: A_ik[0:ts*ts], A_ii[0:ts*ts]) GPU_CLAUSES(collapse(2)))
    for (int m = 0; m < ts; m++) {
        for (int n = 0; n <= m; n++) {
            double sum = 0.0;
            for (int p = 0; p < ts; p++)
                sum += A_ik[m * ts + p] * A_ik[n * ts + p];
            A_ii[m * ts + n] -= sum;
        }
    }
}

/* A_ij = A_ij - A_ik * A_jk^T (general update of an off-diagonal tile). */
static void cholesky_gemm(double *A_ik, double *A_jk, double *A_ij, int ts,
                          char *dep_ik, char *dep_jk, char *dep_ij)
{
    OMP_TARGET_LOOP_TASK(DEPEND(in, *dep_ik, *dep_jk) DEPEND(inout, *dep_ij)
                         MAP(present, alloc: A_ik[0:ts*ts], A_jk[0:ts*ts], A_ij[0:ts*ts]) GPU_CLAUSES(collapse(2)))
    for (int m = 0; m < ts; m++) {
        for (int n = 0; n < ts; n++) {
            double sum = 0.0;
            for (int p = 0; p < ts; p++)
                sum += A_ik[m * ts + p] * A_jk[n * ts + p];
            A_ij[m * ts + n] -= sum;
        }
    }
}

#endif /* CHOLESKY_KERNELS_H */
