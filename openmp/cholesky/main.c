/*
 * main.c - tiled dense Cholesky factorization (A = L L^T) as an OpenMP task DAG.
 * One source, two backends selected at compile time by -DUSE_TARGET (tasking.h):
 *
 *   USE_TARGET == 0 (default): each tile kernel is a CPU task (#pragma omp task).
 *   USE_TARGET == 1:           each tile kernel is a GPU target task, with the
 *                              matrix staged to the device by target enter data
 *                              + per-tile target update pipelines.
 *
 * NOTE: this is intentionally NOT a record/replay taskgraph example -- the tiled
 * factorization's DAG changes shape every step, so there is no fixed per-step
 * graph to record and replay. The reps argument just re-runs the factorization
 * to average the timing.
 *
 * Forked/unified from the earlier task/ (CPU, LAPACK) and target-nowait-depend/
 * (GPU, hand-written) variants; MPI/MPC support has been removed. The kernels
 * are hand-written (kernels.h) so the same code offloads to the device.
 *
 * Usage:  ./cholesky [nt] [ts] [reps] [check]
 *   nt    : number of tiles per dimension (matrix is (nt*ts) x (nt*ts))
 *   ts    : tile size
 *   reps  : number of factorizations to run (for timing average)
 *   check : 1 to verify against a sequential reference, 0 to skip
 * All are optional; compile-time defaults come from -DDEFAULT_NT / -DDEFAULT_TS.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "tasking.h"
#include "alloc.h"
#include "kernels.h"

/* ---- Problem defaults (override with -D on the compiler, see Makefile) ---- */
#ifndef DEFAULT_NT
# define DEFAULT_NT 8       /* number of tiles per dimension */
#endif
#ifndef DEFAULT_TS
# define DEFAULT_TS 256     /* tile size */
#endif
#ifndef DEFAULT_REPS
# define DEFAULT_REPS 1     /* number of factorizations to run */
#endif

/* Address of tile (i,j) inside the contiguous row-major-of-tiles buffer. */
#define BLK_ADDR(A, i, j, nt, ts) (&(A)[((size_t)((i) * (nt) + (j))) * (ts) * (ts)])

/* ---- Build a symmetric positive-definite tiled matrix ---- */
static void init_spd_matrix(double *A, int nt, int ts)
{
    const int N = nt * ts;
    double *G = (double *) malloc((size_t) N * N * sizeof(double));

    for (size_t i = 0; i < (size_t) N * N; i++)
        G[i] = (double) rand() / RAND_MAX;

    /* Symmetrize and make strongly diagonally dominant (=> SPD). */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            double val = 0.5 * (G[i * N + j] + G[j * N + i]);
            G[i * N + j] = val;
            G[j * N + i] = val;
        }
        G[i * N + i] += N * 20.0;
    }

    /* Scatter the flat matrix into the tiled layout. */
    for (int i = 0; i < nt; i++) {
        for (int j = 0; j < nt; j++) {
            double *tile = BLK_ADDR(A, i, j, nt, ts);
            for (int m = 0; m < ts; m++)
                for (int n = 0; n < ts; n++)
                    tile[m * ts + n] = G[(i * ts + m) * N + (j * ts + n)];
        }
    }
    free(G);
}

/* ---- Sequential reference factorization (host, for -check) ----
 * Same arithmetic as the tile kernels, run serially over the tile DAG. */
static void cholesky_seq(double *A, int nt, int ts)
{
    for (int k = 0; k < nt; k++) {
        double *A_kk = BLK_ADDR(A, k, k, nt, ts);
        for (int i = 0; i < ts; i++) {
            for (int j = 0; j < i; j++) {
                double s = 0.0;
                for (int m = 0; m < j; m++) s += A_kk[i * ts + m] * A_kk[j * ts + m];
                A_kk[i * ts + j] = (A_kk[i * ts + j] - s) / A_kk[j * ts + j];
            }
            double s = 0.0;
            for (int m = 0; m < i; m++) s += A_kk[i * ts + m] * A_kk[i * ts + m];
            A_kk[i * ts + i] = sqrt(A_kk[i * ts + i] - s);
        }
        for (int i = k + 1; i < nt; i++) {
            double *A_ik = BLK_ADDR(A, i, k, nt, ts);
            for (int m = 0; m < ts; m++)
                for (int n = 0; n < ts; n++) {
                    double s = 0.0;
                    for (int p = 0; p < n; p++) s += A_ik[m * ts + p] * A_kk[n * ts + p];
                    A_ik[m * ts + n] = (A_ik[m * ts + n] - s) / A_kk[n * ts + n];
                }
        }
        for (int i = k + 1; i < nt; i++) {
            double *A_ik = BLK_ADDR(A, i, k, nt, ts);
            double *A_ii = BLK_ADDR(A, i, i, nt, ts);
            for (int m = 0; m < ts; m++)
                for (int n = 0; n <= m; n++) {
                    double s = 0.0;
                    for (int p = 0; p < ts; p++) s += A_ik[m * ts + p] * A_ik[n * ts + p];
                    A_ii[m * ts + n] -= s;
                }
            for (int j = k + 1; j < i; j++) {
                double *A_jk = BLK_ADDR(A, j, k, nt, ts);
                double *A_ij = BLK_ADDR(A, i, j, nt, ts);
                for (int m = 0; m < ts; m++)
                    for (int n = 0; n < ts; n++) {
                        double s = 0.0;
                        for (int p = 0; p < ts; p++) s += A_ik[m * ts + p] * A_jk[n * ts + p];
                        A_ij[m * ts + n] -= s;
                    }
            }
        }
    }
}

/* Max abs difference on the lower triangle of tiles (the factored part). */
static double lower_triangle_maxdiff(const double *A, const double *R, int nt, int ts)
{
    double maxd = 0.0;
    for (int i = 0; i < nt; i++)
        for (int j = 0; j <= i; j++) {
            const double *a = BLK_ADDR(A, i, j, nt, ts);
            const double *r = BLK_ADDR(R, i, j, nt, ts);
            for (int e = 0; e < ts * ts; e++) {
                double d = fabs(a[e] - r[e]);
                if (d > maxd) maxd = d;
            }
        }
    return maxd;
}

static const char *backend_name(void)
{
#if USE_SYNC
    return "synchronous (blocking)";
#elif USE_TARGET
    return "GPU target tasks";
#else
    return "CPU tasks";
#endif
}

int main(int argc, char **argv)
{
    if (argc > 1 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("usage: %s [nt] [ts] [reps] [check]\n", argv[0]);
        printf("  nt    number of tiles per dimension (default %d)\n", DEFAULT_NT);
        printf("  ts    tile size                     (default %d)\n", DEFAULT_TS);
        printf("  reps  factorizations to run         (default %d)\n", DEFAULT_REPS);
        printf("  check 1 = verify vs sequential ref  (default 0)\n");
        return 0;
    }

    const int nt    = (argc > 1) ? atoi(argv[1]) : DEFAULT_NT;
    const int ts    = (argc > 2) ? atoi(argv[2]) : DEFAULT_TS;
    const int reps  = (argc > 3) ? atoi(argv[3]) : DEFAULT_REPS;
    const int check = (argc > 4) ? atoi(argv[4]) : 0;

    if (nt <= 0 || ts <= 0 || reps <= 0) {
        fprintf(stderr, "error: nt, ts and reps must be positive integers\n");
        return 1;
    }

    const size_t ntiles = (size_t) nt * nt;
    const size_t total  = ntiles * ts * ts;

    /* Device-mapped matrix (pinned on GPU builds) + pristine host copy. */
    double *A     = (double *) host_alloc(total * sizeof(double));
    double *A_org = (double *) malloc(total * sizeof(double));
    /* One dependency "token" per tile; the DAG is expressed on these on both
     * backends (the tiles themselves are device-resident in the GPU build). */
    char *deps = (char *) calloc(ntiles, sizeof(char));
    if (!A || !A_org || !deps) { fprintf(stderr, "allocation failed\n"); return 1; }

    srand(2024);
    init_spd_matrix(A_org, nt, ts);

    printf("Cholesky  backend=%s  nt=%d  ts=%d  N=%d  reps=%d\n",
           backend_name(), nt, ts, nt * ts, reps);

    double *rep_ms = (double *) malloc((size_t) reps * sizeof(double));

    /* One-time device allocation of the whole matrix (no copy yet). */
    OMP_TARGET_ENTER_DATA(MAP(alloc: A[0:total]))

    const double t_total0 = omp_get_wtime();

    #pragma omp parallel
    #pragma omp single
    {
        for (int rep = 0; rep < reps; rep++) {
            /* Fresh input each repetition (previous rep left A factored). */
            memcpy(A, A_org, total * sizeof(double));

            const double t0 = omp_get_wtime();

            {
#if USE_TARGET
                /* H2D pipeline: stage the lower-triangular tiles, each tagged
                 * with its token so compute can start as its tile arrives. */
                for (int i = 0; i < nt; i++)
                    for (int j = 0; j <= i; j++) {
                        double *tile = BLK_ADDR(A, i, j, nt, ts);
                        char   *dep  = &deps[i * nt + j];
                        OMP_TARGET_UPDATE(to(tile[0:ts*ts]) NOWAIT DEPEND(out, *dep))
                    }
#endif

                /* Right-looking tiled factorization DAG. */
                for (int k = 0; k < nt; k++) {
                    double *A_kk = BLK_ADDR(A, k, k, nt, ts);
                    char   *d_kk = &deps[k * nt + k];
                    cholesky_potrf(A_kk, ts, d_kk);

                    for (int i = k + 1; i < nt; i++) {
                        double *A_ik = BLK_ADDR(A, i, k, nt, ts);
                        char   *d_ik = &deps[i * nt + k];
                        cholesky_trsm(A_kk, A_ik, ts, d_kk, d_ik);
                    }

                    for (int i = k + 1; i < nt; i++) {
                        double *A_ik = BLK_ADDR(A, i, k, nt, ts);
                        char   *d_ik = &deps[i * nt + k];
                        double *A_ii = BLK_ADDR(A, i, i, nt, ts);
                        char   *d_ii = &deps[i * nt + i];
                        cholesky_syrk(A_ik, A_ii, ts, d_ik, d_ii);

                        for (int j = k + 1; j < i; j++) {
                            double *A_jk = BLK_ADDR(A, j, k, nt, ts);
                            double *A_ij = BLK_ADDR(A, i, j, nt, ts);
                            char   *d_jk = &deps[j * nt + k];
                            char   *d_ij = &deps[i * nt + j];
                            cholesky_gemm(A_ik, A_jk, A_ij, ts, d_ik, d_jk, d_ij);
                        }
                    }
                }

#if USE_TARGET
                /* D2H pipeline: read the factored lower triangle back as each
                 * tile is finalized. */
                for (int i = 0; i < nt; i++)
                    for (int j = 0; j <= i; j++) {
                        double *tile = BLK_ADDR(A, i, j, nt, ts);
                        char   *dep  = &deps[i * nt + j];
                        OMP_TARGET_UPDATE(from(tile[0:ts*ts]) NOWAIT DEPEND(in, *dep))
                    }
#endif
            }

            #pragma omp taskwait
            const double t1 = omp_get_wtime();
            rep_ms[rep] = (t1 - t0) * 1000.0;
        }
    }

    const double t_total1 = omp_get_wtime();

    OMP_TARGET_EXIT_DATA(MAP(release: A[0:total]))

    /* ---- report ----
     * total time, theoretical FLOPs and the derived GFLOP/s, then the mean and
     * sample stddev of the per-repetition times (dropping rep 0 as warmup). */
    const double Nd    = (double) nt * (double) ts;          /* matrix dimension */
    const double flops = Nd * Nd * Nd / 3.0;                 /* ~ (1/3) N^3 for Cholesky */

    int lo = (reps > 1) ? 1 : 0;                             /* drop rep 0 (warmup) */
    int cnt = reps - lo;
    double mean = 0.0;
    for (int r = lo; r < reps; r++) mean += rep_ms[r];
    if (cnt > 0) mean /= cnt;
    double var = 0.0;
    for (int r = lo; r < reps; r++) { double d = rep_ms[r] - mean; var += d * d; }
    double stddev = (cnt > 1) ? sqrt(var / (cnt - 1)) : 0.0;
    double gflops = (mean > 0.0) ? flops / (mean * 1.0e-3) / 1.0e9 : 0.0;

    printf("total solve time     : %10.6f s\n", t_total1 - t_total0);
    printf("theoretical flops    : %.6e\n", flops);
    printf("performance          : %10.3f GFLOP/s\n", gflops);
    for (int r = 0; r < reps; r++)
        printf("  rep %3d            : %10.3f ms\n", r, rep_ms[r]);
    printf("repetitions (avg)    : %10.3f ms\n", mean);
    printf("repetitions (stddev) : %10.3f ms\n", stddev);

    /* ---- optional verification ---- */
    if (check) {
        double *ref = (double *) malloc(total * sizeof(double));
        memcpy(ref, A_org, total * sizeof(double));
        cholesky_seq(ref, nt, ts);
        double maxd = lower_triangle_maxdiff(A, ref, nt, ts);
        printf("max |A - ref| (lower) = %.3e  ->  %s\n",
               maxd, (maxd < 1e-6) ? "OK" : "MISMATCH");
        free(ref);
    }

    host_free(A);
    free(A_org);
    free(deps);
    free(rep_ms);
    return 0;
}
