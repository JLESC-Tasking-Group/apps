// Generated using OpenCode + Claude Opus 4.6

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define BLK_ADDR(A, i, j, NT, TS) (&A[((i) * (NT) + (j)) * (TS) * (TS)])

// -------------------------------------------------------------------------
// OpenMP Target Offload Kernels (Accepting Runtime Tile Sizes)
// -------------------------------------------------------------------------

void omp_potrf(double *A_kk, int TS, char *dep_kk) {
    #pragma omp target nowait depend(inout: *dep_kk) map(present, alloc: A_kk[0:TS*TS])
    {
        for (int i = 0; i < TS; i++) {
            for (int j = 0; j < i; j++) {
                double sum = 0.0;
                for (int m = 0; m < j; m++) {
                    sum += A_kk[i * TS + m] * A_kk[j * TS + m];
                }
                A_kk[i * TS + j] = (A_kk[i * TS + j] - sum) / A_kk[j * TS + j];
            }
            double sum = 0.0;
            for (int m = 0; m < i; m++) {
                sum += A_kk[i * TS + m] * A_kk[i * TS + m];
            }
            A_kk[i * TS + i] = sqrt(A_kk[i * TS + i] - sum);
        }
    }
}

void omp_trsm(double *A_kk, double *A_ik, int TS, char *dep_kk, char *dep_ik) {
    #pragma omp target teams distribute parallel for collapse(2) nowait \
                depend(in: *dep_kk) depend(inout: *dep_ik) \
                map(present, alloc: A_kk[0:TS*TS], A_ik[0:TS*TS])
    for (int m = 0; m < TS; m++) {
        for (int n = 0; n < TS; n++) {
            double sum = 0.0;
            for (int p = 0; p < n; p++) {
                sum += A_ik[m * TS + p] * A_kk[n * TS + p];
            }
            A_ik[m * TS + n] = (A_ik[m * TS + n] - sum) / A_kk[n * TS + n];
        }
    }
}

void omp_syrk(double *A_ik, double *A_ii, int TS, char *dep_ik, char *dep_ii) {
    #pragma omp target teams distribute parallel for collapse(2) nowait \
                depend(in: *dep_ik) depend(inout: *dep_ii) \
                map(present, alloc: A_ik[0:TS*TS], A_ii[0:TS*TS])
    for (int m = 0; m < TS; m++) {
        for (int n = 0; n <= m; n++) {
            double sum = 0.0;
            for (int p = 0; p < TS; p++) {
                sum += A_ik[m * TS + p] * A_ik[n * TS + p];
            }
            A_ii[m * TS + n] -= sum;
        }
    }
}

void omp_gemm(double *A_ik, double *A_jk, double *A_ij, int TS, char *dep_ik, char *dep_jk, char *dep_ij) {
    #pragma omp target teams distribute parallel for collapse(2) nowait \
                depend(in: *dep_ik, *dep_jk) depend(inout: *dep_ij) \
                map(present, alloc: A_ik[0:TS*TS], A_jk[0:TS*TS], A_ij[0:TS*TS])
    for (int m = 0; m < TS; m++) {
        for (int n = 0; n < TS; n++) {
            double sum = 0.0;
            for (int p = 0; p < TS; p++) {
                sum += A_ik[m * TS + p] * A_jk[n * TS + p];
            }
            A_ij[m * TS + n] -= sum;
        }
    }
}

// -------------------------------------------------------------------------
// Matrix Initialization Helper
// -------------------------------------------------------------------------
void init_spd_matrix(double *A, int NT, int TS) {
    int N = NT * TS;
    double *G = (double *)malloc(N * N * sizeof(double));

    for (int i = 0; i < N * N; i++) {
        G[i] = (double)rand() / RAND_MAX;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            double val = 0.5 * (G[i * N + j] + G[j * N + i]);
            G[i * N + j] = val;
            G[j * N + i] = val;
        }
        G[i * N + i] += N * 20.0;
    }

    for (int i = 0; i < NT; i++) {
        for (int j = 0; j < NT; j++) {
            double *tile = BLK_ADDR(A, i, j, NT, TS);
            for (int m = 0; m < TS; m++) {
                for (int n = 0; n < TS; n++) {
                    tile[m * TS + n] = G[(i * TS + m) * N + (j * TS + n)];
                }
            }
        }
    }
    free(G);
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <NT: number of tiles> <TS: tile size>\n", argv[0]);
        return 1;
    }

    int NT = atoi(argv[1]);
    int TS = atoi(argv[2]);

    if (NT <= 0 || TS <= 0) {
        printf("Error: Matrix tile arguments must be positive integers.\n");
        return 1;
    }

    size_t total_elements = (size_t)NT * NT * TS * TS;
    double *A = (double *)malloc(total_elements * sizeof(double));
    char *deps = (char *)calloc(NT * NT, sizeof(char));

    init_spd_matrix(A, NT, TS);
    printf("Matrix initialized (%d x %d total size via %dx%d grid of tiles).\n", NT*TS, NT*TS, NT, NT);

    double start_time = omp_get_wtime();

    // 1. Map allocation via runtime variables
    #pragma omp target data map(alloc: A[0:total_elements])
    {
        #pragma omp single
        {
            // 2. Dynamic Tiled H2D Data Pipeline
            for (int i = 0; i < NT; i++) {
                for (int j = 0; j <= i; j++) {
                    double *tile = BLK_ADDR(A, i, j, NT, TS);
                    char *dep = &deps[i * NT + j];
                    #pragma omp target update to(tile[0:TS*TS]) nowait depend(out: *dep)
                }
            }

            // 3. Compute DAG
            for (int k = 0; k < NT; k++) {
                double *A_kk = BLK_ADDR(A, k, k, NT, TS);
                char *dep_kk = &deps[k * NT + k];
                omp_potrf(A_kk, TS, dep_kk);

                for (int i = k + 1; i < NT; i++) {
                    double *A_ik = BLK_ADDR(A, i, k, NT, TS);
                    char *dep_ik = &deps[i * NT + k];
                    omp_trsm(A_kk, A_ik, TS, dep_kk, dep_ik);
                }

                for (int i = k + 1; i < NT; i++) {
                    double *A_ik = BLK_ADDR(A, i, k, NT, TS);
                    char *dep_ik = &deps[i * NT + k];

                    double *A_ii = BLK_ADDR(A, i, i, NT, TS);
                    char *dep_ii = &deps[i * NT + i];
                    omp_syrk(A_ik, A_ii, TS, dep_ik, dep_ii);

                    for (int j = k + 1; j < i; j++) {
                        double *A_jk = BLK_ADDR(A, j, k, NT, TS);
                        double *A_ij = BLK_ADDR(A, i, j, NT, TS);
                        char *dep_jk = &deps[j * NT + k];
                        char *dep_ij = &deps[i * NT + j];
                        omp_gemm(A_ik, A_jk, A_ij, TS, dep_ik, dep_jk, dep_ij);
                    }
                }
            }

            // 4. Dynamic Tiled D2H Data Pipeline
            for (int i = 0; i < NT; i++) {
                for (int j = 0; j <= i; j++) {
                    double *tile = BLK_ADDR(A, i, j, NT, TS);
                    char *dep = &deps[i * NT + j];
                    #pragma omp target update from(tile[0:TS*TS]) nowait depend(in: *dep)
                }
            }

            // 5. Host Fence Sync
            #pragma omp taskwait
        }
    }

    double end_time = omp_get_wtime();
    printf("Total Execution Time: %f seconds\n", end_time - start_time);
    printf("Top-Left Verification Element [0][0]: %f\n", A[0]);

    free(A);
    free(deps);
    return 0;
}
