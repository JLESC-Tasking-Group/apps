/* forked from: https://github.com/devreal/cholesky_omptasks */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

# include <omp.h>

#include "kernels.h"

/* PARALLEL (tasks + MPI) - CHOLESKY FACTORIZATION */

#ifdef USING_MPI
# include "interop.h"

static void
get_block_rank(
    int * block_rank,
    int nt)
{
    int row = np;
    int col = np;
    if (np != 1)
    {
        while (1)
        {
            row = row / 2;
            if (row * col == np) break;
            col = col / 2;
            if (row * col == np) break;
        }
    }
    if (mype == 0) printf("row = %d, col = %d\n", row, col);
    int i, j, tmp_rank = 0, offset = 0;
    for (i = 0; i < nt; i++) {
        for (j = 0; j < nt; j++) {
            block_rank[i*nt + j] = tmp_rank + offset;
            tmp_rank++;
            if (tmp_rank >= col) tmp_rank = 0;
        }
        tmp_rank = 0;
        offset = (offset + col >= np) ? 0 : offset + col;
    }
}

void
cholesky_mpi(
        const int ts,
        const int nt,
        double * A[nt][nt],
        double * B,
        double * C[nt],
        int * block_rank)
{
    char * send_flags = malloc(sizeof(char) * np);
    reset_send_flags(send_flags);
    char recv_flag = 0;

    # pragma omp parallel
    {
        # pragma omp single
        {
            for (int k = 0; k < nt; k++)
            {
                if (block_rank[k*nt+k] == mype)
                {
                    #pragma omp task depend(out: A[k][k]) firstprivate(k) priority(TASK_PRIORITY_POTRF(i, j, k, nt))
                        potrf(A[k][k], ts, ts);

                    if (np != 1)
                    {
                        for (int dst = 0; dst < np; dst++)
                        {
                            int send_flag = 0;
                            for (int kk = k+1; kk < nt; kk++)
                            {
                                if (dst == block_rank[k*nt+kk]) { send_flag = 1; break; }
                            }
                            if (send_flag && dst != mype)
                            {
                                # pragma omp task depend(in: A[k][k]) untied priority(TASK_PRIORITY_SEND_K(k, nt))
                                {
                                    MPI_Request req;
                                    MPI_Isend(A[k][k], ts*ts, MPI_DOUBLE, dst, k*nt+k, MPI_COMM_WORLD, &req);
                                    MPIX_Wait(&req, MPI_STATUS_IGNORE);
                                }
                            }
                        }
                    }
                    reset_send_flags(send_flags);
                }

                if (block_rank[k*nt+k] != mype)
                {
                    for (int i = k + 1; i < nt; i++)
                    {
                        if (block_rank[k*nt+i] == mype) recv_flag = 1;
                    }
                    if (recv_flag)
                    {
                        #pragma omp task depend(out: B) firstprivate(k) untied priority(TASK_PRIORITY_RECV_K(k, nt))
                        {
                            MPI_Request req;
                            MPI_Irecv(B, ts*ts, MPI_DOUBLE, block_rank[k*nt+k], k*nt+k, MPI_COMM_WORLD, &req);
                            MPIX_Wait(&req, MPI_STATUS_IGNORE);
                        }
                        recv_flag = 0;
                    }
                }

                for (int i = k + 1; i < nt; i++)
                {
                    if (block_rank[k*nt+i] == mype)
                    {
                        if (block_rank[k*nt+k] == mype)
                        {
                            #pragma omp task depend(in: A[k][k]) depend(out: A[k][i]) firstprivate(k, i) priority(TASK_PRIORITY_TRSM(i, j, k, nt))
                                trsm(A[k][k], A[k][i], ts, ts);
                        }
                        else
                        {
                            #pragma omp task depend(in: B) depend(out: A[k][i]) firstprivate(k, i) priority(TASK_PRIORITY_TRSM(i, j, k, nt))
                                trsm(B, A[k][i], ts, ts);
                        }
                    }

                    if (block_rank[k*nt+i] == mype && np != 1)
                    {
                        for (int ii = k + 1; ii < i; ii++)
                        {
                            if (!send_flags[block_rank[ii*nt+i]]) send_flags[block_rank[ii*nt+i]] = 1;
                        }
                        for (int ii = i + 1; ii < nt; ii++)
                        {
                            if (!send_flags[block_rank[i*nt+ii]]) send_flags[block_rank[i*nt+ii]] = 1;
                        }
                        if (!send_flags[block_rank[i*nt+i]]) send_flags[block_rank[i*nt+i]] = 1;
                        for (int dst = 0; dst < np; dst++)
                        {
                            if (send_flags[dst] && dst != mype)
                            {
                                #pragma omp task depend(in: A[k][i]) firstprivate(k, i, dst) untied priority(TASK_PRIORITY_SEND_I_K(i, k, nt))
                                {
                                    MPI_Request req;
                                    MPI_Isend(A[k][i], ts*ts, MPI_DOUBLE, dst, k*nt+i, MPI_COMM_WORLD, &req);
                                    MPIX_Wait(&req, MPI_STATUS_IGNORE);
                                }
                            }
                        }
                        reset_send_flags(send_flags);
                    }
                    if (block_rank[k*nt+i] != mype)
                    {
                        for (int ii = k + 1; ii < i; ii++)
                        {
                            if (block_rank[ii*nt+i] == mype) recv_flag = 1;
                        }
                        for (int ii = i + 1; ii < nt; ii++)
                        {
                            if (block_rank[i*nt+ii] == mype) recv_flag = 1;
                        }
                        if (block_rank[i*nt+i] == mype) recv_flag = 1;
                        if (recv_flag)
                        {
                            #pragma omp task depend(out: C[i]) firstprivate(k, i) untied priority(TASK_PRIORITY_RECV_I_K(i, k, nt))
                            {
                                MPI_Request req;
                                MPI_Irecv(C[i], ts*ts, MPI_DOUBLE, block_rank[k*nt+i], k*nt+i, MPI_COMM_WORLD, &req);
                                MPIX_Wait(&req, MPI_STATUS_IGNORE);
                            }
                            recv_flag = 0;
                        }
                    }
                }

                for (int i = k + 1; i < nt; i++)
                {
                    for (int j = k + 1; j < i; j++)
                    {
                        if (block_rank[j*nt+i] == mype)
                        {
                            if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] == mype)
                            {
                                #pragma omp task depend(in: A[k][i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i) priority(TASK_PRIORITY_GEMM(i, j, k, nt))
                                    gemm(A[k][i], A[k][j], A[j][i], ts, ts);
                            }
                            else if (block_rank[k*nt+i] != mype && block_rank[k*nt+j] == mype)
                            {
                                #pragma omp task depend(in: C[i], A[k][j]) depend(out: A[j][i]) firstprivate(k, j, i) priority(TASK_PRIORITY_GEMM(i, j, k, nt))
                                    gemm(C[i], A[k][j], A[j][i], ts, ts);
                            }
                            else if (block_rank[k*nt+i] == mype && block_rank[k*nt+j] != mype)
                            {
                                #pragma omp task depend(in: A[k][i], C[j]) depend(out: A[j][i]) firstprivate(k, j, i) priority(TASK_PRIORITY_GEMM(i, j, k, nt))
                                    gemm(A[k][i], C[j], A[j][i], ts, ts);
                            }
                            else
                            {
                                #pragma omp task depend(in: C[i], C[j]) depend(out: A[j][i]) firstprivate(k, j, i) priority(TASK_PRIORITY_GEMM(i, j, k, nt))
                                    gemm(C[i], C[j], A[j][i], ts, ts);
                            }
                        }
                    }

                    if (block_rank[i*nt+i] == mype)
                    {
                        if (block_rank[k*nt+i] == mype)
                        {
                            #pragma omp task depend(in: A[k][i]) depend(out: A[i][i]) firstprivate(k, i) priority(TASK_PRIORITY_SYRK(i, j, k, nt))
                                syrk(A[k][i], A[i][i], ts, ts);
                        }
                        else
                        {
                            #pragma omp task depend(in: C[i]) depend(out: A[i][i]) firstprivate(k, i) priority(TASK_PRIORITY_SYRK(i, j, k, nt))
                                syrk(C[i], A[i][i], ts, ts);
                        }
                    }
                }
            } // for k
        } // single
    } // parallel
    free(send_flags);
}

#endif /* USING_MPI */

/* PARALLEL (tasks) - CHOLESKY FACTORIZATION */
static void
cholesky_par(
        const int ts,
        const int nt,
        double * A[nt][nt],
        double * B,
        double * C[nt])
{
    for (int k = 0; k < nt; k++)
    {
        # pragma omp task depend(out: A[k][k])
            potrf(A[k][k], ts, ts);
        for (int i = k + 1; i < nt; i++)
        {
            # pragma omp task depend(in: A[k][k]) depend(out: A[k][i])
                trsm(A[k][k], A[k][i], ts, ts);
        }
        for (int i = k + 1; i < nt; i++)
        {
            for (int j = k + 1; j < i; j++)
            {
                # pragma omp task depend(in: A[k][i]) depend(out: A[k][j], A[j][i])
                    gemm(A[k][i], A[k][j], A[j][i], ts, ts);
            }
            # pragma omp task depend(in: A[k][i]) depend(out: A[i][i])
                syrk(A[k][i], A[i][i], ts, ts);
        }
    }
}

/* SEQUENTIAL - CHOLESKY FACTORIZATION */
static void
cholesky_seq(
        const int ts,
        const int nt,
        double * A[nt][nt],
        double * B,
        double * C[nt])
{
    for (int k = 0; k < nt; k++)
    {
        potrf(A[k][k], ts, ts);
        for (int i = k + 1; i < nt; i++)
        {
            trsm(A[k][k], A[k][i], ts, ts);
        }
        for (int i = k + 1; i < nt; i++)
        {
            for (int j = k + 1; j < i; j++)
            {
                gemm(A[k][i], A[k][j], A[j][i], ts, ts);
            }
            syrk(A[k][i], A[i][i], ts, ts);
        }
    }
}

int
main(int argc, char ** argv)
{
    if (argc < 4)
    {
        printf("usage: %s matrix_size block_size check\n", argv[0]);
        return 1;
    }

# ifdef MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (provided != MPI_THREAD_MULTIPLE)
    {
        printf("This Compiler does not support MPI_THREAD_MULTIPLE\n");
        return 0;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int * block_rank = (int *)malloc(nt * nt * sizeof(int));
    get_block_rank(block_rank, nt);

# endif /* MPI */

    int     n = atoi(argv[1]); // matrix size
    int    ts = atoi(argv[2]); // tile size
    int check = atoi(argv[3]); // check
    const int nt = n / ts;
    double * A[nt][nt], * B, * C[nt], * Ans[nt][nt];

    /* MATRIX INITIALIZATION */
    for (int i = 0; i < nt; i++)
    {
        for (int j = 0; j < nt; j++)
        {
            A[i][j] = (double *) malloc(ts * ts * sizeof(double));
            initialize_tile(ts * ts, A[i][j]);
            if (check)
            {
                Ans[i][j] = (double *) malloc(ts * ts * sizeof(double));
                memcpy(Ans[i][j], A[i][j], ts * ts * sizeof(double));
            }
        }

        // diagonal dominant
        for (int k = 0 ; k < ts ; k++)
        {
            A[i][i][k * ts + k] = (double) n;
            Ans[i][i][k * ts + k] = (double) n;
        }
        C[i] = (double *) malloc(ts * ts * sizeof(double));
    }
    B = (double *) malloc(ts * ts * sizeof(double));

    # pragma omp parallel
    {
        # pragma omp single
        {
            int num_threads = omp_get_num_threads();
            printf("OpenMP num_threads = %d\n", num_threads);

            if (check)
            {
                double t0 = omp_get_wtime();
                puts("Running check...");
#if USING_MPI
                cholesky_par(ts, nt, Ans, B, C);
#else /* USING_MPI */
                cholesky_seq(ts, nt, Ans, B, C);
#endif /* USING_MPI */
                double t1 = omp_get_wtime();
                printf("check time : %lf s.\n", t1 - t0);
            }

            puts("Running parallel...");
            double t1 = omp_get_wtime();
#if USING_MPI
            cholesky_mpi(ts, nt, A, B, C);
# else /* USING_MPI */
            cholesky_par(ts, nt, A, B, C);
# endif /* USING_MPI */
            # pragma omp taskwait
            double t2 = omp_get_wtime();
            printf("parallel time : %lf s.\n", t2 - t1);
        }
    }

    /* Check */
    if (check)
    {
        int wrong = 0;
        for (int i = 0; i < nt; i++)
        {
            for (int j = 0; j < nt; j++)
            {
                for (int k = 0; k < ts*ts; k++)
                {
                    if (Ans[i][j][k] != A[i][j][k])
                    {
                        wrong = 1;
                        break ;
                    }
                }
            }
        }
        puts(wrong ? "ERROR IN PARALLEL FACTORIZATION" : "Parallel factorization is correct");
    }

    /* FREE MEMORY */
    for (int i = 0; i < nt; i++)
    {
        for (int j = 0; j < nt; j++)
        {
            free(A[i][j]);
            if (check) free(Ans[i][j]);
        }
        free(C[i]);
    }
    free(B);

# ifdef USING_MPI
    MPI_Finalize();
    free(block_rank);
# endif /* USING_MPI */
    return 0;
}
