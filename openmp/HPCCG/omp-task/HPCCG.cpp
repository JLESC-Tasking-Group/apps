//@HEADER
// ************************************************************************
//
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// BSD 3-Clause License
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER
/////////////////////////////////////////////////////////////////////////

// Routine to compute an approximate solution to Ax = b where:

// A - known matrix stored as an HPC_Sparse_Matrix struct

// b - known right hand side vector

// x - On entry is initial guess, on exit new approximate solution

// max_iter - Maximum number of iterations to perform, even if
//            tolerance is not met.

// tolerance - Stop and assert convergence if norm of residual is <=
//             to tolerance.

// niters - On output, the number of iterations actually performed.

/////////////////////////////////////////////////////////////////////////

#include <cassert>
#include <iostream>
#include <map>
using std::cerr;
using std::cout;
using std::endl;
#include <cmath>
#include <cstring>
#include <unistd.h>
#include "HPCCG.hpp"
#include "mytimer.hpp"

#include <omp.h>

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
extern Index_t T1;   /* number of tasks per operations */
Index_t BS;          /* operations block size */

extern Index_t T2; /* number of subtasks per spmv */
Index_t SBS;       /* sub-operations block size */

/* precomputed dependencies */
typedef struct
{
    Index_t * indices;
    Index_t size;
} task_dependency_t;
static task_dependency_t * deps_SPMV;

/* openmpi global variables */

#ifdef USING_MPI
# include <mpi.h>
#endif

# ifdef USING_OMPI
#  define OMP_MPI_SHARED rank, BS, hostname, ompi_mpi_comm_world, ompi_mpi_double, ompi_mpi_op_sum
# elif USING_MPICH
#  define OMP_MPI_SHARED rank, BS
# else
#  define OMP_MPI_SHARED rank, BS
# endif

char hostname[64];
int rank;

#ifdef USING_MPI

#ifdef DEBUG
#  define PRINT(...) do {\
        printf("[%s] [%d] [%d] [%d] ", hostname, getpid(), rank, omp_get_thread_num());\
        printf(__VA_ARGS__);\
        fflush(stdout); \
    } while (0)
#else /* DEBUG */
# define PRINT(...)
#endif /* DEBUG */

/* precomputed dependencies */
static task_dependency_t * dependencies_recv;
static task_dependency_t * dependencies_send;

static inline void
wait_req(MPI_Request * req)
{
    MPI_Status status;
# ifdef USING_TAMPI
    int err, completed;
    do {
        err = MPI_Test(req, &completed, &status);
        # pragma omp taskyield
    } while (!completed);
# else /* USING_TAMPI */
    int err = MPI_Wait(req, &status);
# endif /* USING_TAMPI */

    if (err == MPI_ERR_IN_STATUS)
    {
        assert(status.MPI_ERROR == MPI_SUCCESS);
    }
    else
    {
        assert(err == MPI_SUCCESS);
    }
}

static void
__exchange_externals_deps_deinit(HPC_Sparse_Matrix * A)
{
    Index_t i;

    for (i = 0; i < A->num_send_neighbors; ++i)
        free(dependencies_recv[i].indices);
    free(dependencies_recv);

    for (i = 0; i < A->num_send_neighbors; ++i)
        free(dependencies_send[i].indices);
    free(dependencies_send);
}

static void
__exchange_externals_deps_init(HPC_Sparse_Matrix * A, double * x)
{
    /* dependencies_recv */
    assert(!dependencies_recv);
    dependencies_recv = (task_dependency_t *)malloc(sizeof(task_dependency_t) * A->num_send_neighbors);
    assert(dependencies_recv);
    Index_t offset = 0;
    Index_t i;
    for (i = 0; i < A->num_send_neighbors; ++i)
    {
        # pragma omp task default(none)     \
            firstprivate(A, x, i, offset)   \
            shared(BS, dependencies_recv)
        {
            Index_t n_recv = A->recv_length[i];
            Index_t ndeps = n_recv / BS + (n_recv % BS != 0);
            Index_t * indices = (Index_t *)malloc(sizeof(Index_t) * ndeps);
            Index_t j;
            for (j = 0; j < n_recv; j += BS)
            {
                Index_t index = (A->local_nrow + offset + j) / BS * BS;
                indices[j / BS] = index;
            }
            assert(j / BS == ndeps);

            dependencies_recv[i].size = ndeps;
            dependencies_recv[i].indices = indices;
        }
        offset += A->recv_length[i];
    }

    /* dependencies_send */
    assert(!dependencies_send);
    dependencies_send = (task_dependency_t *) malloc(sizeof(task_dependency_t) * A->num_send_neighbors);
    assert(dependencies_send);
    offset = 0;
    for (i = 0; i < A->num_send_neighbors; ++i)
    {
        # pragma omp task default(none)     \
            firstprivate(A, x, offset, i)   \
            shared(dependencies_send, BS)
        {
            std::map<Index_t, bool> blocks;
            Index_t j;
            for (j = 0; j < A->send_length[i] ; ++j)
            {
                Index_t index = A->elements_to_send[offset + j] / BS * BS;  // retrieve block starting address
                if (blocks.count(index)) continue;
                blocks[index] = true;
            }

            dependencies_send[i].size = blocks.size();
            dependencies_send[i].indices = (Index_t *)malloc(sizeof(Index_t) * blocks.size());

            j = 0;
            for (std::map<Index_t, bool>::iterator it = blocks.begin(); it != blocks.end(); ++it)
            {
                dependencies_send[i].indices[j] = it->first;
                ++j;
            }
        }
        offset += A->send_length[i];
    }
}

void
exchange_externals(HPC_Sparse_Matrix * A, double * x, Index_t it)
{
    //
    //  first post receives, these are immediate receives
    //  Do not wait for result to come, will do that at the
    //  wait call below.
    //

    //
    // Externals are at end of locals
    //

    // Post receives first
    Index_t offset = 0;
    Index_t i;
    for (i = 0; i < A->num_send_neighbors; ++i)
    {
        task_dependency_t * deps = dependencies_recv + i;
        # pragma omp task default(none)                                  \
            firstprivate(A, x, offset, i, it)                           \
            shared(OMP_MPI_SHARED)                                      \
            depend(iterator(i=0:deps->size), out: x[deps->indices[i]])  \
            untied
        {
            MPI_Request req;
            MPI_Irecv(x + A->local_nrow + offset, A->recv_length[i], MPI_DOUBLE, A->neighbors[i], (Index_t)0, MPI_COMM_WORLD, &req);
            wait_req(&req);
        }
        offset += A->recv_length[i];
    }

    //
    // Send to each neighbor
    //
    offset = 0;
    for (i = 0; i < A->num_send_neighbors; ++i)
    {
        task_dependency_t * deps = dependencies_send + i;
        # pragma omp task default(none)                                 \
            firstprivate(A, x, offset, i, it)                           \
            shared(OMP_MPI_SHARED)                                      \
            depend(inout: A->send_buffer[offset])                       \
            depend(iterator(i=0:deps->size), in: x[deps->indices[i]])   \
            untied priority(1)
        {
            //
            // Fill up send buffer
            //
            Index_t j;
            for (j = 0; j < A->send_length[i] ; ++j)
            {
                A->send_buffer[offset + j] = x[A->elements_to_send[offset + j]];
            }
            MPI_Request req;
            MPI_Isend(A->send_buffer + offset, A->send_length[i], MPI_DOUBLE, A->neighbors[i], (Index_t)0, MPI_COMM_WORLD, &req);
            wait_req(&req);
        }
        offset += A->send_length[i];
    }
}
#endif  // USING_MPI

static void
__HPC_sparsemv_deps_deinit(HPC_Sparse_Matrix * A)
{
    (void) A;

    Index_t i;
    for (i = 0 ; i < T1 * T2 ; ++i)
    {
        free(deps_SPMV[i].indices);
    }
    free(deps_SPMV);
}

static void
__HPC_sparsemv_deps_init(HPC_Sparse_Matrix * A, const double * const x, double * y)
{
    const Index_t n = A->local_nrow;
    assert(!deps_SPMV);
    deps_SPMV = (task_dependency_t *) malloc(sizeof(task_dependency_t) * T1 * T2);
    assert(deps_SPMV);
    Index_t t1;
    for (t1 = 0 ; t1 < T1 ; ++t1)
    {
        Index_t t2;
        for (t2 = 0 ; t2 < T2 ; ++t2)
        {
            # pragma omp task default(none)         \
                firstprivate(A, x, y, n, t1, t2)    \
                shared(deps_SPMV, BS, SBS, T2)
            {
                // generate dependencies
                std::map<Index_t, bool> blocks;
                Index_t begin = t1 * BS + t2 * SBS;
                Index_t end = MIN(n, MIN((t1 + 1) * BS, begin + SBS));
                Index_t i;
                for (i = begin ; i < end; ++i)
                {
                    const Index_t * const cur_inds = A->ptr_to_inds_in_row[i];
                    const Index_t cur_nnz = A->nnz_in_row[i];
                    Index_t j;
                    for (j = 0; j < cur_nnz; ++j)
                    {
                        Index_t index = cur_inds[j] / BS * BS;
                        if (blocks.count(index)) continue;
                        blocks[index] = true;
                    }
                }

                task_dependency_t * in = deps_SPMV + (t1*T2+t2);
                in->size = blocks.size();
                in->indices = (Index_t *)malloc(sizeof(Index_t) * blocks.size());

                Index_t j = 0;
                for (std::map<Index_t, bool>::iterator it = blocks.begin(); it != blocks.end(); ++it)
                {
                    in->indices[j] = it->first;
                    ++j;
                }
                assert(j == in->size);
            }
        }
    }
}

int
HPC_sparsemv(
        HPC_Sparse_Matrix * A,
        const double * const x,
        double * const y)
{
    Index_t t1;
    for (t1 = 0 ; t1 < T1 ; ++t1)
    {
        Index_t t2;
        for (t2 = 0 ; t2 < T2 ; ++t2)
        {
            task_dependency_t * deps = deps_SPMV + (t1*T2+t2);
            # pragma omp task default(none)                                 \
                firstprivate(A, x, y, t1, t2)                               \
                shared(BS, SBS)                                             \
                depend(iterator(i=0:deps->size), in: x[deps->indices[i]])   \
                depend(out: y[t1 * BS + t2 * SBS])
            {
                Index_t n = A->local_nrow;
                Index_t begin = t1 * BS + t2 * SBS;
                Index_t end = MIN(n, MIN((t1 + 1) * BS, begin + SBS));
                Index_t i;
                for (i = begin ; i < end; ++i)
                {
                    double sum = 0.0;
                    const double * const cur_vals = A->ptr_to_vals_in_row[i];
                    const Index_t * const cur_inds = A->ptr_to_inds_in_row[i];
                    const Index_t cur_nnz = A->nnz_in_row[i];
                    Index_t j;
                    for (j = 0; j < cur_nnz; j++)
                    {
                        sum += cur_vals[j] * x[cur_inds[j]];
                    }
                    y[i] = sum;
                }
            }
        }
    }
    return 0;
}

int
waxpby(const int n,
        const double alpha,
        const double * const x,
        const double beta,
        const double * const y,
        double * const w)
{
    if (alpha == 1.0)
    {
        for (int i = 0; i < n; i++)
            w[i] = x[i] + beta * y[i];
    }
    else if (beta == 1.0)
    {
        for (int i = 0; i < n; i++)
            w[i] = alpha * x[i] + y[i];
    }
    else
    {
        for (int i = 0; i < n; i++)
            w[i] = alpha * x[i] + beta * y[i];
    }
    return (0);
}

static void
HPCCG_single(HPC_Sparse_Matrix * A,
        const double * const b,
        double * const x,
        const int max_iter,
        const double tolerance,
        int & niters,
        double & normr,
        double * times)
{
#ifdef USING_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    Index_t rank = 0;  // Serial case (not using MPI)
#endif
    gethostname(hostname, sizeof(hostname));

    Index_t nrow = A->local_nrow;
    Index_t ncol = A->local_ncol;

    double * r = new double[nrow];
    double * p = new double[ncol];  // In parallel case, A is rectangular
    double * Ap = new double[nrow];

    // compute block size
    Index_t n = nrow;
    BS = std::ceil((double)n / (double)T1);
    SBS = std::ceil((double)n / (double)(T1 * T2));
    if (rank == 0) printf("T1=" Index_t_FORMAT  "\n", T1);
    if (rank == 0) printf("T2=" Index_t_FORMAT  "\n", T2);
    if (rank == 0) printf("BS=" Index_t_FORMAT  "\n", BS);
    if (rank == 0) printf("SBS=" Index_t_FORMAT "\n", SBS);
    normr = 0.0;

    double rtrans;
    double * rtrans_per_block = (double *) malloc(sizeof(double) * T1);

    double beta;
    double * beta_per_block = (double *) malloc(sizeof(double) * T1);

    Index_t print_freq = max_iter / 10;
    if (print_freq > 50)
        print_freq = 50;
    if (print_freq < 1)
        print_freq = 1;

    double t0 = mytimer();  // Start timing right away
    __HPC_sparsemv_deps_init(A, p, Ap);
#ifdef USING_MPI
    __exchange_externals_deps_init(A, p);
#endif /* USING_MPI */
    # pragma omp taskwait

    double t_end = 0.0;
    double t1 = mytimer();
    double dt2 = t1 - t0;

    // waxpby(nrow, 1.0, x, 0.0, x, p)
    Index_t bs;
    for (bs = 0; bs < n; bs += BS)
    {
        # pragma omp task default(none) \
            firstprivate(bs)            \
            shared(p, x, ncol, BS, n)   \
            depend(in: x[bs])           \
            depend(out: p[bs])
        {
            waxpby(MIN(bs + BS, n) - bs, 1.0, x + bs, 0.0, x + bs, p + bs);
        }
    }

    HPC_sparsemv(A, p, Ap);

    // waxpby(nrow, 1.0, b, -1.0, Ap, r)
    for (bs = 0; bs < n; bs += BS)
    {
        # pragma omp task default(none)                         \
            firstprivate(bs)                                    \
            shared(BS, r, b, Ap, n)                             \
            depend(in: b[bs])                                   \
            depend(iterator(t2=0:T2), in: Ap[bs + t2 * SBS])    \
            depend(out: r[bs])
        {
            waxpby(MIN(bs + BS, n) - bs, 1.0, b + bs, -1.0, Ap + bs, r + bs);
        }
    }


    // ddot(nrow, r, r, &rtrans, t4)
    for (bs = 0; bs < n; bs += BS)
    {
        # pragma omp task default(none)             \
            firstprivate(bs)                        \
            shared(BS, n, r, rtrans_per_block)      \
            depend(in: r[bs])                       \
            depend(out: rtrans_per_block[bs/BS])
        {
            Index_t end = MIN(bs + BS, n);
            Index_t i;
            rtrans_per_block[bs/BS] = 0;
            for (i = bs; i < end; ++i)
            {
                rtrans_per_block[bs/BS] += r[i] * r[i];
            }
        }
    }

    // MPI reduce to get partial sum from every ranks
    # pragma omp task default(none)                         \
        shared(                                             \
            rtrans, normr, r, cout,                         \
            T1, rtrans_per_block, OMP_MPI_SHARED            \
        )                                                   \
        depend(iterator(t1=0:T1), in: rtrans_per_block[t1]) \
        depend(out: rtrans, normr)
    {
        Index_t t1;
        rtrans = 0;
        for (t1 = 0 ; t1 < T1 ; ++t1)
        {
            rtrans += rtrans_per_block[t1];
        }
#ifdef USING_MPI
        MPI_Allreduce(MPI_IN_PLACE, &rtrans, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif /* USING_MPI */

        // update_normr
        normr = sqrt(rtrans);
        if (rank == 0)
        {
            cout << "Initial Residual = " << normr << endl;
        }
    }

#ifdef USING_MPI
    exchange_externals(A, p, 0);
#endif

    /* wrap iterations within a taskgroup, for cancellation */
    # pragma omp taskgroup
    {
        // start iterations
        volatile Index_t continu = 1;
        Index_t k;
        for (k = 1; k < max_iter && continu; ++k)
        {
            if (k == 1)
            {
                // waxpby(nrow, 1.0, r, 0.0, r, p)
                for (bs = 0; bs < n; bs += BS)
                {
                    # pragma omp task default(none) \
                        firstprivate(bs)            \
                        shared(p, r, BS, n)         \
                        depend(in: r[bs])           \
                        depend(out: p[bs])
                    {
                        waxpby(MIN(bs + BS, n) - bs, 1.0, r + bs, 0.0, r + bs, p + bs);
                    }
                }
            }
            else
            {
                for (bs = 0; bs < n; bs += BS)
                {
                    # pragma omp task default(none)                 \
                        firstprivate(bs)                            \
                        shared(rtrans, r, BS, n, rtrans_per_block)  \
                        depend(in: r[bs])                           \
                        depend(out: rtrans_per_block[bs/BS])
                    {
                        Index_t end = MIN(bs + BS, n);
                        Index_t i;
                        rtrans_per_block[bs/BS] = 0;
                        for (i = bs; i < end; ++i)
                        {
                            rtrans_per_block[bs/BS] += r[i] * r[i];
                        }
                    }
                }

                // MPI reduce to get partial sum from every ranks
                # pragma omp task default(none)                         \
                    shared(                                             \
                        beta, rtrans, OMP_MPI_SHARED,                   \
                        rtrans_per_block, T1                            \
                    )                                                   \
                    depend(iterator(t1=0:T1), in: rtrans_per_block[t1]) \
                    depend(out: rtrans, beta)
                {

                    double oldrtrans = rtrans;
                    rtrans = 0;

                    Index_t t1;
                    for (t1 = 0 ; t1 < T1 ; ++t1)
                    {
                        rtrans += rtrans_per_block[t1];
                    }

#ifdef USING_MPI
                    MPI_Allreduce(MPI_IN_PLACE, &rtrans, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif /* USING_MPI */
                    beta = rtrans / oldrtrans;
                }

                // waxpby(nrow, 1.0, r, beta, p, p)
                for (bs = 0; bs < n; bs += BS)
                {
                    # pragma omp task default(none) \
                        firstprivate(bs)            \
                        shared(p, r, beta, BS, n)   \
                        depend(in: r[bs], beta)     \
                        depend(inout: p[bs])
                    {
                        waxpby(MIN(bs + BS, n) - bs, 1.0, r + bs, beta, p + bs, p + bs);
                    }
                }
            }

            // update_normr(rank, k, print_freq, max_iter, tolerance, normr, rtrans, continu)
            # pragma omp task default(none)                         \
                firstprivate(k)                                     \
                shared(                                             \
                    normr, rtrans, continu, tolerance, print_freq,  \
                    max_iter, cout, rank, niters, t_end             \
                )                                                   \
                depend(in: rtrans)                                  \
                depend(out: normr, continu)
            {
                niters = k;
                normr = sqrt(rtrans);
                continu = normr > tolerance;
                if (rank == 0 && (k % print_freq == 0 || k + 1 == max_iter))
                {
                    cout << "Iteration=" << k << " Residual=" << normr << " Tolerance=" << tolerance << endl;
                }
                if (!continu)
                {
                    t_end = mytimer();
                    //# pragma omp cancel taskgroup
                }
            }

#ifdef USING_MPI
            exchange_externals(A, p, k);
#endif
            HPC_sparsemv(A, p, Ap);

            // ddot(nrow, p, Ap, beta)
            for (bs = 0 ; bs < n ; bs += BS)
            {
                # pragma omp task default(none)                         \
                    firstprivate(bs)                                    \
                    shared(p, Ap, beta, n, BS, beta_per_block)          \
                    depend(in: p[bs])                                   \
                    depend(iterator(t2=0:T2), in: Ap[bs + t2 * SBS])    \
                    depend(out: beta_per_block[bs/BS])
                {
                    Index_t end = MIN(bs + BS, n);
                    Index_t i;
                    beta_per_block[bs/BS] = 0;
                    for (i = bs; i < end; ++i)
                    {
                        beta_per_block[bs/BS] += p[i] * Ap[i];
                    }
                }
            }

            // MPI reduce to get partial sum from every ranks
            # pragma omp task default(none)                                 \
                shared(beta, rtrans, OMP_MPI_SHARED, beta_per_block, T1)    \
                depend(iterator(t1=0:T1), in: beta_per_block[t1])           \
                depend(in: rtrans)                                          \
                depend(out: beta)
            {
                Index_t t1;
                beta = 0;
                for (t1 = 0 ; t1 < T1 ; ++t1)
                {
                    beta += beta_per_block[t1];
                }
#ifdef USING_MPI
                MPI_Allreduce(MPI_IN_PLACE, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif /* USING_MPI */
                beta = rtrans / beta;
            }

            for (bs = 0; bs < n; bs += BS)
            {
                // waxpby(nrow, 1.0, x,  alpha,  p, x)
                # pragma omp task default(none) \
                    firstprivate(bs)            \
                    shared(x, beta, p, BS, n)   \
                    depend(in: p[bs], beta)     \
                    depend(inout: x[bs])
                {
                    waxpby(MIN(bs + BS, n) - bs, 1.0, x + bs, beta, p + bs, x + bs);
                }

                // waxpby(nrow, 1.0, r, -alpha, Ap, r)
                # pragma omp task default(none)                         \
                    firstprivate(bs)                                    \
                    shared(r, beta, Ap, BS, n)                          \
                    depend(in: beta)                                    \
                    depend(iterator(t2=0:T2), in: Ap[bs + t2 * SBS])    \
                    depend(inout: r[bs])
                {
                    waxpby(MIN(bs + BS, n) - bs, 1.0, r + bs, -beta, Ap + bs, r + bs);
                }
            }
        } /* for k */

        double dt1 = mytimer() - t1;
        # pragma omp taskwait
        if (t_end == 0.0) t_end = mytimer();
        double dt_cancel = mytimer() - t_end;

        // Store times
        times[0] = t_end - t1;      // Total time. All done...
        times[1] = -1;              // ddot time
        times[2] = -1;              // waxpby time
        times[3] = -1;              // sparsemv time
        times[4] = -1;              // AllReduce time
#ifdef USING_MPI
        times[5] = -1;  // exchange boundary time
#endif
        times[7] = dt1;  // graph generation
        times[8] = dt2;  // dependencies precomputation
        times[9] = dt_cancel; // time to cancel tasks

    } /* taskgroup */

    __HPC_sparsemv_deps_deinit(A);
#ifdef USING_MPI
    __exchange_externals_deps_deinit(A);
#endif /* USING_MPI */
    free(rtrans_per_block);
    free(beta_per_block);
    delete[] p;
    delete[] Ap;
    delete[] r;
}

int
HPCCG(HPC_Sparse_Matrix * A,
        const double * const b,
        double * const x,
        const int max_iter,
        const double tolerance,
        int & niters,
        double & normr,
        double * times)
{
    # pragma omp parallel default(shared)
    {
        # pragma omp single
        {
            HPCCG_single(A, b, x, max_iter, tolerance, niters, normr, times);
        }
    }
    return 0;
}
