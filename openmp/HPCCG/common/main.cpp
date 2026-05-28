
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

// Main routine of a program that reads a sparse matrix, right side
// vector, solution vector and initial guess from a file  in HPC
// format.  This program then calls the HPCCG conjugate gradient
// solver to solve the problem, and then prints results.

// Calling sequence:

// test_HPCCG linear_system_file

// Routines called:

// read_HPC_row - Reads in linear system

// mytimer - Timing routine (compile with -DWALL to get wall clock
//           times

// HPCCG - CG Solver

// compute_residual - Compares HPCCG solution to known solution.

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#ifdef USING_MPI
#include <mpi.h>                  // If this routine is compiled with -DUSING_MPI
                                  // then include mpi.h
#include "make_local_matrix.hpp"  // Also include this function
#endif
#ifdef USING_OMP
#include <omp.h>
#endif
#include "HPCCG.hpp"
#include "HPC_Sparse_Matrix.hpp"
#include "HPC_sparsemv.hpp"
#include "compute_residual.hpp"
#include "dump_matlab_matrix.hpp"
#include "generate_matrix.hpp"
#include "mytimer.hpp"
#include "read_HPC_row.hpp"

#include "YAML_Doc.hpp"
#include "YAML_Element.hpp"

#undef DEBUG

#if defined(USING_OMPSS)
#include <nanos6/debug.h>
#endif /* defined(USING_OMPSS) */

// number of tasks the decompose a `dot` or `saxpy`
// number of block is `n / T1`
size_t T1;

// number of tasks per block for spmv
// number of subblocks is T2
size_t T2;

int main(int argc, char * argv[])
{
    HPC_Sparse_Matrix * A;
    double *x, *b, *xexact;
    int ierr = 0;
    double times[16];
    Index_t nx, ny, nz;

#ifdef USING_MPI
    double t6 = 0.0;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(provided == MPI_THREAD_MULTIPLE);
    int size, rank;  // Number of MPI processes, My process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //  if (size < 100) cout << "Process "<<rank<<" of "<<size<<" is alive."
    //  <<endl;

#else
    int size = 1;  // Serial case (not using MPI)
    int rank = 0;
#endif

#ifdef DEBUG
    if (rank == 0)
    {
        int junk = 0;
        cout << "Press enter to continue" << endl;
        cin >> junk;
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (argc != 2 && argc != 7)
    {
        if (rank == 0)
            cerr << "Usage:" << endl
                 << "Mode 1: " << argv[0] << " nx ny nz max_iter num_tasks num_tasks_per_block" << endl
                 << "     where nx, ny and nz are the local sub-block "
                    "dimensions, or"
                 << endl
                 << "Mode 2: " << argv[0] << " HPC_data_file " << endl
                 << "     where HPC_data_file is a globally accessible file "
                    "containing matrix data."
                 << endl;
        return 1;
    }

    int nthreads = 1;
#ifdef USING_OMP
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            nthreads = omp_get_num_threads();
        }
    }
#elif defined(USING_OMPSS)
    nthreads = nanos6_get_num_cpus();
#endif

    int max_iter = 150;
    if (argc == 7)
    {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        nz = atoi(argv[3]);
        max_iter = atoi(argv[4]);
        T1 = atoi(argv[5]);
        T2 = atoi(argv[6]);
        generate_matrix(nx, ny, nz, &A, &x, &b, &xexact);
    }
    else
    {
#if 0
        read_HPC_row(argv[1], &A, &x, &b, &xexact);
#endif
        puts("nx,ny,nz not initialized");
        return 1;
    }

    bool dump_matrix = false;
    if (dump_matrix && size <= 4)
        dump_matlab_matrix(A, rank);

#ifdef USING_MPI

    // Transform matrix indices from global to local values.
    // Define number of columns for the local matrix.

    t6 = mytimer();
    make_local_matrix(A);
    t6 = mytimer() - t6;
    times[6] = t6;

#endif

    int niters = 0;
    double normr = 0.0;
    double tolerance = 0.0;  // Set tolerance to zero to make all runs do max_iter iterations
    //double tolerance = 10e-12;
    ierr = HPCCG(A, b, x, max_iter, tolerance, niters, normr, times);

    if (ierr)
        cerr << "Error in call to CG: " << ierr << ".\n" << endl;

#ifdef USING_MPI
    double t4 = times[4];
    double t4min = 0.0;
    double t4max = 0.0;
    double t4avg = 0.0;
    MPI_Allreduce(&t4, &t4min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&t4, &t4max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&t4, &t4avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    t4avg = t4avg / ((double)size);
#endif

    // initialize YAML doc

    if (rank == 0)  // Only PE 0 needs to compute and report timing results
    {
        double fniters = niters;
        double fnrow = A->total_nrow;
        double fnnz = A->total_nnz;
        double fnops_ddot = fniters * 4 * fnrow;
        double fnops_waxpby = fniters * 6 * fnrow;
        double fnops_sparsemv = fniters * 2 * fnnz;
        double fnops = fnops_ddot + fnops_waxpby + fnops_sparsemv;

        YAML_Doc doc("hpccg", "1.0");

        doc.add("Parallelism", "");

#ifdef USING_MPI
        doc.get("Parallelism")->add("Number of MPI ranks", size);
#else
        doc.get("Parallelism")->add("MPI not enabled", "");
#endif

#ifdef USING_OMP
        doc.get("Parallelism")->add("Number of OpenMP threads", nthreads);
        doc.get("Parallelism")->add("Number of tasks per loop (T1)", T1);
        doc.get("Parallelism")->add("Number of subtasks per SPMV (T2)", T2);
#elif defined(USING_OMPSS)
        doc.get("Parallelism")->add("Number of Nanos6 threads", nthreads);
        doc.get("Parallelism")->add("Number of tasks per loop (T1)", T1);
        doc.get("Parallelism")->add("Number of subtasks per SPMV (T2)", T2);
#else
        doc.get("Parallelism")->add("OpenMP not enabled", "");
        (void) nthreads;
#endif

        doc.add("Dimensions", "");
        doc.get("Dimensions")->add("nx", nx);
        doc.get("Dimensions")->add("ny", ny);
        doc.get("Dimensions")->add("nz", nz);

        doc.add("Number of iterations", niters);
        doc.add("Final residual", normr);
        doc.add("#********** Performance Summary (times in sec) ***********", "");

        doc.add("Time Summary", "");
        doc.get("Time Summary")->add("COMPUTATION               ", times[0]);
        doc.get("Time Summary")->add("DDOT                      ", times[1]);
        doc.get("Time Summary")->add("WAXPBY                    ", times[2]);
        doc.get("Time Summary")->add("SPARSEMV                  ", times[3]);
        doc.get("Time Summary")->add("GRAPH GENERATION          ", times[7]);
        doc.get("Time Summary")->add("DEPENDENCY PRE-COMPUTATION", times[8]);
        doc.get("Time Summary")->add("GRAPH CANCELLATION        ", times[9]);

        doc.add("FLOPS Summary", "");
        doc.get("FLOPS Summary")->add("Total   ", fnops);
        doc.get("FLOPS Summary")->add("DDOT    ", fnops_ddot);
        doc.get("FLOPS Summary")->add("WAXPBY  ", fnops_waxpby);
        doc.get("FLOPS Summary")->add("SPARSEMV", fnops_sparsemv);

        doc.add("MFLOPS Summary", "");
        doc.get("MFLOPS Summary")->add("Total   ", fnops / times[0] / 1.0E6);
        doc.get("MFLOPS Summary")->add("DDOT    ", fnops_ddot / times[1] / 1.0E6);
        doc.get("MFLOPS Summary")->add("WAXPBY  ", fnops_waxpby / times[2] / 1.0E6);
        doc.get("MFLOPS Summary")->add("SPARSEMV", fnops_sparsemv / (times[3]) / 1.0E6);

#ifdef USING_MPI
        doc.add("DDOT Timing Variations", "");
        doc.get("DDOT Timing Variations")->add("Min DDOT MPI_Allreduce time", t4min);
        doc.get("DDOT Timing Variations")->add("Max DDOT MPI_Allreduce time", t4max);
        doc.get("DDOT Timing Variations")->add("Avg DDOT MPI_Allreduce time", t4avg);

        double totalSparseMVTime = times[3] + times[5] + times[6];
        doc.add("SPARSEMV OVERHEADS", "");
        doc.get("SPARSEMV OVERHEADS")
            ->add("SPARSEMV MFLOPS W OVERHEAD", fnops_sparsemv / (totalSparseMVTime) / 1.0E6);
        doc.get("SPARSEMV OVERHEADS")
            ->add("SPARSEMV PARALLEL OVERHEAD Time", (times[5] + times[6]));
        doc.get("SPARSEMV OVERHEADS")
            ->add("SPARSEMV PARALLEL OVERHEAD Pct",
                  (times[5] + times[6]) / totalSparseMVTime * 100.0);
        doc.get("SPARSEMV OVERHEADS")->add("SPARSEMV PARALLEL OVERHEAD Setup Time", (times[6]));
        doc.get("SPARSEMV OVERHEADS")
            ->add("SPARSEMV PARALLEL OVERHEAD Setup Pct", (times[6]) / totalSparseMVTime * 100.0);
        doc.get("SPARSEMV OVERHEADS")->add("SPARSEMV PARALLEL OVERHEAD Bdry Exch Time", (times[5]));
        doc.get("SPARSEMV OVERHEADS")
            ->add("SPARSEMV PARALLEL OVERHEAD Bdry Exch Pct",
                  (times[5]) / totalSparseMVTime * 100.0);
#endif

        if (rank == 0)
        {  // only PE 0 needs to compute and report timing results
            std::string yaml = doc.generateYAML();
            cout << yaml;
        }
    }

    // Compute difference between known exact solution and computed solution
    // All processors are needed here.
    // double residual;
    //  if ((ierr = compute_residual(A->local_nrow, x, xexact, &residual)))
    //  cerr << "Error in call to compute_residual: " << ierr << ".\n" << endl;

    // if (rank==0)
    //   cout << "Difference between computed and exact  = "
    //        << residual << ".\n" << endl;

    // Finish up
#ifdef USING_MPI
    MPI_Finalize();
#endif
    return 0;
}
