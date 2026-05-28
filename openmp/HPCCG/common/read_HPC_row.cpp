
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

// Routine to read a sparse matrix, right hand side, initial guess,
// and exact solution (as computed by a direct solver).

/////////////////////////////////////////////////////////////////////////

// nrow - number of rows of matrix (on this processor)

#include <iostream>
using std::cerr;
using std::cout;
using std::endl;
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include "read_HPC_row.hpp"
void read_HPC_row(char * data_file,
                  HPC_Sparse_Matrix ** A,
                  double ** x,
                  double ** b,
                  double ** xexact)

{
    FILE * in_file;

    Index_t i, j;
    Index_t total_nrow;
    long long total_nnz;
    Index_t l;
    Index_t * lp = &l;
    double v;
    double * vp = &v;
#ifdef DEBUG
    int debug = 1;
#else
    int debug = 0;
#endif

    printf("Reading matrix info from %s...\n", data_file);

    in_file = fopen(data_file, "r");
    if (in_file == NULL)
    {
        printf("Error: Cannot open file: %s\n", data_file);
        exit(1);
    }

    fscanf(in_file, Index_t_FORMAT, &total_nrow);
    fscanf(in_file, Index_t_FORMAT, &total_nnz);
#ifdef USING_MPI
    int size, rank;  // Number of MPI processes, My process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
    int size = 1;  // Serial case (not using MPI)
    int rank = 0;
#endif
    Index_t n = total_nrow;
    Index_t chunksize = n / size;
    Index_t remainder = n % size;

    Index_t mp = chunksize;
    if (rank < remainder)
        mp++;
    Index_t local_nrow = mp;

    Index_t off = rank * (chunksize + 1);
    if (rank > remainder)
        off -= (rank - remainder);
    Index_t start_row = off;
    Index_t stop_row = off + mp - 1;

    // Allocate arrays that are of length local_nrow
    Index_t * nnz_in_row = new Index_t[local_nrow];
    double ** ptr_to_vals_in_row = new double *[local_nrow];
    Index_t ** ptr_to_inds_in_row = new Index_t *[local_nrow];
    double ** ptr_to_diags = new double *[local_nrow];

    *x = new double[local_nrow];
    *b = new double[local_nrow];
    *xexact = new double[local_nrow];

    // Find nnz for this processor
    Index_t local_nnz = 0;
    Index_t cur_local_row = 0;

    for (i = 0; i < total_nrow; i++)
    {
        fscanf(in_file, Index_t_FORMAT, lp);            /* row #, nnz in row */
        if (start_row <= i && i <= stop_row)  // See if nnz for row should be added
        {
            local_nnz += l;
            nnz_in_row[cur_local_row] = l;
            cur_local_row++;
        }
    }
    assert(cur_local_row == local_nrow);  // cur_local_row should equal local_nrow

    // Allocate arrays that are of length local_nnz
    double * list_of_vals = new double[local_nnz];
    Index_t * list_of_inds = new Index_t[local_nnz];

    // Define pointers into list_of_vals/inds
    ptr_to_vals_in_row[0] = list_of_vals;
    ptr_to_inds_in_row[0] = list_of_inds;
    for (i = 1; i < local_nrow; i++)
    {
        ptr_to_vals_in_row[i] = ptr_to_vals_in_row[i - 1] + nnz_in_row[i - 1];
        ptr_to_inds_in_row[i] = ptr_to_inds_in_row[i - 1] + nnz_in_row[i - 1];
    }

    cur_local_row = 0;
    for (i = 0; i < total_nrow; i++)
    {
        Index_t cur_nnz;
        fscanf(in_file, Index_t_FORMAT, &cur_nnz);
        if (start_row <= i && i <= stop_row)  // See if nnz for row should be added
        {
            if (debug)
                cout << "Process " << rank << " of " << size << " getting row " << i << endl;
            for (j = 0; j < cur_nnz; j++)
            {
                fscanf(in_file, "%lf " Index_t_FORMAT, vp, lp);
                ptr_to_vals_in_row[cur_local_row][j] = v;
                ptr_to_inds_in_row[cur_local_row][j] = l;
            }
            cur_local_row++;
        }
        else
            for (j = 0; j < cur_nnz; j++)
                fscanf(in_file, "%lf " Index_t_FORMAT, vp, lp);
    }

    cur_local_row = 0;
    double xt, bt, xxt;
    for (i = 0; i < total_nrow; i++)
    {
        if (start_row <= i && i <= stop_row)  // See if entry should be added
        {
            if (debug)
                cout << "Process " << rank << " of " << size << " getting RHS " << i << endl;
            fscanf(in_file, "%lf %lf %lf", &xt, &bt, &xxt);
            (*x)[cur_local_row] = xt;
            (*b)[cur_local_row] = bt;
            (*xexact)[cur_local_row] = xxt;
            cur_local_row++;
        }
        else
            fscanf(in_file, "%lf %lf %lf", vp, vp, vp);  // or thrown away
    }

    fclose(in_file);
    if (debug)
        cout << "Process " << rank << " of " << size << " has " << local_nrow;

    if (debug)
        cout << " rows. Global rows " << start_row << " through " << stop_row << endl;

    if (debug)
        cout << "Process " << rank << " of " << size << " has " << local_nnz << " nonzeros."
             << endl;

    *A = new HPC_Sparse_Matrix;  // Allocate matrix struct and fill it
    (*A)->title = 0;
    (*A)->start_row = start_row;
    (*A)->stop_row = stop_row;
    (*A)->total_nrow = total_nrow;
    (*A)->total_nnz = total_nnz;
    (*A)->local_nrow = local_nrow;
    (*A)->local_ncol = local_nrow;
    (*A)->local_nnz = local_nnz;
    (*A)->nnz_in_row = nnz_in_row;
    (*A)->ptr_to_vals_in_row = ptr_to_vals_in_row;
    (*A)->ptr_to_inds_in_row = ptr_to_inds_in_row;
    (*A)->ptr_to_diags = ptr_to_diags;

    return;
}
