
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

#ifdef USING_MPI  // Compile this routine only if running in parallel
#include <iostream>
using std::cerr;
using std::endl;
#include <cstdio>
#include <cstdlib>
#include "exchange_externals.hpp"
#undef DEBUG
void exchange_externals(HPC_Sparse_Matrix* A, const double* x)
{
    int i;

    // Extract Matrix pieces

    Index_t local_nrow = A->local_nrow;
    Index_t num_neighbors = A->num_send_neighbors;
    Index_t * recv_length = A->recv_length;
    Index_t * send_length = A->send_length;
    Index_t * neighbors = A->neighbors;
    double * send_buffer = A->send_buffer;
    Index_t total_to_be_sent = A->total_to_be_sent;
    Index_t * elements_to_send = A->elements_to_send;

    int size, rank;  // Number of MPI processes, My process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //
    //  first post receives, these are immediate receives
    //  Do not wait for result to come, will do that at the
    //  wait call below.
    //

    int MPI_MY_TAG = 99;

    MPI_Request* request = new MPI_Request[num_neighbors];

    //
    // Externals are at end of locals
    //
    double* x_external = (double*)x + local_nrow;

    // Post receives first
    for (i = 0; i < num_neighbors; i++)
    {
        Index_t n_recv = recv_length[i];
        MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG,
                  MPI_COMM_WORLD, request + i);
        x_external += n_recv;
    }

    //
    // Fill up send buffer
    //

    for (i = 0; i < total_to_be_sent; i++)
        send_buffer[i] = x[elements_to_send[i]];

    //
    // Send to each neighbor
    //

    for (i = 0; i < num_neighbors; i++)
    {
        Index_t n_send = send_length[i];
        MPI_Send(send_buffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG,
                 MPI_COMM_WORLD);
        send_buffer += n_send;
    }

    //
    // Complete the reads issued above
    //

    MPI_Status status;
    for (i = 0; i < num_neighbors; i++)
    {
        if (MPI_Wait(request + i, &status))
        {
            cerr << "MPI_Wait error\n" << endl;
            exit(-1);
        }
    }

    delete[] request;
}
#endif  // USING_MPI
