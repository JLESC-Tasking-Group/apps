#ifdef MPI
    #include <TAMPI.h>
    #include <mpi.h>
#endif

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "mpi_scheduled.h"
#include "mpi_utils.h"

static inline int MIN(int a, int b) { return a < b ? a : b; }
static inline int MAX(int a, int b) { return a > b ? a : b; }

struct mpi_user_def_op_args_t {
    void *in;
    void *inout;
    int *len;
    
    #ifdef MPI
        MPI_Datatype *data_type;
    #endif 
};

struct comm_block_t {
    int covered_subgrads_count;
    int *dependencies_offsets;
    int first_data_offset;
    int *slices_length;
    int block_size;
    int root;

    #ifdef MPI
        MPI_Comm *block_slices_comm;
    #endif
};

typedef struct MPIWorker_internal_t {
    int rank;
    int worldsize;

    #ifdef MPI
        MPI_Comm nextbatch;
        MPI_Comm loss_comm;
        MPI_Comm inference_comm;
        MPI_Comm *block_comms;
        MPI_Op parallel_sum;
    #endif 

    int B; // Batch size processed by this worker
    int T; // Sequence size processed by this worker
    int world_B;
    int world_T;

    struct comm_block_t *comm_blocks;
    int comm_blocks_count;
} MPIWorker_internal;

// TAMPI interface
#ifdef MPI

// MPI_Reduce wrapper for the different cases : tampi or tampi/git, OpenMP or OmpSs-2 or sequential
int _TAMPI_Ireduce(const void *send, void *recv, int length, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm) {
#if (defined(OSS) && defined(MPI))
    TAMPI_Ireduce(send, recv, length, type, op, root, comm);
#endif
#if (defined(OMP) && defined(MPI))
    MPI_Reduce(send, recv, length, type, op, root, comm);
#endif

    return MPI_SUCCESS;
}

// MPI_Bcast wrapper for the different cases : tampi or tampi/git, OpenMP or OmpSs-2 or sequential
int _TAMPI_Ibcast(void *ptr, int length, MPI_Datatype type, int root, MPI_Comm comm) {
#if (defined(OSS) && defined(MPI))
    TAMPI_Ibcast(ptr, length, type, root, comm);
#endif
#if (defined(OMP) && defined(MPI))
    MPI_Bcast(ptr, length, type, root, comm);
#endif
    return MPI_SUCCESS;
}

#endif

// Internal functions

// Function to be called by mpi_sum_wrapper (indirectly)
// It performs sum between two arrays in a MPI reduction
// In order to beneficiate from OmpSs-2, this function must be inside nosV runtime
void mpi_ompss_sum(void *_args) {
    struct mpi_user_def_op_args_t *args = (struct mpi_user_def_op_args_t *)_args;
    float *in_float = (float *)args->in;
    float *inout_float = (float *)args->inout;
    int *len = (int *)args->len;
    int mtx;

#pragma oss taskloop concurrent(mtx) firstprivate(in_float, inout_float, len) grainsize(MPI_PARALLEL_SUM_TRIGGER_SIZE) label("MPI_SUM")
    for (int i = 0; i < *len; i++) {
        inout_float[i] += in_float[i];
    }

#pragma oss taskwait on(mtx)
}

// This function is directly called by MPI as a user-defined reduction function.
// Here, we are outside of nosV runtime, thus we will call mpi_launch_on_scheduler
// which will spawn mpi_ompss_sum inside nosV scheduler
#if (defined MPI) && (defined OSS)
void mpi_sum_wrapper(void *in, void *inout, int *len, MPI_Datatype *data_type) {
    assert(*data_type == MPI_FLOAT);

    if (*len > MPI_PARALLEL_SUM_TRIGGER_SIZE) { // Only parallelize the reduction if enough elements to reduce
        struct mpi_user_def_op_args_t op_args = {in, inout, len, data_type};
        mpi_launch_on_scheduler(mpi_ompss_sum, &op_args, "MPI_SUM");
    } else { // Otherwise do a classic sum
        float *inout_float = (float *)inout;
        float *in_float = (float *)in;
        for (int i = 0; i < *len; i++)
            inout_float[i] += in_float[i];
    }
}
#endif

// Calculate how to slice the tokens among the ranks
void mpi_batch_split(MPIWorker worker, int B, int T) {
    assert(MAX(worker->worldsize, B) % MIN(worker->worldsize, B) == 0);

    if (worker->worldsize > B) {
        worker->T = (T * B) / worker->worldsize;
        worker->B = 1;
    } else {
        worker->T = T;
        worker->B = B / worker->worldsize;
    }
}

// Initialize the MPIWorker structure
MPIWorker mpi_init(int *argc, char **argv[], int B, int T) 
{
    #ifdef __cplusplus
    MPIWorker worker = new MPIWorker_internal;
    #else
    MPIWorker worker = malloc(sizeof(MPIWorker_internal));
    #endif

    worker->world_B = B;
    worker->world_T = T;

#ifdef MPI
    int provided;
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
    assert(provided == MPI_THREAD_MULTIPLE);

    MPI_Comm_rank(MPI_COMM_WORLD, &worker->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worker->worldsize);

    MPI_Comm_dup(MPI_COMM_WORLD, &worker->nextbatch);      // Used to scatter the inputs & targets
    MPI_Comm_dup(MPI_COMM_WORLD, &worker->loss_comm);      // Used to reduce the forward pass loss
    MPI_Comm_dup(MPI_COMM_WORLD, &worker->inference_comm); // Used to send generated tokens during inference

    // Calculate the batch size and sequence size to be used on a rank.
    // Compatible for sequence splitting
    mpi_batch_split(worker, B, T);
    if (worker->rank == 0)
        printf("MPI batch split :\t World : (B=%d, T=%d)   ---->   Rank : (B=%d, T=%d)\n", B, T, worker->B, worker->T);

    assert(worker->B * worker->T * worker->worldsize == B * T);
    assert(B % worker->B == 0);
    assert(T % worker->T == 0);
    assert(T == worker->T); // Sequence splitting is not allowed on this distributed version

    // Create our own MPI_SUM Operator, which will run a parallel sum, unlike MPI_SUM which is sequential
    #ifdef OSS
    MPI_Op_create(&mpi_sum_wrapper, 1, &worker->parallel_sum);
    #endif
#else
    worker->B = B;
    worker->T = T;
    worker->rank = 0;
    worker->worldsize = 1;
#endif

    worker->comm_blocks = NULL; // Will be initialized lazily during the update phase
    worker->comm_blocks_count = 0;
    return worker;
}

#ifdef MPI
// Initialize the communication structures
void comm_block_init(struct comm_block_t *comm_block, float **ptrs, int *lengths, int n, int start_offset, int stop_offset, int block_size,
                     float *ref_ptr) {
    // comm_blocks is the communications block to initialize,
    // ptrs are pointers towards each slice of the gradients array, but ordered along the order in which there are called in gpt2_backward
    // lengths are the lengths of these slices
    // n is how many slices the communcation block needs to contain
    // start_offset is the offset to add to the first pointer to get the start of our slice
    // stop_offset is how many elements the last slice of communication block will contain
    // block_size is the summed length of all the slices contained inside the communication block
    // ref_ptr is the pointer towards the beginning of the grads_memory array

    // We will mostly work with offset and not pointers, as the offsets are the same for the params_memory and grads_memory arrays
    // The dependencies_offsets are both used to register where the dependencies are, and where the first elements of the sliced arrays
    // are, with exeption for the first element, where those two are not at the same place.
    comm_block->dependencies_offsets = malloc(n * sizeof(int));
    comm_block->covered_subgrads_count = n;
    comm_block->first_data_offset = (int)(ptrs[0] - ref_ptr) + start_offset;
    if (n == 1 && stop_offset == 0) // If we have only one slice in the communication block
        comm_block->block_size = lengths[0] - start_offset;
    else
        comm_block->block_size = block_size;

    for (int i = 0; i < n; i++)
        comm_block->dependencies_offsets[i] = (int)(ptrs[i] - ref_ptr);

    comm_block->slices_length = malloc(n * sizeof(int));
    memcpy(comm_block->slices_length, lengths, n * sizeof(int));
    comm_block->slices_length[n - 1] = stop_offset ? stop_offset : comm_block->slices_length[n - 1];
    comm_block->slices_length[0] -= start_offset;
    comm_block->block_slices_comm = malloc(n * sizeof(MPI_Comm));
    for (int i = 0; i < n; i++)
        MPI_Comm_dup(MPI_COMM_WORLD, comm_block->block_slices_comm + i);
}

// Free a communcation block
void comm_block_destroy(struct comm_block_t *comm_block) {
    free(comm_block->dependencies_offsets);
    free(comm_block->slices_length);
    for (int i = 0; i < comm_block->covered_subgrads_count; i++)
        MPI_Comm_free(comm_block->block_slices_comm + i);
}

// Destroy the communication block array
void comm_destroy(struct comm_block_t *comm_blocks, int n) {
    for (int i = 0; i < n; i++)
        comm_block_destroy(comm_blocks + i);
    free(comm_blocks);
}

// Scattering token inputs or targets
void mpi_scatter_inputs(MPIWorker worker, int *inputs) {
    if (worker->rank == 0)
        MPI_Scatter(inputs, worker->T * worker->B, MPI_INT, MPI_IN_PLACE, worker->T * worker->B, MPI_INT, 0, worker->nextbatch);
    else
        MPI_Scatter(inputs, worker->T * worker->B, MPI_INT, inputs, worker->T * worker->B, MPI_INT, 0, worker->nextbatch);
}

#endif

// Free all MPI related structures
void mpi_finalize(MPIWorker worker) {
#ifdef MPI
    // if (worker->comm_blocks != NULL)
    //     comm_destroy(worker->comm_blocks, worker->comm_blocks_count);
    MPI_Comm_free(&worker->nextbatch);
    MPI_Comm_free(&worker->loss_comm);
    MPI_Comm_free(&worker->inference_comm);
    #ifdef OSS
    MPI_Op_free(&worker->parallel_sum);
    #endif

    int comms = (int) worker->comm_blocks;
    for (int i = 0; i < comms; ++i)
        MPI_Comm_free(&worker->block_comms[i]);

    MPI_Finalize();
#endif
    free(worker);
}

#ifdef MPI

// Calculate which subslices a block contains, where it starts and stops
// The aim is to be able to retrieve a block of size block_size
int get_comm_block_caracteristics(int *offset_out, int *subgrads_sizes, int subgrads_count, int offset, int block_size) {
    int covered_memzone = 0;
    int curr_size = subgrads_sizes[0] - offset;
    for (int i = 0; i < subgrads_count; i++) {
        if (covered_memzone + curr_size >= block_size) {
            *offset_out = block_size - covered_memzone + (i == 0 ? offset : 0);
            return i + 1;
        }
        covered_memzone += curr_size;
        curr_size = subgrads_sizes[MIN(i + 1, subgrads_count - 1)];
    }
    *offset_out = 0;
    return subgrads_count;
}

// Performs a optimized reduction if we are using tampi/git and OmpSs-2 otherwise falls back to classic MPI_SUM
void mpi_reduce(MPIWorker worker, float *ptr, int length, int root, MPI_Comm *comm) {
#if defined OSS && defined TAMPI_VERSION_MAJOR
    MPI_Op op = worker->parallel_sum;
#else
    MPI_Op op = MPI_SUM;
#endif
    if (worker->rank == root) {
        _TAMPI_Ireduce(MPI_IN_PLACE, ptr, length, MPI_FLOAT, op, root, *comm);
    } else {
        _TAMPI_Ireduce(ptr, ptr, length, MPI_FLOAT, op, root, *comm);
    }
}

// broadcast wrapper
void mpi_broadcast(float *ptr, int length, int root, MPI_Comm *comm) { _TAMPI_Ibcast(ptr, length, MPI_FLOAT, root, *comm); }

void mpi_reduce_block(MPIWorker worker, float *ptr, int length, int root, int comm_idx)
{
    mpi_reduce(worker, ptr, length, root, &worker->block_comms[comm_idx]);
}

void mpi_bcast_block(MPIWorker worker, float *ptr, int length, int root, int comm_idx)
{
    mpi_broadcast(ptr, length, root, &worker->block_comms[comm_idx]);
}

#endif

// Interface

#ifdef MPI

// Performs a reduction of the gradients, by using the communcations blocks
void mpi_share_gradients(float *grads_memory, MPIWorker worker) {
    for (int block_id = 0; block_id < worker->comm_blocks_count; block_id++) { // for each communication block
        struct comm_block_t *block = worker->comm_blocks + block_id;
        for (int block_slice_id = 0; block_slice_id < block->covered_subgrads_count; block_slice_id++) { // For each slice in this block
            float *slice_ptr = grads_memory + (block_slice_id ? block->dependencies_offsets[block_slice_id] : block->first_data_offset);
            int slice_length = block->slices_length[block_slice_id];
            float *grads_dependency_ptr = grads_memory + block->dependencies_offsets[block_slice_id];
#pragma oss task firstprivate(worker, slice_ptr, slice_length, block, block_slice_id, grads_dependency_ptr, block_id)                                     \
    concurrent(worker->comm_blocks->root) in(*grads_dependency_ptr)in(block->root)
            mpi_reduce(worker, slice_ptr, slice_length, block->root, block->block_slices_comm + block_slice_id); // Reduce the slice
        }
    }
}

// Broadcasts the parameters among the ranks
void mpi_broadcast_parameters(float *params_memory, MPIWorker worker) {
    for (int block_id = 0; block_id < worker->comm_blocks_count; block_id++) { // for each communication block
        struct comm_block_t *block = worker->comm_blocks + block_id;
        for (int block_slice_id = 0; block_slice_id < block->covered_subgrads_count; block_slice_id++) { // For each slice in this block
            float *slice_ptr = params_memory + (block_slice_id ? block->dependencies_offsets[block_slice_id] : block->first_data_offset);
            int slice_length = block->slices_length[block_slice_id];
            float *params_dependency_ptr = params_memory + block->dependencies_offsets[block_slice_id];
#pragma oss task firstprivate(slice_ptr, slice_length, block, block_slice_id) in(*params_dependency_ptr, block->root)in(worker->comm_blocks->root)
            mpi_broadcast(slice_ptr, slice_length, block->root, block->block_slices_comm + block_slice_id); // Broadcast the slice
        }
    }
}

// Reduces the loss in gpt2_forward to print it out
void mpi_loss_reduce(MPIWorker worker, float *send, float *recv) {
#if defined OSS && defined MPI
    TAMPI_Ireduce(send, recv, 1, MPI_FLOAT, MPI_SUM, 0, worker->loss_comm);
#endif
#if defined OMP && defined MPI
    MPI_Reduce(send, recv, 1, MPI_FLOAT, MPI_SUM, 0, worker->loss_comm);
#endif
}

#endif

// As each ranki is responsible for only a slice of the parameters, a rank only have to allocate as much memory as we have parameters to update
size_t mpi_get_adam_buffer_size(MPIWorker worker) {
    size_t adam_memory_size = 0;
    for (int block_id = 0; block_id < worker->comm_blocks_count; block_id++) {
        struct comm_block_t *block = worker->comm_blocks + block_id;
        adam_memory_size += block->root == worker->rank ? block->block_size : 0;
    }
    return adam_memory_size;
}

// Retrieve all the parameters slices the rank is responsible to update, and update them
void mpi_update_params_slice(void *model, void (*update_func)(void *, int, int, int, float, float, float, float, float, int *, int, int *),
                             MPIWorker worker, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int *t) {

    int adam_offset = 0;
    for (int block_id = 0; block_id < worker->comm_blocks_count; block_id++) { // For each communication block
        struct comm_block_t *block = worker->comm_blocks + block_id;
        if (block->root == worker->rank) { // If the rank is reponsible for this block's update
            for (int block_slice_id = 0; block_slice_id < block->covered_subgrads_count; block_slice_id++) { // For each block's slice
                int slice_length = block->slices_length[block_slice_id];
                int slice_offset = block_slice_id ? block->dependencies_offsets[block_slice_id] : block->first_data_offset;

                for (int taskloop = 0; taskloop < slice_length; taskloop += UPDATE_TASKLOOP_GRAINSIZE) { // Flatten taskloop (for taskiter)
                    int subslice_size = MIN(UPDATE_TASKLOOP_GRAINSIZE, slice_length - taskloop);
                    update_func(model, slice_offset, adam_offset, subslice_size, learning_rate, beta1, beta2, eps, weight_decay, t,
                                block->dependencies_offsets[block_slice_id], &block->root); // Update the slice

                    slice_offset += subslice_size;
                    adam_offset += subslice_size;
                }
            }
        }
    }
}

#ifdef MPI
// Initialize the communication block array
void mpi_init_communications(MPIWorker worker, int comms)
{
    worker->block_comms = malloc(comms * sizeof(MPI_Comm));
    for (int i = 0; i < comms; ++i) {
        MPI_Comm_dup(MPI_COMM_WORLD, &worker->block_comms[i]);
    }

    worker->comm_blocks = (void *)comms;

    // printf("Creating %d commns\n", comms);

    // assert(model_num_parameters < INT_MAX); // Limitation from MPI

    // // We are defining memory blocks along the gradients array. A memory block is not contiguous in memory,
    // // but its concatenated size will be block_size, with exception for the last one which will be shorter.
    // int blocks_count = model_num_parameters / block_size + (model_num_parameters % block_size ? 1 : 0);
    // int tmp_subgrads_count = arr_size;

    // assert(subgrads_count < tmp_subgrads_count); // If fails, just increase tmp_subgrads_count (temporal buffer size)

    // worker->comm_blocks = malloc(blocks_count * sizeof(struct comm_block_t));
    // worker->comm_blocks_count = blocks_count;
    // int next_block_offset = 0;
    // int last_block_offset = 0;
    // int tot_blocks_covered = 0;
    // for (int block = 0; block < blocks_count; block++) { // Initialize all communication blocks
    //     last_block_offset = next_block_offset;
    //     int subgrads_covered = get_comm_block_caracteristics(&next_block_offset, subgrads_sizes + tot_blocks_covered, subgrads_count - tot_blocks_covered,
    //                                                          last_block_offset, block_size);

    //     comm_block_init(worker->comm_blocks + block, subgrads_ptrs + tot_blocks_covered, subgrads_sizes + tot_blocks_covered, subgrads_covered,
    //                     last_block_offset, next_block_offset, block_size, grads_memory);
    //     worker->comm_blocks[block].root = block % worker->worldsize;
    //     tot_blocks_covered += subgrads_covered - (next_block_offset ? 1 : 0);
    // }
    // assert(tot_blocks_covered == subgrads_count); // Check if we have the correct number of sub-blocks (or block's slices) we expected
}

// Share the produced token during an inference iteration
void mpi_inference_share_gen_token(MPIWorker worker, int sender, int receiver, int next_token, int *gen_token) {
    if (worker->rank == sender) {
        if (sender == receiver) // This rank is the next computing rank : no need for communication
            *gen_token = next_token;
        else // Send generated token to next computing rank
            MPI_Send(&next_token, 1, MPI_INT, receiver, 0, worker->inference_comm);
        if (worker->rank != 0 && receiver != 0) { // Only rank 0 is allowed to print tokens
            MPI_Send(&next_token, 1, MPI_INT, 0, 0, worker->inference_comm);
        }
    }
    if (worker->rank == receiver && sender != receiver)
        MPI_Recv(gen_token, 1, MPI_INT, sender, 0, worker->inference_comm, MPI_STATUS_IGNORE);
    else if (worker->rank == 0 && receiver != 0)
        MPI_Recv(gen_token, 1, MPI_INT, sender, 0, worker->inference_comm, MPI_STATUS_IGNORE);
}
#endif

// Returns the batch size computed on one rank
int mpi_get_rank_B(MPIWorker worker) { return worker->B; }

// Returns the token sequence size computed on one rank
int mpi_get_rank_T(MPIWorker worker) { return worker->T; }

// Returns the batch size
int mpi_get_world_B(MPIWorker worker) { return worker->world_B; }

// Returns the token sequence size
int mpi_get_world_T(MPIWorker worker) { return worker->world_T; }

// Returns rank's id
int mpi_get_rank(MPIWorker worker) { return worker->rank; }

// Returns worldsize
int mpi_get_worldsize(MPIWorker worker) { return worker->worldsize; }

// Returns weither the communications blocks have already been initialized
int mpi_is_comm_initialized(MPIWorker worker) { return (worker->comm_blocks != NULL); }

// #include <assert.h>
// #include <limits.h>
// #include <omp.h>
// #include <stdio.h>
// #include <string.h>

// #include "mpi_scheduled.h"
// #include "mpi_utils.h"

// static inline int MIN(int a, int b) { return a < b ? a : b; }
// static inline int MAX(int a, int b) { return a > b ? a : b; }

// struct mpi_user_def_op_args_t {
//     void *in;
//     void *inout;
//     int *len;
// };

// struct comm_block_t {
//     int covered_subgrads_count;
//     int *dependencies_offsets;
//     int first_data_offset;
//     int *slices_length;
//     int block_size;
//     int root;
// };

// typedef struct MPIWorker_internal_t {
//     int rank;
//     int worldsize;
//     int B; // Batch size processed by this worker
//     int T; // Sequence size processed by this worker
//     int world_B;
//     int world_T;

//     struct comm_block_t *comm_blocks;
//     int comm_blocks_count;
// } MPIWorker_internal;

// // TAMPI interface

// // MPI_Reduce wrapper for the different cases : tampi or tampi/git, OpenMP or OmpSs-2 or sequential
// // int _TAMPI_Ireduce(const void *send, void *recv, int length, MPI_Datatype type, MPI_Op op, int root, MPI_Comm comm) {
// // #if (defined(OSS) && defined(MPI))
// //     TAMPI_Ireduce(send, recv, length, type, op, root, comm);
// // #endif
// // #if (defined(OMP) && defined(MPI))
// //     MPI_Reduce(send, recv, length, type, op, root, comm);
// // #endif

// //     return MPI_SUCCESS;
// // }

// // // MPI_Bcast wrapper for the different cases : tampi or tampi/git, OpenMP or OmpSs-2 or sequential
// // int _TAMPI_Ibcast(void *ptr, int length, MPI_Datatype type, int root, MPI_Comm comm) {
// // #if (defined(OSS) && defined(MPI))
// //     TAMPI_Ibcast(ptr, length, type, root, comm);
// // #endif
// // #if (defined(OMP) && defined(MPI))
// //     MPI_Bcast(ptr, length, type, root, comm);
// // #endif
// //     return MPI_SUCCESS;
// // }

// // Internal functions

// // Function to be called by mpi_sum_wrapper (indirectly)
// // It performs sum between two arrays in a MPI reduction
// // In order to beneficiate from OmpSs-2, this function must be inside nosV runtime
// void mpi_ompss_sum(void *_args) 
// {
//     struct mpi_user_def_op_args_t *args = (struct mpi_user_def_op_args_t *)_args;
//     float *in_float = (float *)args->in;
//     float *inout_float = (float *)args->inout;
//     int *len = (int *)args->len;
//     int mtx;

//     #pragma omp taskgroup
//     {
//         for (int i = 0; i < *len; i += MPI_PARALLEL_SUM_TRIGGER_SIZE) {
//             #pragma omp task firstprivate(i, in_float, inout_float, len)
//             {
//                 int end = (i + MPI_PARALLEL_SUM_TRIGGER_SIZE < *len)
//                             ? i + MPI_PARALLEL_SUM_TRIGGER_SIZE : *len;
//                 for (int j = i; j < end; j++) {
//                     inout_float[j] += in_float[j];
//                 }
//             }
//         }
//     }

//     #pragma omp taskwait 
// }

// // This function is directly called by MPI as a user-defined reduction function.
// // Here, we are outside of nosV runtime, thus we will call mpi_launch_on_scheduler
// // which will spawn mpi_ompss_sum inside nosV scheduler
// #if (defined MPI) && (defined OSS)
// void mpi_sum_wrapper(void *in, void *inout, int *len, MPI_Datatype *data_type) {
//     assert(*data_type == MPI_FLOAT);

//     if (*len > MPI_PARALLEL_SUM_TRIGGER_SIZE) { // Only parallelize the reduction if enough elements to reduce
//         struct mpi_user_def_op_args_t op_args = {in, inout, len, data_type};
//         mpi_launch_on_scheduler(mpi_ompss_sum, &op_args, "MPI_SUM");
//     } else { // Otherwise do a classic sum
//         float *inout_float = (float *)inout;
//         float *in_float = (float *)in;
//         for (int i = 0; i < *len; i++)
//             inout_float[i] += in_float[i];
//     }
// }
// #endif

// // Calculate how to slice the tokens among the ranks
// void mpi_batch_split(MPIWorker worker, int B, int T) {
//     assert(MAX(worker->worldsize, B) % MIN(worker->worldsize, B) == 0);

//     if (worker->worldsize > B) {
//         worker->T = (T * B) / worker->worldsize;
//         worker->B = 1;
//     } else {
//         worker->T = T;
//         worker->B = B / worker->worldsize;
//     }
// }

// // Initialize the MPIWorker structure
// MPIWorker mpi_init(int *argc, char **argv[], int B, int T) {
//     MPIWorker worker = malloc(sizeof(MPIWorker_internal));
//     worker->world_B = B;
//     worker->world_T = T;

// #ifdef MPI
//     int provided;
//     MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
//     assert(provided == MPI_THREAD_MULTIPLE);

//     MPI_Comm_rank(MPI_COMM_WORLD, &worker->rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &worker->worldsize);

//     MPI_Comm_dup(MPI_COMM_WORLD, &worker->nextbatch);      // Used to scatter the inputs & targets
//     MPI_Comm_dup(MPI_COMM_WORLD, &worker->loss_comm);      // Used to reduce the forward pass loss
//     MPI_Comm_dup(MPI_COMM_WORLD, &worker->inference_comm); // Used to send generated tokens during inference

//     // Calculate the batch size and sequence size to be used on a rank.
//     // Compatible for sequence splitting
//     mpi_batch_split(worker, B, T);
//     if (worker->rank == 0)
//         printf("MPI batch split :\t World : (B=%d, T=%d)   ---->   Rank : (B=%d, T=%d)\n", B, T, worker->B, worker->T);

//     assert(worker->B * worker->T * worker->worldsize == B * T);
//     assert(B % worker->B == 0);
//     assert(T % worker->T == 0);
//     assert(T == worker->T); // Sequence splitting is not allowed on this distributed version

//     // Create our own MPI_SUM Operator, which will run a parallel sum, unlike MPI_SUM which is sequential
//     #ifdef OSS
//     MPI_Op_create(&mpi_sum_wrapper, 1, &worker->parallel_sum);
//     #endif
// #else
//     worker->B = B;
//     worker->T = T;
//     worker->rank = 0;
//     worker->worldsize = 1;
// #endif

//     worker->comm_blocks = NULL; // Will be initialized lazily during the update phase
//     worker->comm_blocks_count = 0;
//     return worker;
// }

// #ifdef MPI
// // Initialize the communication structures
// void comm_block_init(struct comm_block_t *comm_block, float **ptrs, int *lengths, int n, int start_offset, int stop_offset, int block_size,
//                      float *ref_ptr) {
//     // comm_blocks is the communications block to initialize,
//     // ptrs are pointers towards each slice of the gradients array, but ordered along the order in which there are called in gpt2_backward
//     // lengths are the lengths of these slices
//     // n is how many slices the communcation block needs to contain
//     // start_offset is the offset to add to the first pointer to get the start of our slice
//     // stop_offset is how many elements the last slice of communication block will contain
//     // block_size is the summed length of all the slices contained inside the communication block
//     // ref_ptr is the pointer towards the beginning of the grads_memory array

//     // We will mostly work with offset and not pointers, as the offsets are the same for the params_memory and grads_memory arrays
//     // The dependencies_offsets are both used to register where the dependencies are, and where the first elements of the sliced arrays
//     // are, with exeption for the first element, where those two are not at the same place.
//     comm_block->dependencies_offsets = malloc(n * sizeof(int));
//     comm_block->covered_subgrads_count = n;
//     comm_block->first_data_offset = (int)(ptrs[0] - ref_ptr) + start_offset;
//     if (n == 1 && stop_offset == 0) // If we have only one slice in the communication block
//         comm_block->block_size = lengths[0] - start_offset;
//     else
//         comm_block->block_size = block_size;

//     for (int i = 0; i < n; i++)
//         comm_block->dependencies_offsets[i] = (int)(ptrs[i] - ref_ptr);

//     comm_block->slices_length = malloc(n * sizeof(int));
//     memcpy(comm_block->slices_length, lengths, n * sizeof(int));
//     comm_block->slices_length[n - 1] = stop_offset ? stop_offset : comm_block->slices_length[n - 1];
//     comm_block->slices_length[0] -= start_offset;
//     comm_block->block_slices_comm = malloc(n * sizeof(MPI_Comm));
//     for (int i = 0; i < n; i++)
//         MPI_Comm_dup(MPI_COMM_WORLD, comm_block->block_slices_comm + i);
// }

// // Free a communcation block
// void comm_block_destroy(struct comm_block_t *comm_block) {
//     free(comm_block->dependencies_offsets);
//     free(comm_block->slices_length);
//     for (int i = 0; i < comm_block->covered_subgrads_count; i++)
//         MPI_Comm_free(comm_block->block_slices_comm + i);
// }

// // Destroy the communication block array
// void comm_destroy(struct comm_block_t *comm_blocks, int n) {
//     for (int i = 0; i < n; i++)
//         comm_block_destroy(comm_blocks + i);
//     free(comm_blocks);
// }

// // Scattering token inputs or targets
// void mpi_scatter_inputs(MPIWorker worker, int *inputs) {
//     if (worker->rank == 0)
//         MPI_Scatter(inputs, worker->T * worker->B, MPI_INT, MPI_IN_PLACE, worker->T * worker->B, MPI_INT, 0, worker->nextbatch);
//     else
//         MPI_Scatter(inputs, worker->T * worker->B, MPI_INT, inputs, worker->T * worker->B, MPI_INT, 0, worker->nextbatch);
// }
// #endif

// // Free all MPI related structures
// void mpi_finalize(MPIWorker worker) {
// #ifdef MPI
//     // if (worker->comm_blocks != NULL)
//     //     comm_destroy(worker->comm_blocks, worker->comm_blocks_count);
//     MPI_Comm_free(&worker->nextbatch);
//     MPI_Comm_free(&worker->loss_comm);
//     MPI_Comm_free(&worker->inference_comm);
//     #ifdef OSS
//     MPI_Op_free(&worker->parallel_sum);
//     #endif

//     int comms = (int) worker->comm_blocks;
//     for (int i = 0; i < comms; ++i)
//         MPI_Comm_free(&worker->block_comms[i]);

//     MPI_Finalize();
// #endif
//     free(worker);
// }

// // Calculate which subslices a block contains, where it starts and stops
// // The aim is to be able to retrieve a block of size block_size
// int get_comm_block_caracteristics(int *offset_out, int *subgrads_sizes, int subgrads_count, int offset, int block_size) {
//     int covered_memzone = 0;
//     int curr_size = subgrads_sizes[0] - offset;
//     for (int i = 0; i < subgrads_count; i++) {
//         if (covered_memzone + curr_size >= block_size) {
//             *offset_out = block_size - covered_memzone + (i == 0 ? offset : 0);
//             return i + 1;
//         }
//         covered_memzone += curr_size;
//         curr_size = subgrads_sizes[MIN(i + 1, subgrads_count - 1)];
//     }
//     *offset_out = 0;
//     return subgrads_count;
// }

// // Performs a optimized reduction if we are using tampi/git and OmpSs-2 otherwise falls back to classic MPI_SUM
// // void mpi_reduce(MPIWorker worker, float *ptr, int length, int root, MPI_Comm *comm) {
// // #if defined OSS && defined TAMPI_VERSION_MAJOR
// //     MPI_Op op = worker->parallel_sum;
// // #else
// //     MPI_Op op = MPI_SUM;
// // #endif
// //     if (worker->rank == root) {
// //         _TAMPI_Ireduce(MPI_IN_PLACE, ptr, length, MPI_FLOAT, op, root, *comm);
// //     } else {
// //         _TAMPI_Ireduce(ptr, ptr, length, MPI_FLOAT, op, root, *comm);
// //     }
// // }

// // broadcast wrapper
// // void mpi_broadcast(float *ptr, int length, int root, MPI_Comm *comm) { _TAMPI_Ibcast(ptr, length, MPI_FLOAT, root, *comm); }

// // void mpi_reduce_block(MPIWorker worker, float *ptr, int length, int root, int comm_idx)
// // {
// //     mpi_reduce(worker, ptr, length, root, &worker->block_comms[comm_idx]);
// // }

// // void mpi_bcast_block(MPIWorker worker, float *ptr, int length, int root, int comm_idx)
// // {
// //     mpi_broadcast(ptr, length, root, &worker->block_comms[comm_idx]);
// // }

// // Interface

// // Performs a reduction of the gradients, by using the communcations blocks
// // void mpi_share_gradients(float *grads_memory, MPIWorker worker) 
// // {
// //     for (int block_id = 0; block_id < worker->comm_blocks_count; block_id++) // for each communication block 
// //     {
// //         struct comm_block_t *block = worker->comm_blocks + block_id;
// //         for (int block_slice_id = 0; block_slice_id < block->covered_subgrads_count; block_slice_id++) // For each slice in this block 
// //         { 
// //             float *slice_ptr = grads_memory + (block_slice_id ? block->dependencies_offsets[block_slice_id] : block->first_data_offset);
// //             int slice_length = block->slices_length[block_slice_id];
// //             float *grads_dependency_ptr = grads_memory + block->dependencies_offsets[block_slice_id];

// //             #pragma oss task firstprivate(worker, slice_ptr, slice_length, block, block_slice_id, grads_dependency_ptr, block_id)                                     \
// //                         concurrent(worker->comm_blocks->root) in(*grads_dependency_ptr)in(block->root)
// //             mpi_reduce(worker, slice_ptr, slice_length, block->root, block->block_slices_comm + block_slice_id); // Reduce the slice
// //         }
// //     }
// // }

// // // Broadcasts the parameters among the ranks
// // void mpi_broadcast_parameters(float *params_memory, MPIWorker worker) {
// //     for (int block_id = 0; block_id < worker->comm_blocks_count; block_id++) { // for each communication block
// //         struct comm_block_t *block = worker->comm_blocks + block_id;
// //         for (int block_slice_id = 0; block_slice_id < block->covered_subgrads_count; block_slice_id++) { // For each slice in this block
// //             float *slice_ptr = params_memory + (block_slice_id ? block->dependencies_offsets[block_slice_id] : block->first_data_offset);
// //             int slice_length = block->slices_length[block_slice_id];
// //             float *params_dependency_ptr = params_memory + block->dependencies_offsets[block_slice_id];
        
// //             #pragma oss task firstprivate(slice_ptr, slice_length, block, block_slice_id) in(*params_dependency_ptr, block->root)in(worker->comm_blocks->root)
// //             mpi_broadcast(slice_ptr, slice_length, block->root, block->block_slices_comm + block_slice_id); // Broadcast the slice
            
// //         }
// //     }
// // }

// // // Reduces the loss in gpt2_forward to print it out
// // void mpi_loss_reduce(MPIWorker worker, float *send, float *recv) {
// // #if defined OSS && defined MPI
// //     TAMPI_Ireduce(send, recv, 1, MPI_FLOAT, MPI_SUM, 0, worker->loss_comm);
// // #endif
// // #if defined OMP && defined MPI
// //     MPI_Reduce(send, recv, 1, MPI_FLOAT, MPI_SUM, 0, worker->loss_comm);
// // #endif
// // }

// // // As each ranki is responsible for only a slice of the parameters, a rank only have to allocate as much memory as we have parameters to update
// // size_t mpi_get_adam_buffer_size(MPIWorker worker) {
// //     size_t adam_memory_size = 0;
// //     for (int block_id = 0; block_id < worker->comm_blocks_count; block_id++) {
// //         struct comm_block_t *block = worker->comm_blocks + block_id;
// //         adam_memory_size += block->root == worker->rank ? block->block_size : 0;
// //     }
// //     return adam_memory_size;
// // }

// // // Retrieve all the parameters slices the rank is responsible to update, and update them
// // void mpi_update_params_slice(void *model, void (*update_func)(void *, int, int, int, float, float, float, float, float, int *, int, int *),
// //                              MPIWorker worker, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int *t) {

// //     int adam_offset = 0;
// //     for (int block_id = 0; block_id < worker->comm_blocks_count; block_id++) { // For each communication block
// //         struct comm_block_t *block = worker->comm_blocks + block_id;
// //         if (block->root == worker->rank) { // If the rank is reponsible for this block's update
// //             for (int block_slice_id = 0; block_slice_id < block->covered_subgrads_count; block_slice_id++) { // For each block's slice
// //                 int slice_length = block->slices_length[block_slice_id];
// //                 int slice_offset = block_slice_id ? block->dependencies_offsets[block_slice_id] : block->first_data_offset;

// //                 for (int taskloop = 0; taskloop < slice_length; taskloop += UPDATE_TASKLOOP_GRAINSIZE) { // Flatten taskloop (for taskiter)
// //                     int subslice_size = MIN(UPDATE_TASKLOOP_GRAINSIZE, slice_length - taskloop);
// //                     update_func(model, slice_offset, adam_offset, subslice_size, learning_rate, beta1, beta2, eps, weight_decay, t,
// //                                 block->dependencies_offsets[block_slice_id], &block->root); // Update the slice

// //                     slice_offset += subslice_size;
// //                     adam_offset += subslice_size;
// //                 }
// //             }
// //         }
// //     }
// // }

// #ifdef MPI
// // Initialize the communication block array
// void mpi_init_communications(MPIWorker worker, int comms)
// {
//     worker->block_comms = malloc(comms * sizeof(MPI_Comm));
//     for (int i = 0; i < comms; ++i) {
//         MPI_Comm_dup(MPI_COMM_WORLD, &worker->block_comms[i]);
//     }

//     worker->comm_blocks = (void *)comms;

//     // printf("Creating %d commns\n", comms);

//     // assert(model_num_parameters < INT_MAX); // Limitation from MPI

//     // // We are defining memory blocks along the gradients array. A memory block is not contiguous in memory,
//     // // but its concatenated size will be block_size, with exception for the last one which will be shorter.
//     // int blocks_count = model_num_parameters / block_size + (model_num_parameters % block_size ? 1 : 0);
//     // int tmp_subgrads_count = arr_size;

//     // assert(subgrads_count < tmp_subgrads_count); // If fails, just increase tmp_subgrads_count (temporal buffer size)

//     // worker->comm_blocks = malloc(blocks_count * sizeof(struct comm_block_t));
//     // worker->comm_blocks_count = blocks_count;
//     // int next_block_offset = 0;
//     // int last_block_offset = 0;
//     // int tot_blocks_covered = 0;
//     // for (int block = 0; block < blocks_count; block++) { // Initialize all communication blocks
//     //     last_block_offset = next_block_offset;
//     //     int subgrads_covered = get_comm_block_caracteristics(&next_block_offset, subgrads_sizes + tot_blocks_covered, subgrads_count - tot_blocks_covered,
//     //                                                          last_block_offset, block_size);

//     //     comm_block_init(worker->comm_blocks + block, subgrads_ptrs + tot_blocks_covered, subgrads_sizes + tot_blocks_covered, subgrads_covered,
//     //                     last_block_offset, next_block_offset, block_size, grads_memory);
//     //     worker->comm_blocks[block].root = block % worker->worldsize;
//     //     tot_blocks_covered += subgrads_covered - (next_block_offset ? 1 : 0);
//     // }
//     // assert(tot_blocks_covered == subgrads_count); // Check if we have the correct number of sub-blocks (or block's slices) we expected
// }

// // Share the produced token during an inference iteration
// void mpi_inference_share_gen_token(MPIWorker worker, int sender, int receiver, int next_token, int *gen_token) {
//     if (worker->rank == sender) {
//         if (sender == receiver) // This rank is the next computing rank : no need for communication
//             *gen_token = next_token;
//         else // Send generated token to next computing rank
//             MPI_Send(&next_token, 1, MPI_INT, receiver, 0, worker->inference_comm);
//         if (worker->rank != 0 && receiver != 0) { // Only rank 0 is allowed to print tokens
//             MPI_Send(&next_token, 1, MPI_INT, 0, 0, worker->inference_comm);
//         }
//     }
//     if (worker->rank == receiver && sender != receiver)
//         MPI_Recv(gen_token, 1, MPI_INT, sender, 0, worker->inference_comm, MPI_STATUS_IGNORE);
//     else if (worker->rank == 0 && receiver != 0)
//         MPI_Recv(gen_token, 1, MPI_INT, sender, 0, worker->inference_comm, MPI_STATUS_IGNORE);
// }
// #endif

// // Returns the batch size computed on one rank
// int mpi_get_rank_B(MPIWorker worker) { return worker->B; }

// // Returns the token sequence size computed on one rank
// int mpi_get_rank_T(MPIWorker worker) { return worker->T; }

// // Returns the batch size
// int mpi_get_world_B(MPIWorker worker) { return worker->world_B; }

// // Returns the token sequence size
// int mpi_get_world_T(MPIWorker worker) { return worker->world_T; }

// // Returns rank's id
// int mpi_get_rank(MPIWorker worker) { return worker->rank; }

// // Returns worldsize
// int mpi_get_worldsize(MPIWorker worker) { return worker->worldsize; }

// // Returns weither the communications blocks have already been initialized
// int mpi_is_comm_initialized(MPIWorker worker) { return (worker->comm_blocks != NULL); }
