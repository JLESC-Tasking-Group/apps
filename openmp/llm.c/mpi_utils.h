#ifndef __MPI_UTILS_H__
#define __MPI_UTILS_H__

// Settings

#define MPI_PRIORITY 30
#define NB_UPDATE_TOTSLICES 64
#define NB_UPDATE_COMMS (NB_UPDATE_TOTSLICES * 2)
#define MAX_RANKS_PER_SEQUENCE 16
#define MPI_PARALLEL_SUM_TRIGGER_SIZE 30000
#define UPDATE_TASKLOOP_GRAINSIZE (1024 * 16)

// Definitions

#include <stdlib.h>
typedef struct MPIWorker_internal_t *MPIWorker;

// General functions
#ifdef __cplusplus
extern "C" {
#endif

MPIWorker mpi_init(int *argc, char **argv[], int B, int T);
void mpi_finalize(MPIWorker worker);
void mpi_scatter_inputs(MPIWorker worker, int *inputs);
void mpi_share_gradients(float *grads_memory, MPIWorker worker);
size_t mpi_get_adam_buffer_size(MPIWorker worker);
void mpi_update_params_slice(void *model, void (*update_func)(void *, int, int, int, float, float, float, float, float, int *, int, int *),
                             MPIWorker worker, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int *t);
void mpi_broadcast_parameters(float *params_memory, MPIWorker worker);
void mpi_init_communications(MPIWorker worker, int comms);
void mpi_loss_reduce(MPIWorker worker, float *send, float *recv);
void mpi_inference_share_gen_token(MPIWorker worker, int sender, int receiver, int next_token, int *gen_token) ;


void mpi_reduce_block(MPIWorker worker, float *ptr, int length, int root, int comm_idx);
void mpi_bcast_block(MPIWorker worker, float *ptr, int length, int root, int comm_idx);
// Utils

int mpi_get_rank_B(MPIWorker worker);
int mpi_get_rank_T(MPIWorker worker);
int mpi_get_world_B(MPIWorker worker);
int mpi_get_world_T(MPIWorker worker);
int mpi_get_rank(MPIWorker worker);
int mpi_get_worldsize(MPIWorker worker);
int mpi_is_comm_initialized(MPIWorker worker);

#ifdef __cplusplus
}
#endif

#endif // __MPI_UTILS_H__
