/*
This file trains the GPT-2 model.
This version is the clean, minimal, reference. As such:
- it runs on CPU.
- it does not make the code too complex; it is readable.
- it does not use any processor-specific instructions, intrinsics and such.
- it _does_ use a few OpenMP pragmas because this is a large speedup at very
low cost There will be other versions of this code that specialize it and make
it fast.
*/
#include <assert.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "tracer.h"

// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch,
// dataloader_free
#include "llmc/dataloader.h"
// defines ENABLE_TASKITER, MB_SUBSIZE, MT_SUBSIZE, MATMUL_LOOP_UNROLL
#include "ompss-2_settings.h"
// defines : dep_init, dep_finish, dep_set_slice_shape, dep_reset, all OmpSs-2 wrapper functions
#include "ompss-2_wrappers.h"
// defines TIMERS_INIT, TICK, TOCK, GET_TIME
#include "timer.h"
// defines mpi_init mpi_finalize mpi_scatter_inputs mpi_share_gradients mpi_get_adam_buffer_size mpi_update_params_slice mpi_broadcast_parameters
// mpi_init_ring_communications mpi_loss_reduce mpi_inference_share_gen_token mpi_get_rank_B mpi_get_rank_T mpi_get_world_B mpi_get_world_T mpi_get_rank
// mpi_get_worldsize mpi_is_comm_initialized
#include "mpi_utils.h"

enum Datasets { TINYSHAKESPEARE = 0, TINYSTORIES, DATASETS_MAX_INT };

// Task / target / taskgraph macros and the USE_TARGET / USE_TASKGRAPH / USE_SYNC
// / USE_OMPSS / USE_XKOMP / USE_REPLAYABLE toggles are shared with the other
// apps/openmp benchmarks in ../tasking.h (resolved via the Makefile's -I..).
#include "tasking.h"

// OpenMP: omp_get_wtime() & omp_get_num_threads(). OmpSs-2: nanos6 API (nodes.h).
#if USE_OMPSS
# include <nodes.h>
#else
# include <omp.h>
#endif

// Used for strong/weak scaling
#define PERF_TESTING 1

#ifndef NB_STEPS
    #define NB_STEPS 8
#endif

#ifndef SEQUENCE_SIZE
    #define SEQUENCE_SIZE 64
#endif

#ifndef BATCH_SIZE
    #define BATCH_SIZE 4
#endif

#ifndef DATASET
    #define TRAINING_DATASET TINYSHAKESPEARE
    #define VALIDATION_DATASET TINYSHAKESPEARE
#else
    #define TRAINING_DATASET DATASET
    #define VALIDATION_DATASET DATASET
#endif

// Tile size along the token dimension. On the host it is 8 (task granularity);
// on the device it is 1 so every offloaded loop exposes one GPU thread per token
// (B*T threads) instead of one thread per 8-token tile.
#ifndef GRAN_TMP
# define GRAN_TMP (USE_TARGET ? 1 : 8)
#endif /* GRAN_TMP */

#ifndef OC_SPLIT
# define OC_SPLIT 6
#endif /* OC_SPLIT */

#ifndef OC_BACK_SPLIT
#define OC_BACK_SPLIT 10
#endif /* OC_BACK_SPLIT */

#define min(a, b) ((a) < (b) ? (a) : (b))

#undef MATMUL_LOOP_UNROLL
#define MATMUL_LOOP_UNROLL 8

#if USE_OMPSS

    // OmpSs-2 runtime (NODES/nanos6) reports the number of available CPUs.
    #define GET_NUM_CPUS() nanos6_get_num_cpus()

    // No omp_get_wtime() under OmpSs-2: use a monotonic wall clock.
    static inline double _wall_time(void)
    {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
    }
    #define WALL_TIME() _wall_time()

#elif defined(_OPENMP)

    int _omp_get_num_th()
    {
        int num_cpus = 0;
        #pragma omp parallel
        {
            num_cpus = omp_get_num_threads();
        }
        return num_cpus;
    }
    #define GET_NUM_CPUS() _omp_get_num_th()
    #define WALL_TIME()    omp_get_wtime()

#else
    #define GET_NUM_CPUS() 1
    #define WALL_TIME()    omp_get_wtime()
#endif

static inline int MAX_INT(int a, int b) { return ((a) > (b) ? a : b); }

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

// In the target backend these per-tile kernels are called from inside offloaded
// "target teams distribute parallel for" regions, so they must be compiled for
// the device as well.
#if USE_TARGET
#pragma omp begin declare target
#endif

void encoder_forward(float *out, int *inp, float *wte, float *wpe, int B, int T, int C, struct SliceHandler slh)
{
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing
    // token & position inp is (B,T) of integers, holding the token ids at each
    // (b,t) position wte is (V,C) of token embeddings, short for "weight token
    // embeddings" wpe is (maxT,C) of position embeddings, short for "weight
    // positional embedding"

    // slh.t_start is the sliced loop's starting point
    // slh.dim_starts[x] is the start offset of the current nested taskloop iteration
    // slh.dim_stops[x] is the stop offset of the current nested taskloop iteration
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    // fprintf(stderr, "B: [%d, %d], T[%04d, %04d] | DIMS [[%d, %d], [%d, %d]]\n", b_start, b_end, t_start, t_end, slh.dims_starts[0], slh.dims_stops[0], slh.dims_starts[1], slh.dims_stops[1]);
    for (int b = b_start; b < b_end; b++)
    {     // 0:B
        for (int t = t_start; t < t_end; t++)
        { // 0:T
            // seek to the output position in out[b,t,:]
            float *out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float *wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float *wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++)
            {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}


void encoder_backward(float *dwte, float *dwpe, float *dout, int *inp, int B, int T, int C, struct SliceHandler slh)
{
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    for (int b = b_start; b < b_end; b++) {     // 0:B
        for (int t = t_start; t < t_end; t++) { // 0:T
            float *dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float *dwte_ix = dwte + ix * C;
            float *dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];

                ATOMIC
                dwte_ix[i] += d;

                ATOMIC
                dwpe_t[i] += d;
            }
        }
    }
}

void layernorm_forward(float *out, float *mean, float *rstd, float *inp, float *weight, float *bias, int B, int T, int C, struct SliceHandler slh) {
    // reference:
    // https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html both inp
    // and out are (B,T,C) of the activations mean and rstd are (B,T) buffers, to
    // be used later in backward pass at each position (b,t) of the input, the
    // C-dimensional vector of activations gets normalized, then scaled and
    // shifted

    float eps = 1e-5f;
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    for (int b = b_start; b < b_end; b++) {     // 0:B
        for (int t = t_start; t < t_end; t++) { // 0:T
            // seek to the input position inp[b,t,:]
            float *x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m / C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v / C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float *out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++)
            {
                float n = (s * (x[i] - m));        // normalize
                float o = n * weight[i] + bias[i]; // scale and shift

                out_bt[i] = o;                     // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

void layernorm_backward(float *dinp, float *dweight, float *dbias, float *dout, float *inp, float *weight, float *mean, float *rstd, int B, int T, int C,
                        struct SliceHandler slh) {
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    for (int b = b_start; b < b_end; b++) {     // 0:B
        for (int t = t_start; t < t_end; t++) { // 0:T
            float *dout_bt = dout + b * T * C + t * C;
            float *inp_bt = inp + b * T * C + t * C;
            float *dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias

                ATOMIC
                dbias[i] += dout_bt[i];

                // gradient contribution to weight
                ATOMIC
                dweight[i] += norm_bti * dout_bt[i];

                // gradient contribution to input
                float dval = 0.0f;
                dval += dnorm_i;                    // term 1
                dval -= dnorm_mean;                 // term 2
                dval -= norm_bti * dnorm_norm_mean; // term 3
                dval *= rstd_bt;                    // final scale
                dinp_bt[i] += dval;
            }
        }
    }
}

void matmul_forward_naive(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C, int OC, struct SliceHandler *slh) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    int b_start = slh->b_start;
    int b_end = slh->b_end;
    int t_start = slh->t_start;
    int t_end = slh->t_end;
    int o_start = slh->dims_starts[0];
    int o_end = slh->dims_stops[0];

    for (int b = b_start; b < b_end; b++) {     // 0:B
        for (int t = t_start; t < t_end; t++) { // 0:T
            int bt = b * T + t;
            for (int o = o_start; o < o_end; o++) { // 0:OC
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++)
                {
                    val += inp[bt * C + i] * weight[o * C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

void matmul_forward(float *out, const float *inp, const float *weight, const float *bias, int B, int T, int C, int OC, struct SliceHandler slh) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

#if USE_TARGET
    // On the device every thread handles a single (token, output-channel) tile,
    // so the 8-token-block unrolled path does not apply: use the tile-respecting
    // naive kernel (one output element per GPU thread).
    matmul_forward_naive(out, inp, weight, bias, B, T, C, OC, &slh);
    return;
#endif

    // make sure the tiled loop will be correct or fallback to naive version
    if (B * T % MATMUL_LOOP_UNROLL != 0)
    {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC, &slh);
        return;
    }

// collapse the B and T loops into one and turn it into a strided loop.
// then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL
// many times oss taskloop label("Matmul forward")

    int obt = slh.b_start * T + slh.t_start;

    for (int o = slh.dims_starts[0]; o < slh.dims_stops[0]; o++) { // 0:OC
        float result[MATMUL_LOOP_UNROLL];
        // initialize the bias, if it exists
        for (int ibt = 0; ibt < MATMUL_LOOP_UNROLL; ibt++)
        {
            result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
        }
        // inner loops. Because we do LOOP_UNROLL steps of inner bt, we
        // can cache the value of weight[i + o * C] and reuse it. we
        // compile with -Ofast, so the compiler will turn the inner
        // loop into FMAs
        for (int i = 0; i < C; i++) {
            float w = weight[i + o * C];
            for (int ibt = 0; ibt < MATMUL_LOOP_UNROLL; ibt++)
            {
                int bt = obt + ibt;
                result[ibt] += inp[bt * C + i] * w;
            }
        }
        //  write back results to main memory
        for (int ibt = 0; ibt < MATMUL_LOOP_UNROLL; ibt++)
        {
            int bt = obt + ibt;
            out[bt * OC + o] = result[ibt];
        }
    }
}


void matmul_input_backward(float *dinp, const float *dout, const float *weight, int B, int T, int C, int OC, struct SliceHandler slh)
{
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];


    // T: 64
    // B: 4

    // backward into inp first, parallelize over B,T
    for (int b = b_start; b < b_end; b++)
    {
        for (int t = t_start; t < t_end; t++)
        {
            const float *dout_bt = dout + b * T * OC + t * OC;
            float *dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++)
            {
                const float *wrow = weight + o * C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++)
                {
                    // #pragma omp atomic
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
}

void matmul_params_backward(float *dweight, float *dbias, const float *dout, const float *inp, int B, int T, int C, int OC, struct SliceHandler slh) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    int o_start = slh.dims_starts[0];
    int o_end = slh.dims_stops[0];

    // backward into weight/bias, parallelize over output channels OC
    for (int o = o_start; o < o_end; o++)
    {
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                const float *dout_bt = dout + b * T * OC + t * OC;
                const float *inp_bt = inp + b * T * C + t * C;
                float *dwrow = dweight + o * C;
                float d = dout_bt[o];

                if (dbias != NULL)
                {
                    // #pragma omp atomic
                    dbias[o] += d;
                }
                for (int i = 0; i < C; i++)
                {
                    // #pragma omp atomic
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

void attention_forward(float *out, float *preatt, float *att, float *inp, int B, int T, int C, int NH, struct SliceHandler slh) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)

    int C3 = C * 3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    for (int b = b_start; b < b_end; b++) {     // 0:B
        for (int t = t_start; t < t_end; t++) { // 0:T
            for (int h = 0; h < NH; h++) {      // 0:NH
                float *query_t = inp + b * T * C3 + t * C3 + h * hs;
                float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float *att_bth = att + b * NH * T * T + h * T * T + t * T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical
                // stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set
                        // to zero here only doing this explicitly for debugging
                        // and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of
                // attention
                float *out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) {
                    out_bth[i] = 0.0f;
                }
                for (int t2 = 0; t2 <= t; t2++) {
                    float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

void attention_backward_Q(float *dinp, float *dpreatt, float *datt, float *dout, float *inp, float *att, int B, int T, int C, int NH,
                          struct SliceHandler slh) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)

    // attention_backward has been seperated in two kernels
    // This allows us to increase the collapse depth of our parallelization

    int C3 = C * 3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    for (int b = b_start; b < b_end; b++) {     // 0:B
        for (int t = t_start; t < t_end; t++) { // 0:T
            for (int h = 0; h < NH; h++) {      // 0:NH

                float *att_bth = att + b * NH * T * T + h * T * T + t * T;
                float *datt_bth = datt + b * NH * T * T + h * T * T + t * T;
                float *dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                float *dquery_t = dinp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float *dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input
                // (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                for (int t2 = 0; t2 <= t; t2++) {
                    float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

void attention_backward_KV(float *dinp, float *dpreatt, float *dout, float *inp, float *att, int B, int T, int C, int NH, struct SliceHandler slh)
{
    // attention_backward has been seperated in two kernels
    // This allows us to increase the collapse depth of our parallelization

    int C3 = C * 3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    for (int b = b_start; b < b_end; b++) {        // 0:B
        for (int t2 = t_start; t2 < t_end; t2++) { // 0:t_mpi_stop
            for (int h = 0; h < NH; h++) {         // 0:NH
                for (int i = 0; i < hs; i++) {     // 0:hs

                    float *dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float *dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;

                    // backward pass 1, the query @ key matmul
                    for (int t = MAX_INT(t2, 0); t < T; t++) {
                        float *att_bth = att + b * NH * T * T + h * T * T + t * T;
                        float *dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                        float *query_t = inp + b * T * C3 + t * C3 + h * hs;
                        float *dout_bth = dout + b * T * C + t * C + h * hs;

                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }
            }
        }
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float *out, float *inp, int B, int T, int C, struct SliceHandler slh)
{
    // (approximate) GeLU elementwise non-linearity in the MLP block of
    // Transformer
    int b_start = slh.b_start;
    int b_end = slh.b_end;
    int t_start = slh.t_start + slh.dims_starts[0];
    int t_end = slh.t_start + slh.dims_stops[0];

    // fprintf(stderr, "B: [%d, %d], T[%04d, %04d] | B: [%d, %d], T[%04d, %04d] | DIMS [[%d, %d], [%d, %d]]\n", b_start, b_end, t_start, t_end, slh.b_start, slh.b_end, slh.t_start, slh.t_end, slh.dims_starts[0], slh.dims_stops[0], slh.dims_starts[1], slh.dims_stops[1]);
    for (int b = b_start; b < b_end; b++) { // 0:B
        int i_start = b * T * C + t_start * C;
        int i_end = b * T * C + t_end * C;
        // OMP_FOR(GELU_FORWARD_trace_label, 1, )
        for (int i = i_start; i < i_end; i++) { // 0:T*C
            float x = inp[i];
            float cube = 0.044715f * x * x * x;
            out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
        }
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this
// flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__ ((optimize ("no-finite-math-only")))
#endif
void gelu_backward (float *dinp, float *inp, float *dout, int B, int T, int C, struct SliceHandler slh)
{
    int b_start = slh.b_start;
    int b_end = slh.b_end;
    int t_start = slh.t_start + slh.dims_starts[0];
    int t_end = slh.t_start + slh.dims_stops[0];

    for (int b = b_start; b < b_end; b++) { // 0:B
        int i_start = b * T * C + t_start * C;
        int i_end = b * T * C + t_end * C;
        for (int i = i_start; i < i_end; i++) { // 0:T*C
            {
                float x = inp[i];
                float tanh_arg = GELU_SCALING_FACTOR * (x + 0.044715f * x * x * x);
                float tanh_out = tanhf(tanh_arg);
                float coshf_out = coshf(tanh_arg);
                float sech_out = 1.0f / (coshf_out * coshf_out);
                float local_grad = 0.5f * (1.0f + tanh_out) + sech_out * (1.5f * tanh_arg - GELU_SCALING_FACTOR * x);
                dinp[i] += local_grad * dout[i];
            }
        }
    }
}
#pragma float_control(pop)

void residual_forward(float *out, float *inp1, float *inp2, int B, int T, int C, struct SliceHandler slh) {
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start;
    int t_end = slh.t_end;

    for (int b = b_start; b < b_end; b++) { // 0:B
        int i_start = b * T * C + t_start * C;
        int i_end = b * T * C + t_end * C;
        for (int i = i_start; i < i_end; i++) { // 0:T*C
            out[i] = inp1[i] + inp2[i];
        }
    }
}

void residual_backward(float *dinp1, float *dinp2, float *dout, int B, int T, int C, struct SliceHandler slh) {
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start;
    int t_end = slh.t_end;

    for (int b = b_start; b < b_end; b++) { // 0:B
        int i_start = b * T * C + t_start * C;
        int i_end = b * T * C + t_end * C;
        for (int i = i_start; i < i_end; i++) { // 0:T*C

            dinp1[i] += dout[i];
            dinp2[i] += dout[i];
        }
    }
}

void softmax_forward(float *probs, float *logits, int B, int T, int V, int Vp, struct SliceHandler slh)
{
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t
    // position) input: logits is (B,T,Vp) of the unnormalized log probabilities Vp
    // is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    for (int b = b_start; b < b_end; b++) {     // 0:B
        for (int t = t_start; t < t_end; t++) { // 0:T
            // probs <- softmax(logits)
            float *logits_bt = logits + b * T * Vp + t * Vp;
            float *probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

void crossentropy_forward(float *losses, float *probs, int *targets, int B, int T, int Vp, struct SliceHandler slh)
{
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    for (int b = b_start; b < b_end; b++) {     // 0:B
        for (int t = t_start; t < t_end; t++) { // 0:T
            // loss = -log(probs[target])
            float *probs_bt = probs + b * T * Vp + t * Vp;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

void crossentropy_softmax_backward(float *dlogits, float *dlosses, float *probs, int *targets, int B, int T, int V, int Vp, struct SliceHandler slh)
{
    // backwards through both softmax and crossentropy
    int b_start = slh.b_start + slh.dims_starts[0];
    int b_end = slh.b_start + slh.dims_stops[0];
    int t_start = slh.t_start + slh.dims_starts[1];
    int t_end = slh.t_start + slh.dims_stops[1];

    for (int b = b_start; b < b_end; b++)
    {     // 0:B
        for (int t = t_start; t < t_end; t++)
        { // 0:T
            float *dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float *probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

#if USE_TARGET
#pragma omp end declare target
#endif

#if USE_TARGET
// ----------------------------------------------------------------------------
// Fine-grained device attention kernels: one GPU thread per (batch, head, token)
// instead of one per (batch, token) with the heads looped serially. Every output
// is disjoint per (b, h, t) so there are no atomics. Kept as nowait target tasks
// with the same dependency bases as the host versions.
// ----------------------------------------------------------------------------
static void gpu_attention_forward(float *out, float *preatt, float *att, const float *inp,
                                  int B, int T, int C, int NH)
{
    const int C3 = C * 3;
    const int hs = C / NH;
    const float scale = 1.0f / sqrtf((float)hs);

    OMP_TARGET_LOOP_TASK(collapse(3) \
        DEPEND(in, inp[0]) DEPEND(out, out[0], preatt[0], att[0]))
    for (int b = 0; b < B; b++)
        for (int h = 0; h < NH; h++)
            for (int t = 0; t < T; t++)
            {
                const float *query_t = inp + b * T * C3 + t * C3 + h * hs;
                float *preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                float *att_bth    = att    + b * NH * T * T + h * T * T + t * T;

                float maxval = -10000.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) val += query_t[i] * key_t2[i];
                    val *= scale;
                    if (val > maxval) maxval = val;
                    preatt_bth[t2] = val;
                }

                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                for (int t2 = 0; t2 < T; t2++)
                    att_bth[t2] = (t2 <= t) ? att_bth[t2] * expsum_inv : 0.0f;

                float *out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) out_bth[i] += att_btht2 * value_t2[i];
                }
            }
}

static void gpu_attention_backward_Q(float *dinp, float *dpreatt, float *datt, const float *dout,
                                     const float *inp, const float *att, int B, int T, int C, int NH)
{
    const int C3 = C * 3;
    const int hs = C / NH;
    const float scale = 1.0f / sqrtf((float)hs);

    OMP_TARGET_LOOP_TASK(collapse(3) \
        DEPEND(in, inp[0], att[0], dout[0]) DEPEND(inout, dinp[0], dpreatt[0], datt[0]))
    for (int b = 0; b < B; b++)
        for (int h = 0; h < NH; h++)
            for (int t = 0; t < T; t++)
            {
                const float *att_bth = att + b * NH * T * T + h * T * T + t * T;
                float *datt_bth      = datt    + b * NH * T * T + h * T * T + t * T;
                float *dpreatt_bth   = dpreatt + b * NH * T * T + h * T * T + t * T;
                float *dquery_t      = dinp + b * T * C3 + t * C3 + h * hs;
                const float *dout_bth = dout + b * T * C + t * C + h * hs;

                for (int t2 = 0; t2 <= t; t2++) {
                    const float *value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                    for (int i = 0; i < hs; i++) datt_bth[t2] += value_t2[i] * dout_bth[i];
                }
                // softmax backward in O(t) instead of O(t^2):
                //   dpreatt[t3] = att[t3] * (datt[t3] - sum_t2 att[t2]*datt[t2])
                float dot = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) dot += att_bth[t2] * datt_bth[t2];
                for (int t3 = 0; t3 <= t; t3++)
                    dpreatt_bth[t3] += att_bth[t3] * (datt_bth[t3] - dot);
                for (int t2 = 0; t2 <= t; t2++) {
                    const float *key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C;
                    for (int i = 0; i < hs; i++) dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                }
            }
}

static void gpu_attention_backward_KV(float *dinp, const float *dpreatt, const float *dout,
                                      const float *inp, const float *att, int B, int T, int C, int NH)
{
    const int C3 = C * 3;
    const int hs = C / NH;
    const float scale = 1.0f / sqrtf((float)hs);

    // One thread per (b, h, key t2): accumulate dk/dv over all queries t >= t2.
    OMP_TARGET_LOOP_TASK(collapse(3) \
        DEPEND(in, inp[0], att[0], dout[0], dpreatt[0]) DEPEND(inout, dinp[0]))
    for (int b = 0; b < B; b++)
        for (int h = 0; h < NH; h++)
            for (int t2 = 0; t2 < T; t2++)
            {
                float *dkey_t2   = dinp + b * T * C3 + t2 * C3 + h * hs + C;
                float *dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C * 2;
                for (int i = 0; i < hs; i++) {
                    float dk = 0.0f, dv = 0.0f;
                    for (int t = t2; t < T; t++) {
                        const float *att_bth     = att     + b * NH * T * T + h * T * T + t * T;
                        const float *dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T;
                        const float *query_t     = inp  + b * T * C3 + t * C3 + h * hs;
                        const float *dout_bth    = dout + b * T * C  + t * C  + h * hs;
                        dk += query_t[i] * dpreatt_bth[t2] * scale;
                        dv += att_bth[t2] * dout_bth[i];
                    }
                    dkey_t2[i]   += dk;
                    dvalue_t2[i] += dv;
                }
            }
}
#endif

// ----------------------------------------------------------------------------
// GPT-2 model definition

typedef struct {
    int max_seq_len;       // max sequence length, e.g. 1024
    int vocab_size;        // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers;        // number of layers, e.g. 12
    int num_heads;         // number of heads in attention, e.g. 12
    int channels;          // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float *wte;      // (V, C)
    float *wpe;      // (maxT, C)
    float *ln1w;     // (L, C)
    float *ln1b;     // (L, C)
    float *qkvw;     // (L, 3*C, C)
    float *qkvb;     // (L, 3*C)
    float *attprojw; // (L, C, C)
    float *attprojb; // (L, C)
    float *ln2w;     // (L, C)
    float *ln2b;     // (L, C)
    float *fcw;      // (L, 4*C, C)
    float *fcb;      // (L, 4*C)
    float *fcprojw;  // (L, C, 4*C)
    float *fcprojb;  // (L, C)
    float *lnfw;     // (C)
    float *lnfb;     // (C)
} ParameterTensors;

void fill_in_parameter_sizes(size_t *param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C;           // wte
    param_sizes[1] = maxT * C;         // wpe
    param_sizes[2] = L * C;            // ln1w
    param_sizes[3] = L * C;            // ln1b
    param_sizes[4] = L * (3 * C) * C;  // qkvw
    param_sizes[5] = L * (3 * C);      // qkvb
    param_sizes[6] = L * C * C;        // attprojw
    param_sizes[7] = L * C;            // attprojb
    param_sizes[8] = L * C;            // ln2w
    param_sizes[9] = L * C;            // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C);     // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C;           // fcprojb
    param_sizes[14] = C;               // lnfw
    param_sizes[15] = C;               // lnfb
}

// allocate memory for the parameters and point the individual tensors to the
// right places
float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes)
{
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float *params_memory = (float *)mallocCheck(num_parameters * sizeof(float));
    // assign all the tensors
    float **ptrs[] = {&params->wte,  &params->wpe,  &params->ln1w, &params->ln1b, &params->qkvw,    &params->qkvb,    &params->attprojw, &params->attprojb,
                      &params->ln2w, &params->ln2b, &params->fcw,  &params->fcb,  &params->fcprojw, &params->fcprojb, &params->lnfw,     &params->lnfb};
    float *params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float *encoded;   // (B, T, C)
    float *ln1;       // (L, B, T, C)
    float *ln1_mean;  // (L, B, T)
    float *ln1_rstd;  // (L, B, T)
    float *qkv;       // (L, B, T, 3*C)
    float *atty;      // (L, B, T, C)
    float *preatt;    // (L, B, NH, T, T)
    float *att;       // (L, B, NH, T, T)
    float *attproj;   // (L, B, T, C)
    float *residual2; // (L, B, T, C)
    float *ln2;       // (L, B, T, C)
    float *ln2_mean;  // (L, B, T)
    float *ln2_rstd;  // (L, B, T)
    float *fch;       // (L, B, T, 4*C)
    float *fch_gelu;  // (L, B, T, 4*C)
    float *fcproj;    // (L, B, T, C)
    float *residual3; // (L, B, T, C)
    float *lnf;       // (B, T, C)
    float *lnf_mean;  // (B, T)
    float *lnf_rstd;  // (B, T)
    float *logits;    // (B, T, V)
    float *probs;     // (B, T, V)
    float *losses;    // (B, T)
} ActivationTensors;

float *malloc_and_point_activations(ActivationTensors *acts, size_t *act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float *acts_memory = (float *)mallocCheck(num_activations * sizeof(float));
    float **ptrs[] = {&acts->encoded,   &acts->ln1,       &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv,      &acts->atty,  &acts->preatt,   &acts->att,
                      &acts->attproj,   &acts->residual2, &acts->ln2,      &acts->ln2_mean, &acts->ln2_rstd, &acts->fch,   &acts->fch_gelu, &acts->fcproj,
                      &acts->residual3, &acts->lnf,       &acts->lnf_mean, &acts->lnf_rstd, &acts->logits,   &acts->probs, &acts->losses};
    float *acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float *params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float *grads_memory;
    // buffers for the AdamW optimizer
    float *m_memory;
    float *v_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float *acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float *grads_acts_memory;

    // other run state configuration
    int batch_size;  // the batch size (B) of current forward pass
    int seq_len;     // the sequence length (T) of current forward pass
    int *inputs;     // the input tokens for the current forward pass
    int *targets;    // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with
                     // the mean loss
} GPT2;


void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint_path)
{
    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    if (model_file == NULL)
    {
        fprintf(stderr, "Error opening model file\n", NULL);
        exit(1);
    }
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) {
        fprintf(stderr, "Bad magic model file\n", NULL);
        exit(1);
    }
    if (model_header[1] != 3) {
        fprintf(stderr, "Bad version in model file\n", NULL);
        fprintf(stderr, "---> HINT: try to re-run `python train_gpt2.py`\n", NULL);
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    fprintf(stderr, "[GPT-2]\n", NULL);
    fprintf(stderr, "max_seq_len: %zu\n", maxT);
    fprintf(stderr, "vocab_size: %zu\n", V);
    fprintf(stderr, "padded_vocab_size: %zu\n", Vp);
    fprintf(stderr, "num_layers: %zu\n", L);
    fprintf(stderr, "num_heads: %zu\n", NH);
    fprintf(stderr, "channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    fprintf(stderr, "num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

float gpt2_estimate_mfu(GPT2 *model, int num_tokens, float dt, unsigned int num_cpus) {
    /*
    Estimate model flops utilization (MFU)
    ref: Section 2.1 of https://arxiv.org/pdf/2001.08361
    Note: Ideally, the N here would be only the parameters that actually
    participate in matrix multiplications. In this N, we are over-estimating by
    including LayerNorm params, biases, and the position embedding weights,
    but these are very small terms. Also keep in mind that we would want to exclude
    the token embedding weights, but in GPT-2 these are weight shared, so they
    participate in the classifier matmul, so they are correct to be included in N.
    Note 2: The first term (6 * N) in flops_per_token is all weight matmuls, the
    second is the attention matmul, which is also usually a small contribution.
    */
    size_t N = model->num_parameters;
    int L = model->config.num_layers;
    int C = model->config.channels;
    int T = model->seq_len;
    size_t flops_per_token = 6 * N + (size_t)6 * L * C * T;
    size_t flops_per_step = flops_per_token * num_tokens;
    // express our flops throughput as ratio of A100 bfloat16 peak flops
    float flops_achieved = (float)flops_per_step * (1.0f / dt); // per second
    const int double_precision = 2;
    const int MN5_cpus_per_node = 112;
    const int MN5_num_procs_per_node = 2;
    const float intel_xeon_platinium_8480_plus_gflops =
        2329.6f; // https://www.intel.com/content/dam/support/us/en/documents/processors/APP-for-Intel-Xeon-Processors.pdf
    float flops_promised =
        ((float)num_cpus / MN5_cpus_per_node) * double_precision * MN5_num_procs_per_node * intel_xeon_platinium_8480_plus_gflops * 1e9f;
    if (flops_promised < 0) {
        return -1.f; // don't know
    }
    float mfu = flops_achieved / flops_promised;
    return mfu;
}

// Eagerly allocate every model buffer (activations, gradients, activation
// gradients, and the Adam moment buffers) for a given (B,T). gpt2_forward /
// gpt2_zero_grad / gpt2_update normally do this lazily on first use; we do it up
// front so the buffers exist to be mapped onto the device before the loop. Safe
// to call on the host path too (it just makes the allocation eager).
void gpt2_allocate(GPT2 *model, size_t B, size_t T)
{
    size_t Vp = model->config.padded_vocab_size;
    size_t L  = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C  = model->config.channels;
    size_t C4 = 4 * C;

    if (model->acts_memory == NULL) {
        model->batch_size = B;
        model->seq_len = T;
        model->act_sizes[0]  = B * T * C;          // encoded
        model->act_sizes[1]  = L * B * T * C;      // ln1
        model->act_sizes[2]  = L * B * T;          // ln1_mean
        model->act_sizes[3]  = L * B * T;          // ln1_rstd
        model->act_sizes[4]  = L * B * T * 3 * C;  // qkv
        model->act_sizes[5]  = L * B * T * C;      // atty
        model->act_sizes[6]  = L * B * NH * T * T; // preatt
        model->act_sizes[7]  = L * B * NH * T * T; // att
        model->act_sizes[8]  = L * B * T * C;      // attproj
        model->act_sizes[9]  = L * B * T * C;      // residual2
        model->act_sizes[10] = L * B * T * C;      // ln2
        model->act_sizes[11] = L * B * T;          // ln2_mean
        model->act_sizes[12] = L * B * T;          // ln2_rstd
        model->act_sizes[13] = L * B * T * C4;     // fch
        model->act_sizes[14] = L * B * T * C4;     // fch_gelu
        model->act_sizes[15] = L * B * T * C;      // fcproj
        model->act_sizes[16] = L * B * T * C;      // residual3
        model->act_sizes[17] = B * T * C;          // lnf
        model->act_sizes[18] = B * T;              // lnf_mean
        model->act_sizes[19] = B * T;              // lnf_rstd
        model->act_sizes[20] = B * T * Vp;         // logits
        model->act_sizes[21] = B * T * Vp;         // probs
        model->act_sizes[22] = B * T;              // losses
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
            num_activations += model->act_sizes[i];
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        model->inputs  = (int *)mallocCheck(B * T * sizeof(int));
        model->targets = (int *)mallocCheck(B * T * sizeof(int));
    }
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
    }
    if (model->m_memory == NULL) {
        model->m_memory = (float *)mallocCheck(model->num_parameters * sizeof(float));
        model->v_memory = (float *)mallocCheck(model->num_parameters * sizeof(float));
    }
}

void gpt2_forward(GPT2 *model, MPIWorker worker, int *inputs, int *targets, size_t B, size_t T, int *dep_handler)
{
    // Since this is purely used for allocation and error checking, I'm going to
    // keep small portion sequential.

    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        fprintf(stderr, "Error: model was not initialized properly.\n", NULL);
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;
    size_t C3 = 3 * C;
    size_t C4 = 4 * C;


    // allocate space for all the activations if needed (done here, lazily)
    if (model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        model->act_sizes[0] = B * T * C;          // encoded
        model->act_sizes[1] = L * B * T * C;      // ln1
        model->act_sizes[2] = L * B * T;          // ln1_mean
        model->act_sizes[3] = L * B * T;          // ln1_rstd
        model->act_sizes[4] = L * B * T * 3 * C;  // qkv
        model->act_sizes[5] = L * B * T * C;      // atty
        model->act_sizes[6] = L * B * NH * T * T; // preatt
        model->act_sizes[7] = L * B * NH * T * T; // att
        model->act_sizes[8] = L * B * T * C;      // attproj
        model->act_sizes[9] = L * B * T * C;      // residual2
        model->act_sizes[10] = L * B * T * C;     // ln2
        model->act_sizes[11] = L * B * T;         // ln2_mean
        model->act_sizes[12] = L * B * T;         // ln2_rstd
        model->act_sizes[13] = L * B * T * C4;    // fch
        model->act_sizes[14] = L * B * T * C4;    // fch_gelu
        model->act_sizes[15] = L * B * T * C;     // fcproj
        model->act_sizes[16] = L * B * T * C;     // residual3
        model->act_sizes[17] = B * T * C;         // lnf
        model->act_sizes[18] = B * T;             // lnf_mean
        model->act_sizes[19] = B * T;             // lnf_rstd
        model->act_sizes[20] = B * T * Vp;        // logits
        model->act_sizes[21] = B * T * Vp;        // probs
        model->act_sizes[22] = B * T;             // losses
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        if (mpi_get_rank(worker) == 0)
            fprintf(stderr, "num_activations: %zu\n\n\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int *)mallocCheck(B * T * sizeof(int));
        model->targets = (int *)mallocCheck(B * T * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this
        // is safest
        if (B != model->batch_size || T != model->seq_len)
        {
            fprintf(stderr, "Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }

    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float *acts_lnf = acts.lnf;
    float *acts_lnf_mean = acts.lnf_mean;
    float *acts_lnf_rstd = acts.lnf_rstd;
    float *acts_losses = acts.losses;
    float *acts_probs = acts.probs;
    float *acts_logits = acts.logits;
    float *acts_encoded = acts.encoded;
    float *params_lnfw = params.lnfw;
    float *params_lnfb = params.lnfb;
    float *params_wte = params.wte;
    float *params_wpe = params.wpe;
    float *residual;

    // In the target backend the model runs entirely on device-resident buffers.
    // The freshly loaded tokens are cached into the (mapped) model->inputs /
    // model->targets and every forward kernel reads those. On the host path we
    // read the raw dataloader buffers exactly as before.
#if USE_TARGET
    int *fwd_inputs  = model->inputs;
    int *fwd_targets = model->targets;
#else
    int *fwd_inputs  = inputs;
    int *fwd_targets = targets;
#endif


    ////////////////////////////////////////////////////////////////////////////////////////////////////


    // validate inputs, all indices must be in the range [0, V)
#if USE_TARGET
    // Host-side validate + cache of the freshly loaded tokens, then a single
    // H2D of the (mapped) device copies. This is the only per-iteration H2D.
    OMP_HOST_TASK( DEPEND(in, inputs[0], targets[0]) \
                   DEPEND(out, model->inputs[0], model->targets[0]) )
    {
        for (int i = 0; i < B * T; i++)
        {
            assert(0 <= inputs[i] && inputs[i] < V);
            model->inputs[i] = inputs[i];
            if (targets != NULL)
            {
                assert(0 <= targets[i] && targets[i] < V);
                model->targets[i] = targets[i];
            }
        }
    }
    #pragma omp target update to(model->inputs[0:B*T], model->targets[0:B*T]) nowait \
        DEPEND(inout, model->inputs[0], model->targets[0])
#else
    for (int i = 0; i < B * T; i += GRAN_TMP)
    {
        // OMPT_SET_LABEL("validate inputs");
        OMP_TASK( DEPEND(out, model->inputs[i: GRAN_TMP], model->targets[i: GRAN_TMP]) \
                DEPEND(in, inputs[i: GRAN_TMP], targets[i: GRAN_TMP])                         \
                firstprivate(i))
        {
            assert(0 <= inputs[i] && inputs[i] < V);
            if (targets != NULL) {
                assert(0 <= targets[i] && targets[i] < V);
            }

            // cache the inputs/targets
            memcpy(&model->inputs[i], &inputs[i], GRAN_TMP * sizeof(int));
            if (targets != NULL)
            {
                memcpy(&model->targets[i], &targets[i], GRAN_TMP * sizeof(int));
            }
        }
    }
#endif


    // forward pass
    OMP_TARGET_LOOP_TASK(collapse(2) \
            DEPEND(in, fwd_inputs[0], params_wte[0], params_wpe[0]) \
            DEPEND(out, acts_encoded[0]))
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t += GRAN_TMP)
        {
            // OMPT_SET_LABEL("encoder_forward 1");
            OMP_TASK( firstprivate(b, t) \
                    DEPEND(in, fwd_inputs[b * T + t: GRAN_TMP]) \
                    DEPEND(out, acts_encoded[b * T * C + t * C: GRAN_TMP * C]) \
                    DEPEND(in, params_wte[0: Vp * C]) \
                    DEPEND(in, params_wpe[t * C: GRAN_TMP * C]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = slh.dims_starts[1] = 0;
                slh.b_start = b;
                slh.b_end = b + 1;
                slh.dims_stops[0] = 1;
                slh.t_start = t;
                slh.t_end = slh.t_start + GRAN_TMP;
                slh.dims_stops[1] = GRAN_TMP;
                encoder_forward(acts_encoded, fwd_inputs, params_wte, params_wpe, B, T, C, slh);
            }
        }
    }


    // dep_encoder_forward(encoder_forward, acts_encoded, inputs, params_wte, params_wpe, B, T, C, dep_handler); // encoding goes into residual[0]

    for (int l = 0; l < L; l++)
    {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;

        // get the pointers of the weights for this layer
        float *l_ln1w = params.ln1w + l * C;
        float *l_ln1b = params.ln1b + l * C;
        float *l_qkvw = params.qkvw + l * 3 * C * C;
        float *l_qkvb = params.qkvb + l * 3 * C;
        float *l_attprojw = params.attprojw + l * C * C;
        float *l_attprojb = params.attprojb + l * C;
        float *l_ln2w = params.ln2w + l * C;
        float *l_ln2b = params.ln2b + l * C;
        float *l_fcw = params.fcw + l * C4 * C;
        float *l_fcb = params.fcb + l * C4;
        float *l_fcprojw = params.fcprojw + l * C * C4;
        float *l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float *l_ln1 = acts.ln1 + l * B * T * C;
        float *l_ln1_mean = acts.ln1_mean + l * B * T;
        float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float *l_qkv = acts.qkv + l * B * T * 3 * C;
        float *l_atty = acts.atty + l * B * T * C;
        float *l_preatt = acts.preatt + l * B * NH * T * T;
        float *l_att = acts.att + l * B * NH * T * T;
        float *l_attproj = acts.attproj + l * B * T * C;
        float *l_residual2 = acts.residual2 + l * B * T * C;
        float *l_ln2 = acts.ln2 + l * B * T * C;
        float *l_ln2_mean = acts.ln2_mean + l * B * T;
        float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float *l_fch = acts.fch + l * B * T * C4;
        float *l_fch_gelu = acts.fch_gelu + l * B * T * C4;
        float *l_fcproj = acts.fcproj + l * B * T * C;
        float *l_residual3 = acts.residual3 + l * B * T * C;

        // now do the forward pass

        OMP_TARGET_LOOP_TASK(collapse(2) \
                DEPEND(in, residual[0], l_ln1w[0], l_ln1b[0]) \
                DEPEND(out, l_ln1[0], l_ln1_mean[0], l_ln1_rstd[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("layernorm_forward 1");
                OMP_TASK( firstprivate(b, t)       \
                        DEPEND(in, residual[b * T * C + t * C: GRAN_TMP * C]) \
                        DEPEND(out, l_ln1[b * T * C + t * C: GRAN_TMP * C], l_ln1_mean[b * T + t: GRAN_TMP], l_ln1_rstd[b * T + t: GRAN_TMP]) \
                        DEPEND(in, l_ln1w[0: C], l_ln1b[0: C]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C, slh);
                }
            }
        }

        int OC3_GRANULARITY = (USE_TARGET ? 1 : C3 / OC_SPLIT);
        OMP_TARGET_LOOP_TASK(collapse(3) \
                DEPEND(in, l_ln1[0], l_qkvw[0], l_qkvb[0]) \
                DEPEND(out, l_qkv[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                for (int o = 0; o < C3; o += OC3_GRANULARITY)
                {
                    // OMPT_SET_LABEL("matmul_forward 1");
                    OMP_TASK( firstprivate(b, t, o, OC3_GRANULARITY)                                                                                 \
                                    DEPEND(out, l_qkv[b * T * C3 + t * C3 + o: OC3_GRANULARITY])                                                            \
                                    DEPEND(in, l_ln1[b * T * C + t * C: GRAN_TMP * C], l_qkvw[o * C: OC3_GRANULARITY * C], l_qkvb[o: OC3_GRANULARITY]))
                    {
                        // fprintf(stderr, "MF1: %07d\n", b * T * C3 + t * C3 + o);
                        struct SliceHandler slh;
                        slh.dims_starts[0] = o;
                        slh.b_start = b;
                        slh.b_end = b + 1;
                        slh.dims_stops[0] = min(o + OC3_GRANULARITY, C3);
                        slh.t_start = t;
                        slh.t_end = slh.t_start + GRAN_TMP;
                        slh.dims_stops[1] = GRAN_TMP;
                        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, C3, slh);
                    }
                }
            }
        }

#if USE_TARGET
        gpu_attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("attention_forward 1");
                OMP_TASK( firstprivate(b,t)                                                                                                              \
                                DEPEND_MULTI(in, (i_t=0:t+1, i_o=0:C3:OC3_GRANULARITY), l_qkv[b * T * C3 + i_t * C3 + i_o : OC3_GRANULARITY])                 \
                                DEPEND(out, l_atty[b * T * C + t * C: GRAN_TMP * C])                                                                            \
                                DEPEND(inout, l_preatt[b * NH * T * T + t * T: NH * T * T * GRAN_TMP], l_att[b * NH * T * T + t * T: NH * T * T * GRAN_TMP]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH, slh);

                }
            }
        }
#endif


        //////////////////////////////////////////////////////////////////////////

        const int OC_GRANULARITY = (USE_TARGET ? 1 : C / OC_SPLIT);
        OMP_TARGET_LOOP_TASK(collapse(3) \
                DEPEND(in, l_atty[0], l_attprojw[0], l_attprojb[0]) \
                DEPEND(out, l_attproj[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                for (int o = 0; o < C; o += OC_GRANULARITY)
                {
                    // OMPT_SET_LABEL("matmul_forward");
                    OMP_TASK( firstprivate(b,t,o)                                                                                                        \
                                    DEPEND(in, l_atty[b * T * C + t * C: GRAN_TMP * C], l_attprojw[o * C: OC_GRANULARITY * C], l_attprojb[o: OC_GRANULARITY])  \
                                    DEPEND(out, l_attproj[b * T * C + t * C + o: OC_GRANULARITY] ))
                    {
                        struct SliceHandler slh;
                        slh.dims_starts[0] = o;
                        slh.b_start = b;
                        slh.b_end = b + 1;
                        slh.dims_stops[0] = min(o + OC_GRANULARITY, C);
                        slh.t_start = t;
                        slh.t_end = slh.t_start + GRAN_TMP;
                        slh.dims_stops[1] = GRAN_TMP;
                        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C, slh);
                    }
                }
            }
        }


        //////////////////////////////////////////////////////////////////////////

        OMP_TARGET_LOOP_TASK(collapse(2) \
                DEPEND(in, l_attproj[0], residual[0]) \
                DEPEND(out, l_residual2[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("residual_forward");
                OMP_TASK( firstprivate(b, t) \
                                DEPEND_MULTI(in, (i_o=0:C:OC_GRANULARITY), l_attproj[b * T * C + t * C + i_o : OC_GRANULARITY]) \
                                DEPEND(in, residual[b * T * C + t * C : GRAN_TMP * C]) \
                                DEPEND(out, l_residual2[b * T * C + t * C: GRAN_TMP * C]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    residual_forward(l_residual2, residual, l_attproj, B, T, C, slh);

                }
            }
        }


        //////////////////////////////////////////////////////////////////////////

        // dep_residual_forward(residual_forward, l_residual2, residual, l_attproj, B, T, C, dep_handler);

        OMP_TARGET_LOOP_TASK(collapse(2) \
                DEPEND(in, l_residual2[0], l_ln2w[0], l_ln2b[0]) \
                DEPEND(out, l_ln2[0], l_ln2_mean[0], l_ln2_rstd[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("layernorm_forward");
                OMP_TASK( firstprivate(b, t)                                                                                                     \
                                DEPEND(in, l_residual2[b * T * C + t * C: GRAN_TMP * C], l_ln2w[0: C], l_ln2b[0: C])                                    \
                                DEPEND(out, l_ln2[b * T * C + t * C: GRAN_TMP * C], l_ln2_mean[b * T + t: GRAN_TMP], l_ln2_rstd[b * T + t: GRAN_TMP]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C, slh);
                }
            }
        }


        const int OC4_GRANULARITY = (USE_TARGET ? 1 : C4 / OC_SPLIT);

        OMP_TARGET_LOOP_TASK(collapse(3) \
                DEPEND(in, l_ln2[0], l_fcw[0], l_fcb[0]) \
                DEPEND(out, l_fch[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                for (int o = 0; o < C4; o += OC4_GRANULARITY)
                {
                    // OMPT_SET_LABEL("matmul_forward");
                    OMP_TASK( firstprivate(b, t, o, B, T, C, C4)                                                                                 \
                                    DEPEND(in, l_ln2[b * T * C + t * C: GRAN_TMP * C], l_fcw[o * C: OC4_GRANULARITY * C], l_fcb[o: OC4_GRANULARITY])   \
                                    DEPEND(out, l_fch[b * T * C4 + t * C4 + o: OC4_GRANULARITY]))
                    {
                        struct SliceHandler slh;
                        slh.dims_starts[0] = o;
                        slh.b_start = b;
                        slh.b_end = b + 1;
                        slh.dims_stops[0] = min(o + OC4_GRANULARITY, C4);
                        slh.t_start = t;
                        slh.t_end = slh.t_start + GRAN_TMP;
                        slh.dims_stops[1] = GRAN_TMP;
                        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, C4, slh);
                    }
                }
            }
        }

        //////////////////////////////////////////////////////////////////////////

        OMP_TARGET_LOOP_TASK(collapse(2) \
                DEPEND(in, l_fch[0]) \
                DEPEND(out, l_fch_gelu[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("gelu_forward");
                OMP_TASK( firstprivate(b, t)                                                                                 \
                                DEPEND_MULTI(in, (o=0:C4:OC4_GRANULARITY), l_fch[b * T * C4 + t * C4 + o : OC4_GRANULARITY])       \
                                DEPEND(out, l_fch_gelu[b * T * C4 + t * C4: GRAN_TMP * C4]))
                {
                    // Does not use dims[1] for some god unknown reason. Must set t granularity in [0]
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = GRAN_TMP;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    gelu_forward(l_fch_gelu, l_fch, B, T, C4, slh);

                }
            }
        }


        //////////////////////////////////////////////////////////////////////////

        OMP_TARGET_LOOP_TASK(collapse(3) \
                DEPEND(in, l_fch_gelu[0], l_fcprojw[0], l_fcprojb[0]) \
                DEPEND(out, l_fcproj[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                for (int o = 0; o < C; o += OC_GRANULARITY)
                {
                    // OMPT_SET_LABEL("matmul_forward");
                    OMP_TASK( firstprivate(b, t, o, B, T, C4, C)  \
                            DEPEND(out, l_fcproj[b * T * C + t * C + o: OC_GRANULARITY]) \
                            DEPEND(in, l_fch_gelu[b * T * C4 + t * C4: GRAN_TMP * C4], l_fcprojw[o * C4: OC_GRANULARITY * C4], l_fcprojb[o: OC_GRANULARITY]))
                    {
                        struct SliceHandler slh;
                        slh.dims_starts[0] = o;
                        slh.b_start = b;
                        slh.b_end = b + 1;
                        slh.dims_stops[0] = min(o + OC_GRANULARITY, C);
                        slh.t_start = t;
                        slh.t_end = slh.t_start + GRAN_TMP;
                        slh.dims_stops[1] = GRAN_TMP;
                        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, C4, C, slh);
                    }
                }
            }
        }

        //////////////////////////////////////////////////////////////////////////

        OMP_TARGET_LOOP_TASK(collapse(2) \
                DEPEND(in, l_fcproj[0], l_residual2[0]) \
                DEPEND(out, l_residual3[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("residual_forward");
                OMP_TASK( firstprivate(b, t) \
                                DEPEND_MULTI(in, (i_o=0:C:OC_GRANULARITY), l_fcproj[b * T * C + t * C + i_o : OC_GRANULARITY]) \
                                DEPEND(in, l_residual2[b * T * C + t * C : GRAN_TMP * C]) \
                                DEPEND(out, l_residual3[b * T * C + t * C: GRAN_TMP * C]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    residual_forward(l_residual3, l_residual2, l_fcproj, B, T, C, slh);
                }
            }
        }

        // exit(1);
    }

    residual = acts.residual3 + (L - 1) * B * T * C; // last residual is in residual3

    OMP_TARGET_LOOP_TASK(collapse(2) \
            DEPEND(in, residual[0], params_lnfw[0], params_lnfb[0]) \
            DEPEND(out, acts_lnf[0], acts_lnf_mean[0], acts_lnf_rstd[0]))
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t += GRAN_TMP)
        {
            // OMPT_SET_LABEL("layernorm_forward");
            OMP_TASK( firstprivate(b, t)                                                                                                         \
                        DEPEND(in, residual[b * T * C + t * C: GRAN_TMP * C], params_lnfw[0: C], params_lnfb[0: C])                            \
                        DEPEND(out, acts_lnf[b * T * C + t * C: GRAN_TMP * C], acts_lnf_mean[b * T + t: GRAN_TMP], acts_lnf_rstd[b * T + t: GRAN_TMP]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = slh.dims_starts[1] = 0;
                slh.b_start = b;
                slh.b_end = b + 1;
                slh.dims_stops[0] = 1;
                slh.t_start = t;
                slh.t_end = slh.t_start + GRAN_TMP;
                slh.dims_stops[1] = GRAN_TMP;
                layernorm_forward(acts_lnf, acts_lnf_mean, acts_lnf_rstd, residual, params_lnfw, params_lnfb, B, T, C, slh);
            }
        }
    }

    const int VP_GRANULARITY = (USE_TARGET ? 1 : Vp / OC_SPLIT);

    OMP_TARGET_LOOP_TASK(collapse(3) \
            DEPEND(in, acts_lnf[0], params_wte[0]) \
            DEPEND(out, acts_logits[0]))
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t += GRAN_TMP)
        {
            for (int o = 0; o < Vp; o += VP_GRANULARITY)
            {
                // OMPT_SET_LABEL("matmul_forward");
                OMP_TASK( firstprivate(b, t, o)                                                                      \
                            DEPEND(in, acts_lnf[b * T * C + t * C: GRAN_TMP * C], params_wte[o * C: VP_GRANULARITY * C])    \
                            DEPEND(out, acts_logits[b * T * Vp + t * Vp + o: VP_GRANULARITY]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = o;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = min(o + VP_GRANULARITY, Vp);
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    matmul_forward(acts_logits, acts_lnf, params_wte, NULL, B, T, C, Vp, slh);
                }
            }
        }
    }

    // dep_matmul_forward(matmul_forward, acts_logits, acts_lnf, params_wte, NULL, B, T, C, Vp, dep_handler);

    OMP_TARGET_LOOP_TASK(collapse(2) \
            DEPEND(in, acts_logits[0]) \
            DEPEND(out, acts_probs[0]))
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t += GRAN_TMP)
        {
            // for (int o = 0; o < Vp; o += VP_GRANULARITY)
            // {
            // OMPT_SET_LABEL("softmax_forward");
            OMP_TASK( firstprivate(b, t) \
                    DEPEND_MULTI(in, (o=0:Vp:VP_GRANULARITY), acts_logits[b * T * Vp + t * Vp + o : VP_GRANULARITY]) \
                    DEPEND(out, acts_probs[b * T * Vp + t * Vp: GRAN_TMP * Vp]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = slh.dims_starts[1] = 0;
                slh.b_start = b;
                slh.b_end = b + 1;
                slh.dims_stops[0] = 1;
                slh.t_start = t;
                slh.t_end = slh.t_start + GRAN_TMP;
                slh.dims_stops[1] = GRAN_TMP;
                softmax_forward(acts_probs, acts_logits, B, T, V, Vp, slh);
            }
            // }
        }
    }

    // TODO: Loss calculations

    // also forward the cross-entropy loss function if we have the targets
    if (targets != NULL)
    {
        OMP_TARGET_LOOP_TASK(collapse(2) \
                DEPEND(in, acts_probs[0], fwd_targets[0]) \
                DEPEND(out, acts_losses[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("crossentropy_forward");
                OMP_TASK( firstprivate(b, t) \
                            DEPEND(out, acts_losses[b * T + t: GRAN_TMP])   \
                            DEPEND(in, acts_probs[b * T * Vp + t * Vp: GRAN_TMP * Vp], fwd_targets[b * T + t: GRAN_TMP]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    crossentropy_forward(acts_losses, acts_probs, fwd_targets, B, T, Vp, slh);
                }
            }
        }

        // OMPT_SET_LABEL("zero mean loss"); (loop-less: plain target region on GPU)
        OMP_TARGET_TASK( DEPEND(out, model->mean_loss))
        {
            model->mean_loss = 0.0f;
        }


        float *loss = &model->mean_loss;
        (void) loss; // unused in the target backend (loss lives in model->mean_loss)
        OMP_TARGET_LOOP_TASK(collapse(2) \
                DEPEND(in, acts_losses[0]) \
                DEPEND(inout, model->mean_loss))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("loss calculation");
                OMP_TASK( firstprivate(loss, b, t, T) \
                        DEPEND(in, acts_losses[b * T + t: GRAN_TMP]) \
                        DEPEND(inout, model->mean_loss))
                {
                    float l = 0.0f;
                    for (int g = 0; g < GRAN_TMP; g++)
                    {
                        l += acts_losses[b * T + t + g];
                    }

                    // accumulate into the (device-resident) scalar loss
                    ATOMIC
                    model->mean_loss += l;
                }
            }
        }

        // OMPT_SET_LABEL("assign loss"); (loop-less: plain target region on GPU)
        OMP_TARGET_TASK( DEPEND(inout, model->mean_loss) firstprivate(B, T))
        {
            model->mean_loss = model->mean_loss / (B * T);
        }
    }
    else
    {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}


void gpt2_zero_grad(GPT2 *model)
{
    // lazily allocate the memory for gradients of the weights and activations,
    // if needed
    if (model->grads_memory == NULL)
    {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
    }

    int B = model->batch_size;
    int T = model->seq_len;
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;
    int C3 = C * 3;
    int C4 = C * 4;

    int grads_partitioning[] = {1, 1, L, L, L, L, L, L, L, L, L, L, L, L, 1, 1};
    // The subpartitions of grads structures in the backward pass
    int grads_subpartition[] = {
        /*wte*/ (Vp / OC_BACK_SPLIT) * C,
        /*wpe*/ GRAN_TMP * C,
        /*ln1w*/ C,
        /*ln1b*/ C,
        /*qkvw*/ (C3 / OC_BACK_SPLIT) * C3,
        /*qkvb*/ (C3 / OC_BACK_SPLIT),
        /*attprojw*/ (C / OC_BACK_SPLIT) * C,
        /*attprojb*/ (C / OC_BACK_SPLIT),
        /*ln2w*/ C,
        /*ln2b*/ C,
        /*fcw*/ (C4 / OC_BACK_SPLIT) * C4,
        /*fcb*/ (C4 / OC_BACK_SPLIT),
        /*fcprojw*/ (C / OC_BACK_SPLIT) * C,
        /*fcprojb*/ (C / OC_BACK_SPLIT),
        /*lnfw*/ C,
        /*lnfb*/ C,
    };
    int grads_acts_partitioning[] = {1, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, 1, 1, 1, 1, 1, 1};
    int grads_acts_subpartition[] = {
        /*encoded*/ GRAN_TMP * C,
        /*ln1*/ GRAN_TMP * C,
        /*ln1_mean*/ -1,
        /*ln1_rstd*/ -1,
        /*qkv*/ GRAN_TMP * C3,
        /*atty*/ GRAN_TMP * C,
        /*preatt*/ NH * T * T,
        /*att*/ NH * T * T,
        /*attproj*/ GRAN_TMP * C,
        /*residual2*/ GRAN_TMP * C,
        /*ln2*/ GRAN_TMP * C,
        /*ln2_mean*/ -1,
        /*ln2_rstd*/ -1,
        /*fch*/ GRAN_TMP * C4,
        /*fch_gelu*/ GRAN_TMP * C4,
        /*fcproj*/ GRAN_TMP * C,
        /*residual3*/ GRAN_TMP * C,
        /*lnf*/ GRAN_TMP * C,
        /*lnf_mean*/ -1,
        /*lnf_rstd*/ -1,
        /*logits*/ GRAN_TMP * Vp,
        /*probs*/ -1,
        /*losses*/ GRAN_TMP,
    };
    float *memory_ptr = model->grads_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        for (int l = 0; l < grads_partitioning[i]; l++)
        {
            size_t memory_size = model->param_sizes[i] / ((size_t)grads_partitioning[i]);
            int subpart = grads_subpartition[i];

#if USE_TARGET
            // Device zeroing: one offloaded kernel per slice. The slice base
            // memory_ptr[0] matches the per-tensor base used by the backward
            // accumulators and the optimizer, so zero -> backward -> update (and
            // the cross-iteration update -> zero WAR) is ordered by base address.
            (void) subpart;
            OMP_TARGET_LOOP_TASK(DEPEND(out, memory_ptr[0]))
            for (size_t j = 0; j < memory_size; j++)
                memory_ptr[j] = 0.0f;
#else
            if (subpart < 0)
            {
                if (subpart < 0)
                {
                    // OMPT_SET_LABEL("zero grads_memory 1");
                    OMP_TASK( firstprivate(memory_ptr, memory_size) \
                            DEPEND(out, memory_ptr[0]))
                    { memset(memory_ptr, 0, memory_size * sizeof(float)); }
                }
            }
            else
            {
                for (size_t j = 0; j < memory_size; j += subpart)
                {
                    size_t actual = (j + subpart <= memory_size) ? subpart : (memory_size - j);

                    // OMPT_SET_LABEL("zero grads_memory 2");
                    OMP_TASK( firstprivate(memory_ptr, j, actual) \
                            DEPEND(out, memory_ptr[j]))       // base == consumer's chunk base
                    { memset(memory_ptr + j, 0, actual * sizeof(float)); }
                }
            }
#endif

            memory_ptr += memory_size;
        }
    }

    memory_ptr = model->grads_acts_memory;
    for (int i = 0; i < NUM_ACTIVATION_TENSORS; i++)
    {
        for (int l = 0; l < grads_acts_partitioning[i]; l++)
        {
            size_t memory_size = model->act_sizes[i] / ((size_t)grads_acts_partitioning[i]);
            int subpart = grads_acts_subpartition[i];

#if USE_TARGET
            (void) subpart;
            OMP_TARGET_LOOP_TASK(DEPEND(out, memory_ptr[0]))
            for (size_t j = 0; j < memory_size; j++)
                memory_ptr[j] = 0.0f;
#else
            if (subpart < 0)
            {
                if (subpart < 0)
                {
                    // OMPT_SET_LABEL("zero grads_acts_memory 1");
                    OMP_TASK( firstprivate(memory_ptr, memory_size) \
                            DEPEND(out, memory_ptr[0]))
                    { memset(memory_ptr, 0, memory_size * sizeof(float)); }
                }
            }
            else
            {
                for (size_t j = 0; j < memory_size; j += subpart)
                {
                    size_t actual = (j + subpart <= memory_size) ? subpart : (memory_size - j);
                    // OMPT_SET_LABEL("zero grads_acts_memory 2");
                    OMP_TASK( firstprivate(memory_ptr, j, actual) \
                            DEPEND(out, memory_ptr[j]))       // base == consumer's chunk base
                    { memset(memory_ptr + j, 0, actual * sizeof(float)); }
                }
            }
#endif

            memory_ptr += memory_size;
        }
    }
}

#if USE_TARGET
// Fine-grained device matmul_input_backward: one GPU thread per (token, input
// channel), reducing over the output channels (no atomics). Replaces the coarse
// one-thread-per-row kernel. dinp is (B,T,Cin), dout is (B,T,OC), weight is
// (OC, Cin); computes dinp += weight^T . dout.
static void gpu_matmul_input_backward(float *dinp, const float *dout, const float *weight,
                                      int B, int T, int Cin, int OC)
{
    OMP_TARGET_LOOP_TASK(collapse(2) \
        DEPEND(in, dout[0], weight[0]) DEPEND(inout, dinp[0]))
    for (int bt = 0; bt < B * T; bt++)
        for (int i = 0; i < Cin; i++)
        {
            float s = 0.0f;
            for (int o = 0; o < OC; o++)
                s += weight[o * Cin + i] * dout[bt * OC + o];
            dinp[bt * Cin + i] += s;
        }
}

// Fine-grained device matmul_params_backward: one GPU thread per (out-channel,
// in-channel) reducing over tokens for the weight grads (no atomics), plus one
// thread per out-channel for the bias grads. Replaces the coarse OC-thread
// kernel. dweight is (OC, Cin), dbias is (OC) [may be NULL], dout is (B,T,OC),
// inp is (B,T,Cin); computes dweight += dout^T . inp, dbias += sum_tokens dout.
static void gpu_matmul_params_backward(float *dweight, float *dbias, const float *dout,
                                       const float *inp, int B, int T, int Cin, int OC)
{
    OMP_TARGET_LOOP_TASK(collapse(2) \
        DEPEND(in, dout[0], inp[0]) DEPEND(inout, dweight[0]))
    for (int o = 0; o < OC; o++)
        for (int i = 0; i < Cin; i++)
        {
            float s = 0.0f;
            for (int bt = 0; bt < B * T; bt++)
                s += inp[bt * Cin + i] * dout[bt * OC + o];
            dweight[o * Cin + i] += s;
        }

    if (dbias != NULL)
    {
        OMP_TARGET_LOOP_TASK(DEPEND(in, dout[0]) DEPEND(inout, dbias[0]))
        for (int o = 0; o < OC; o++)
        {
            float s = 0.0f;
            for (int bt = 0; bt < B * T; bt++)
                s += dout[bt * OC + o];
            dbias[o] += s;
        }
    }
}

// Fine-grained elementwise device kernels (one GPU thread per element) for the
// otherwise 256-thread (b,t)-tiled backward ops.
static void gpu_gelu_backward(float *dinp, const float *inp, const float *dout, long N)
{
    OMP_TARGET_LOOP_TASK(DEPEND(in, inp[0], dout[0]) DEPEND(inout, dinp[0]))
    for (long i = 0; i < N; i++) {
        float x = inp[i];
        float tanh_arg = GELU_SCALING_FACTOR * (x + 0.044715f * x * x * x);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + sech_out * (1.5f * tanh_arg - GELU_SCALING_FACTOR * x);
        dinp[i] += local_grad * dout[i];
    }
}

static void gpu_residual_backward(float *dinp1, float *dinp2, const float *dout, long N)
{
    OMP_TARGET_LOOP_TASK(DEPEND(in, dout[0]) DEPEND(inout, dinp1[0], dinp2[0]))
    for (long i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

// crossentropy+softmax backward is elementwise over (token, vocab): one thread
// per (bt, i) reducing nothing -> collapse(2) over B*T*V.
static void gpu_crossentropy_softmax_backward(float *dlogits, const float *dlosses, const float *probs,
                                              const int *targets, int B, int T, int V, int Vp)
{
    OMP_TARGET_LOOP_TASK(collapse(2) \
        DEPEND(in, dlosses[0], probs[0], targets[0]) DEPEND(inout, dlogits[0]))
    for (int bt = 0; bt < B * T; bt++)
        for (int i = 0; i < V; i++) {
            float dloss = dlosses[bt];
            int ix = targets[bt];
            float p = probs[(long)bt * Vp + i];
            float indicator = (i == ix) ? 1.0f : 0.0f;
            dlogits[(long)bt * Vp + i] += (p - indicator) * dloss;
        }
}
#endif

void gpt2_backward(GPT2 *model, int *dep_handler)
{
    // convenience shortcuts (and size_t to help prevent int overflow)
    int B = model->batch_size;
    int T = model->seq_len;
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;
    int C3 = C * 3;
    int C4 = C * 4;

    // backward pass: go in the reverse order of the forward pass, and call
    // backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;
    float *grads_acts_logits = grads_acts.logits;
    float *grads_acts_losses = grads_acts.losses;
    float *grads_acts_lnf = grads_acts.lnf;
    float *grads_acts_encoded = grads_acts.encoded;
    float *acts_probs = acts.probs;
    float *acts_lnf = acts.lnf;
    float *acts_lnf_mean = acts.lnf_mean;
    float *acts_lnf_rstd = acts.lnf_rstd;
    float *grads_wte = grads.wte;
    float *grads_lnfw = grads.lnfw;
    float *grads_lnfb = grads.lnfb;
    float *grads_wpe = grads.wpe;
    float *params_wte = params.wte;
    float *params_lnfw = params.lnfw;
    float *custom_nullptr = NULL;
    int *model_targets = model->targets;
    int *model_inputs = model->inputs;

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in
    // the batch
    float dloss_mean = 1.0f / (BATCH_SIZE * SEQUENCE_SIZE);

    OMP_TARGET_LOOP_TASK(collapse(2) \
            DEPEND(out, grads_acts_losses[0]))
    for(int b = 0; b < B; ++b)
    {
        for(int t = 0; t < T; t += GRAN_TMP)
        {
            // OMPT_SET_LABEL("grads_acts_losses");
            OMP_TASK( firstprivate(b, t) \
                    DEPEND(in, dloss_mean) \
                    DEPEND(out, grads_acts_losses[b * T + t: GRAN_TMP]))
                {
                    for (int k = 0; k < GRAN_TMP; ++k)
                    {
                        grads_acts_losses[b * T + t + k] = dloss_mean;
                    }
                }
        }
    }

#if USE_TARGET
    gpu_crossentropy_softmax_backward(grads_acts_logits, grads_acts_losses, acts_probs, model_targets, B, T, V, Vp);
#else
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t += GRAN_TMP)
        {
            // OMPT_SET_LABEL("crossentropy_softmax_backward");
            OMP_TASK( firstprivate(b, t) \
                        DEPEND(inout, grads_acts_logits[b * T * Vp + t * Vp: GRAN_TMP * Vp]) \
                        DEPEND(in, grads_acts_losses[b * T + t: GRAN_TMP], model_targets[b * T + t: GRAN_TMP], acts_probs[b * T * Vp + t * Vp: GRAN_TMP * Vp]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = slh.dims_starts[1] = 0;
                slh.b_start = b;
                slh.b_end = b + 1;
                slh.dims_stops[0] = 1;
                slh.t_start = t;
                slh.t_end = slh.t_start + GRAN_TMP;
                slh.dims_stops[1] = GRAN_TMP;
                crossentropy_softmax_backward(grads_acts_logits, grads_acts_losses, acts_probs, model_targets, B, T, V, Vp, slh);
            }
        }
    }
#endif

    // Deps are fork-join due to change in split

    // dep_crossentropy_softmax_backward(crossentropy_softmax_backward, grads_acts_logits, grads_acts_losses, acts_probs, model_targets, B, T, V, Vp,
    //                                   dep_handler);

    // params has OC split, input not?
    // TODO Temp OC Split to 10, but may be even better if higher
    const int VP_GRANULARITY = (USE_TARGET ? 1 : Vp / OC_BACK_SPLIT);

#if USE_TARGET
    gpu_matmul_params_backward(grads_wte, custom_nullptr, grads_acts_logits, acts_lnf, B, T, C, Vp);
#else
    for (int o = 0; o < Vp; o += VP_GRANULARITY)
    {
        // OMPT_SET_LABEL("matmul_params_backward");
        OMP_TASK( firstprivate(o)    \
                DEPEND(out, grads_wte[o * C : VP_GRANULARITY * C]) \
                DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), grads_acts_logits[b * T * Vp + t * Vp : GRAN_TMP * Vp]) \
                DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), acts_lnf[b * T * C + t * C : GRAN_TMP * C]))
        {
            struct SliceHandler slh;
            slh.dims_starts[0] = o;
            slh.dims_stops[0] = min(o + VP_GRANULARITY, Vp);
            matmul_params_backward(grads_wte, custom_nullptr, grads_acts_logits, acts_lnf, B, T, C, Vp, slh);
        }
    }

#endif
#if USE_TARGET
    gpu_matmul_input_backward(grads_acts_lnf, grads_acts_logits, params_wte, B, T, C, Vp);
#else
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t += GRAN_TMP)
        {
            // OMPT_SET_LABEL("matmul_input_backward");
            OMP_TASK( firstprivate(b, t)                                           \
                        DEPEND(inout, grads_acts_lnf[b * T * C + t * C: GRAN_TMP * C])    \
                        DEPEND(in, grads_acts_logits[b * T * Vp + t * Vp: GRAN_TMP * Vp]) \
                        DEPEND(in, params_wte[0 : Vp * C]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = slh.dims_starts[1] = 0;
                slh.b_start = b;
                slh.b_end = b + 1;
                slh.dims_stops[0] = 1;
                slh.t_start = t;
                slh.t_end = slh.t_start + GRAN_TMP;
                slh.dims_stops[1] = GRAN_TMP;
                matmul_input_backward(grads_acts_lnf, grads_acts_logits, params_wte, B, T, C, Vp, slh);
            }
        }
    }
#endif

    float *residual = acts.residual3 + (L - 1) * B * T * C;        // last layer's residual
    float *dresidual = grads_acts.residual3 + (L - 1) * B * T * C; // write to last layer's residual

    OMP_TARGET_LOOP_TASK(collapse(2) \
            DEPEND(in, grads_acts_lnf[0], residual[0], params_lnfw[0], acts_lnf_mean[0], acts_lnf_rstd[0]) \
            DEPEND(inout, dresidual[0], grads_lnfw[0], grads_lnfb[0]))
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t += GRAN_TMP)
        {
            // OMPT_SET_LABEL("layernorm_backward");
            OMP_TASK( firstprivate(b, t) \
                        DEPEND(inoutset, grads_lnfw[0:C], grads_lnfb[0:C]) \
                        DEPEND(in, grads_acts_lnf[b * T * C + t * C: GRAN_TMP * C], residual[b * T * C + t * C: GRAN_TMP * C], params_lnfw[0: C], acts_lnf_mean[b * T + t: GRAN_TMP], acts_lnf_rstd[b * T + t: GRAN_TMP]) \
                        DEPEND(inout, dresidual[b * T * C + t * C: GRAN_TMP * C]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = slh.dims_starts[1] = 0;
                slh.b_start = b;
                slh.b_end = b + 1;
                slh.dims_stops[0] = 1;
                slh.t_start = t;
                slh.t_end = slh.t_start + GRAN_TMP;
                slh.dims_stops[1] = GRAN_TMP;
                layernorm_backward(dresidual, grads_lnfw, grads_lnfb, grads_acts_lnf, residual, params_lnfw, acts_lnf_mean, acts_lnf_rstd, B, T, C, slh);
            }
        }
    }



    // dep_layernorm_backward(layernorm_backward, dresidual, grads_lnfw, grads_lnfb, grads_acts_lnf, residual, params_lnfw, acts_lnf_mean, acts_lnf_rstd, B,
    //                        T, C, dep_handler);

    for (int l = L - 1; l >= 0; l--)
    {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l - 1) * B * T * C;

        // get the pointers of the weights for this layer
        float *l_ln1w = params.ln1w + l * C;
        float *l_qkvw = params.qkvw + l * C3 * C;
        float *l_attprojw = params.attprojw + l * C * C;
        float *l_ln2w = params.ln2w + l * C;
        float *l_fcw = params.fcw + l * C4 * C;
        float *l_fcprojw = params.fcprojw + l * C * C4;
        // get the pointers of the gradients of the weights for this layer
        float *dl_ln1w = grads.ln1w + l * C;
        float *dl_ln1b = grads.ln1b + l * C;
        float *dl_qkvw = grads.qkvw + l * C3 * C;
        float *dl_qkvb = grads.qkvb + l * C3;
        float *dl_attprojw = grads.attprojw + l * C * C;
        float *dl_attprojb = grads.attprojb + l * C;
        float *dl_ln2w = grads.ln2w + l * C;
        float *dl_ln2b = grads.ln2b + l * C;
        float *dl_fcw = grads.fcw + l * C4 * C;
        float *dl_fcb = grads.fcb + l * C4;
        float *dl_fcprojw = grads.fcprojw + l * C * C4;
        float *dl_fcprojb = grads.fcprojb + l * C;
        // get the pointers of the activations for this layer
        float *l_ln1 = acts.ln1 + l * B * T * C;
        float *l_ln1_mean = acts.ln1_mean + l * B * T;
        float *l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float *l_qkv = acts.qkv + l * B * T * C3;
        float *l_atty = acts.atty + l * B * T * C;
        float *l_att = acts.att + l * B * NH * T * T;
        float *l_residual2 = acts.residual2 + l * B * T * C;
        float *l_ln2 = acts.ln2 + l * B * T * C;
        float *l_ln2_mean = acts.ln2_mean + l * B * T;
        float *l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float *l_fch = acts.fch + l * B * T * C4;
        float *l_fch_gelu = acts.fch_gelu + l * B * T * C4;
        // get the pointers of the gradients of the activations for this layer
        float *dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float *dl_qkv = grads_acts.qkv + l * B * T * C3;
        float *dl_atty = grads_acts.atty + l * B * T * C;
        float *dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float *dl_att = grads_acts.att + l * B * NH * T * T;
        float *dl_attproj = grads_acts.attproj + l * B * T * C;
        float *dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float *dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float *dl_fch = grads_acts.fch + l * B * T * C4;
        float *dl_fch_gelu = grads_acts.fch_gelu + l * B * T * C4;
        float *dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float *dl_residual3 = grads_acts.residual3 + l * B * T * C;
        // backprop this layer

#if USE_TARGET
        gpu_residual_backward(dl_residual2, dl_fcproj, dl_residual3, (long)B * T * C);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("residual_backward");
                OMP_TASK( firstprivate(b, t) \
                            DEPEND(inout, dl_residual2[b * T * C + t * C: GRAN_TMP * C], \
                                          dl_fcproj[b * T * C + t * C: GRAN_TMP * C]) \
                            DEPEND(in, dl_residual3[b * T * C + t * C: GRAN_TMP * C]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    residual_backward(dl_residual2, dl_fcproj, dl_residual3, B, T, C, slh);
                }
            }
        }
#endif

        const int OC_GRANULARITY = (USE_TARGET ? 1 : C / OC_BACK_SPLIT);
        const int OC3_GRANULARITY_b = C3 / OC_SPLIT;
        const int OC4_GRANULARITY_b = C4 / OC_SPLIT;

#if USE_TARGET
        gpu_matmul_params_backward(dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, B, T, C4, C);
#else
        for (int o = 0; o < C; o += OC_GRANULARITY)
        {
            // OMPT_SET_LABEL("matmul_params_backward");
            OMP_TASK( firstprivate(o) \
                    DEPEND(out, dl_fcprojw[o * C: OC_GRANULARITY * C], dl_fcprojb[o: OC_GRANULARITY]) \
                    DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), dl_fcproj[b * T * C + t * C : GRAN_TMP * C]) \
                    DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), l_fch_gelu[b * T * C4 + t * C4 : GRAN_TMP * C4]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = o;
                slh.dims_stops[0] = min(o + OC_GRANULARITY, C);
                matmul_params_backward(dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, B, T, C4, C, slh);
            }
        }
#endif

#if USE_TARGET
        gpu_matmul_input_backward(dl_fch_gelu, dl_fcproj, l_fcprojw, B, T, C4, C);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("matmul_input_backward");
                OMP_TASK( firstprivate(b, t) \
                        DEPEND(in, dl_fcproj[b * T * C + t * C: GRAN_TMP * C], l_fcprojw[0: C * C4]) \
                        DEPEND(inout, dl_fch_gelu[b * T * C4 + t * C4: GRAN_TMP * C4]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    matmul_input_backward(dl_fch_gelu, dl_fcproj, l_fcprojw, B, T, C4, C, slh);
                }
            }
        }
#endif
#if USE_TARGET
        gpu_gelu_backward(dl_fch, l_fch, dl_fch_gelu, (long)B * T * C4);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("gelu_backward");
                OMP_TASK( firstprivate(b, t) \
                            DEPEND(in, l_fch[b * T * C4 + t * C4: GRAN_TMP * C4], dl_fch_gelu[b * T * C4 + t * C4: GRAN_TMP * C4]) \
                            DEPEND(inout, dl_fch[b * T * C4 + t * C4: GRAN_TMP * C4]))
                {
                    struct SliceHandler slh;
                    // GELU does not use dims_[starts,stops][1]
                    slh.dims_starts[0] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[0] = GRAN_TMP;
                    gelu_backward(dl_fch, l_fch, dl_fch_gelu, B, T, C4, slh);
                }
            }
        }
#endif


        // dep_residual_backward(residual_backward, dl_residual2, dl_fcproj, dl_residual3, B, T, C, dep_handler);
        // dep_matmul_params_backward(matmul_params_backward, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, B, T, C4, C, dep_handler, dl_residual3);
        // dep_matmul_input_backward(matmul_input_backward, dl_fch_gelu, dl_fcproj, l_fcprojw, B, T, C4, C, dep_handler);
        // dep_gelu_backward(gelu_backward, dl_fch, l_fch, dl_fch_gelu, B, T, C4, dep_handler);

        // const int OC4_GRANULARITY = C4 / OC_BACK_SPLIT;
        const int OC4_GRANULARITY = (USE_TARGET ? 1 : C4 / OC_BACK_SPLIT);
#if USE_TARGET
        gpu_matmul_params_backward(dl_fcw, dl_fcb, dl_fch, l_ln2, B, T, C, C4);
#else
        for (int o = 0; o < C4; o += OC4_GRANULARITY)
        {
            // OMPT_SET_LABEL("matmul_params_backward");
            OMP_TASK( \
                firstprivate(o) \
                DEPEND(out, dl_fcw[o * C4 : OC4_GRANULARITY * C4], dl_fcb[o : OC4_GRANULARITY])              \
                DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), dl_fch[b * T * C4 + t * C4: GRAN_TMP * C4]) \
                DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), l_ln2[b * T * C + t * C: GRAN_TMP * C]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = o;
                slh.dims_stops[0] = min(o + OC4_GRANULARITY, C4);
                matmul_params_backward(dl_fcw, dl_fcb, dl_fch, l_ln2, B, T, C, C4, slh);
            }
        }
#endif

#if USE_TARGET
        gpu_matmul_input_backward(dl_ln2, dl_fch, l_fcw, B, T, C, C4);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("matmul_input_backward");
                OMP_TASK( firstprivate(b, t) \
                    DEPEND(inout, dl_ln2[b * T * C + t * C : GRAN_TMP * C]) \
                    DEPEND(in, dl_fch[b * T * C4 + t * C4: GRAN_TMP * C4]) \
                    DEPEND(in, l_fcw[0 : C4*C]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    matmul_input_backward(dl_ln2, dl_fch, l_fcw, B, T, C, C4, slh);
                }
            }
        }
#endif

        OMP_TARGET_LOOP_TASK(collapse(2) \
                DEPEND(in, dl_ln2[0], l_residual2[0], l_ln2w[0], l_ln2_mean[0], l_ln2_rstd[0]) \
                DEPEND(inout, dl_residual2[0], dl_ln2w[0], dl_ln2b[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("layernorm_backward");
                OMP_TASK( firstprivate(b, t) \
                    DEPEND(inout, dl_residual2[b * T * C + t * C: GRAN_TMP * C]) \
                    DEPEND(inoutset, dl_ln2w[0:C], dl_ln2b[0:C]) \
                    DEPEND(in, dl_ln2[b * T * C + t * C: GRAN_TMP * C], l_residual2[b * T * C + t * C: GRAN_TMP * C]) \
                    DEPEND(in, l_ln2w[0: C], l_ln2_mean[b * T + t: GRAN_TMP], l_ln2_rstd[b * T + t: GRAN_TMP]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C, slh);
                }
            }
        }

#if USE_TARGET
        gpu_residual_backward(dresidual, dl_attproj, dl_residual2, (long)B * T * C);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("residual_backward");
                OMP_TASK( firstprivate(b,t) \
                    DEPEND(inout, dresidual[b * T * C + t * C: GRAN_TMP * C]) \
                    DEPEND(inout, dl_attproj[b * T * C + t * C: GRAN_TMP * C]) \
                    DEPEND(in, dl_residual2[b * T * C + t * C: GRAN_TMP * C]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    residual_backward(dresidual, dl_attproj, dl_residual2, B, T, C, slh);
                }
            }
        }
#endif
#if USE_TARGET
        gpu_matmul_params_backward(dl_attprojw, dl_attprojb, dl_attproj, l_atty, B, T, C, C);
#else
        for (int o = 0; o < C; o += OC_GRANULARITY)
        {
            // OMPT_SET_LABEL("matmul_params_backward");
            OMP_TASK( firstprivate(o) \
                    DEPEND(out, dl_attprojw[o * C: OC_GRANULARITY * C]) \
                    DEPEND(out, dl_attprojb[o : OC_GRANULARITY]) \
                    DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), dl_attproj[b * T * C + t * C: GRAN_TMP * C]) \
                    DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), l_atty[b * T * C + t * C: GRAN_TMP * C]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = o;
                slh.dims_stops[0] = min(o + OC_GRANULARITY, C);
                matmul_params_backward(dl_attprojw, dl_attprojb, dl_attproj, l_atty, B, T, C, C, slh);
            }
        }
#endif

#if USE_TARGET
        gpu_matmul_input_backward(dl_atty, dl_attproj, l_attprojw, B, T, C, C);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("matmul_input_backward");
                OMP_TASK( firstprivate(b, t) \
                    DEPEND(out, dl_atty[b * T * C + t * C : GRAN_TMP * C]) \
                    DEPEND(in, dl_attproj[b * T * C + t * C : GRAN_TMP * C]) \
                    DEPEND(in, l_attprojw[0 : C * C]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    matmul_input_backward(dl_atty, dl_attproj, l_attprojw, B, T, C, C, slh);
                }
            }
        }
#endif

        // dep_matmul_params_backward(matmul_params_backward, dl_fcw, dl_fcb, dl_fch, l_ln2, B, T, C, C4, dep_handler, l_fch);
        // dep_matmul_input_backward(matmul_input_backward, dl_ln2, dl_fch, l_fcw, B, T, C, C4, dep_handler);
        // dep_layernorm_backward(layernorm_backward, dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C,
        //                        dep_handler);
        // dep_residual_backward(residual_backward, dresidual, dl_attproj, dl_residual2, B, T, C, dep_handler);
        // dep_matmul_params_backward(matmul_params_backward, dl_attprojw, dl_attprojb, dl_attproj, l_atty, B, T, C, C, dep_handler, dl_residual2);
        // dep_matmul_input_backward(matmul_input_backward, dl_atty, dl_attproj, l_attprojw, B, T, C, C, dep_handler);

        // Maybe we can simplifi the l_qkv dep

        // The deps on preatt and att are a bit heavy, but needed since the Q and KV passes
        // work forward and backwards in those structures.

        // const int OC3_GRANULARITY = C3 / OC_BACK_SPLIT;
        const int OC3_GRANULARITY = (USE_TARGET ? 1 : C3 / OC_BACK_SPLIT);
        // #pragma omp taskgroup
        // {
#if USE_TARGET
        gpu_attention_backward_Q(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("attention_backward_Q");
                OMP_TASK( firstprivate(b, t) \
                        DEPEND_MULTI(in, (i_t=0:t:GRAN_TMP), l_qkv[b * T * C3 + i_t * C3: GRAN_TMP * C3]) \
                        DEPEND(in, l_att[b * NH * T * T + t * T: NH * T * T * GRAN_TMP]) \
                        DEPEND(inout, dl_atty[b * T * C + t * C: GRAN_TMP * C]) \
                        DEPEND(inout, dl_qkv[b * T * C3 + t * C3: GRAN_TMP * C3]) \
                        DEPEND(inoutset, dl_preatt[b * NH * T * T: NH * T * T], dl_att[b * NH * T * T: NH * T * T]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    attention_backward_Q(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH, slh);
                }
            }
        }
#endif
        // }


        // dep_attention_backward_Q(attention_backward_Q, dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH, dep_handler);

#if USE_TARGET
        gpu_attention_backward_KV(dl_qkv, dl_preatt, dl_atty, l_qkv, l_att, B, T, C, NH);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("attention_backward_KV");
                OMP_TASK( firstprivate(b, t) \
                    DEPEND(inout, dl_qkv[b * T * C3 + t * C3 : GRAN_TMP * C3]) \
                    DEPEND(in, dl_atty[b * T * C + t * C : GRAN_TMP * C], l_qkv[b * T * C3 + t * C3 : GRAN_TMP * C3]) \
                    DEPEND(in, dl_preatt[b * NH * T * T : NH * T * T], dl_att[b * NH * T * T : NH * T * T]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    attention_backward_KV(dl_qkv, dl_preatt, dl_atty, l_qkv, l_att, B, T, C, NH, slh);
                }
            }
        }
#endif


        // dep_attention_backward_KV(attention_backward_KV, dl_qkv, dl_preatt, dl_atty, l_qkv, l_att, B, T, C, NH, dep_handler);

#if USE_TARGET
        gpu_matmul_params_backward(dl_qkvw, dl_qkvb, dl_qkv, l_ln1, B, T, C, C3);
#else
        for (int o = 0; o < C3; o += OC3_GRANULARITY)
        {
            // OMPT_SET_LABEL("matmul_params_backward");
            OMP_TASK( firstprivate(o) \
                DEPEND(out, dl_qkvw[o * C3 : OC3_GRANULARITY * C3], dl_qkvb[o : OC3_GRANULARITY]) \
                DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), dl_qkv[b * T * C3 + t * C3: GRAN_TMP * C3]) \
                DEPEND_MULTI(in, (b=0:B, t=0:T:GRAN_TMP), l_ln1[b * T * C + t * C: GRAN_TMP * C]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = o;
                slh.dims_stops[0] = min(o + OC3_GRANULARITY, C3);
                matmul_params_backward(dl_qkvw, dl_qkvb, dl_qkv, l_ln1, B, T, C, C3, slh);
            }
        }
#endif
#if USE_TARGET
        gpu_matmul_input_backward(dl_ln1, dl_qkv, l_qkvw, B, T, C, C3);
#else
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("matmul_input_backward");
                OMP_TASK( firstprivate(b, t) \
                    DEPEND(inout, dl_ln1[b * T * C + t * C : GRAN_TMP * C]) \
                    DEPEND(in, dl_qkv[b * T * C3 + t * C3 : GRAN_TMP * C3], l_qkvw[0 : C3 * C]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    matmul_input_backward(dl_ln1, dl_qkv, l_qkvw, B, T, C, C3, slh);
                }
            }
        }
#endif

        OMP_TARGET_LOOP_TASK(collapse(2) \
                DEPEND(in, dl_ln1[0], residual[0], l_ln1w[0], l_ln1_mean[0], l_ln1_rstd[0]) \
                DEPEND(inout, dresidual[0], dl_ln1w[0], dl_ln1b[0]))
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t += GRAN_TMP)
            {
                // OMPT_SET_LABEL("layernorm_backward");
                OMP_TASK( firstprivate(b, t)                                                                 \
                    DEPEND(inout, dresidual[b * T * C + t * C: GRAN_TMP * C]) \
                    DEPEND(inoutset, dl_ln1w[0:C], dl_ln1b[0:C])      \
                    DEPEND(in, dl_ln1[b * T * C + t * C: GRAN_TMP * C], \
                            residual[b * T * C + t * C: GRAN_TMP * C], \
                            l_ln1w[0: C], \
                            l_ln1_mean[b * T + t: GRAN_TMP], \
                            l_ln1_rstd[b * T + t: GRAN_TMP]))
                {
                    struct SliceHandler slh;
                    slh.dims_starts[0] = slh.dims_starts[1] = 0;
                    slh.b_start = b;
                    slh.b_end = b + 1;
                    slh.dims_stops[0] = 1;
                    slh.t_start = t;
                    slh.t_end = slh.t_start + GRAN_TMP;
                    slh.dims_stops[1] = GRAN_TMP;
                    layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C, slh);
                }
            }
        }
    }

    // int o=0:Vp:VP_GRANULARITY
    OMP_TARGET_LOOP_TASK(collapse(2) \
            DEPEND(in, grads_acts_encoded[0], model_inputs[0]) \
            DEPEND(inout, grads_wte[0], grads_wpe[0]))
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t += GRAN_TMP)
        {
            // OMPT_SET_LABEL("encoder_backward");
            OMP_TASK( firstprivate(b, t)            \
                DEPEND(inoutset, grads_wte[0: C*T], grads_wpe[t*C: GRAN_TMP * C])  \
                DEPEND(in, grads_acts_encoded[b*T + t : GRAN_TMP], model_inputs[b * T * C + t * C : GRAN_TMP * C]))
            {
                struct SliceHandler slh;
                slh.dims_starts[0] = slh.dims_starts[1] = 0;
                slh.b_start = b;
                slh.b_end = b + 1;
                slh.dims_stops[0] = 1;
                slh.t_start = t;
                slh.t_end = slh.t_start + GRAN_TMP;
                slh.dims_stops[1] = GRAN_TMP;
                encoder_backward(grads_wte, grads_wpe, grads_acts_encoded, model_inputs, B, T, C, slh);
            }
        }
    }
}


// Update function for distributed implementation
// Updates a slice of parameters
void gpt2_update_slice_iter(
    float *params, float *grads, float *m_memory, float *v_memory,
    size_t size, int params_subpartition, int params_subpartition_bw, int grads_subpartition,
    float learning_rate, float beta1, float beta2, float eps, float weight_decay, int *step)
{
    // Update
    // OMPT_SET_LABEL("update");
    // On the device this whole slice is a single offloaded kernel; the slice
    // bases (grads[0], params[0], m/v_memory[0]) match the per-tensor bases used
    // by the backward accumulators / next forward, so the ordering holds.
    // On the device we run one thread per parameter (step by 1); on the host we
    // keep the coarse task tiling. (Without this the device launched ~10 threads,
    // each doing millions of serial Adam updates -- the step-1 bottleneck.)
    size_t kstep = (USE_TARGET ? (size_t)1 : (size_t)grads_subpartition);
    OMP_TARGET_LOOP_TASK(DEPEND(in, grads[0], step[0]) \
            DEPEND(inout, params[0], m_memory[0], v_memory[0]))
    for (size_t k = 0; k < size; k += kstep)
    {
        size_t len = (USE_TARGET ? (size_t)1 : ((k + (size_t)grads_subpartition <= size) ? (size_t)grads_subpartition : (size - k)));
        OMP_TASK( \
            firstprivate(params, grads, m_memory, v_memory, k, len, \
                        learning_rate, beta1, beta2, eps, weight_decay, step) \
            DEPEND(in, grads[k : len], step[0]) \
            DEPEND(inout, params[k : len], m_memory[k : len], v_memory[k : len]))
        {

            float bias_1 = (1.0 - powf(beta1, *step));
            float bias_2 = (1.0 - powf(beta2, *step));

            for (size_t j = k; j < k + len; ++j)
            {
                float param = params[j];
                float grad  = grads[j];
                float m = beta1 * m_memory[j] + (1.0f - beta1) * grad;
                float v = beta2 * v_memory[j] + (1.0f - beta2) * grad * grad;
                m_memory[j] = m;
                v_memory[j] = v;
                params[j] -= learning_rate * ((m / bias_1) / (sqrtf(v / bias_2) + eps) + weight_decay * param);
            }
        }
    }

    // #pragma omp task                                                                                            \
    //     depend(inout: m_memory[0], v_memory[0])                                                                 \
    //     depend(iterator(k=0:size:params_subpartition_bw), inout: params[k: params_subpartition_bw])               \
    //     depend(iterator(k=0:size:params_subpartition),    inout: params[k: params_subpartition])                  \
    //     depend(iterator(k=0:size:grads_subpartition),     in: grads[k: grads_subpartition])                     \
    //     depend(in: step[0])                                                                                     \
    //     firstprivate(params, grads, size, m_memory, v_memory, learning_rate, beta1, beta2, eps, weight_decay)
    // {
    //     for (size_t j = 0; j < size; ++j)
    //     {
    //         float param = params[j];
    //         float grad = grads[j];

    //         // update the first moment (momentum)
    //         float m = beta1 * m_memory[j] + momentum_1 * grad;
    //         // update the second moment (RMSprop)
    //         float v = beta2 * v_memory[j] + momentum_2 * grad * grad;
    //         // bias-correct both moments
    //         float m_hat = m / bias_1;
    //         float v_hat = v / bias_2;

    //         // update
    //         m_memory[j] = m;
    //         v_memory[j] = v;
    //         params[j] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    //     }
    // }
}


// This isn't called within train_gpt2, and test_gpt2.
void gpt2_update_slice(void *_model, int slice_offset, int adam_offset, int size, float learning_rate, float beta1, float beta2, float eps,
                       float weight_decay, int *t, int dependency_offset, int *conc_dep) {
    GPT2 *model = (GPT2 *)_model;
    float *grads_slice_ptr = model->grads_memory + slice_offset;
    float *params_slice_ptr = model->params_memory + slice_offset;
    float *m_memory_ptr = model->m_memory + adam_offset;
    float *v_memory_ptr = model->v_memory + adam_offset;
    float *params_dependency_ptr = model->params_memory + dependency_offset;
    float *grads_dependency_ptr = model->grads_memory + dependency_offset;
    // OSS_TASK(task label("update_slice") firstprivate(model, grads_slice_ptr, params_slice_ptr, m_memory_ptr, v_memory_ptr, params_dependency_ptr, grads_dependency_ptr,
    //                            slice_offset, adam_offset, size, learning_rate, beta1, beta2, eps, weight_decay, t, dependency_offset, conc_dep)
    //              in(*params_dependency_ptr, *grads_dependency_ptr) concurrent(*t, *conc_dep)) 

    OMP_TASK( \
            firstprivate(model, grads_slice_ptr, params_slice_ptr,      \
                m_memory_ptr, v_memory_ptr, params_dependency_ptr,      \
                grads_dependency_ptr, slice_offset, adam_offset, size,  \
                learning_rate, beta1, beta2, eps, weight_decay, t,      \
                dependency_offset, conc_dep)                            \
            DEPEND(in, *params_dependency_ptr, *grads_dependency_ptr)   \
            DEPEND(inoutset, *t, *conc_dep))
    {
        for (size_t i = 0; i < size; i++) {

            float param = params_slice_ptr[i];
            float grad = grads_slice_ptr[i];

            // update the first moment (momentum)
            float m = beta1 * m_memory_ptr[i] + (1.0f - beta1) * grad;
            // update the second moment (RMSprop)
            float v = beta2 * v_memory_ptr[i] + (1.0f - beta2) * grad * grad;
            // bias-correct both moments
            float m_hat = m / (1.0f - powf(beta1, *t));
            float v_hat = v / (1.0f - powf(beta2, *t));

            // update
            m_memory_ptr[i] = m;
            v_memory_ptr[i] = v;
            params_slice_ptr[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
        }
    }
}


// current model's memory is following the layers names.
// we will fill ptrs_out with pointers toward the model's memory,
// in the order in which the activation memory is used during the backward pass
// subgrads_sizes will contain the sizes of each slice
int grads_array_reshape(GPT2 *model, float **ptrs_out, int *subgrads_sizes) {
    int ptrs_count = 0;
    int sizes_count = 0;

    size_t maxT = model->config.max_seq_len;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t C = model->config.channels;
    size_t C3 = C * 3;
    size_t C4 = C * 4;

    // backward pass: go in the reverse order of the forward pass, and call
    // backward() functions
    ParameterTensors grads = model->grads;

    ptrs_out[ptrs_count++] = grads.lnfw; // lnfw
    subgrads_sizes[sizes_count++] = C;
    ptrs_out[ptrs_count++] = grads.lnfb; // lnfb
    subgrads_sizes[sizes_count++] = C;

    for (int l = L - 1; l >= 0; l--) {

        // get the pointers of the gradients of the weights for this layer
        ptrs_out[ptrs_count++] = grads.fcprojb + l * C; // fcprojb
        subgrads_sizes[sizes_count++] = C;
        ptrs_out[ptrs_count++] = grads.fcprojw + l * C * C4; // fcprojw
        subgrads_sizes[sizes_count++] = C * C4;
        ptrs_out[ptrs_count++] = grads.fcb + l * C4; // fcb
        subgrads_sizes[sizes_count++] = C4;
        ptrs_out[ptrs_count++] = grads.fcw + l * C4 * C; // fcw
        subgrads_sizes[sizes_count++] = C4 * C;
        ptrs_out[ptrs_count++] = grads.ln2b + l * C; // ln2b
        subgrads_sizes[sizes_count++] = C;
        ptrs_out[ptrs_count++] = grads.ln2w + l * C; // ln2w
        subgrads_sizes[sizes_count++] = C;
        ptrs_out[ptrs_count++] = grads.attprojb + l * C; // attprojb
        subgrads_sizes[sizes_count++] = C;
        ptrs_out[ptrs_count++] = grads.attprojw + l * C * C; // attprojw
        subgrads_sizes[sizes_count++] = C * C;
        ptrs_out[ptrs_count++] = grads.qkvb + l * C3; // qkvb
        subgrads_sizes[sizes_count++] = C3;
        ptrs_out[ptrs_count++] = grads.qkvw + l * C3 * C; // qkvw
        subgrads_sizes[sizes_count++] = C3 * C;
        ptrs_out[ptrs_count++] = grads.ln1b + l * C; // ln1b
        subgrads_sizes[sizes_count++] = C;
        ptrs_out[ptrs_count++] = grads.ln1w + l * C; // ln1w
        subgrads_sizes[sizes_count++] = C;
    }

    ptrs_out[ptrs_count++] = grads.wpe; // wpe
    subgrads_sizes[sizes_count++] = maxT * C;
    ptrs_out[ptrs_count++] = grads.wte; // wte
    subgrads_sizes[sizes_count++] = Vp * C;

    size_t sum_sizes = 0;
    for (int i = 0; i < sizes_count; i++) {
        sum_sizes += subgrads_sizes[i];
    }
    assert(ptrs_count == sizes_count);
    assert(sum_sizes == model->num_parameters);

    return ptrs_count;
}


// Classic update function for non-distributed implementations
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int *t)
{
    // reference:
    // https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float *)mallocCheck(model->num_parameters * sizeof(float));
        model->v_memory = (float *)mallocCheck(model->num_parameters * sizeof(float));
    }

    int B = model->batch_size;
    int T = model->seq_len;
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;
    int C3 = C * 3;
    int C4 = C * 4;

    int grads_partitioning[] = {1, 1, L, L, L, L, L, L, L, L, L, L, L, L, 1, 1};
    int grads_subpartition[] = {
        /*wte*/ (Vp / OC_BACK_SPLIT) * C,
        /*wpe*/ GRAN_TMP * C,
        /*ln1w*/ C,
        /*ln1b*/ C,
        /*qkvw*/ (C3 / OC_BACK_SPLIT) * C3,
        /*qkvb*/ (C3 / OC_BACK_SPLIT),
        /*attprojw*/ (C / OC_BACK_SPLIT) * C,
        /*attprojb*/ (C / OC_BACK_SPLIT),
        /*ln2w*/ C,
        /*ln2b*/ C,
        /*fcw*/ (C4 / OC_BACK_SPLIT) * C4,
        /*fcb*/ (C4 / OC_BACK_SPLIT),
        /*fcprojw*/ (C / OC_BACK_SPLIT) * C,
        /*fcprojb*/ (C / OC_BACK_SPLIT),
        /*lnfw*/ C,
        /*lnfb*/ C,
    };

    int params_subpartition[] = {
        /*wte*/ Vp * C,
        /*wpe*/ GRAN_TMP * C,
        /*ln1w*/ C,
        /*ln1b*/ C,
        /*qkvw*/ (C3 / OC_SPLIT) * C,
        /*qkvb*/ (C3 / OC_SPLIT),
        /*attprojw*/ (C / OC_SPLIT) * C,
        /*attprojb*/ (C / OC_SPLIT),
        /*ln2w*/ C,
        /*ln2b*/ C,
        /*fcw*/ (C4 / OC_SPLIT) * C,
        /*fcb*/ (C4 / OC_SPLIT),
        /*fcprojw*/ (C / OC_SPLIT) * C4,
        /*fcprojb*/ (C / OC_SPLIT),
        /*lnfw*/ C,
        /*lnfb*/ C,
    };

    int params_subpartition_backward[] = {
        /*wte*/ Vp * C,
        /*wpe*/ GRAN_TMP * C,
        /*ln1w*/ C,
        /*ln1b*/ C,
        /*qkvw*/ C3 * C,
        /*qkvb*/ C3,
        /*attprojw*/ C * C,
        /*attprojb*/ C,
        /*ln2w*/ C,
        /*ln2b*/ C,
        /*fcw*/ C4 * C,
        /*fcb*/ C4,
        /*fcprojw*/ C * C4,
        /*fcprojb*/ C,
        /*lnfw*/ C,
        /*lnfb*/ C,
    };



    float *params_ptr = model->params_memory;
    float *grads_ptr = model->grads_memory;
    size_t offset = 0;

    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        for (int l = 0; l < grads_partitioning[i]; l++)
        {
            size_t memory_size = model->param_sizes[i] / ((size_t)grads_partitioning[i]);
            int params_sub = params_subpartition[i];
            int params_sub_bw = params_subpartition_backward[i];
            int grads_sub = grads_subpartition[i];

            gpt2_update_slice_iter(
                params_ptr, grads_ptr, &model->m_memory[offset], &model->v_memory[offset],
                memory_size, params_sub, params_sub_bw, grads_sub,
                learning_rate, beta1, beta2, eps, weight_decay, t
            );

            params_ptr += memory_size;
            grads_ptr += memory_size;
            offset += memory_size;
        }
    }

    // OMPT_SET_LABEL("t++"); (loop-less: a plain target region on the device so
    // the step counter *t stays resident on the device with the optimizer state)
    OMP_TARGET_TASK( DEPEND(inout, t[0:1], model))
    {
        (*t)++;
    }
}

// // General update function for distributed implementations
// // Performs communications and parameters update
// void gpt2_update(GPT2 *model, MPIWorker worker, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int *t) {
//     if (!mpi_is_comm_initialized(worker)) { // Lazy initialization
//         int buff_max_size = 150;            // has to be lower than the number of parameters subslices. We could set it to NUM_PARAMETERS_TENSORS*L
//         float *subgrads_ptrs[buff_max_size];
//         int subgrads_sizes[buff_max_size];
//         int subgrads_count = grads_array_reshape(model, subgrads_ptrs, subgrads_sizes);

//         // This value is more related to dependencies than the blocksize we send
//         // const int communication_block_size = 1900000; // Adjusted for 4 ranks on 56 CPUs, B=4, T=64
//         mpi_init_communications(worker, NUM_PARAMETER_TENSORS * model->config.num_layers);
//     }

//     int B = model->batch_size;
//     int T = model->seq_len;
//     int V = model->config.vocab_size;
//     int Vp = model->config.padded_vocab_size;
//     int L = model->config.num_layers;
//     int NH = model->config.num_heads;
//     int C = model->config.channels;
//     int C3 = C * 3;
//     int C4 = C * 4;

//     int grads_partitioning[] = {1, 1, L, L, L, L, L, L, L, L, L, L, L, L, 1, 1};
//     int grads_subpartition[] = {
//         /*wte*/ (Vp / OC_BACK_SPLIT) * C,
//         /*wpe*/ GRAN_TMP * C,
//         /*ln1w*/ C,
//         /*ln1b*/ C,
//         /*qkvw*/ (C3 / OC_BACK_SPLIT) * C3,
//         /*qkvb*/ (C3 / OC_BACK_SPLIT),
//         /*attprojw*/ (C / OC_BACK_SPLIT) * C,
//         /*attprojb*/ (C / OC_BACK_SPLIT),
//         /*ln2w*/ C,
//         /*ln2b*/ C,
//         /*fcw*/ (C4 / OC_BACK_SPLIT) * C4,
//         /*fcb*/ (C4 / OC_BACK_SPLIT),
//         /*fcprojw*/ (C / OC_BACK_SPLIT) * C,
//         /*fcprojb*/ (C / OC_BACK_SPLIT),
//         /*lnfw*/ C,
//         /*lnfb*/ C,
//     };

//     int params_subpartition[] = {
//         /*wte*/ Vp * C,
//         /*wpe*/ GRAN_TMP * C,
//         /*ln1w*/ C,
//         /*ln1b*/ C,
//         /*qkvw*/ (C3 / OC_SPLIT) * C,
//         /*qkvb*/ (C3 / OC_SPLIT),
//         /*attprojw*/ (C / OC_SPLIT) * C,
//         /*attprojb*/ (C / OC_SPLIT),
//         /*ln2w*/ C,
//         /*ln2b*/ C,
//         /*fcw*/ (C4 / OC_SPLIT) * C4,
//         /*fcb*/ (C4 / OC_SPLIT),
//         /*fcprojw*/ (C / OC_SPLIT) * C4,
//         /*fcprojb*/ (C / OC_SPLIT),
//         /*lnfw*/ C,
//         /*lnfb*/ C,
//     };

//     int params_subpartition_backward[] = {
//         /*wte*/ Vp * C,
//         /*wpe*/ GRAN_TMP * C,
//         /*ln1w*/ C,
//         /*ln1b*/ C,
//         /*qkvw*/ C3 * C,
//         /*qkvb*/ C3,
//         /*attprojw*/ C * C,
//         /*attprojb*/ C,
//         /*ln2w*/ C,
//         /*ln2b*/ C,
//         /*fcw*/ C4 * C,
//         /*fcb*/ C4,
//         /*fcprojw*/ C * C4,
//         /*fcprojb*/ C,
//         /*lnfw*/ C,
//         /*lnfb*/ C,
//     };

//     if (model->m_memory == NULL) {
//         // size_t adam_memory_size = mpi_get_adam_buffer_size(worker);
//         // MMAP the needed memory, with an idea of it not being allocated if not touched.
//         model->m_memory = mmap(NULL, model->num_parameters * sizeof(float), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
//         model->v_memory = mmap(NULL, model->num_parameters * sizeof(float), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
//     }

//     float *params_ptr = model->params_memory;
//     float *grads_ptr = model->grads_memory;
//     size_t offset = 0;

//     // First, reduce the gradients. The partition is simpler than the original, since
//     // each node will be the root for a different parameter, instead of having different
//     // partitions for MPI

//     int globalId = 0;
//     int worldSize = mpi_get_worldsize(worker);
//     int rank = mpi_get_rank(worker);

//     for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
//         for (int l = 0; l < grads_partitioning[i]; l++) {
//             int rootOfCurrentBlock = globalId % worldSize;
//             size_t memory_size = model->param_sizes[i] / ((size_t)grads_partitioning[i]);
//             int params_sub = params_subpartition[i];
//             int params_sub_bw = params_subpartition_backward[i];
//             int grads_sub = grads_subpartition[i];

//             // Communicators
//             // #pragma oss task label("mpi_reduce") \
//             //     inout(params_subpartition[globalId]) \
//             //     inout({grads_ptr[k; grads_sub], k=0;memory_size:grads_sub}) in(*t)

//             // Communicators
//             #pragma omp task \
//                 depend(inout: params_subpartition[globalId], {grads_ptr[k; grads_sub], k=0;memory_size:grads_sub}) \
//                 depend(in: *t)
//             {
//                 // fprintf(stderr, "Rank %d Block Reduce %d Root %d\n", rank, globalId, rootOfCurrentBlock);
//                 mpi_reduce_block(worker, grads_ptr, memory_size, rootOfCurrentBlock, globalId);
//             }

//             if (rank == rootOfCurrentBlock) {
//                 gpt2_update_slice_iter(
//                     params_ptr, grads_ptr, &model->m_memory[offset], &model->v_memory[offset],
//                     memory_size, params_sub, params_sub_bw, grads_sub,
//                     learning_rate, beta1, beta2, eps, weight_decay, t
//                 );

//                 offset += memory_size;
//             }

//             // #pragma oss task label("mpi_broadcast") inout(params_subpartition[globalId]) \
//                 inout({params_ptr[k; params_sub], k=0;memory_size:params_sub}) \
//                 inout({params_ptr[k; params_sub_bw], k=0;memory_size:params_sub_bw}) in(*t)

//             // mpi_broadcast
//             #pragma omp task \
//                 depend(inout: params_subpartition[globalId], {params_ptr[k; params_sub], k=0;memory_size:params_sub}, {params_ptr[k; params_sub_bw], k=0;memory_size:params_sub_bw}) \
//                 depend(in: *t)
//             {
//                 // fprintf(stderr, "Rank %d Block BCAST %d Root %d\n", rank, globalId, rootOfCurrentBlock);
//                 mpi_bcast_block(worker, params_ptr, memory_size, rootOfCurrentBlock, globalId); // Broadcast the slice
//             }

//             params_ptr += memory_size;
//             grads_ptr += memory_size;

//             globalId++;
//         }
//     }

//     // mpi_reduce(worker, slice_ptr, slice_length, block->root, block->block_slices_comm + block_slice_id);

//     // mpi_share_gradients(model->grads_memory, worker);                                                             // Reduce the gradients
//     // mpi_update_params_slice(model, gpt2_update_slice, worker, learning_rate, beta1, beta2, eps, weight_decay, t); // Update parameters
//     // mpi_broadcast_parameters(model->params_memory, worker);                                                       // Broadcast parameters

//     OSS_TASK(task label("training_step") inout(*t) firstprivate(t))
//     {
//         // fprintf(stderr, "Comms DONE\n", NULL);
//         (*t)++;
//     } // training step += 1

// }

// Rank 0 loads next batch and scatter the inputs and targets among all the ranks
// If B=16 and worldsize=4, rank i will get tokens from token sequence 'rank*B/worldsize' to '(rank+1)*B/worldsize'
void dataloader_next_scattered_batch(DataLoader *loader, MPIWorker worker)
{
    // Broadcasting B and T is useless for now. Maybe remove it later
    dataloader_next_batch(loader);
}

void gpt2_free(GPT2 *model) {
    util_free(model->params_memory, model->num_parameters * sizeof(float));
    util_free(model->grads_memory, model->num_parameters * sizeof(float));

    util_free(model->m_memory, model->num_parameters * sizeof(float));
    util_free(model->v_memory, model->num_parameters * sizeof(float));

    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += model->act_sizes[i];
    }

    util_free(model->acts_memory, num_activations * sizeof(float));
    util_free(model->grads_acts_memory, num_activations * sizeof(float));
    util_free(model->inputs, BATCH_SIZE * SEQUENCE_SIZE * sizeof(int));
    util_free(model->targets, BATCH_SIZE * SEQUENCE_SIZE * sizeof(int));
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
int main(int argc, char *argv[])
{
    TIMERS_INIT(2); // applicaiton, iteration
    TICK(0);

    // build the GPT-2 model from a checkpoint
    setvbuf(stdout, NULL, _IOFBF, 0);
    GPT2 model;
    DataLoader train_loader, val_loader;
    Tokenizer tokenizer;
    MPIWorker worker;
    double runtimes[NB_STEPS] = {0.};
    double tok_s[NB_STEPS] = {0.};
    unsigned int cpu_count = GET_NUM_CPUS();
    unsigned int mpi_cpu_count;

    int B = BATCH_SIZE;    // batch size 4 (i.e. 4 independent token sequences will be
                           // trained on)
    int T = SEQUENCE_SIZE; // sequence length 64 (i.e. each sequence is 64 tokens long).
                           // must be <= maxT, which is 1024 for GPT-2
    worker = mpi_init(&argc, &argv, B, T);
    const int rank = mpi_get_rank(worker);

    int val_num_batches = 5;

    // some memory for generating samples from the model
    unsigned long long rng_state = 1337;
    int *gen_tokens = (int *)mallocCheck(B * T * sizeof(int));
    const int genT = 64; // number of steps of inference we will do
    assert(T % MT_SUBSIZE == 0);

    if (rank == 0) {
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        // build the DataLoaders from tokens files.
        const char *tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
        const char *tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
        const char *tiny_stories[2] = {tiny_stories_train, tiny_stories_val};
        const char *tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
        const char *tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
        const char *tiny_shakespeare[2] = {tiny_shakespeare_train, tiny_shakespeare_val};
        const char *sets[2];
        for (int set_id = 0; set_id < 2; set_id++) { // training & validation set
            switch (TRAINING_DATASET) {
            case TINYSHAKESPEARE:
                sets[set_id] = tiny_shakespeare[set_id];
                break;
            case TINYSTORIES:
                sets[set_id] = tiny_stories[set_id];
                break;
            default:
                fprintf(stderr, "Wrong training/validation dataset given\n", NULL);
                exit(EXIT_FAILURE);
            }
        }
        if (access(sets[0], F_OK) == -1 || access(sets[1], F_OK) == -1) {
            fprintf(stderr, "Can't access to dataset\n", NULL);
            exit(EXIT_FAILURE);
        }
        const char *train_tokens = sets[0];
        const char *val_tokens = sets[1];

        dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 0);
        dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);

        fprintf(stderr, "train dataset num_batches: %zu\n", train_loader.num_tokens / (B * T));
        fprintf(stderr, "val dataset num_batches: %zu\n", val_loader.num_tokens / (B * T));

        // build the Tokenizer
        tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");
    }

    mpi_cpu_count = cpu_count;

    // Handlers that manage the dependencies for slicing and parameters update
    // Validation and inference have to use a different handler because of some concurrent accesses
    int rank_B = mpi_get_rank_B(worker);
    int rank_T = mpi_get_rank_T(worker);
    int *dep_handler = dep_init(rank_B, rank_T, MB_SUBSIZE, MT_SUBSIZE);
    int *validation_dep_handler = dep_init(rank_B, rank_T, MB_SUBSIZE, MT_SUBSIZE);
    int *inference_dep_handler = dep_init(rank_B, rank_T, MB_SUBSIZE, MT_SUBSIZE);
    dep_set_slice_shape(dep_handler, MB_SUBSIZE, MT_SUBSIZE);
    dep_set_slice_shape(validation_dep_handler, MB_SUBSIZE, MT_SUBSIZE);
    dep_set_slice_shape(inference_dep_handler, MB_SUBSIZE, MT_SUBSIZE);

    // train
    int step = 1;
    TICK(1); // Iteration runtime clock
    fflush(stdout);
    // #pragma omp taskwait

#if USE_TARGET
    // Preallocate every model buffer on the host, then stage it resident on the
    // device. Only the weights (and the first input tokens) need an initial H2D;
    // gradients, Adam moments and activations are allocated device-side and never
    // copied back except the final weights (see the target update / exit data
    // after the loop). Per iteration only the new tokens go H2D and the scalar
    // loss comes D2H (host-fed dataloader + host-printed metrics).
    const size_t dev_B  = (size_t) mpi_get_rank_B(worker);
    const size_t dev_T  = (size_t) mpi_get_rank_T(worker);
    const size_t dev_BT = dev_B * dev_T;
    gpt2_allocate(&model, dev_B, dev_T);
    const size_t dev_Np = model.num_parameters;
    const size_t dev_Na = model.num_activations;
    #pragma omp target enter data                                            \
        map(to:    model.params_memory[0:dev_Np],                           \
                   model.inputs[0:dev_BT], model.targets[0:dev_BT],         \
                   step, model.mean_loss)                                    \
        map(alloc: model.grads_memory[0:dev_Np],                            \
                   model.m_memory[0:dev_Np], model.v_memory[0:dev_Np],      \
                   model.acts_memory[0:dev_Na],                             \
                   model.grads_acts_memory[0:dev_Na])
#endif

    double start = WALL_TIME();

    // OmpSs-2 tasks are created from the implicit main task (no parallel/single
    // region); an explicit taskwait after the loop provides the barrier instead.
#if !USE_OMPSS
    # pragma omp parallel
    # pragma omp single
#endif
    {
        // loss: How much an guess was wrong
        // gradient: For every weight, how much to change it to make loss smaller

        // OmpSs-2 has no `taskgraph`; `taskiter` applied to the step loop records
        // the per-iteration task graph once and replays it across iterations
        // (the OmpSs equivalent of OpenMP's taskgraph, see train_gpt2_ompss.c).
#if USE_TASKGRAPH && USE_OMPSS
        # pragma oss taskiter label("taskiter_create")                                             \
            shared(model, train_loader, worker, dep_handler, runtimes, tok_s, mpi_cpu_count, B, T, step) \
            firstprivate(rank)
#endif
        for (int taskiter_step = 0; taskiter_step < NB_STEPS; ++taskiter_step)
        {
            // Decide between using a compiler that implements the taskgraph construct
            // Can choose between Julian's LLVM implementation or XKOMP's.
            // (OmpSs-2 uses `oss taskiter` on the loop above instead of taskgraph.)
            #if USE_TASKGRAPH && !USE_OMPSS
            # if USE_XKOMP
            constexpr xkomp_taskgraph_id_t    gid   = 0;
            constexpr xkomp_taskgraph_flags_t flags = XKOMP_TASKGRAPH_FLAG_NONE;
            pragma_omp_taskgraph(gid, flags, [&] (void)
            # else /* USE_XKOMP */
            #  pragma omp taskgraph graph_id(0)
            # endif /* USE_XKOMP */
            #endif /* USE_TASKGRAPH && !USE_OMPSS */
            {
                // dataloader
                // grabs the next chunk of training data (inputs, targets)
                // OMPT_SET_LABEL("Dataloader");
                // Always a host task: it reads the tokens file into host memory.
                // In the target backend gpt2_forward then stages them H2D.
                OMP_HOST_TASK(                                                       \
                    DEPEND_MULTI(out, (i=0:B*T:GRAN_TMP), train_loader.targets[i : GRAN_TMP]) \
                    DEPEND_MULTI(out, (i=0:B*T:GRAN_TMP), train_loader.inputs[i : GRAN_TMP])  \
                    shared(train_loader, worker))
                {
                    dataloader_next_scattered_batch(&train_loader, worker);
                }

                // Have the model make a guess and assign loss
                gpt2_forward(&model, worker, train_loader.inputs, train_loader.targets, mpi_get_rank_B(worker), mpi_get_rank_T(worker), dep_handler);

                // erase the last step's gradients. We will use this to write the new gradient
                gpt2_zero_grad(&model);

                // Compute the gradients
                gpt2_backward(&model, dep_handler);

                // Update the weights with the computed gradients
                gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, &step);

                // Calculate metrics and output to the terminal
                // OMPT_SET_LABEL("metrics");
#if USE_TARGET
                // Bring the scalar loss back to the host (only per-iteration D2H),
                // then print from a host task. Use the host loop counter for
                // indexing/printing since the step counter lives on the device.
                #pragma omp target update from(model.mean_loss) nowait \
                    DEPEND(inout, model.mean_loss)
                OMP_HOST_TASK(DEPEND(in, model.mean_loss) firstprivate(taskiter_step))
                {
                    double time_elapsed_it_s = TOCK(1);
                    double tokens_per_seconds = BATCH_SIZE * SEQUENCE_SIZE / time_elapsed_it_s;
                    runtimes[taskiter_step] = time_elapsed_it_s;
                    tok_s[taskiter_step] = tokens_per_seconds;
                    if (rank == 0)
                        fprintf(stderr, "Step %d :\tIteration runtime : %0.1lf ms, \t\t tokens/s : %0.1lf, \t\t tokens/(s.cpus) : %0.2lf, \t\t Loss : %f \t\t MFU : %0.2f "
                                "%%\n",
                                taskiter_step + 1, time_elapsed_it_s * 1000, tokens_per_seconds, tokens_per_seconds / mpi_cpu_count, model.mean_loss,
                                100 * gpt2_estimate_mfu(&model, B * T, time_elapsed_it_s, mpi_cpu_count));
                    fflush(stdout);
                    TICK(1);
                }
#else
                OMP_TASK(DEPEND(in, model.mean_loss))
                {
                    double time_elapsed_it_s = TOCK(1);
                    double tokens_per_seconds = BATCH_SIZE * SEQUENCE_SIZE / time_elapsed_it_s;
                    runtimes[step - 2] = time_elapsed_it_s;
                    tok_s[step - 2] = tokens_per_seconds;
                    if (rank == 0)
                        fprintf(stderr, "Step %d :\tIteration runtime : %0.1lf ms, \t\t tokens/s : %0.1lf, \t\t tokens/(s.cpus) : %0.2lf, \t\t Loss : %f \t\t MFU : %0.2f "
                                "%%\n",
                                step - 1, time_elapsed_it_s * 1000, tokens_per_seconds, tokens_per_seconds / mpi_cpu_count, model.mean_loss,
                                100 * gpt2_estimate_mfu(&model, B * T, time_elapsed_it_s, mpi_cpu_count));
                    fflush(stdout);
                    TICK(1);
                }
#endif
            }
            # if USE_TASKGRAPH && !USE_OMPSS && USE_XKOMP
            );
            # endif /* USE_TASKGRAPH && !USE_OMPSS && USE_XKOMP */

        } /* taskiter loop */
    } /* single, parallel  */
    // The parallel region's implicit barrier guarantees all target tasks have
    // completed here. Under OmpSs-2 there is no such region, so wait explicitly.
#if USE_OMPSS
    #pragma oss taskwait
#endif

#if USE_TARGET
    // Final D2H: bring the trained weights (and last loss) back, then release
    // all device resident buffers. This is the only D2H of the large buffers.
    #pragma omp target update from(model.params_memory[0:dev_Np], model.mean_loss)
    #pragma omp target exit data                                             \
        map(release: model.params_memory[0:dev_Np],                         \
                     model.grads_memory[0:dev_Np],                          \
                     model.m_memory[0:dev_Np], model.v_memory[0:dev_Np],    \
                     model.acts_memory[0:dev_Na],                           \
                     model.grads_acts_memory[0:dev_Na],                     \
                     model.inputs[0:dev_BT], model.targets[0:dev_BT],       \
                     step, model.mean_loss)
#endif

    double end = WALL_TIME();
    double total_time = end - start;
    printf("Took %f s\n", total_time);

    // Calculate the metrics
    int average_mask_offset = 2; // Warmup iterations for metrics average calculation
    if (NB_STEPS >= average_mask_offset)
    {
        double average_runtime_s = 0.;
        double average_tok_s = 0.;
        double mpi_average_tok_s;
        double mpi_average_runtime_s;
        for (int i = average_mask_offset; i < NB_STEPS; i++) {
            average_runtime_s += runtimes[i] / (NB_STEPS - average_mask_offset);
            average_tok_s += tok_s[i] / (NB_STEPS - average_mask_offset);
        }

        mpi_average_runtime_s = average_runtime_s;
        mpi_average_tok_s = average_tok_s;

        if (rank == 0)
        {
            const int worldsize = mpi_get_worldsize(worker);
            fprintf(stderr, "\n\nAverage runtime per iteration : \t%0.1lf ms\n", mpi_average_runtime_s * 1000 / worldsize);
            fprintf(stderr, "Average tokens/s : \t\t\t%0.1lf toks/s\n", mpi_average_tok_s / worldsize);
            fprintf(stderr, "Average tokens/(s.cpus) : \t\t%0.2lf toks/(s.cpus)\n", mpi_average_tok_s / (mpi_cpu_count * worldsize));
            fprintf(stderr, "Average MFU : \t\t\t\t%0.2f %%\n", 100 * gpt2_estimate_mfu(&model, B * T, mpi_average_runtime_s / worldsize, mpi_cpu_count));
        }
    }


    // free
    dep_finish(dep_handler);
    dep_finish(validation_dep_handler);
    dep_finish(inference_dep_handler);
    if (rank == 0)
    {
        dataloader_free(&train_loader);
        dataloader_free(&val_loader);
        tokenizer_free(&tokenizer);
    }
    else
    {
        int rank_B = mpi_get_rank_B(worker);
        int rank_T = mpi_get_rank_T(worker);
        util_free(train_loader.targets, rank_B * rank_T * sizeof(int));
        util_free(train_loader.inputs, rank_B * rank_T * sizeof(int));
        util_free(val_loader.targets, rank_B * rank_T * sizeof(int));
        util_free(val_loader.inputs, rank_B * rank_T * sizeof(int));
    }
    gpt2_free(&model);
    util_free(gen_tokens, B * T * sizeof(int));

    const double application_time = TOCK(0);
    if (rank == 0)
        fprintf(stderr, "Application total runtime : \t\t%0.1f ms\n\n", application_time * 1000);

    fflush(stdout);

    return EXIT_SUCCESS;
}
#endif
