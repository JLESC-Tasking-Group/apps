// xkomp: supported  (taskgraph over CPU tasks: AXPBY with a firstprivate STRUCT)
//
// Aggregate-firstprivate variant of xkomp_taskgraph_axpby_host_local.cc: the
// per-block scale->axpy chain captures a small trivially-copyable struct
// `firstprivate(blk)` (block offset + the alpha/beta scalars) instead of a plain
// scalar. This exercises the packed task-body ABI's aggregate path -- the packed
// kernel memcpys the struct from the buffer, which the individual-parameter
// kernel cannot do (an aggregate-by-value parameter is C-ABI coerced, breaking
// fusion/dedup). Only meaningful under -fopenmp-task-jit-abi=packed (registered
// by xkomp_add_packed_test); the arrays are captured `shared` (SHARED_ADDR). Per
// pass y <- alpha*x + beta*y; validated against a serial host reference.
#include "common.h"

#include <xkomp/xkomp.h>
#include <xkomp/xkomp++.h>

XKRT_NAMESPACE_USE;

#define NB    8
#define BS    64
#define N     (NB * BS)
#define ITERS 5
#define ALPHA 0.7f
#define BETA  0.5f

// trivially-copyable aggregate captured firstprivate (12 bytes: int + 2 floats)
struct blk_t { int off; float alpha; float beta; };

int
main(void)
{
    float x[N];
    float y[N];
    float ref[N];
    for (int i = 0; i < N; ++i) { x[i] = (float) i + 0.1f; y[i] = (float) i + 1.0f; ref[i] = y[i]; }

    for (int it = 0; it < ITERS; ++it)
        for (int i = 0; i < N; ++i) { ref[i] = BETA * ref[i]; ref[i] = ALPHA * x[i] + ref[i]; }

    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            for (int it = 0; it < ITERS; ++it)
            {
                constexpr xkomp_taskgraph_id_t    gid   = 0;
                constexpr xkomp_taskgraph_flags_t flags = XKOMP_TASKGRAPH_FLAG_NONE;

                pragma_omp_taskgraph(gid, flags, [&] (void)
                {
                    for (int b = 0; b < NB; ++b)
                    {
                        const blk_t blk = { b * BS, ALPHA, BETA };

                        // (1) scale: y = beta*y  -- predecessor
                        #pragma omp task depend(out: y[blk.off]) firstprivate(blk) shared(y) default(none)
                        for (int i = 0; i < BS; ++i)
                            y[i + blk.off] = blk.beta * y[i + blk.off];

                        // (2) axpy: y = alpha*x + y  -- after the scale (same &y[blk.off])
                        #pragma omp task depend(out: y[blk.off]) firstprivate(blk) shared(x, y) default(none)
                        for (int i = 0; i < BS; ++i)
                            y[i + blk.off] = blk.alpha * x[i + blk.off] + y[i + blk.off];
                    }
                });
            }
        }
    }

    for (int i = 0; i < N; ++i)
        CHECK_NEAR(y[i], ref[i], 1e-3);

    TEST_PASS();
    return 0;
}
