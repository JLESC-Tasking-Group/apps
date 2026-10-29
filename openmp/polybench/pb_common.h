/*
 * pb_common.h - shared harness for the apps/openmp PolyBench stencil ports.
 *
 * The upstream PolyBench common utilities (polybench.h / polybench.c) are not
 * vendored in this tree, and their VLA array-type macros do not play well with
 * OpenMP target map()/depend() clauses (which want flat, pointer-addressable
 * buffers). This header provides a small, self-contained replacement tuned for
 * the tasking.h backends:
 *
 *   - flat row-major arrays allocated through the pinned/target allocator
 *     (so the same buffer is both host-usable and GPU-mappable),
 *   - the per-iteration timing summary + FOM print the evaluate.py harness
 *     parses (iteration 0 / iterations 2.. (avg|stddev), Elapsed time, FOM),
 *   - a cheap checksum over the live-out array for self-consistency checks.
 *
 * Problem size is compile-time: each kernel .c defines its own default sizes,
 * overridable with -DPB_N=<n> / -DPB_TSTEPS=<t> (and -DPB_TMAX for fdtd). The
 * number of tasks per loop is the runtime knob -nb <n> (default 32), matching
 * the other harness apps.
 */
#ifndef PB_COMMON_H
#define PB_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#include "tasking.h"

#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#define SCALAR_VAL(x) ((DATA_TYPE)(x))

/* ---- Target-capable allocator (pinned host memory when USE_TARGET) --------- */
#if USE_TARGET
# ifdef ALLOC_CUDA
#  include <cuda_runtime.h>
# elif defined(ALLOC_HIP)
#  define __HIP_PLATFORM_AMD__
#  include <hip/hip_runtime.h>
# endif
#endif

static inline DATA_TYPE *pb_alloc(size_t n)
{
    DATA_TYPE *p = NULL;
#if USE_TARGET && defined(ALLOC_CUDA)
    cudaMallocHost((void **)&p, sizeof(DATA_TYPE) * n);
#elif USE_TARGET && defined(ALLOC_HIP)
    hipHostMalloc((void **)&p, sizeof(DATA_TYPE) * n, hipHostMallocDefault);
#else
    p = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * n);
#endif
    return p;
}

static inline void pb_free(DATA_TYPE *p)
{
    if (!p) return;
#if USE_TARGET && defined(ALLOC_CUDA)
    cudaFreeHost(p);
#elif USE_TARGET && defined(ALLOC_HIP)
    hipHostFree(p);
#else
    free(p);
#endif
}

/* ---- Block decomposition: nb tiles over a [0,n) range --------------------- */
/* Tile size for a loop of `extent` iterations split into `nb` tasks. */
static inline int pb_block_size(int extent, int nb)
{
    if (nb < 1) nb = 1;
    int bs = extent / nb + (extent % nb != 0);
    return bs < 1 ? 1 : bs;
}

/* ---- Command line (-i iters, -nb tasks-per-loop, -v verify dump) ----------- */
typedef struct {
    int iters;   /* -i : TSTEPS cap for timing (0 -> use the compiled TSTEPS) */
    int nb;      /* -nb: tasks per loop            */
    int verify;  /* -v : print the live-out checksum */
} pb_opts;

static inline void pb_parse_args(int argc, char **argv, pb_opts *o)
{
    o->iters = 0;
    o->nb    = 32;
    o->verify = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-nb") && a + 1 < argc)      o->nb    = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-i") && a + 1 < argc)  o->iters = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-v"))                  o->verify = 1;
        else if (!strcmp(argv[a], "-h")) {
            printf("usage: %s [-i tsteps] [-nb tasks-per-loop] [-v]\n", argv[0]);
            exit(0);
        }
    }
}

/* ---- Per-iteration timing + FOM summary (parseable by evaluate.py) -------- */
/* iter_times[] holds the per-timestep wall-clock (ms). lattice_updates is the
 * number of stencil point-updates per timestep (for the FOM). */
static inline void pb_report(const char *name, int backend_taskgraph,
                             const char *backend, int nb,
                             double *iter_times, int its,
                             double elapsed_s, double lattice_updates_per_step)
{
    printf("app=%s\n", name);
    printf("backend=%s taskgraph=%d\n", backend, backend_taskgraph);
    printf("Num tasks per loop: %d\n", nb);
    printf("Elapsed time         = %10.6f\n", elapsed_s);

    if (its > 0) {
        const int lo  = (its > 2) ? 2 : (its > 1 ? 1 : 0);
        const int cnt = its - lo;
        double mean = 0.0;
        for (int r = lo; r < its; r++) mean += iter_times[r];
        if (cnt > 0) mean /= cnt;
        double var = 0.0;
        for (int r = lo; r < its; r++) { double d = iter_times[r] - mean; var += d * d; }
        const double sd = (cnt > 1) ? sqrt(var / (cnt - 1)) : 0.0;

        printf("iteration 0          : %10.3f ms  (record)\n", iter_times[0]);
        printf("iteration 1          : %10.3f ms  (1st replay)\n",
               (its > 1) ? iter_times[1] : iter_times[0]);
        printf("iterations 2.. (avg)    : %10.3f ms\n", mean);
        printf("iterations 2.. (stddev) : %10.3f ms\n", sd);

        /* figure of merit: lattice-site updates per second (steady state). */
        const double fom = (mean > 0.0) ? lattice_updates_per_step / (mean / 1000.0) : 0.0;
        printf("FOM                  = %10.8g (lup/s)\n", fom);
    }
}

/* ---- Cheap checksum of a flat array (self-consistency verification) -------- */
static inline double pb_checksum(const DATA_TYPE *a, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)a[i];
    return s;
}

#endif /* PB_COMMON_H */
