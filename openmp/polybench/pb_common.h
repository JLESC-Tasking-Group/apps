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
 *   - the per-instance timing summary (../tg_report.h, the same lines as
 *     LULESH), FOM and `answer` the evaluate.py harness parses,
 *   - a cheap checksum over the live-out array, printed as the `answer` every
 *     configuration must reproduce.
 *
 * Problem size is compile-time: each kernel .c defines its own default sizes,
 * overridable with -DPB_N=<n> / -DPB_TSTEPS=<t> (and -DPB_TMAX for fdtd). The
 * number of tasks per loop is the runtime knob -nb <n> (default 32) and the
 * timesteps folded into one taskgraph instance is -u <n> (default 1), matching
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
#include "tg_report.h"

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

/* ---- Command line (-i iters, -nb tasks-per-loop, -u unroll, -v) ----------- */
typedef struct {
    int iters;   /* -i : timesteps to run (0 -> use the compiled TSTEPS) */
    int nb;      /* -nb: tasks per loop            */
    int unroll;  /* -u : timesteps folded into one taskgraph instance */
    int verify;  /* -v : print per-array checksums */
} pb_opts;

static inline void pb_parse_args(int argc, char **argv, pb_opts *o)
{
    o->iters  = 0;
    o->nb     = 32;
    o->unroll = 1;
    o->verify = 0;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "-nb") && a + 1 < argc)      o->nb     = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-i") && a + 1 < argc)  o->iters  = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-u") && a + 1 < argc)  o->unroll = atoi(argv[++a]);
        else if (!strcmp(argv[a], "-v"))                  o->verify = 1;
        else if (!strcmp(argv[a], "-h")) {
            printf("usage: %s [-i tsteps] [-nb tasks-per-loop] [-u unroll] [-v]\n", argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "%s: unknown option '%s' (-h for help)\n", argv[0], argv[a]);
            exit(1);
        }
    }
    if (o->nb < 1)     o->nb = 1;
    if (o->unroll < 1) o->unroll = 1;
}

/* Timesteps to run: -i (or the compiled default) rounded down to a whole number
 * of -u instances. Exits if that leaves nothing. */
static inline int pb_tsteps(const pb_opts *o, int compiled_default)
{
    const long t = tg_round_iters(o->iters > 0 ? o->iters : compiled_default, o->unroll);
    if (t <= 0) exit(1);
    return (int) t;
}

/* ---- Timing + FOM + answer summary (parseable by evaluate.py) -------------- */
/* inst_ms[] holds ms per timestep, one entry per taskgraph instance (see
 * tg_report.h). lattice_updates_per_step is the number of stencil point-updates
 * per timestep (for the FOM); answer is the live-out checksum. */
static inline void pb_report(const char *name, int nb,
                             const double *inst_ms, long ninst, long unroll,
                             double elapsed_s, double lattice_updates_per_step,
                             double answer)
{
    printf("app=%s\n", name);
    printf("backend=%s taskgraph=%d\n", tg_backend_name(), USE_TASKGRAPH);
    printf("Num tasks per loop: %d\n", nb);
    const double mean = tg_report_instances(inst_ms, ninst, unroll);
    printf("Elapsed time (s)     : %10.6f\n", elapsed_s);
    /* figure of merit: lattice-site updates per second (steady state). */
    const double fom = (mean > 0.0) ? lattice_updates_per_step / (mean / 1000.0) : 0.0;
    printf("FOM (lup/s)          : %10.8g\n", fom);
    printf("answer (checksum)    : %.15e\n", answer);
}

/* ---- Cheap checksum of a flat array (self-consistency verification) -------- */
static inline double pb_checksum(const DATA_TYPE *a, size_t n)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) s += (double)a[i];
    return s;
}

#endif /* PB_COMMON_H */
