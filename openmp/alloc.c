/*
 * alloc.c - implementation of the shared host-memory allocator (see alloc.h).
 *
 * Deliberately does NOT include tasking.h: cholesky ships its own divergent
 * local copy, and the only thing needed here is USE_TARGET, which every app
 * passes with -DUSE_TARGET=... (defaulted to 0 below when absent).
 *
 * Compiled as C++ when the app is built with a C++ driver (xkcxx/clang++ treat
 * .c as C++); alloc.h's extern "C" guards keep the linkage consistent either way.
 */
#include "alloc.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef USE_TARGET          /* 0: host CPU tasks     1: GPU target tasks */
# define USE_TARGET 0
#endif

#if USE_TARGET
# if defined(ALLOC_CUDA)
#  include <cuda_runtime.h>
# elif defined(ALLOC_HIP)
#  define __HIP_PLATFORM_AMD__
#  include <hip/hip_runtime.h>
# else
#  include <omp.h>
#  define HOST_ALLOC_OMP_PINNED 1
# endif
#endif

#if defined(HOST_ALLOC_OMP_PINNED)
/* Lazily-created pinned allocator (all allocations happen before the parallel
 * region, so no synchronization is needed here). */
static omp_allocator_handle_t host_pinned = omp_null_allocator;

static void host_pinned_init(void)
{
    if (host_pinned != omp_null_allocator) return;
    omp_alloctrait_t traits[1];
    traits[0].key   = omp_atk_pinned;
    traits[0].value = omp_atv_true;
    host_pinned = omp_init_allocator(omp_default_mem_space, 1, traits);
    if (host_pinned == omp_null_allocator)
        fprintf(stderr, "alloc: pinned OpenMP allocator unavailable, using malloc\n");
}
#endif

void *host_alloc(size_t bytes)
{
#if USE_TARGET
# if defined(ALLOC_CUDA)
    void *p = NULL;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) p = NULL;
    return p;
# elif defined(ALLOC_HIP)
    void *p = NULL;
    if (hipHostMalloc(&p, bytes, hipHostMallocDefault) != hipSuccess) p = NULL;
    return p;
# else
    host_pinned_init();
    if (host_pinned != omp_null_allocator) return omp_alloc(bytes, host_pinned);
    return malloc(bytes);
# endif
#else
    return malloc(bytes);
#endif
}

void host_free(void *ptr)
{
    if (!ptr) return;
#if USE_TARGET
# if defined(ALLOC_CUDA)
    cudaFreeHost(ptr);
# elif defined(ALLOC_HIP)
    hipHostFree(ptr);
# else
    if (host_pinned != omp_null_allocator) omp_free(ptr, host_pinned);
    else free(ptr);
# endif
#else
    free(ptr);
#endif
}
