/*
 * kalloc.cpp - implementation of the host memory allocator abstraction.
 */
#include "kalloc.h"
#include "tasking.h" /* for USE_TARGET */

#include <stdio.h>
#include <stdlib.h>

#if USE_TARGET
# if defined(ALLOC_CUDA)
#  include <cuda_runtime.h>
# elif defined(ALLOC_HIP)
#  define __HIP_PLATFORM_AMD__
#  include <hip/hip_runtime.h>
# else
#  include <omp.h>
#  define KR_ALLOC_OMP_PINNED 1
# endif
#endif

#if defined(KR_ALLOC_OMP_PINNED)
/* Lazily-created pinned allocator (all allocations happen before the parallel
 * region, so no synchronization is needed here). */
static omp_allocator_handle_t kr_pinned = omp_null_allocator;

static void kr_pinned_init(void)
{
    if (kr_pinned != omp_null_allocator) return;
    omp_alloctrait_t traits[1];
    traits[0].key   = omp_atk_pinned;
    traits[0].value = omp_atv_true;
    kr_pinned = omp_init_allocator(omp_default_mem_space, 1, traits);
    if (kr_pinned == omp_null_allocator)
        fprintf(stderr, "kalloc: pinned OpenMP allocator unavailable, using malloc\n");
}
#endif

void *kr_alloc(size_t bytes)
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
    kr_pinned_init();
    if (kr_pinned != omp_null_allocator) return omp_alloc(bytes, kr_pinned);
    return malloc(bytes);
# endif
#else
    return malloc(bytes);
#endif
}

void kr_free(void *ptr)
{
    if (!ptr) return;
#if USE_TARGET
# if defined(ALLOC_CUDA)
    cudaFreeHost(ptr);
# elif defined(ALLOC_HIP)
    hipHostFree(ptr);
# else
    if (kr_pinned != omp_null_allocator) omp_free(ptr, kr_pinned);
    else free(ptr);
# endif
#else
    free(ptr);
#endif
}
