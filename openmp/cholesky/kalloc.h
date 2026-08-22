/*
 * kalloc.h - host memory allocator abstraction for the Cholesky app.
 *
 * ch_alloc/ch_free return host-accessible memory:
 *   - CPU build (USE_TARGET == 0): plain malloc/free.
 *   - GPU build (USE_TARGET == 1): page-locked (pinned) host memory, so the
 *     OpenMP target H2D/D2H transfers of the mapped matrix are fast and can
 *     overlap compute. The pinned backend is chosen at compile time:
 *         -DALLOC_CUDA   -> cudaMallocHost / cudaFreeHost
 *         -DALLOC_HIP    -> hipHostMalloc  / hipHostFree
 *         (default)      -> omp_alloc with the omp_atk_pinned trait, which the
 *                          OpenMP runtime backs with CUDA / HIP / Level Zero.
 *
 * Use ch_alloc only for buffers that are mapped onto the device; host-only
 * scratch can keep using plain malloc.
 */
#ifndef CHOLESKY_KALLOC_H
#define CHOLESKY_KALLOC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *ch_alloc(size_t bytes);
void  ch_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* CHOLESKY_KALLOC_H */
