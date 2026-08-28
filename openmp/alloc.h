/*
 * alloc.h - shared host-memory allocator for the apps/openmp benchmarks
 * (krylov, lulesh, cholesky, MNMGDatalog). Previously duplicated per app as
 * kr_alloc (krylov), ch_alloc (cholesky) and AllocateForTarget (lulesh).
 *
 * host_alloc/host_free return host-accessible memory:
 *   - CPU build (USE_TARGET == 0): plain malloc/free.
 *   - GPU build (USE_TARGET == 1): page-locked (pinned) host memory, so the
 *     OpenMP target H2D/D2H transfers of the mapped buffers are fast and can
 *     overlap compute. The pinned backend is chosen at compile time:
 *         -DALLOC_CUDA   -> cudaMallocHost / cudaFreeHost
 *         -DALLOC_HIP    -> hipHostMalloc  / hipHostFree
 *         (default)      -> omp_alloc with the omp_atk_pinned trait, which the
 *                          OpenMP runtime backs with CUDA / HIP / Level Zero.
 *
 * Use host_alloc only for buffers that are mapped onto the device; host-only
 * scratch can keep using plain malloc. Device-only buffers that are never
 * touched by the host should use omp_target_alloc instead (see MNMGDatalog).
 *
 * Build: add ../alloc.c to the app's sources; ../common.mk already passes -I..
 * (cholesky, which does not include common.mk, sets -I.. itself).
 */
#ifndef OPENMP_ALLOC_H
#define OPENMP_ALLOC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *host_alloc(size_t bytes);
void  host_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif /* OPENMP_ALLOC_H */
