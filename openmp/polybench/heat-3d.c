/*
 * heat-3d - PolyBench 7-point 3D Jacobi stencil on the shared tasking.h backends.
 * One timestep is two ping-pong sweeps (A->B, B->A); each sweep is tiled into
 * -nb blocks over the outer index i (planes of size n*n).
 *
 * Dependency convention (see jacobi-2d.c): block b keys its I/O on the aligned
 * address X[b*RBS*plane]; the 7-point stencil reads neighbour planes i-1/i+1, so
 * a tile names its neighbour blocks b-1/b+1 as `in` deps too. Flat row-major
 * A[(i*n + j)*n + k]; the PolyBench heat-3d spatial N is small (3D), default 120.
 */
#include "pb_common.h"

#ifndef PB_N
# define PB_N 120
#endif
#ifndef PB_TSTEPS
# define PB_TSTEPS 500
#endif

static void init_array(int n, DATA_TYPE *A, DATA_TYPE *B)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                A[(i * n + j) * n + k] = B[(i * n + j) * n + k] =
                    (DATA_TYPE)(i + j + (n - k)) * 10 / n;
}

int main(int argc, char **argv)
{
    pb_opts opts;
    pb_parse_args(argc, argv, &opts);

    const int n      = PB_N;
    const int tsteps = (opts.iters > 0) ? opts.iters : PB_TSTEPS;
    const size_t npts = (size_t)n * n * n;
    const int plane = n * n;

    DATA_TYPE *A = pb_alloc(npts);
    DATA_TYPE *B = pb_alloc(npts);
    init_array(n, A, B);

    const int RBS  = pb_block_size(n, opts.nb);
    const int nblk = (n + RBS - 1) / RBS;

#if USE_SYNC
    const char *backend = "synchronous (blocking)";
#elif USE_TARGET
    const char *backend = "GPU target tasks";
#else
    const char *backend = "CPU tasks";
#endif

    double *iter_times = (double *)malloc(sizeof(double) * (tsteps > 0 ? tsteps : 1));
    double prev_ts, t0, tf;

    OMP_TARGET_ENTER_DATA(MAP(to: A[0:npts], B[0:npts]))

#pragma omp parallel
{
#pragma omp single
{
    t0 = omp_get_wtime();
    prev_ts = t0;
    for (int t = 0; t < tsteps; t++) {
        TASKGRAPH_BEGIN
        {
            for (int b = 0; b < nblk; b++) {
                const int rb = b * RBS < 1 ? 1 : b * RBS;
                const int re = (b + 1) * RBS > n - 1 ? n - 1 : (b + 1) * RBS;
                if (rb >= re) continue;
                const int bm = (b > 0) ? (b - 1) : b;
                const int bp = (b < nblk - 1) ? (b + 1) : b;
                OMP_TILE(DEPEND(in, A[b * RBS * plane], A[bm * RBS * plane], A[bp * RBS * plane])
                         DEPEND(out, B[b * RBS * plane]),
                         MAP(present: A[0:npts], B[0:npts]) num_teams((re - rb)) thread_limit(256),
                         firstprivate(A, B, rb, re, n, plane))
                for (int i = rb; i < re; i++)
                    for (int j = 1; j < n - 1; j++)
                        for (int k = 1; k < n - 1; k++) {
                            const size_t c = (size_t)(i * n + j) * n + k;
                            B[c] = SCALAR_VAL(0.125) * (A[c + plane] - SCALAR_VAL(2.0) * A[c] + A[c - plane])
                                 + SCALAR_VAL(0.125) * (A[c + n]     - SCALAR_VAL(2.0) * A[c] + A[c - n])
                                 + SCALAR_VAL(0.125) * (A[c + 1]     - SCALAR_VAL(2.0) * A[c] + A[c - 1])
                                 + A[c];
                        }
            }
            for (int b = 0; b < nblk; b++) {
                const int rb = b * RBS < 1 ? 1 : b * RBS;
                const int re = (b + 1) * RBS > n - 1 ? n - 1 : (b + 1) * RBS;
                if (rb >= re) continue;
                const int bm = (b > 0) ? (b - 1) : b;
                const int bp = (b < nblk - 1) ? (b + 1) : b;
                OMP_TILE(DEPEND(in, B[b * RBS * plane], B[bm * RBS * plane], B[bp * RBS * plane])
                         DEPEND(out, A[b * RBS * plane]),
                         MAP(present: A[0:npts], B[0:npts]) num_teams((re - rb)) thread_limit(256),
                         firstprivate(A, B, rb, re, n, plane))
                for (int i = rb; i < re; i++)
                    for (int j = 1; j < n - 1; j++)
                        for (int k = 1; k < n - 1; k++) {
                            const size_t c = (size_t)(i * n + j) * n + k;
                            A[c] = SCALAR_VAL(0.125) * (B[c + plane] - SCALAR_VAL(2.0) * B[c] + B[c - plane])
                                 + SCALAR_VAL(0.125) * (B[c + n]     - SCALAR_VAL(2.0) * B[c] + B[c - n])
                                 + SCALAR_VAL(0.125) * (B[c + 1]     - SCALAR_VAL(2.0) * B[c] + B[c - 1])
                                 + B[c];
                        }
            }
        }
        TASKGRAPH_END

        OMP_HOST_TASK(firstprivate(t, iter_times) shared(prev_ts) DEPEND(in, A[0]))
        {
            const double now = omp_get_wtime();
            iter_times[t] = (now - prev_ts) * 1000.0;
            prev_ts = now;
        }
    }
#pragma omp taskwait
    tf = omp_get_wtime();
} /* single */
} /* parallel */

    OMP_TARGET_UPDATE(from(A[0:npts]))
    OMP_TARGET_EXIT_DATA(MAP(release: A[0:npts], B[0:npts]))

    const double lup = (double)(n - 2) * (n - 2) * (n - 2) * 2.0;
    pb_report("heat-3d", USE_TASKGRAPH, backend, opts.nb,
              iter_times, tsteps, tf - t0, lup);
    if (opts.verify)
        printf("checksum             = %.10e\n", pb_checksum(A, npts));

    free(iter_times);
    pb_free(A);
    pb_free(B);
    return 0;
}
